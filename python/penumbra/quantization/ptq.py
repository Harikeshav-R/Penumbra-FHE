"""Per-layer weight/bias quantizers — the reusable form of the examples' inline PTQ math.

The Phase-2/4 examples (``examples/mnist/train_quantize_export.py``, ``cnn_export.py``) quantize
each layer by hand. This module lifts that math into functions a model adapter can call, so a
new use case never re-derives a scale (``PROJECT.md`` §8, §12). It owns the **weight and bias**
quantizers plus the requant ``(mult, shift, round_bias)`` calibration that approximates a real
scale ratio for the generalized ``Requant`` op.

The one subtlety worth internalizing is the **bias convention**, which depends on what the layer
consumes:

* **Linear/Conv directly on encrypted, scaled input** (the common case). The accumulator
  ``sum(w_q * x_q)`` lives in units of ``input_scale * weight_scale`` — call it ``acc_scale``.
  A plaintext bias added into that accumulator must be in the same units, so
  ``b_q = round(b_f / acc_scale)``. This is :func:`quantize_linear` / :func:`quantize_conv`'s
  default (mirrors ``train_quantize_export.py``).

* **Linear on already-integer features** (e.g. a head reading the *integer* pooled activations of
  an earlier layer, whose effective input scale is 1). Then ``acc_scale == weight_scale``, so the
  bias shares the weight scale: ``b_q = round(b_f / weight_scale)``. This is
  :func:`quantize_linear_integer_input` (mirrors ``cnn_export.py``'s head). Passing
  ``in_scale=1.0`` to :func:`quantize_linear` is the same thing — this helper just names the
  intent.

All math is symmetric (:mod:`penumbra.quantization.spec`) and integer-exact: the quantized
weights/biases these functions return are exactly what the FHE op computes on, so the
quantized-cleartext path built from them is the golden oracle the FHE path must match
bit-for-bit (``AGENTS.md`` §1.1). NumPy only.
"""

from __future__ import annotations

import numpy as np

from penumbra.bitwidth import MESSAGE_BITS, requant_internal_bits
from penumbra.quantization.spec import (
    QuantSpec,
    symmetric_spec,
    symmetric_spec_per_channel,
)


def _quantize_bias_acc(b_f: np.ndarray, acc_scale: float) -> np.ndarray:
    """Quantize a float bias into accumulator units ``round(b_f / acc_scale)`` -> int64.

    Biases are not clamped to the weight's integer range: they are added into a *wide*
    accumulator, and the bit-width tracker sizes the radix from the bias magnitude
    (:func:`penumbra.bitwidth.output_bits`). Clamping here would silently corrupt a legitimately
    large bias; sizing the accumulator to fit it is the correct, loud-on-overflow behavior.
    """
    return np.round(np.asarray(b_f, dtype=np.float64) / acc_scale).astype(np.int64)


def quantize_linear(
    w_f: np.ndarray,
    b_f: np.ndarray | None,
    in_scale: float,
    *,
    bits: int,
    per_channel: bool = False,
) -> tuple[np.ndarray, np.ndarray, QuantSpec | list[QuantSpec]]:
    """Quantize a ``Linear`` layer's weights and bias against a known input scale.

    Mirrors ``train_quantize_export.py`` exactly (that example quantizes a single weight row;
    this generalizes to ``(n_out, n_in)``):

    * ``w_q = w_spec.quantize(w_f)`` with a signed symmetric weight scale,
    * ``acc_scale = in_scale * w_spec.scale`` (the accumulator's units),
    * ``b_q = round(b_f / acc_scale)`` (the bias in accumulator units).

    Args:
        w_f: float weights, shape ``(n_out, n_in)``.
        b_f: float bias, shape ``(n_out,)``, or ``None`` for a bias-free layer (-> all-zero).
        in_scale: the input tensor's quantization scale (from the upstream activation spec).
        bits: signed weight bit-width.
        per_channel: if ``True``, one weight scale **per output row** (``axis=0``); each row's
            ``acc_scale_i = in_scale * w_scale_i`` and ``b_q[i] = round(b_f[i] / acc_scale_i)``.
            Per-channel keeps a small-magnitude output row from being crushed by a large one — the
            main accuracy lever per-tensor leaves on the table (``PROJECT.md`` §8, ROADMAP P5).

    Returns:
        ``(w_q, b_q, spec_or_specs)`` where ``w_q`` is int64 ``(n_out, n_in)``, ``b_q`` is int64
        ``(n_out,)``, and ``spec_or_specs`` is one :class:`QuantSpec` (per-tensor) or a list of
        ``n_out`` specs (per-channel). The spec(s) are returned so callers can derive downstream
        scales (e.g. the next layer's input scale).
    """
    w_f = np.asarray(w_f, dtype=np.float64)
    if w_f.ndim != 2:
        raise ValueError(
            f"quantize_linear expects 2-D weights (n_out, n_in), got shape {w_f.shape}"
        )
    n_out = w_f.shape[0]
    b_f = np.zeros(n_out, dtype=np.float64) if b_f is None else np.asarray(b_f, dtype=np.float64)
    if b_f.shape != (n_out,):
        raise ValueError(
            f"bias shape {b_f.shape} does not match n_out={n_out} from weights {w_f.shape}"
        )

    if not per_channel:
        w_spec = symmetric_spec(w_f, bits, signed=True)
        w_q = w_spec.quantize(w_f)
        acc_scale = in_scale * w_spec.scale
        b_q = _quantize_bias_acc(b_f, acc_scale)
        return w_q, b_q, w_spec

    # Per-channel: one spec per output row; quantize and bias-scale each row with its own scale.
    specs = symmetric_spec_per_channel(w_f, bits, signed=True, axis=0)
    w_q = np.stack([specs[i].quantize(w_f[i]) for i in range(n_out)])
    b_q = np.array(
        [_quantize_bias_acc(b_f[i], in_scale * specs[i].scale) for i in range(n_out)],
        dtype=np.int64,
    )
    return w_q, b_q, specs


def quantize_conv(
    w_f: np.ndarray,
    *,
    bits: int,
    in_scale: float | None = None,
    b_f: np.ndarray | None = None,
    per_channel: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, QuantSpec | list[QuantSpec]]:
    """Quantize a ``Conv2d`` layer's weights (and optional bias) to the IR's flat layout.

    Mirrors ``cnn_export.py``'s conv weight quantization (``symmetric_spec(CONV_FILTERS, ...,
    signed=True)`` then flatten). ``w_f`` has shape ``(out_ch, in_ch, kh, kw)`` and ``w_q`` is
    returned **flattened to ``(out_ch, in_ch*kh*kw)``** row-major — exactly the
    ``[out_channels][in_channels*kernel_h*kernel_w]`` layout :class:`~penumbra.ir.Conv2dSpec`
    expects (in-channel / kernel-row / kernel-col fastest), which is what ``reshape(out_ch, -1)``
    produces for a contiguous ``(out_ch, in_ch, kh, kw)`` array.

    The Phase-4 conv carries no bias; pass ``b_f``/``in_scale`` only if the conv has one. When a
    bias is present it is quantized into accumulator units like :func:`quantize_linear`
    (``acc_scale = in_scale * weight_scale``), so ``in_scale`` is required in that case.

    Args:
        w_f: float kernels, shape ``(out_ch, in_ch, kh, kw)``.
        bits: signed weight bit-width.
        in_scale: input scale; required iff ``b_f`` is given (to size the bias).
        b_f: optional float bias, shape ``(out_ch,)``.
        per_channel: if ``True``, one weight scale per **output channel** (``axis=0``).

    Returns:
        ``(w_q, b_q, spec_or_specs)`` — ``w_q`` int64 ``(out_ch, in_ch*kh*kw)``; ``b_q`` is int64
        ``(out_ch,)`` if a bias was given else ``None``; spec(s) as in :func:`quantize_linear`.
    """
    w_f = np.asarray(w_f, dtype=np.float64)
    if w_f.ndim != 4:
        raise ValueError(
            f"quantize_conv expects 4-D weights (out_ch, in_ch, kh, kw), got shape {w_f.shape}"
        )
    out_ch = w_f.shape[0]
    w_flat = w_f.reshape(out_ch, -1)  # [out_ch][in_ch*kh*kw], in-channel/row/col fastest

    if b_f is not None and in_scale is None:
        raise ValueError("quantize_conv needs in_scale to quantize a bias into accumulator units")

    if not per_channel:
        w_spec = symmetric_spec(w_f, bits, signed=True)
        w_q = w_spec.quantize(w_flat)
        b_q = None
        if b_f is not None:
            b_f = np.asarray(b_f, dtype=np.float64)
            if b_f.shape != (out_ch,):
                raise ValueError(f"conv bias shape {b_f.shape} does not match out_ch={out_ch}")
            b_q = _quantize_bias_acc(b_f, in_scale * w_spec.scale)  # type: ignore[arg-type]
        return w_q, b_q, w_spec

    # Per-channel: one spec per output channel (computed on the full 4-D kernel, axis 0), applied
    # to the corresponding flattened row.
    specs = symmetric_spec_per_channel(w_f, bits, signed=True, axis=0)
    w_q = np.stack([specs[i].quantize(w_flat[i]) for i in range(out_ch)])
    b_q = None
    if b_f is not None:
        b_f = np.asarray(b_f, dtype=np.float64)
        if b_f.shape != (out_ch,):
            raise ValueError(f"conv bias shape {b_f.shape} does not match out_ch={out_ch}")
        b_q = np.array(
            [_quantize_bias_acc(b_f[i], in_scale * specs[i].scale) for i in range(out_ch)],  # type: ignore[operator]
            dtype=np.int64,
        )
    return w_q, b_q, specs


def quantize_linear_integer_input(
    w_f: np.ndarray,
    b_f: np.ndarray | None,
    *,
    bits: int,
    per_channel: bool = False,
) -> tuple[np.ndarray, np.ndarray, QuantSpec | list[QuantSpec]]:
    """Quantize a ``Linear`` head that consumes **already-integer** features (input scale 1).

    The ``cnn_export.py`` head reads the *integer* pooled activations directly, so its effective
    input scale is 1 and the accumulator units collapse to the weight scale. The bias therefore
    shares the weight scale: ``b_q = round(b_f / w_scale)`` (per-tensor) or
    ``b_q[i] = round(b_f[i] / w_scale_i)`` (per-channel). This is the argmax-preserving
    quantization the example uses for the softmax head.

    It is exactly :func:`quantize_linear` with ``in_scale=1.0``; this wrapper exists to make the
    "integer-feature head" convention explicit at the call site (the two bias conventions are the
    most error-prone part of layer quantization — see the module docstring).
    """
    return quantize_linear(w_f, b_f, 1.0, bits=bits, per_channel=per_channel)


def choose_requant_params(
    acc_scale: float,
    out_scale: float,
    *,
    out_bits: int = MESSAGE_BITS,
    max_mult_bits: int = 5,
    input_bits: int | None = None,
    radix_capacity_bits: int | None = None,
) -> tuple[int, int, int]:
    """Approximate the real rescale ``M = acc_scale / out_scale`` as ``(mult, shift)`` + round bias.

    The generalized ``Requant`` computes ``clamp((max(x,0)*mult + round_bias) >> shift, ...)``
    (``runtime/src/ops/requant.rs``), so an arbitrary real scale ratio ``M`` is realized by a
    fixed-point multiplier ``mult / 2**shift``. This picks the smallest-error ``(mult, shift)``:

    * If ``M`` is (near) a negative power of two, prefer the **pure shift** ``mult = 1`` — that
      reproduces the Phase-4 path byte-identically and adds no radix width.
    * Otherwise search ``mult`` over ``[1, 2**max_mult_bits)`` and pick ``shift`` so
      ``mult / 2**shift`` is closest to ``M`` (a standard fixed-point quantization). ``mult`` is
      capped at ``max_mult_bits`` so the internal peak ``max(x,0)*mult`` does not blow the radix
      (the bit-width tracker enforces this; see :func:`penumbra.bitwidth.requant_internal_bits`).

    ``round_bias = 2**(shift-1)`` gives round-to-nearest (``0`` when ``shift == 0``). When
    ``input_bits``/``radix_capacity_bits`` are given, the chosen params are checked to fit the
    radix's internal-peak budget and an over-budget multiplier is rejected loudly (``AGENTS.md``
    §1.3, §1.4) rather than silently overflowing under FHE.

    Returns ``(mult, shift, round_bias)``.
    """
    if not np.isfinite(acc_scale) or acc_scale <= 0:
        raise ValueError(f"acc_scale must be finite and positive, got {acc_scale}")
    if not np.isfinite(out_scale) or out_scale <= 0:
        raise ValueError(f"out_scale must be finite and positive, got {out_scale}")
    if max_mult_bits < 1:
        raise ValueError(f"max_mult_bits must be >= 1, got {max_mult_bits}")

    ratio = acc_scale / out_scale  # the real rescale M (expected <= 1: narrowing a wide acc)
    if ratio <= 0:
        raise ValueError(f"rescale ratio must be positive, got {ratio}")

    # Candidate 1: the pure power-of-two shift (mult = 1). shift = round(-log2(ratio)), clamped
    # to >= 0 (a Requant only narrows). This is the legacy, zero-extra-width path.
    pow2_shift = max(0, int(round(-np.log2(ratio))))
    best = (1, pow2_shift, _err(1, pow2_shift, ratio))

    # Candidate 2+: a genuine fixed-point multiplier. For each mult, the best shift is the one
    # making mult / 2**shift closest to ratio, i.e. shift = round(log2(mult / ratio)).
    mult_ceiling = 1 << max_mult_bits
    for mult in range(2, mult_ceiling):
        shift = int(round(np.log2(mult / ratio)))
        if shift < 0:
            continue  # would *amplify*; a Requant only narrows
        err = _err(mult, shift, ratio)
        if err < best[2]:
            best = (mult, shift, err)

    mult, shift, _ = best
    round_bias = (1 << (shift - 1)) if shift > 0 else 0

    # Optional loud budget check: the transient peak max(x,0)*mult + round_bias must fit the
    # radix even though the output is tiny (the whole point of the internal-peak rule).
    if input_bits is not None and radix_capacity_bits is not None:
        peak = requant_internal_bits(input_bits, mult, round_bias)
        if peak > radix_capacity_bits:
            raise ValueError(
                f"chosen Requant multiplier {mult} needs {peak} transient bits for a "
                f"{input_bits}-bit input, exceeding the {radix_capacity_bits}-bit radix; "
                "reduce max_mult_bits, reduce input precision, or widen num_blocks"
            )
    return mult, shift, round_bias


def _err(mult: int, shift: int, ratio: float) -> float:
    """Relative error of approximating ``ratio`` by ``mult / 2**shift`` (the rescale fidelity)."""
    approx = mult / (1 << shift)
    return abs(approx - ratio) / ratio
