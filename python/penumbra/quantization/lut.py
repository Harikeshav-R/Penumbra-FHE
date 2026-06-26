"""Lookup-table generation in the quantized-integer domain (``PROJECT.md`` §5, §8).

A programmable bootstrap (PBS) is the only way to apply a nonlinearity to a ciphertext, and
it does so via a **lookup table** indexed by a single radix block's integer value — the
*expensive* regime where runtime ≈ number of bootstraps (``PROJECT.md`` §5). This module is
the quantization service's owner of producing those tables: it turns a float activation (ReLU,
sigmoid, ...) plus the chosen input/output scales into the integer table the runtime applies
**bit-exactly** (``runtime/src/ops/activation.rs``, ``runtime/src/ops/requant.rs``).

The tables live entirely in the **quantized-integer** domain. The library — never the user —
owns generating them consistent with the scales (``PROJECT.md`` §8, §12): an off-by-scale here
silently wrecks accuracy, so the math (dequantize -> apply float fn -> requantize -> clamp) is
written once, here, and validated against the backend's hard constraints before a bad table
can ever reach a PBS (``AGENTS.md`` §1.4).

## The backend LUT contract (mirrored here so a bad table fails in Python first)

Both the ``Activation`` and ``Requant`` ops assert the same shape on the table they apply
(``runtime/src/ops/activation.rs``, ``runtime/src/ops/requant.rs``), because
``MESSAGE_BITS = 2`` (``runtime/src/keys.rs`` -> the ``PARAM_MESSAGE_2_CARRY_2`` profile)
means the message space is exactly ``{0, 1, 2, 3}``:

* the table MUST have exactly ``2**MESSAGE_BITS`` entries (cover the full message space — a
  short table silently maps the uncovered values to ``0``);
* every entry MUST be ``< 2**MESSAGE_BITS`` and ``>= 0`` (a single shortint block;
  ``apply_lookup_table`` reduces modulo the message modulus, so ``5`` would silently wrap to
  ``1`` — wrong-but-confident ciphertext).

:func:`validate_lut` reproduces those panics as a Python ``ValueError`` and is called at the
end of every generator here, so an export bug fails loudly *before* keygen, naming the
problem, rather than producing a wrong ciphertext (``AGENTS.md`` §1.4).

## Activation input domain (narrow, single-block)

``Activation`` applies its LUT to ``blocks()[0]`` only — a **single** radix block — so its
true input domain is the narrow message space ``[0, 2**MESSAGE_BITS)`` of *post-Requant*
values, not a wide accumulator (``runtime/src/ops/activation.rs`` "Domain" notes and the
``output_bits`` assert). :func:`make_activation_lut` therefore enumerates exactly that domain:
it is the only thing the backend ``Activation`` consumes. A wide accumulator must be
``Requant``-ed down first (Phase 4 / :mod:`penumbra.compile`).
"""

from __future__ import annotations

from collections.abc import Callable

from penumbra.bitwidth import MESSAGE_BITS
from penumbra.quantization.spec import QuantSpec

# The full message space of one radix block, ``{0, ..., 2**MESSAGE_BITS - 1}`` — both the
# domain a LUT is indexed over and the (inclusive) ceiling every entry must respect. Derived
# from ``MESSAGE_BITS`` (``runtime/src/keys.rs``); never hardcode the literal.
_MESSAGE_SPACE = 1 << MESSAGE_BITS
_MAX_ENTRY = _MESSAGE_SPACE - 1


def validate_lut(lut: list[int]) -> None:
    """Raise ``ValueError`` if ``lut`` violates the backend's single-block LUT contract.

    Mirrors the Rust panics in ``runtime/src/ops/activation.rs`` and
    ``runtime/src/ops/requant.rs``: the table must cover the full ``2**MESSAGE_BITS``-entry
    message space, and every entry must lie in ``[0, 2**MESSAGE_BITS)``. Called at the end of
    the generators so a malformed table fails here in Python — with an actionable message
    naming the offending index — before it can reach a PBS and silently wrap (``AGENTS.md``
    §1.4).
    """
    if len(lut) != _MESSAGE_SPACE:
        raise ValueError(
            f"LUT must cover the full {MESSAGE_BITS}-bit message space "
            f"({_MESSAGE_SPACE} entries); got {len(lut)}. The backend indexes the table by a "
            "single radix block's value, so a short/long table is a domain mismatch "
            "(runtime/src/ops/activation.rs)."
        )
    for idx, entry in enumerate(lut):
        if entry < 0:
            raise ValueError(
                f"LUT entry lut[{idx}] = {entry} is negative; the backend stores each entry as "
                "an unsigned shortint block, so every output must be >= 0."
            )
        if entry >= _MESSAGE_SPACE:
            raise ValueError(
                f"LUT entry lut[{idx}] = {entry} does not fit one shortint block: every output "
                f"must be < {_MESSAGE_SPACE} (the {MESSAGE_BITS}-bit message space). "
                "apply_lookup_table reduces modulo the message modulus, so a larger value "
                "would silently wrap (runtime/src/ops/activation.rs)."
            )


def lut_output_bits(lut: list[int]) -> int:
    """Minimum bits to represent the table's largest entry (``>= 1``).

    Mirror of the Rust ``lut_output_bits`` helper (``runtime/src/ops/activation.rs``,
    ``runtime/src/ops/requant.rs``): the true output width of a table. An all-zero (or
    single-value) table still occupies a 1-bit representable value, so the width is clamped up
    to ``1`` — useful for setting an op's ``output_bits`` so it never *under-counts* the table
    (the backend asserts ``output_bits >= lut_output_bits(lut)``).
    """
    max_entry = max(lut) if lut else 0
    return max(max_entry.bit_length(), 1)


def make_activation_lut(
    fn: Callable[[float], float], in_spec: QuantSpec, out_spec: QuantSpec
) -> list[int]:
    """Build the integer activation table for ``fn`` under the given input/output scales.

    For each integer input value ``v`` in the input block's message domain
    ``[0, 2**MESSAGE_BITS)`` (the narrow, single-block domain the backend ``Activation``
    consumes — see the module docstring and ``runtime/src/ops/activation.rs``):

    1. **dequantize** to float: ``x = v * in_spec.scale``;
    2. apply the float activation: ``y = fn(x)`` (e.g. ReLU, a clamped ReLU, sigmoid);
    3. **requantize** to the output domain: ``q = round(y / out_spec.scale)``;
    4. **clamp** to ``[0, 2**MESSAGE_BITS - 1]`` so the result fits one shortint block.

    Returns a length-``2**MESSAGE_BITS`` list of ints. The table is generated entirely in the
    quantized-integer domain consistent with the scales — getting the scales right here is the
    whole game (``PROJECT.md`` §8): an off-by-scale silently wrecks accuracy. The result is
    :func:`validate_lut`-checked before return, so a table the backend would reject fails
    loudly here first (``AGENTS.md`` §1.4).
    """
    lut: list[int] = []
    for v in range(_MESSAGE_SPACE):
        x = v * in_spec.scale
        y = fn(x)
        # ``round`` of the rescaled output, then clamp into the single-block range. We use
        # Python's banker's-rounding ``round`` to match ``np.round`` (the quantizer in
        # ``QuantSpec.quantize``), keeping LUT generation consistent with weight/bias
        # quantization — both must land in the same integer domain (``PROJECT.md`` §8).
        q = int(round(y / out_spec.scale))
        lut.append(max(0, min(q, _MAX_ENTRY)))
    validate_lut(lut)
    return lut


def identity_clamp_lut(out_bits: int) -> list[int]:
    """The ``Requant`` clamp table: ``min(v, 2**out_bits - 1)`` over the message space.

    This is the canonical generator for the identity-over-range clamp LUT — exactly what
    ``penumbra.compile._clamp_lut`` builds inline. It is provided here so the compile pass can
    reuse the single source of truth (this module owns LUT generation); the behaviour is pinned
    identical by the tests. The ``Requant`` op saturates the wide value at ``2**out_bits - 1``
    *before* the PBS, so over its in-range domain this LUT is the identity; it exists so the
    table covers the whole message space and clamps any residual high value
    (``runtime/src/ops/requant.rs``).

    ``out_bits`` is the narrowed output width (``<= MESSAGE_BITS``); the ceiling is
    ``2**out_bits - 1``. The result is :func:`validate_lut`-checked before return.
    """
    ceil = (1 << out_bits) - 1
    lut = [min(v, ceil) for v in range(_MESSAGE_SPACE)]
    validate_lut(lut)
    return lut
