"""Symmetric quantization specs — the float<->int scale primitives.

A :class:`QuantSpec` is a single tensor's symmetric quantization scheme: one scale that maps
``q = round(x / scale)`` and back. This is the lowest layer of the quantization service; the
calibrator (:mod:`penumbra.quantization.calibration`) chooses the *ranges* a spec is built
from, and the per-layer quantizers (:mod:`penumbra.quantization.ptq`) consume specs to produce
the int weights/biases the IR carries.

Symmetric (zero-point-free) quantization is the project default: weights are always symmetric
(``PROJECT.md`` §8, ROADMAP Phase 5), and the FHE backend computes on a symmetric *signed*
radix (``runtime/src/keys.rs``). Asymmetric activation quantization (a zero-point) is a gated
later refinement.

LUTs and requant params must be generated in the quantized-integer domain consistent with
these scales — an off-by-scale here silently wrecks accuracy (``PROJECT.md`` §8). The
quantized-cleartext output built from these specs is the golden oracle the FHE path must match
bit-for-bit (``AGENTS.md`` §1.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantSpec:
    """A symmetric quantization scheme for one tensor.

    ``scale`` maps float <-> int as ``q = round(x / scale)`` and ``x ~= scale * q``.
    ``bits`` is the (signed, for weights) integer bit-width; ``signed`` selects the range.
    """

    scale: float
    bits: int
    signed: bool

    def __post_init__(self) -> None:
        # Validate invariants on construction so misuse fails loudly here with a clear
        # message, rather than later as a cryptic shift/division error deep in quantize()
        # or in fixture generation (this is part of the public quantization API).
        if self.bits <= 0:
            raise ValueError(f"QuantSpec.bits must be positive, got {self.bits}")
        if self.signed and self.bits < 2:
            # A signed range needs one bit for sign plus at least one magnitude bit;
            # bits == 1 leaves qmax == 0 and no representable positive value.
            raise ValueError(
                f"signed QuantSpec needs bits >= 2 (sign + magnitude), got {self.bits}"
            )
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"QuantSpec.scale must be finite and positive, got {self.scale}")

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize a float array to integers under this spec (rounded + clamped)."""
        q = np.round(np.asarray(x, dtype=np.float64) / self.scale)
        return np.clip(q, self.qmin, self.qmax).astype(np.int64)


def symmetric_spec(values: np.ndarray, bits: int, *, signed: bool) -> QuantSpec:
    """Choose a symmetric per-tensor scale so ``max|values|`` maps to the range edge.

    This is the simplest PTQ: one scale per tensor from the observed magnitude. Per-channel
    scales (better accuracy for weights) are :func:`symmetric_spec_per_channel`.
    """
    values = np.asarray(values, dtype=np.float64)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    edge = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    # The range edge collapses to 0 for degenerate bit-widths (e.g. signed bits == 1), which
    # would make scale = peak / 0. Fail loudly before dividing (`AGENTS.md` §1.4); QuantSpec
    # re-checks the same invariant on construction.
    if edge == 0:
        raise ValueError(
            f"bits={bits}, signed={signed} yields a zero-width range (no representable "
            "nonzero value); use a larger bit-width"
        )
    # Guard against a degenerate all-zero tensor: a unit scale keeps everything at 0.
    scale = peak / edge if peak > 0 else 1.0
    return QuantSpec(scale=scale, bits=bits, signed=signed)


def symmetric_spec_per_channel(
    values: np.ndarray, bits: int, *, signed: bool, axis: int = 0
) -> list[QuantSpec]:
    """One symmetric :class:`QuantSpec` per slice along ``axis`` (per-channel weights).

    ROADMAP Phase 5: "start per-tensor, offer per-channel for weights." A weight tensor is
    quantized with one scale **per output channel** (``axis=0`` of a ``[out][in...]`` weight),
    which keeps small-magnitude channels from being crushed by a large-magnitude one — the
    main accuracy lever per-tensor leaves on the table.

    The downstream per-channel quantizer folds these per-row scales into the bias scaling and a
    single shared :class:`~penumbra.ir.RequantSpec` shift, so per-channel needs no backend or IR
    change (``phase5-decisions``): the radix stays symmetric and signed.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("per-channel quantization needs an array with at least one axis")
    moved = np.moveaxis(values, axis, 0)
    return [symmetric_spec(moved[i], bits, signed=signed) for i in range(moved.shape[0])]


def linear_logit_int(x_q: np.ndarray, w_q: np.ndarray, bias_q: np.ndarray | int) -> np.ndarray:
    """The quantized-integer logit ``w_q . x_q + bias_q`` — the cleartext oracle.

    All-integer by construction: this is exactly the arithmetic the FHE ``Linear`` op
    performs, so the two must agree bit-for-bit (``AGENTS.md`` §1.1). ``x_q`` may be a
    single sample (1-D) or a batch (2-D, one sample per row).
    """
    x_q = np.asarray(x_q, dtype=np.int64)
    w_q = np.asarray(w_q, dtype=np.int64)
    return x_q @ w_q + np.asarray(bias_q, dtype=np.int64)
