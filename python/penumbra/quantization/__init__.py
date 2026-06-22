"""Quantization service: float graph -> int graph + scales + lookup tables.

Quantization is ~80% of the engineering effort and where ML accuracy lives or dies
(``PROJECT.md`` §8). The library owns it as a **service** so non-crypto users never
compute scales by hand — they supply calibration data, not scales (``PROJECT.md`` §12).

Two paths:
    - **PTQ** (post-training quantization): quantize a trained float model using
      calibration data to choose per-tensor (then optionally per-channel) scales. Start
      here.
    - **QAT** (quantization-aware training): wrap **Brevitas** rather than writing our
      own quantizer (``PROJECT.md`` §8, §15). Needed for harder models.

Bit-width budget link (``PROJECT.md`` §9): keep activation bit-widths small (<=6-8 bits);
LUTs must be generated in the quantized-integer domain consistent with the chosen scales
— an off-by-scale here silently wrecks accuracy.

Verification invariant: the quantized-cleartext output is the oracle — FHE must match it
bit-for-bit (``AGENTS.md`` §1.1). The quantization service must never break this.

This module is a **minimal Phase-2 helper** — just enough symmetric PTQ to quantize a
linear classifier by hand and produce a fixture for the golden test. The full service
(calibration, per-channel scales, Brevitas-backed QAT, automatic LUT generation) is Phase
5; do not grow this beyond Phase-2 needs.
"""

from __future__ import annotations

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
    scales (better accuracy for weights) are a Phase-5 refinement (``ROADMAP`` Phase 5).
    """
    values = np.asarray(values, dtype=np.float64)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    edge = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    # Guard against a degenerate all-zero tensor: a unit scale keeps everything at 0.
    scale = peak / edge if peak > 0 else 1.0
    return QuantSpec(scale=scale, bits=bits, signed=signed)


def linear_logit_int(x_q: np.ndarray, w_q: np.ndarray, bias_q: np.ndarray | int) -> np.ndarray:
    """The quantized-integer logit ``w_q . x_q + bias_q`` — the cleartext oracle.

    All-integer by construction: this is exactly the arithmetic the FHE ``Linear`` op
    performs, so the two must agree bit-for-bit (``AGENTS.md`` §1.1). ``x_q`` may be a
    single sample (1-D) or a batch (2-D, one sample per row).
    """
    x_q = np.asarray(x_q, dtype=np.int64)
    w_q = np.asarray(w_q, dtype=np.int64)
    return x_q @ w_q + np.asarray(bias_q, dtype=np.int64)
