"""Accuracy + sensitivity reporting — see the quantization gap, find the worst layers.

Quantization trades accuracy for the small integers FHE needs (``PROJECT.md`` §8). This module
lets a user *see* that trade honestly:

* :func:`accuracy_report` compares **float** vs **quantized-integer** accuracy on a test set and
  reports the **gap** (float − quantized) — the accuracy lost to low-bit integers. There is no
  separate "FHE accuracy": TFHE is exact, so the FHE output equals the quantized-cleartext output
  bit-for-bit (``AGENTS.md`` §1.1), and the golden tests guarantee it. Inventing an "FHE accuracy"
  column would either duplicate the quantized number or imply a discrepancy that cannot exist, so
  we deliberately do **not** (matching ``docs/BENCHMARKS.md``'s methodology).

* :func:`layer_sqnr_report` measures each layer's **SQNR** (signal-to-quantization-noise ratio,
  in dB) — how much a layer's output degrades under quantization. Low-SQNR layers are the ones
  worth spending more bits on (or per-channel scales), so this is the knob a user tunes against.

All host-side and cleartext (no ciphertext, no crypto dependency). NumPy only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AccuracyReport:
    """Float vs quantized accuracy on a labelled test set, and the quantization gap.

    ``float_accuracy`` / ``quantized_accuracy`` are fractions in ``[0, 1]``; ``gap`` is their
    difference (``float − quantized``, the accuracy lost to quantization, usually ``>= 0``).
    """

    float_accuracy: float
    quantized_accuracy: float

    @property
    def gap(self) -> float:
        return self.float_accuracy - self.quantized_accuracy

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"float={self.float_accuracy:.4f}  quantized={self.quantized_accuracy:.4f}  "
            f"gap={self.gap:+.4f}  (FHE == quantized, bit-for-bit)"
        )


def accuracy_report(
    float_predict: Callable[[np.ndarray], np.ndarray],
    quant_predict: Callable[[np.ndarray], np.ndarray],
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> AccuracyReport:
    """Compare float vs quantized predictions on ``(x_test, y_test)`` and report the gap.

    ``float_predict`` / ``quant_predict`` map a batch ``(N, ...)`` to predicted labels ``(N,)``.
    The FHE accuracy is *not* a separate measurement — it equals the quantized accuracy exactly
    (the golden invariant); see the module docstring.
    """
    y_test = np.asarray(y_test)
    fa = float(np.mean(np.asarray(float_predict(x_test)) == y_test))
    qa = float(np.mean(np.asarray(quant_predict(x_test)) == y_test))
    return AccuracyReport(float_accuracy=fa, quantized_accuracy=qa)


def sqnr_db(reference: np.ndarray, approx: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio in dB between a reference and its quantized approx.

    ``SQNR = 10 * log10( sum(ref^2) / sum((ref - approx)^2) )``. Higher is better (less noise);
    a perfect match is ``+inf``. A zero reference (no signal) returns ``+inf`` if the approx also
    vanishes, else ``-inf`` (all noise, no signal) — the honest degenerate readings.
    """
    reference = np.asarray(reference, dtype=np.float64)
    approx = np.asarray(approx, dtype=np.float64)
    signal = float(np.sum(reference**2))
    noise = float(np.sum((reference - approx) ** 2))
    if noise == 0.0:
        return float("inf")
    if signal == 0.0:
        return float("-inf")
    return 10.0 * np.log10(signal / noise)


def layer_sqnr_report(
    float_outputs: dict[str, np.ndarray],
    quant_outputs: dict[str, np.ndarray],
) -> dict[str, float]:
    """Per-layer SQNR (dB): how much each named layer's output degrades under quantization.

    ``float_outputs`` / ``quant_outputs`` map a layer name to that layer's output over the same
    inputs — the float output and its dequantized-quantized counterpart (``scale * int``), in the
    same units. Returns ``{layer_name: sqnr_db}``. The lowest-SQNR layers are the ones a user
    should give more bits or per-channel scales — the sensitivity knob (``PROJECT.md`` §8, §9).

    Names present in only one of the two maps are skipped (with no entry); both maps should share
    the layers you want compared.
    """
    return {
        name: sqnr_db(float_outputs[name], quant_outputs[name])
        for name in float_outputs
        if name in quant_outputs
    }
