"""Tests for the accuracy + SQNR sensitivity harness (``penumbra.quantization.accuracy``).

These pin the honest-reporting contract: float vs quantized accuracy + gap (no fabricated FHE
number), and per-layer SQNR for finding the layers most worth more bits. NumPy-only, hermetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from penumbra.quantization.accuracy import (
    AccuracyReport,
    accuracy_report,
    layer_sqnr_report,
    sqnr_db,
)


def test_accuracy_report_gap():
    """The gap is float − quantized accuracy over the test set."""
    x = np.arange(10)
    y = np.zeros(10, dtype=int)

    def float_predict(xb):
        return np.zeros(len(xb), dtype=int)  # perfect

    def quant_predict(xb):
        out = np.zeros(len(xb), dtype=int)
        out[:2] = 1  # two wrong
        return out

    rep = accuracy_report(float_predict, quant_predict, x, y)
    assert isinstance(rep, AccuracyReport)
    assert rep.float_accuracy == 1.0
    assert rep.quantized_accuracy == pytest.approx(0.8)
    assert rep.gap == pytest.approx(0.2)


def test_sqnr_perfect_match_is_inf():
    ref = np.array([1.0, -2.0, 3.0])
    assert sqnr_db(ref, ref.copy()) == float("inf")


def test_sqnr_more_noise_is_lower_db():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    near = ref + 0.01
    far = ref + 1.0
    assert sqnr_db(ref, near) > sqnr_db(ref, far)


def test_sqnr_zero_signal_edge_cases():
    zero = np.zeros(4)
    assert sqnr_db(zero, zero) == float("inf")  # no signal, no noise
    assert sqnr_db(zero, np.ones(4)) == float("-inf")  # no signal, all noise


def test_layer_sqnr_report_ranks_layers():
    """A noisier layer gets a lower SQNR; only shared layer names are reported."""
    ref = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([1.0, 1.0, 1.0])}
    quant = {
        "a": np.array([1.0, 2.0, 3.0]),  # exact -> +inf
        "b": np.array([1.5, 0.5, 1.2]),  # noisy -> finite
        "c": np.array([0.0]),  # not in ref -> skipped
    }
    report = layer_sqnr_report(ref, quant)
    assert set(report) == {"a", "b"}  # 'c' skipped (not in both)
    assert report["a"] == float("inf")
    assert report["a"] > report["b"]
