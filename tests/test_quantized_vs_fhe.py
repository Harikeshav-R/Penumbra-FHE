"""Golden-invariant support test, Python side (``AGENTS.md`` §1.1).

The bit-for-bit FHE-vs-cleartext gate itself lives in Rust
(``runtime/tests/golden_logreg.rs``) — it needs the TFHE runtime. This Python test guards
the *other* half of the contract: that the committed fixture's ``expected_labels`` are
exactly what the quantized-integer reference produces. If the fixture ever drifts from the
reference arithmetic, this fails here (fast, no FHE) rather than surfacing as a confusing
Rust golden violation.

No heavy ML deps and no network: it only reads the committed JSON and recomputes integer
dot-products with NumPy (a core dependency).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import penumbra
from penumbra.quantization import QuantSpec, linear_logit_int, symmetric_spec

FIXTURE = Path(__file__).resolve().parent.parent / "examples" / "mnist" / "phase2_fixture.json"


def test_package_imports_and_has_version():
    assert isinstance(penumbra.__version__, str)
    assert penumbra.__version__


def test_fixture_labels_match_quantized_reference():
    fx = json.loads(FIXTURE.read_text())

    weights = np.array(fx["linear"]["weights"][0], dtype=np.int64)
    bias = int(fx["linear"]["bias"][0])
    threshold = int(fx["argmax"]["threshold"])
    inputs = np.array(fx["test_inputs"], dtype=np.int64)
    expected = np.array(fx["expected_labels"], dtype=np.int64)

    logits = linear_logit_int(inputs, weights, bias)
    labels = (logits >= threshold).astype(np.int64)

    assert labels.tolist() == expected.tolist(), (
        "fixture expected_labels disagree with the quantized-integer reference — "
        "regenerate the fixture or fix the quantization"
    )


def test_fixture_fits_bit_width_budget():
    """The accumulator must fit the radix the fixture declares (``PROJECT.md`` §9)."""
    fx = json.loads(FIXTURE.read_text())
    num_blocks = int(fx["num_blocks"])
    message_bits = 2  # PARAM_MESSAGE_2_CARRY_2 message space
    capacity = num_blocks * message_bits

    weights = np.array(fx["linear"]["weights"][0], dtype=np.int64)
    bias = int(fx["linear"]["bias"][0])
    inputs = np.array(fx["test_inputs"], dtype=np.int64)
    logits = linear_logit_int(inputs, weights, bias)

    peak = int(np.max(np.abs(logits)))
    needed = peak.bit_length() + 1  # +1 for sign
    assert needed <= capacity, (
        f"accumulator needs {needed} signed bits but the radix holds {capacity} "
        f"({num_blocks} blocks x {message_bits} bits)"
    )


def test_activation_lut_is_self_consistent():
    fx = json.loads(FIXTURE.read_text())
    act = fx["activation"]
    lut = act["lut"]
    got = [lut[v] for v in act["test_inputs"]]
    assert got == act["expected"]


def test_symmetric_spec_roundtrip():
    """The quantization helper's scale/clamp behave as the export script relies on."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=64)
    spec = symmetric_spec(w, bits=4, signed=True)
    q = spec.quantize(w)
    assert isinstance(spec, QuantSpec)
    assert q.min() >= spec.qmin and q.max() <= spec.qmax


def test_quant_spec_rejects_invalid_specs():
    """Invalid specs must fail loudly on construction, not as cryptic later errors."""
    with pytest.raises(ValueError):
        QuantSpec(scale=1.0, bits=0, signed=False)
    with pytest.raises(ValueError):
        QuantSpec(scale=1.0, bits=1, signed=True)  # signed needs sign + magnitude
    with pytest.raises(ValueError):
        QuantSpec(scale=0.0, bits=4, signed=True)  # non-positive scale
    with pytest.raises(ValueError):
        QuantSpec(scale=float("nan"), bits=4, signed=True)  # non-finite scale


def test_symmetric_spec_rejects_zero_width_range():
    """A bit-width whose range edge collapses to 0 must raise, not divide by zero."""
    with pytest.raises(ValueError):
        symmetric_spec(np.array([1.0, -2.0]), bits=1, signed=True)
