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
from penumbra.ir import ArgmaxSpec, Graph, LinearSpec
from penumbra.quantization import QuantSpec, linear_logit_int, symmetric_spec

FIXTURE = Path(__file__).resolve().parent.parent / "examples" / "mnist" / "phase2_fixture.json"


def test_package_imports_and_has_version():
    assert isinstance(penumbra.__version__, str)
    assert penumbra.__version__


def _linear_argmax_params(fx: dict) -> tuple[np.ndarray, int, int]:
    """Recover (weights, bias, threshold) from the embedded IR graph (Phase 3).

    The model lives under ``fx["graph"]`` as a ``Linear → Argmax`` op graph; pull the single
    logit row, bias, and threshold from the typed spec so these tests and the runtime read
    the same source of model truth (``AGENTS.md`` §5).
    """
    g = Graph.from_dict(fx["graph"])
    fc, head = g.nodes
    assert isinstance(fc.op, LinearSpec) and isinstance(head.op, ArgmaxSpec)
    weights = np.array(fc.op.weights[0], dtype=np.int64)
    return weights, int(fc.op.bias[0]), int(head.op.threshold)


def test_fixture_labels_match_quantized_reference():
    fx = json.loads(FIXTURE.read_text())

    weights, bias, threshold = _linear_argmax_params(fx)
    inputs = np.array(fx["test_inputs"], dtype=np.int64)
    expected = np.array(fx["expected_labels"], dtype=np.int64)

    logits = linear_logit_int(inputs, weights, bias)
    labels = (logits >= threshold).astype(np.int64)

    assert labels.tolist() == expected.tolist(), (
        "fixture expected_labels disagree with the quantized-integer reference — "
        "regenerate the fixture or fix the quantization"
    )


def test_fixture_fits_bit_width_budget():
    """Mirror the Rust load-time budget check (``Linear::output_bits``, ``PROJECT.md`` §9).

    The Rust runtime enforces the accumulator budget at *load time* using a
    DATA-INDEPENDENT, worst-case-over-declared-ranges formula — it never sees the
    committed test inputs. This test recomputes that exact same formula so the two
    languages cannot drift (``AGENTS.md`` §5): if it passes here, the declared
    ``input_bits``/``weight_bits`` ranges provably fit the radix for *any* input, not
    merely for the 4 committed samples.

    Worst-case signed width of one ``Linear`` row of length ``n``:
      - each product is bounded by ``input_bits + weight_bits`` magnitude bits;
      - summing ``n`` of them grows the magnitude by ``ceil(log2(n))`` bits, which for
        a sum of ``n`` terms is ``(n - 1).bit_length()``;
      - adding the bias contributes ``max(sum_bits, bias_bits)`` magnitude bits plus a
        possible carry, and the signed accumulator needs one sign bit on top — hence
        ``+2`` (carry from the bias add + sign).
    """
    fx = json.loads(FIXTURE.read_text())
    g = Graph.from_dict(fx["graph"])
    num_blocks = g.num_blocks
    message_bits = 2  # PARAM_MESSAGE_2_CARRY_2 message space
    capacity = num_blocks * message_bits  # signed radix width

    input_bits = g.input_bits
    fc = g.nodes[0]
    assert isinstance(fc.op, LinearSpec)
    weight_bits = fc.op.weight_bits
    weights = np.array(fc.op.weights[0], dtype=np.int64)
    bias = int(fc.op.bias[0])

    n = len(weights)
    # Magnitude-bit growth from summing n products; matches the Rust
    # `usize::BITS - (n - 1).leading_zeros()` for n > 1, and is 0 for n <= 1.
    sum_growth = 0 if n <= 1 else (n - 1).bit_length()
    sum_bits = input_bits + weight_bits + sum_growth
    bias_bits = abs(bias).bit_length()  # 0 when bias == 0
    needed = max(sum_bits, bias_bits) + 2  # +1 carry from bias add, +1 sign bit
    assert needed <= capacity, (
        f"worst-case accumulator needs {needed} signed bits but the radix holds "
        f"{capacity} ({num_blocks} blocks x {message_bits} bits): "
        f"sum_bits={sum_bits} (input_bits={input_bits} + weight_bits={weight_bits} "
        f"+ sum_growth={sum_growth} for n={n}), bias_bits={bias_bits}"
    )

    # Secondary, DATA-DEPENDENT sanity check: the actual peak over the committed
    # inputs must of course also fit (a weaker condition the worst-case bound implies,
    # kept as a cross-check that the fixture data is consistent with its declared ranges).
    inputs = np.array(fx["test_inputs"], dtype=np.int64)
    logits = linear_logit_int(inputs, weights, bias)
    peak_needed = int(np.max(np.abs(logits))).bit_length() + 1  # +1 for sign
    assert peak_needed <= needed, (
        f"committed inputs peak needs {peak_needed} signed bits, exceeding the "
        f"worst-case bound {needed} — fixture data violates its declared ranges"
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
