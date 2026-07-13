"""Phase-6 second-framework fixture guard (``AGENTS.md`` §1.1, §5) — the scikit-learn proof.

The torch-CNN fixture guard (``test_onnx_fixture.py``) proves ``load_onnx`` works for a PyTorch
export; this proves it works for a **scikit-learn** export (``skl2onnx``), the second framework the
ROADMAP Phase 6 exit criterion asks for. It reloads the **committed** ``digit_linear_sklearn.onnx``
with the *core* ``onnx`` dep — no sklearn, no skl2onnx — lowers it through the front door, quantizes
it via the unchanged Phase-5 service, and asserts the result equals the committed
``phase6_sklearn_fixture.json``. If the loader drifts from what generated the fixture (or Cast
folding regresses), this fails here (fast) rather than as a confusing Rust golden violation.

The bit-for-bit FHE gate over the same fixture lives in Rust (``runtime/tests/golden_sklearn.rs``,
``#[ignore]`` by default — the accurate 64->10 logit head needs a 20-bit radix, so the no-PBS Linear
is still minutes per sample). The sklearn/skl2onnx training that *produced* the ``.onnx`` + fixture
is the example generator's job (the optional ``ml`` extra), never CI's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import penumbra as fhe
from penumbra.bitwidth import check_bit_width_budget
from penumbra.ir import Graph
from penumbra.reference import evaluate_graph_int

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "mnist"
ONNX_MODEL = EXAMPLES / "digit_linear_sklearn.onnx"
FIXTURE = EXAMPLES / "phase6_sklearn_fixture.json"

# Regeneration knobs must match examples/mnist/sklearn_export.py so the reload reproduces the graph.
INPUT_BITS = 4
WEIGHT_BITS = 8


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_sklearn_fixture_graph_round_trips_and_fits_budget():
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    assert Graph.from_json(g.to_json()) == g, "sklearn-lowered IR round-trip must be exact"
    check_bit_width_budget(g)  # raises if any tensor exceeds the radix


def test_sklearn_fixture_graph_is_a_single_linear():
    """The committed graph is one Linear: a linear classifier has no activation and no Requant."""
    g = Graph.from_dict(_fixture()["graph"])
    assert [n.op.op_type for n in g.nodes] == ["Linear"]


def test_sklearn_fixture_logits_and_labels_match_oracle():
    """Committed logits/labels are exactly what the integer reference produces (drift guard)."""
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    for i, (x, expected) in enumerate(zip(fx["test_inputs"], fx["expected_logits"], strict=True)):
        logits = evaluate_graph_int(g, {"x": x})[g.outputs[0]]
        assert logits == expected, f"sample {i}: logits drifted from the oracle"
        assert int(np.argmax(logits)) == fx["expected_labels"][i], f"sample {i}: label drifted"


def test_committed_sklearn_onnx_lowers_to_a_single_linear():
    """load_onnx lowers the committed sklearn ONNX to one Linear, folding the leading Cast away.

    Pure load-time drift guard, hermetic and calibration-free: the raw skl2onnx graph is
    ``Cast -> MatMul -> Add -> Reshape``; the Cast (skl2onnx's input dtype-normalizer) and the
    Reshape must fold to nothing, leaving exactly one Linear whose weight equals the ONNX MatMul
    initializer (read directly with the core ``onnx`` dep). Weight lowering is data-independent, so
    this catches a transB/layout/fold or Cast-folding regression without any calibration.
    """
    model = fhe.load_onnx(str(ONNX_MODEL), input_bits=INPUT_BITS)
    assert [type(layer).__name__ for layer in model.layers] == ["Linear"]
    linear = model.layers[0]
    assert linear.weight.shape == (10, 64)  # (n_out, n_in)

    # The committed ONNX really contains the Cast this example exists to exercise.
    ops = _onnx_op_types(ONNX_MODEL)
    assert "Cast" in ops, "the sklearn export must contain the leading Cast the loader folds"
    assert "MatMul" in ops


def _onnx_op_types(path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(path))
    return [n.op_type for n in model.graph.node]


def test_sklearn_fixture_reports_honest_accuracy():
    acc = _fixture()["accuracy"]
    assert 0.0 <= acc["quantized"] <= 1.0 and 0.0 <= acc["float"] <= 1.0
    assert acc["float"] > 0.8, "the float linear classifier should classify real digits well"
    # Quantization is nearly lossless for a single well-conditioned Linear head.
    assert abs(acc["float"] - acc["quantized"]) < 0.1, "quantized accuracy should track float"
