"""Phase-11 tabular MLP fixture guard, Python side (``AGENTS.md`` §1.1, §5).

The committed ``phase11_tabular_mlp_fixture.json`` is the gate the Rust TFHE and CKKS golden tests
evaluate. This test guards the *emit* end hermetically:

1. The committed graph fits its radix budget and round-trips through ``ir.py``.
2. Re-running ``insert_requants`` on the committed graph is a no-op (idempotence).
3. The committed ``expected_logits`` and ``expected_labels`` are **exactly** what
   :func:`penumbra.reference.evaluate_graph_int` produces on ``test_inputs``.
4. Loading ``breast_cancer_mlp.onnx`` through ``load_onnx`` lowers to the expected float layers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import penumbra as fhe
from penumbra.bitwidth import check_bit_width_budget
from penumbra.compile import insert_requants
from penumbra.ir import Graph, RequantSpec
from penumbra.layers import Activation, Linear
from penumbra.reference import evaluate_graph_int

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "tabular"
ONNX_MODEL = EXAMPLES / "breast_cancer_mlp.onnx"
FIXTURE = EXAMPLES / "phase11_tabular_mlp_fixture.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def test_tabular_mlp_fixture_graph_round_trips_and_fits_budget():
    """The committed graph deserializes, fits its radix budget, and round-trips."""
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    assert Graph.from_json(g.to_json()) == g, "IR round-trip through JSON must be exact"
    check_bit_width_budget(g)


def test_tabular_mlp_fixture_graph_structure_and_idempotent():
    """Graph structure is Linear -> Requant -> Linear; Requant has fused ReLU (clamp_lo == 0)."""
    g = Graph.from_dict(_fixture()["graph"])
    op_types = [n.op.op_type for n in g.nodes]
    assert op_types == ["Linear", "Requant", "Linear"]

    rq = g.nodes[1].op
    assert isinstance(rq, RequantSpec)
    assert rq.clamp_lo == 0, f"expected clamp_lo == 0 (fused ReLU), got {rq.clamp_lo}"

    assert insert_requants(g) == g, "insert_requants must be idempotent on the committed graph"


def test_tabular_mlp_fixture_logits_and_labels_match_oracle():
    """Committed logits/labels are exactly what the integer reference produces (drift guard)."""
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    in_name = g.inputs[0]
    out_name = g.outputs[0]
    for i, (x, exp_logits) in enumerate(zip(fx["test_inputs"], fx["expected_logits"], strict=True)):
        out = evaluate_graph_int(g, {in_name: x})
        logits = out[out_name]
        assert logits == exp_logits, f"sample {i}: logits drifted from oracle"
        assert int(np.argmax(logits)) == fx["expected_labels"][i], f"sample {i}: label drifted"


def test_committed_tabular_mlp_onnx_lowers_to_expected_float_layers():
    """load_onnx lowers breast_cancer_mlp.onnx to [Linear, Activation, Linear]."""
    fx = _fixture()
    input_bits = fx["bit_plan"]["input_bits"]
    model = fhe.load_onnx(str(ONNX_MODEL), input_bits=input_bits)
    assert len(model.layers) == 3
    assert isinstance(model.layers[0], Linear)
    assert isinstance(model.layers[1], Activation)
    assert isinstance(model.layers[2], Linear)

    assert model.layers[0].weight.shape == (8, 30)
    assert model.layers[2].weight.shape == (2, 8)


def test_tabular_mlp_fixture_reports_honest_accuracy():
    """Accuracy report records float and quantized test accuracy within expected bounds."""
    acc = _fixture()["accuracy"]
    assert 0.0 <= acc["quantized"] <= 1.0
    assert 0.0 <= acc["float"] <= 1.0
    assert acc["float"] >= 0.90, "the float MLP should achieve >= 0.90 test accuracy"
    assert acc["quantized"] >= 0.85, "quantized accuracy should be >= 0.85"
