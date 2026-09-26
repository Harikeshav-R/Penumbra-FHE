"""Phase-8 Branch MLP fixture guard, Python side (``AGENTS.md`` §1.1, §5).

The committed ``phase8_branch_fixture.json`` is the gate the Rust TFHE and CKKS golden tests
evaluate. This test guards the *emit* end hermetically:

1. The committed graph fits its radix budget and round-trips through ``ir.py``.
2. Re-running ``insert_requants`` on the committed graph is a no-op (idempotence).
3. The committed ``expected_logits`` and ``expected_labels`` are **exactly** what
   :func:`penumbra.reference.evaluate_graph_int` produces on ``test_inputs``.
4. Loading ``digit_branch_mlp.onnx`` through ``load_onnx`` lowers to the expected DAG nodes.
5. Fixture records honest float and quantized accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import penumbra as fhe
from penumbra.bitwidth import check_bit_width_budget
from penumbra.compile import insert_requants
from penumbra.ir import ConcatSpec, Graph, SplitSpec
from penumbra.layers import Activation, Add, Concat, Linear, Split
from penumbra.reference import evaluate_graph_int

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "mnist"
ONNX_MODEL = EXAMPLES / "digit_branch_mlp.onnx"
FIXTURE = EXAMPLES / "phase8_branch_fixture.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def test_branch_fixture_graph_round_trips_and_fits_budget():
    """The committed graph deserializes, fits its radix budget, and round-trips."""
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    assert Graph.from_json(g.to_json()) == g, "IR round-trip through JSON must be exact"
    check_bit_width_budget(g)


def test_branch_fixture_graph_structure_and_idempotent():
    """Graph structure contains Split, Concat, Add, and Requant; insert_requants is idempotent."""
    g = Graph.from_dict(_fixture()["graph"])
    op_types = [n.op.op_type for n in g.nodes]
    assert "Split" in op_types, "Split op must be present in IR graph"
    assert "Concat" in op_types, "Concat op must be present in IR graph"
    assert "Add" in op_types, "Add op must be present in IR graph"
    assert "Linear" in op_types, "Linear op must be present in IR graph"
    assert "Requant" in op_types, "Requant op must be present in IR graph"

    split_node = next(n for n in g.nodes if isinstance(n.op, SplitSpec))
    assert isinstance(split_node.op, SplitSpec)
    assert split_node.op.sizes == [8, 8]
    concat_node = next(n for n in g.nodes if isinstance(n.op, ConcatSpec))
    assert isinstance(concat_node.op, ConcatSpec)
    assert concat_node.op.sizes == [8, 8]

    assert insert_requants(g) == g, "insert_requants must be idempotent on the committed graph"


def test_branch_fixture_logits_and_labels_match_oracle():
    """Committed logits/labels are exactly what the integer reference produces (drift guard)."""
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    for i, (x, exp_logits) in enumerate(zip(fx["test_inputs"], fx["expected_logits"], strict=True)):
        out = evaluate_graph_int(g, {"x": x})
        logits = out[g.outputs[0]]
        assert logits == exp_logits, f"sample {i}: logits drifted from oracle"
        assert int(np.argmax(logits)) == fx["expected_labels"][i], f"sample {i}: label drifted"


def test_committed_branch_onnx_lowers_to_expected_nodes():
    """load_onnx lowers the committed digit_branch_mlp.onnx to the branching DAG."""
    model = fhe.load_onnx(str(ONNX_MODEL), input_bits=4)
    layer_types = [type(n.layer) for n in model.nodes]
    assert Linear in layer_types
    assert Activation in layer_types
    assert Split in layer_types
    assert Concat in layer_types
    assert Add in layer_types


def test_branch_fixture_reports_honest_accuracy():
    """Accuracy report records float and quantized test accuracy within expected bounds."""
    acc = _fixture()["accuracy"]
    assert 0.0 <= acc["quantized"] <= acc["float"] <= 1.0
    assert acc["float"] >= 0.90, "the float MLP should achieve >= 0.90 test accuracy"
    assert acc["quantized"] >= 0.70, "quantized accuracy should be >= 0.70"
