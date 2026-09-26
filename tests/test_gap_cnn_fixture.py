"""Phase-8 padded-pool + GAP CNN fixture guard, Python side (``AGENTS.md`` §1.1, §5).

The committed ``phase8_gap_cnn_fixture.json`` is the gate the Rust TFHE and CKKS golden tests
evaluate. This test guards the *emit* end hermetically:

1. The committed graph fits its radix budget and round-trips through ``ir.py``.
2. Re-running ``insert_requants`` on the committed graph is a no-op (idempotence).
3. The committed ``expected_logits`` and ``expected_labels`` are **exactly** what
   :func:`penumbra.reference.evaluate_graph_int` produces on ``test_inputs``.
4. Loading ``digit_gap_cnn.onnx`` through ``load_onnx`` lowers to Conv2d, Activation,
   Pool(pad=1), Pool(global), Linear.
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
from penumbra.ir import Graph, PoolSpec
from penumbra.layers import Activation, Conv2d, Linear, Pool
from penumbra.reference import evaluate_graph_int

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "mnist"
ONNX_MODEL = EXAMPLES / "digit_gap_cnn.onnx"
FIXTURE = EXAMPLES / "phase8_gap_cnn_fixture.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def test_gap_cnn_fixture_graph_round_trips_and_fits_budget():
    """The committed graph deserializes, fits its radix budget, and round-trips."""
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    assert Graph.from_json(g.to_json()) == g, "IR round-trip through JSON must be exact"
    check_bit_width_budget(g)


def test_gap_cnn_fixture_graph_structure_and_idempotent():
    """Graph structure is Conv2d -> Requant -> Pool(pad=1) -> Pool(global) -> Linear."""
    g = Graph.from_dict(_fixture()["graph"])
    op_types = [n.op.op_type for n in g.nodes]
    assert op_types == ["Conv2d", "Requant", "Pool", "Pool", "Linear"]

    pool_nodes = [n.op for n in g.nodes if isinstance(n.op, PoolSpec)]
    assert len(pool_nodes) == 2
    assert pool_nodes[0].padding == 1
    assert pool_nodes[0].pool_h == 3
    assert pool_nodes[1].padding == 0
    assert pool_nodes[1].pool_h == pool_nodes[1].in_h

    assert insert_requants(g) == g, "insert_requants must be idempotent on the committed graph"


def test_gap_cnn_fixture_logits_and_labels_match_oracle():
    """Committed logits/labels are exactly what the integer reference produces (drift guard)."""
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    for i, (x, exp_logits) in enumerate(zip(fx["test_inputs"], fx["expected_logits"], strict=True)):
        out = evaluate_graph_int(g, {"x": x})
        logits = out[g.outputs[0]]
        assert logits == exp_logits, f"sample {i}: logits drifted from oracle"
        assert int(np.argmax(logits)) == fx["expected_labels"][i], f"sample {i}: label drifted"


def test_committed_gap_cnn_onnx_lowers_with_padded_pool():
    """load_onnx lowers to 5 layers (Conv2d, Activation, Pool, Pool, Linear) with padded pool."""
    model = fhe.load_onnx(str(ONNX_MODEL), input_bits=4)
    layer_types = [type(n.layer) for n in model.nodes]
    assert layer_types == [Conv2d, Activation, Pool, Pool, Linear]
    p1 = model.layers[2]
    p2 = model.layers[3]
    assert isinstance(p1, Pool) and isinstance(p2, Pool)
    assert p1.padding == 1
    assert p2.padding == 0


def test_gap_cnn_fixture_reports_honest_accuracy():
    """Accuracy report records float and quantized test accuracy within expected bounds."""
    acc = _fixture()["accuracy"]
    assert 0.0 <= acc["quantized"] <= 1.0
    assert 0.0 <= acc["float"] <= 1.0
    assert acc["float"] >= 0.75, "the float CNN should achieve >= 0.75 test accuracy"
    assert acc["quantized"] >= 0.50, "quantized accuracy should be >= 0.50"
