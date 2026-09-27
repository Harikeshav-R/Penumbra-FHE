"""Phase-8 tree ensemble fixture guards (AGENTS.md §1.1, §5).

Hermetic, always-on drift guard for both committed tree fixtures:
    - examples/trees/phase8_trees_fixture.json (scikit-learn RandomForestClassifier)
    - examples/trees/phase8_xgb_fixture.json (XGBoost XGBClassifier)

No sklearn or xgboost imports required — verifies the committed fixtures against
penumbra-core IR and the Python reference oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from penumbra.bitwidth import check_bit_width_budget
from penumbra.ir import Graph
from penumbra.reference import evaluate_graph_int

TREES_DIR = Path(__file__).resolve().parent.parent / "examples" / "trees"
TREES_FIXTURE = TREES_DIR / "phase8_trees_fixture.json"
XGB_FIXTURE = TREES_DIR / "phase8_xgb_fixture.json"


@pytest.mark.parametrize("path", [TREES_FIXTURE, XGB_FIXTURE], ids=["sklearn", "xgboost"])
def test_tree_fixture_graph_round_trips_and_fits_budget(path: Path) -> None:
    fx = json.loads(path.read_text())
    g = Graph.from_dict(fx["graph"])
    assert Graph.from_json(g.to_json()) == g, "tree-lowered IR round-trip must be exact"
    check_bit_width_budget(g)


@pytest.mark.parametrize("path", [TREES_FIXTURE, XGB_FIXTURE], ids=["sklearn", "xgboost"])
def test_tree_fixture_nodes_sequence_is_four_stages(path: Path) -> None:
    fx = json.loads(path.read_text())
    g = Graph.from_dict(fx["graph"])
    assert [n.op.op_type for n in g.nodes] == ["Compare", "Linear", "Compare", "Linear"]


@pytest.mark.parametrize("path", [TREES_FIXTURE, XGB_FIXTURE], ids=["sklearn", "xgboost"])
def test_tree_fixture_logits_and_labels_match_oracle(path: Path) -> None:
    fx = json.loads(path.read_text())
    g = Graph.from_dict(fx["graph"])
    for i, (x, expected) in enumerate(zip(fx["test_inputs"], fx["expected_logits"], strict=True)):
        logits = evaluate_graph_int(g, {"x": x})[g.outputs[0]]
        assert logits == expected, f"sample {i}: logits drifted from oracle"
        assert int(np.argmax(logits)) == fx["expected_labels"][i], f"sample {i}: label drifted"


@pytest.mark.parametrize("path", [TREES_FIXTURE, XGB_FIXTURE], ids=["sklearn", "xgboost"])
def test_tree_fixture_reports_honest_accuracy(path: Path) -> None:
    acc = json.loads(path.read_text())["accuracy"]
    assert 0.0 <= acc["quantized"] <= 1.0 and 0.0 <= acc["float"] <= 1.0
    assert acc["float"] > 0.9, "float tree ensemble on breast cancer should exceed 90% accuracy"
    gap = abs(acc["float"] - acc["quantized"])
    assert gap < 0.05, f"quantized accuracy should track float, gap was {gap:.4f}"
