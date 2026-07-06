"""Phase-5 real-digits fixture guard, Python side (``AGENTS.md`` §1.1, §5).

The bit-for-bit FHE gate for the real-digit CNN lives in Rust (``runtime/tests/golden_digits.rs``,
``#[ignore]`` by default — minutes per sample). This Python test guards the *other* half, in CI on
every run with no FHE and no ML stack: the committed ``phase5_digits_fixture.json`` is
self-consistent — its ``expected_logits``/``expected_labels`` are exactly what the quantized-integer
reference (:func:`penumbra.reference.evaluate_graph_int`, the golden oracle) produces, the graph
round-trips and fits its radix budget, and re-running the compile pass is a no-op (the committed
graph already carries its auto-inserted Requant). If the fixture drifts from the oracle, this fails
here (fast) rather than as a confusing Rust golden violation.

NumPy-only and no network: it reads the committed JSON and recomputes the integer CNN with the
library's reference evaluator. The torch/sklearn training that *produced* the fixture is the
example generator's job (the optional ``ml`` extra), never CI's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from penumbra.bitwidth import check_bit_width_budget
from penumbra.compile import insert_requants
from penumbra.ir import Graph
from penumbra.reference import evaluate_graph_int

FIXTURE = (
    Path(__file__).resolve().parent.parent / "examples" / "mnist" / "phase5_digits_fixture.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_digits_graph_round_trips_and_fits_budget():
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    assert Graph.from_json(g.to_json()) == g, "digits IR round-trip must be exact"
    check_bit_width_budget(g)  # raises if any tensor / Requant internal peak exceeds the radix


def test_digits_graph_is_conv_requant_linear_and_idempotent():
    """The committed graph is Conv2d -> Requant(fused ReLU) -> Linear, with the Requant present."""
    g = Graph.from_dict(_fixture()["graph"])
    kinds = [n.op.op_type for n in g.nodes]
    assert kinds == ["Conv2d", "Requant", "Linear"], kinds
    # Re-running the compile pass must be a no-op on the already-requantized committed graph.
    assert insert_requants(g) == g, "insert_requants must be idempotent on the committed graph"


def test_digits_fixture_logits_and_labels_match_oracle():
    """Committed logits/labels are exactly what the integer reference produces (drift guard)."""
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    for i, (x, expected) in enumerate(zip(fx["test_inputs"], fx["expected_logits"], strict=True)):
        logits = evaluate_graph_int(g, {"x": x})[g.outputs[0]]
        assert logits == expected, f"sample {i}: logits drifted from the oracle"
        assert int(np.argmax(logits)) == fx["expected_labels"][i], f"sample {i}: label drifted"


def test_digits_fixture_reports_honest_accuracy():
    """The committed accuracy metadata is present and honest (float >= quantized, real gap)."""
    acc = _fixture()["accuracy"]
    assert 0.0 <= acc["quantized"] <= acc["float"] <= 1.0
    # 2-bit activations are genuinely lossy; the gap is real, not a bug. Just sanity-bound it.
    assert acc["float"] > 0.8, "the float CNN should classify real digits well"
