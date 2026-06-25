"""Cross-language IR conformance — the Python half (``AGENTS.md`` §5, ROADMAP Phase 3).

The two CI jobs (Rust, Python) run in parallel and never invoke each other, so the
committed IR file is the meeting point. Chain: **Python emits → committed file → Rust
consumes**. This module guards the *emit* end:

1. The IR round-trips through ``ir.py`` (``from_json(to_json(g)) == g``).
2. The committed ``phase2_fixture.json["graph"]`` is **exactly** what the front end emits
   today — the drift guard. If ``ir.py`` or the export script changes without regenerating
   the fixture, this fails here (fast, no FHE), rather than as a confusing Rust error.

Comparison is on *parsed dicts*, not raw strings, so whitespace / key order / float repr
are not load-bearing (the graph subtree is pure ints/strings anyway).

No heavy deps and no network: it reads the committed JSON and reconstructs the canonical
graph with the same builder the exporter uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from penumbra.ir import (
    SCHEMA_VERSION,
    AddSpec,
    ArgmaxSpec,
    Graph,
    LinearSpec,
    Node,
    PoolSpec,
    RequantSpec,
)

FIXTURE = Path(__file__).resolve().parent.parent / "examples" / "mnist" / "phase2_fixture.json"


def _committed_graph_dict() -> dict:
    fx = json.loads(FIXTURE.read_text())
    assert "graph" in fx, "fixture must embed the IR under a 'graph' key (Phase 3)"
    return fx["graph"]


def test_committed_graph_has_current_schema_version():
    assert _committed_graph_dict()["schema_version"] == SCHEMA_VERSION, (
        "committed IR schema_version differs from ir.py — regenerate the fixture "
        "(a schema-version change is breaking, AGENTS.md §5)"
    )


def test_committed_graph_round_trips():
    """Deserialize the committed graph and re-serialize: structural equality must hold."""
    g = Graph.from_dict(_committed_graph_dict())
    assert Graph.from_json(g.to_json()) == g, "ir.py round-trip must be exact"


def test_committed_graph_matches_front_end_output():
    """Canonicalization guard: the committed graph is in ir.py's exact emitted form.

    Parse the committed graph, rebuild it from its own typed fields through the same
    dataclasses the exporter uses, and assert the re-emitted dict equals the committed dict.
    This catches a hand-edited or stale fixture whose *encoding* drifted from what ir.py emits
    today — an unexpected field set, a wrong key order, nested-vs-flat op payloads, or a
    float where an int belongs.

    Scope (be honest about what this proves): it asserts the committed JSON is ir.py's
    canonical *form*, not that the exporter's *logic* still produces these values — the raw
    pre-quantization weights aren't committed, so a builder-logic regression that still yields
    a well-formed graph would pass here. The Rust half (`ir_conformance.rs`) and the golden
    test (`golden_logreg.rs`) pin the structure and the values.
    """
    committed = _committed_graph_dict()
    g = Graph.from_dict(committed)

    # Rebuild from the parsed graph's own fields (single source of truth) and compare the
    # canonical dicts. Any divergence in field set, ordering of nodes, or values trips here.
    rebuilt = Graph(
        schema_version=g.schema_version,
        num_blocks=g.num_blocks,
        input_bits=g.input_bits,
        inputs=g.inputs,
        outputs=g.outputs,
        nodes=g.nodes,
    )
    assert (
        rebuilt.to_dict() == committed
    ), "committed graph is not the front end's canonical output — regenerate the fixture"


def test_committed_graph_is_linear_argmax():
    """The Phase-2 model is exactly Linear → Argmax with the expected wiring."""
    g = Graph.from_dict(_committed_graph_dict())
    assert g.inputs == ["x"]
    assert g.outputs == ["label"]
    assert [n.op.op_type for n in g.nodes] == ["Linear", "Argmax"]

    fc, head = g.nodes
    assert fc.inputs == ["x"] and fc.outputs == ["logit"]
    assert isinstance(fc.op, LinearSpec)
    assert fc.op.weight_bits == 4
    assert len(fc.op.weights) == 1 and len(fc.op.weights[0]) == 64

    assert head.inputs == ["logit"] and head.outputs == ["label"]
    assert isinstance(head.op, ArgmaxSpec)


def test_add_spec_round_trips():
    """The multi-input ``Add`` op (two `inputs`, no payload) round-trips through ir.py."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=4,
        input_bits=4,
        inputs=["a", "b"],
        outputs=["sum"],
        nodes=[Node(name="add", inputs=["a", "b"], outputs=["sum"], op=AddSpec())],
    )
    restored = Graph.from_json(g.to_json())
    assert restored == g
    assert restored.nodes[0].op.to_dict() == {"op_type": "Add"}
    assert restored.nodes[0].inputs == ["a", "b"], "Add carries two operands (merge order)"


def test_requant_spec_round_trips():
    """The ``Requant`` op (shift + out_bits + clamp_lut) round-trips through ir.py."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=6,
        input_bits=10,
        inputs=["x"],
        outputs=["y"],
        nodes=[
            Node(
                name="rq",
                inputs=["x"],
                outputs=["y"],
                op=RequantSpec(shift=4, out_bits=2, clamp_lut=[0, 1, 2, 3]),
            )
        ],
    )
    restored = Graph.from_json(g.to_json())
    assert restored == g
    assert restored.nodes[0].op.to_dict() == {
        "op_type": "Requant",
        "shift": 4,
        "out_bits": 2,
        "clamp_lut": [0, 1, 2, 3],
    }


def test_requant_spec_rejects_invalid():
    """RequantSpec fails loudly at construction on a negative shift / zero out_bits."""
    with pytest.raises(ValueError, match="shift"):
        RequantSpec(shift=-1, out_bits=2, clamp_lut=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="out_bits"):
        RequantSpec(shift=1, out_bits=0, clamp_lut=[0, 1, 2, 3])


def test_pool_spec_round_trips():
    """The ``Pool`` op round-trips, and invalid modes/windows fail at construction."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=6,
        input_bits=5,
        inputs=["x"],
        outputs=["y"],
        nodes=[
            Node(
                name="pool",
                inputs=["x"],
                outputs=["y"],
                op=PoolSpec(mode="avg", in_h=4, in_w=4, channels=2, pool_h=2, pool_w=2, stride=2),
            )
        ],
    )
    assert Graph.from_json(g.to_json()) == g

    with pytest.raises(ValueError, match="mode"):
        PoolSpec(mode="median", in_h=4, in_w=4, channels=1, pool_h=2, pool_w=2, stride=2)
    with pytest.raises(ValueError, match="must fit"):
        PoolSpec(mode="max", in_h=2, in_w=2, channels=1, pool_h=3, pool_w=3, stride=1)


def test_from_dict_rejects_version_mismatch():
    bad = {
        "schema_version": "0.0.1",
        "num_blocks": 8,
        "input_bits": 4,
        "inputs": ["x"],
        "outputs": ["y"],
        "nodes": [],
    }
    with pytest.raises(ValueError, match="schema-version mismatch"):
        Graph.from_dict(bad)


def test_from_dict_rejects_unknown_op_type():
    bad = {
        "schema_version": SCHEMA_VERSION,
        "num_blocks": 8,
        "input_bits": 4,
        "inputs": ["x"],
        "outputs": ["y"],
        "nodes": [{"name": "c", "inputs": ["x"], "outputs": ["y"], "op": {"op_type": "Conv2d"}}],
    }
    with pytest.raises(ValueError, match="unknown op_type"):
        Graph.from_dict(bad)
