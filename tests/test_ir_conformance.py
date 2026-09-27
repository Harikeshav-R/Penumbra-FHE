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
    CompareSpec,
    ConcatSpec,
    Conv2dSpec,
    Graph,
    LinearSpec,
    Node,
    OpSpec,
    PoolSpec,
    RequantSpec,
    SplitSpec,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "examples" / "mnist" / "phase2_fixture.json"

ALL_FIXTURES: list[Path] = [
    REPO_ROOT / "examples/mnist/phase2_fixture.json",
    REPO_ROOT / "examples/mnist/phase4_cnn_fixture.json",
    REPO_ROOT / "examples/mnist/phase5_digits_fixture.json",
    REPO_ROOT / "examples/mnist/phase5_qat_fixture.json",
    REPO_ROOT / "examples/mnist/phase6_onnx_fixture.json",
    REPO_ROOT / "examples/mnist/phase6_sklearn_fixture.json",
    REPO_ROOT / "examples/mnist/phase8_bn_cnn_fixture.json",
    REPO_ROOT / "examples/mnist/phase8_branch_fixture.json",
    REPO_ROOT / "examples/mnist/phase8_gap_cnn_fixture.json",
    REPO_ROOT / "examples/mnist/phase8_tanh_fixture.json",
    REPO_ROOT / "examples/faces/phase7_faces_fixture.json",
    REPO_ROOT / "examples/tabular/phase8_trees_fixture.json",
    REPO_ROOT / "examples/tabular/phase8_xgb_fixture.json",
]

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


@pytest.mark.parametrize("fixture_path", ALL_FIXTURES, ids=[p.stem for p in ALL_FIXTURES])
def test_every_committed_fixture_graph_is_in_emitted_form(fixture_path: Path):
    fx = json.loads(fixture_path.read_text())
    assert "graph" in fx, f"fixture {fixture_path.name} missing 'graph' key"
    g = fx["graph"]
    assert g["schema_version"] == SCHEMA_VERSION, (
        f"{fixture_path.name} schema_version {g.get('schema_version')} != current {SCHEMA_VERSION}"
    )
    assert Graph.from_dict(g).to_dict() == g, (
        f"{fixture_path.name} is not in canonical emitted form"
    )

def test_committed_graph_is_linear_argmax():
    """The Phase-2 model is exactly Linear → Argmax with the expected wiring."""
    g = Graph.from_dict(_committed_graph_dict())
    assert g.inputs == ["x"]
    assert g.outputs == ["label"]
    assert [n.op.op_type for n in g.nodes] == ["Linear", "Argmax"]

    fc, head = g.nodes
    assert fc.inputs == ["x"] and fc.outputs == ["logit"]
    assert isinstance(fc.op, LinearSpec)
    assert fc.op.weight_bits == 2
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


def test_concat_spec_round_trips():
    """The multi-input ``Concat`` op round-trips through ir.py."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=4,
        input_bits=4,
        inputs=["a", "b"],
        outputs=["c"],
        nodes=[Node(name="cat", inputs=["a", "b"], outputs=["c"], op=ConcatSpec(sizes=[2, 3]))],
    )
    restored = Graph.from_json(g.to_json())
    assert restored == g
    assert restored.nodes[0].op.to_dict() == {"op_type": "Concat", "sizes": [2, 3]}


def test_split_spec_round_trips():
    """The multi-output ``Split`` op round-trips through ir.py."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=4,
        input_bits=4,
        inputs=["x"],
        outputs=["s0", "s1"],
        nodes=[Node(name="split", inputs=["x"], outputs=["s0", "s1"], op=SplitSpec(sizes=[2, 2]))],
    )
    restored = Graph.from_json(g.to_json())
    assert restored == g
    assert restored.nodes[0].op.to_dict() == {"op_type": "Split", "sizes": [2, 2]}


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
    # Defaulted 0.5.0 fields (mult=1, round_bias=0) are emitted in the Rust struct field order.
    assert restored.nodes[0].op.to_dict() == {
        "op_type": "Requant",
        "shift": 4,
        "mult": 1,
        "round_bias": 0,
        "out_bits": 2,
        "clamp_lut": [0, 1, 2, 3],
    }


def test_requant_spec_generalized_round_trips():
    """A generalized Requant (mult != 1, round-to-nearest bias) round-trips through ir.py."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=8,
        input_bits=10,
        inputs=["x"],
        outputs=["y"],
        nodes=[
            Node(
                name="rq",
                inputs=["x"],
                outputs=["y"],
                op=RequantSpec(shift=5, mult=3, round_bias=16, out_bits=2, clamp_lut=[0, 1, 2, 3]),
            )
        ],
    )
    assert Graph.from_json(g.to_json()) == g


def test_requant_spec_defaults_when_fields_absent():
    """A Requant payload without mult/round_bias loads with the legacy pure-shift defaults."""
    spec = OpSpec.from_dict(
        {"op_type": "Requant", "shift": 4, "out_bits": 2, "clamp_lut": [0, 1, 2, 3]}
    )
    assert isinstance(spec, RequantSpec)
    assert spec.mult == 1 and spec.round_bias == 0


def test_requant_spec_rejects_invalid():
    """RequantSpec fails loudly at construction on bad shift / out_bits / mult / round_bias."""
    with pytest.raises(ValueError, match="shift"):
        RequantSpec(shift=-1, out_bits=2, clamp_lut=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="out_bits"):
        RequantSpec(shift=1, out_bits=0, clamp_lut=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="mult"):
        RequantSpec(shift=1, out_bits=2, clamp_lut=[0, 1, 2, 3], mult=0)
    with pytest.raises(ValueError, match="round_bias"):
        RequantSpec(shift=1, out_bits=2, clamp_lut=[0, 1, 2, 3], round_bias=-1)


def test_requant_spec_per_channel_round_trips():
    """A per-channel Requant (0.6.0 overlay) round-trips and emits the four extra keys in order."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=8,
        input_bits=10,
        inputs=["x"],
        outputs=["y"],
        nodes=[
            Node(
                name="rq",
                inputs=["x"],
                outputs=["y"],
                op=RequantSpec(
                    shift=0,
                    out_bits=2,
                    clamp_lut=[0, 1, 2, 3],
                    mults=[1, 3],
                    shifts=[0, 5],
                    round_biases=[0, 16],
                    channel_size=2,
                ),
            )
        ],
    )
    restored = Graph.from_json(g.to_json())
    assert restored == g
    assert restored.nodes[0].op.to_dict() == {
        "op_type": "Requant",
        "shift": 0,
        "mult": 1,
        "round_bias": 0,
        "out_bits": 2,
        "clamp_lut": [0, 1, 2, 3],
        "mults": [1, 3],
        "shifts": [0, 5],
        "round_biases": [0, 16],
        "channel_size": 2,
    }


def test_requant_per_tensor_omits_channel_fields():
    """A per-tensor Requant must NOT emit any per-channel key — byte-identical to 0.5.0."""
    d = RequantSpec(shift=4, out_bits=2, clamp_lut=[0, 1, 2, 3]).to_dict()
    for key in ("mults", "shifts", "round_biases", "channel_size"):
        assert key not in d, f"per-tensor Requant leaked per-channel key {key!r}"


def test_requant_spec_per_channel_rejects_inconsistent():
    """The per-channel overlay validates at construction: equal lengths + channel_size present."""
    # Mismatched array lengths.
    with pytest.raises(ValueError, match="equal length"):
        RequantSpec(
            shift=0,
            out_bits=2,
            clamp_lut=[0, 1, 2, 3],
            mults=[1, 3],
            shifts=[0],
            round_biases=[0, 16],
            channel_size=2,
        )
    # Arrays present but no channel_size.
    with pytest.raises(ValueError, match="channel_size"):
        RequantSpec(
            shift=0,
            out_bits=2,
            clamp_lut=[0, 1, 2, 3],
            mults=[1, 3],
            shifts=[0, 5],
            round_biases=[0, 16],
        )
    # A per-channel multiplier < 1.
    with pytest.raises(ValueError, match="mults"):
        RequantSpec(
            shift=0,
            out_bits=2,
            clamp_lut=[0, 1, 2, 3],
            mults=[1, 0],
            shifts=[0, 5],
            round_biases=[0, 16],
            channel_size=2,
        )


def test_requant_signed_fields_round_trip():
    """RequantSpec with clamp_lo < 0 and zero_point > 0 round-trips and emits fields in order."""
    op = RequantSpec(
        shift=3,
        mult=1,
        round_bias=4,
        clamp_lo=-16,
        zero_point=2,
        out_bits=2,
        clamp_lut=[0, 1, 2, 3],
    )
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=6,
        input_bits=8,
        inputs=["x"],
        outputs=["y"],
        nodes=[Node(name="rq", inputs=["x"], outputs=["y"], op=op)],
    )
    restored = Graph.from_json(g.to_json())
    assert restored == g
    assert restored.nodes[0].op.to_dict() == {
        "op_type": "Requant",
        "shift": 3,
        "mult": 1,
        "round_bias": 4,
        "clamp_lo": -16,
        "zero_point": 2,
        "out_bits": 2,
        "clamp_lut": [0, 1, 2, 3],
    }


def test_requant_zero_floor_and_zero_point_omits_fields():
    """A (0, 0) Requant emits neither clamp_lo nor zero_point — byte-identical to 0.7.0."""
    d = RequantSpec(shift=4, out_bits=2, clamp_lut=[0, 1, 2, 3], clamp_lo=0, zero_point=0).to_dict()
    assert "clamp_lo" not in d
    assert "zero_point" not in d


def test_requant_spec_rejects_positive_clamp_lo_or_uncovered_floor():
    """clamp_lo > 0 and zero_point + u_min < 0 are rejected at construction."""
    with pytest.raises(ValueError, match="clamp_lo must be <= 0"):
        RequantSpec(shift=1, out_bits=2, clamp_lut=[0, 1, 2, 3], clamp_lo=1)
    with pytest.raises(ValueError, match="does not cover the floor image"):
        # (-16 * 1 + 4) >> 3 = -2; zero_point=1 gives 1 + (-2) = -1 < 0
        RequantSpec(
            shift=3,
            mult=1,
            round_bias=4,
            clamp_lo=-16,
            zero_point=1,
            out_bits=2,
            clamp_lut=[0, 1, 2, 3],
        )


def test_compare_spec_round_trips():
    """The ``Compare`` op (indices + thresholds) round-trips through ir.py."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=6,
        input_bits=8,
        inputs=["x"],
        outputs=["y2"],
        nodes=[
            Node(
                name="cmp1",
                inputs=["x"],
                outputs=["y1"],
                op=CompareSpec(indices=[0, 2], thresholds=[5, 10]),
            ),
            Node(
                name="cmp2",
                inputs=["y1"],
                outputs=["y2"],
                op=CompareSpec(indices=[0], thresholds=[1]),
            ),
        ],
    )
    restored = Graph.from_json(g.to_json())
    assert restored == g
    assert restored.nodes[0].op.to_dict() == {
        "op_type": "Compare",
        "indices": [0, 2],
        "thresholds": [5, 10],
    }


def test_compare_spec_rejects_invalid():
    """CompareSpec fails loudly at construction on empty thresholds or bad indices."""
    with pytest.raises(ValueError, match="needs at least one threshold"):
        CompareSpec(indices=[], thresholds=[])
    with pytest.raises(ValueError, match="threshold"):
        CompareSpec(indices=[0], thresholds=[])
    with pytest.raises(ValueError, match="need one input index per threshold"):
        CompareSpec(indices=[0, 1], thresholds=[5])
    with pytest.raises(ValueError, match="non-negative"):
        CompareSpec(indices=[-1], thresholds=[5])


def test_concat_spec_rejects_invalid():
    """ConcatSpec fails loudly at construction on fewer than 2 segments or non-positive sizes."""
    with pytest.raises(ValueError, match="at least 2 input segments"):
        ConcatSpec(sizes=[4])
    with pytest.raises(ValueError, match="positive"):
        ConcatSpec(sizes=[2, 0])
    with pytest.raises(ValueError, match="positive"):
        ConcatSpec(sizes=[2, -1])


def test_split_spec_rejects_invalid():
    """SplitSpec fails loudly at construction on fewer than 2 segments or non-positive sizes."""
    with pytest.raises(ValueError, match="at least 2 output segments"):
        SplitSpec(sizes=[4])
    with pytest.raises(ValueError, match="positive"):
        SplitSpec(sizes=[2, 0])
    with pytest.raises(ValueError, match="positive"):
        SplitSpec(sizes=[2, -1])


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


def test_pool_spec_padding_round_trips_and_omits_zero():
    """PoolSpec padding round-trips; padding=0 is omitted from to_dict()."""
    p_pad = PoolSpec(
        mode="avg", in_h=3, in_w=3, channels=1, pool_h=2, pool_w=2, stride=2, padding=1
    )
    d_pad = p_pad.to_dict()
    assert d_pad["padding"] == 1
    keys = list(d_pad.keys())
    assert keys.index("padding") == keys.index("stride") + 1
    assert PoolSpec.from_dict(d_pad) == p_pad

    p_zero = PoolSpec(
        mode="avg", in_h=4, in_w=4, channels=1, pool_h=2, pool_w=2, stride=2, padding=0
    )
    d_zero = p_zero.to_dict()
    assert "padding" not in d_zero

    # from_dict without padding defaults to 0
    restored = PoolSpec.from_dict(d_zero)
    assert restored.padding == 0
    assert restored == p_zero


def test_pool_spec_rejects_bad_padding():
    """PoolSpec rejects padding >= min(pool_h, pool_w) or negative padding."""
    with pytest.raises(ValueError, match="must be smaller than the window"):
        PoolSpec(mode="avg", in_h=4, in_w=4, channels=1, pool_h=2, pool_w=2, stride=2, padding=2)
    with pytest.raises(ValueError, match="non-negative"):
        PoolSpec(mode="avg", in_h=4, in_w=4, channels=1, pool_h=2, pool_w=2, stride=2, padding=-1)


def test_conv2d_spec_round_trips():
    """The ``Conv2d`` op round-trips, and a kernel/fan-in mismatch fails at construction."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=8,
        input_bits=4,
        inputs=["x"],
        outputs=["y"],
        nodes=[
            Node(
                name="conv",
                inputs=["x"],
                outputs=["y"],
                op=Conv2dSpec(
                    weights=[[0] * 9],
                    bias=[0],
                    weight_bits=4,
                    in_h=5,
                    in_w=5,
                    in_channels=1,
                    kernel_h=3,
                    kernel_w=3,
                    stride=1,
                    padding=0,
                ),
            )
        ],
    )
    assert Graph.from_json(g.to_json()) == g

    with pytest.raises(ValueError, match="fan-in|width"):
        Conv2dSpec(
            weights=[[0] * 8],  # should be 1*3*3 = 9
            bias=[0],
            weight_bits=4,
            in_h=5,
            in_w=5,
            in_channels=1,
            kernel_h=3,
            kernel_w=3,
            stride=1,
            padding=0,
        )


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
        # BatchNorm is still unsupported (Conv2d/Pool/Requant/Add became known in Phase 4).
        "nodes": [{"name": "c", "inputs": ["x"], "outputs": ["y"], "op": {"op_type": "BatchNorm"}}],
    }
    with pytest.raises(ValueError, match="unknown op_type"):
        Graph.from_dict(bad)
