//! Cross-language IR conformance — the Rust half (`AGENTS.md` §5, ROADMAP Phase 3).
//!
//! Python emits the IR graph → it is committed to `examples/mnist/phase2_fixture.json`
//! under the `"graph"` key → this test (and the runtime) consume it. The Python half
//! (`tests/test_ir_conformance.py`) asserts the committed file *is* the front end's current
//! output (the drift guard); this half asserts the Rust runtime deserializes it into the
//! expected typed graph. Together they keep `ir.py` ↔ `ir.rs` in lockstep.
//!
//! No keygen, no FHE — this is a pure (de)serialization + structure check, so it runs
//! instantly even in debug.

use std::path::PathBuf;

use penumbra_fhe_runtime::{Graph, OpSpec, SCHEMA_VERSION};

fn fixture_graph_json() -> String {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../examples/mnist/phase2_fixture.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    let fx: serde_json::Value = serde_json::from_str(&text).expect("fixture is valid JSON");
    fx["graph"].to_string()
}

/// The committed IR graph deserializes into the typed `Graph` the runtime expects, with the
/// matching schema version and the Phase-2 `Linear → Argmax` structure.
#[test]
fn committed_ir_deserializes_to_expected_graph() {
    let graph = Graph::from_json(&fixture_graph_json()).expect("committed IR graph deserializes");

    assert_eq!(
        graph.schema_version, SCHEMA_VERSION,
        "committed IR must match this runtime's schema version"
    );
    assert_eq!(
        graph.num_blocks, 8,
        "Phase-2 fixture uses an 8-block (16-bit) radix"
    );
    assert_eq!(graph.input_bits, 4);
    assert_eq!(graph.inputs, vec!["x".to_string()]);
    assert_eq!(graph.outputs, vec!["label".to_string()]);

    assert_eq!(graph.nodes.len(), 2, "Phase-2 model is Linear → Argmax");

    // Node 0: Linear, single weight row, 4-bit weights, wiring x → logit.
    let fc = &graph.nodes[0];
    assert_eq!(fc.name, "fc");
    assert_eq!(fc.inputs, vec!["x".to_string()]);
    assert_eq!(fc.outputs, vec!["logit".to_string()]);
    match &fc.op {
        OpSpec::Linear {
            weights,
            bias,
            weight_bits,
        } => {
            assert_eq!(*weight_bits, 4);
            assert_eq!(weights.len(), 1, "one logit row");
            assert_eq!(weights[0].len(), 64, "64 features");
            assert_eq!(bias.len(), 1);
        }
        other => panic!("node 0 must be Linear, got {}", other.op_type()),
    }

    // Node 1: Argmax, wiring logit → label.
    let head = &graph.nodes[1];
    assert_eq!(head.name, "head");
    assert_eq!(head.inputs, vec!["logit".to_string()]);
    assert_eq!(head.outputs, vec!["label".to_string()]);
    assert!(
        matches!(head.op, OpSpec::Argmax { .. }),
        "node 1 must be Argmax, got {}",
        head.op.op_type()
    );
}

/// The committed graph round-trips through Rust (de)serialization unchanged — the same
/// structural-equality guard the Python side runs, so both languages agree on the format.
#[test]
fn committed_ir_round_trips_in_rust() {
    let graph = Graph::from_json(&fixture_graph_json()).expect("deserializes");
    let restored = Graph::from_json(&graph.to_json()).expect("re-deserializes");
    assert_eq!(graph, restored, "Rust IR round-trip must be exact");
}
