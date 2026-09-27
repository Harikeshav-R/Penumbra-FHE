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
use serde_json::Value;
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
        graph.num_blocks, 6,
        "Phase-2 fixture uses a 6-block (12-bit) radix"
    );
    assert_eq!(graph.input_bits, 2);
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
            assert_eq!(*weight_bits, 2);
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

/// Every committed fixture graph re-serializes identically to the Python-emitted JSON.
#[test]
fn every_committed_fixture_graph_reserializes_identically() {
    let fixtures = [
        "examples/mnist/phase2_fixture.json",
        "examples/mnist/phase4_cnn_fixture.json",
        "examples/mnist/phase5_digits_fixture.json",
        "examples/mnist/phase5_qat_fixture.json",
        "examples/mnist/phase6_onnx_fixture.json",
        "examples/mnist/phase6_sklearn_fixture.json",
        "examples/mnist/phase8_bn_cnn_fixture.json",
        "examples/mnist/phase8_branch_fixture.json",
        "examples/mnist/phase8_gap_cnn_fixture.json",
        "examples/mnist/phase8_tanh_fixture.json",
        "examples/faces/phase7_faces_fixture.json",
        "examples/trees/phase8_trees_fixture.json",
        "examples/trees/phase8_xgb_fixture.json",
        "examples/tabular/phase11_tabular_mlp_fixture.json",
    ];

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    for rel_path in fixtures {
        let path = root.join(rel_path);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read fixture at {}: {e}", path.display()));
        let fx: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("invalid JSON in {}: {e}", path.display()));
        let committed_graph = &fx["graph"];
        assert!(
            !committed_graph.is_null(),
            "fixture {rel_path} has no 'graph' key"
        );

        let parsed_graph = Graph::from_json(&committed_graph.to_string())
            .unwrap_or_else(|e| panic!("failed to deserialize Graph from {rel_path}: {e}"));
        let reserialized_json = parsed_graph.to_json();
        let reserialized_val: Value = serde_json::from_str(&reserialized_json)
            .unwrap_or_else(|e| panic!("invalid re-serialized JSON for {rel_path}: {e}"));

        assert_eq!(
            reserialized_val, *committed_graph,
            "IR drift detected on {rel_path}: Rust re-serialization differs from committed JSON"
        );
    }

    // Also cover every model graph in the committed property corpus
    let corpus_path = root.join("tests/fixtures/property_corpus.json");
    let text = std::fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("cannot read corpus at {}: {e}", corpus_path.display()));
    let corpus: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("invalid JSON in {}: {e}", corpus_path.display()));
    let models = corpus["models"].as_array().expect("models array");
    for model in models {
        let name = model["name"].as_str().unwrap_or("unknown");
        let committed_graph = &model["graph"];
        let parsed_graph = Graph::from_json(&committed_graph.to_string()).unwrap_or_else(|e| {
            panic!("failed to deserialize Graph from corpus model {name}: {e}")
        });
        let reserialized_json = parsed_graph.to_json();
        let reserialized_val: Value = serde_json::from_str(&reserialized_json)
            .unwrap_or_else(|e| panic!("invalid re-serialized JSON for corpus model {name}: {e}"));
        assert_eq!(
            reserialized_val, *committed_graph,
            "IR drift detected on corpus model {name}: Rust re-serialization differs from committed JSON"
        );
    }
}
