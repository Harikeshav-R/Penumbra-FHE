//! Integration test verifying measured PBS counter on TFHE backend.
//!
//! Falsifies the claim that "Linear is PBS-free": radix arithmetic issues carry-propagation
//! bootstraps internally, which are now measured and reported.

use std::collections::HashMap;

use penumbra_core::backend::{Backend, EvalCtx};
use penumbra_core::eval::evaluate_graph_profiled;
use penumbra_core::ir::{Graph, Node, OpSpec, SCHEMA_VERSION};
use penumbra_core::profile::GraphProfile;
use penumbra_tfhe::TfheBackend;

#[test]
fn test_linear_measured_pbs_greater_than_zero() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 3,
        input_bits: 2,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "fc".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Linear {
                weights: vec![vec![1, 1]],
                bias: vec![0],
                weight_bits: 2,
            },
        }],
    };

    let backend = TfheBackend::default();
    let (ck, sk) = backend.keygen(graph.num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks: graph.num_blocks,
    };

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), backend.encrypt(&ck, &[1, 1]));

    let mut profile = GraphProfile::default();
    let outputs = evaluate_graph_profiled(&backend, &ctx, &graph, inputs, &mut profile)
        .expect("evaluate_graph_profiled should succeed");

    let y_ct = outputs.get("y").expect("output y present");
    let got = backend.decrypt_vec(&ck, y_ct);
    assert_eq!(got, vec![2], "1*1 + 1*1 + 0 = 2");

    assert_eq!(profile.nodes.len(), 1);
    let fc_node = &profile.nodes[0];
    assert_eq!(fc_node.name, "fc");

    let measured_map: HashMap<&str, u64> = fc_node.measured.iter().copied().collect();
    let pbs_count = *measured_map
        .get("pbs")
        .expect("measured counters must contain 'pbs'");
    assert!(
        pbs_count > 0,
        "fc Linear node must measure > 0 PBS ops (got {pbs_count}), falsifying 'Linear is PBS-free'"
    );

    let totals = profile.measured_totals();
    assert_eq!(totals.get("pbs"), Some(&pbs_count));
}
