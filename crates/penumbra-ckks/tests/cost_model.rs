#![cfg(feature = "ckks")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use penumbra_ckks::ops::conv2d_matrix;
use penumbra_ckks::params::DEFAULT_PARAMS;
use penumbra_ckks::CkksBackend;
use penumbra_core::backend::Backend;
use penumbra_core::ir::{Graph, OpSpec};

fn load_fixture_graph(rel_path: &str) -> Graph {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel_path);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
    Graph::from_json(&v["graph"].to_string()).expect("valid Graph")
}

#[test]
fn test_ckks_cost_model_cnn_nodes() {
    let backend = CkksBackend::default();
    let graph = load_fixture_graph("../../examples/mnist/phase4_cnn_fixture.json");

    // 1. Build 'conv' node through CkksBackend::build_op
    let conv_node = graph
        .nodes
        .iter()
        .find(|n| n.name == "conv")
        .expect("conv node");
    let op_conv = backend.build_op(&conv_node.op).expect("build conv op");
    let cost_conv: BTreeMap<&'static str, u64> = op_conv.cost(&[]).into_iter().collect();

    let rots = cost_conv.get("rotations").copied().unwrap_or(0);
    assert!(rots > 0, "conv node should have rotations > 0");
    assert_eq!(cost_conv.get("rescales"), Some(&1));
    assert_eq!(cost_conv.get("depth_levels"), Some(&1));

    // Verify against direct prepare_linear_map
    if let OpSpec::Conv2d {
        weights,
        bias,
        in_h,
        in_w,
        in_channels,
        kernel_h,
        kernel_w,
        stride,
        padding,
        ..
    } = &conv_node.op
    {
        let m = conv2d_matrix(
            weights,
            bias,
            *in_h,
            *in_w,
            *in_channels,
            *kernel_h,
            *kernel_w,
            *stride,
            *padding,
        )
        .expect("conv2d matrix");
        let prepared = backend
            .prepare_linear_map(&m)
            .expect("prepare_linear_map");
        assert_eq!(rots, prepared.rotation_count() as u64);
    } else {
        panic!("conv node is not OpSpec::Conv2d");
    }

    // 2. Build 'conv__requant' node
    let rq_node = graph
        .nodes
        .iter()
        .find(|n| n.name == "conv__requant")
        .expect("conv__requant node");
    let op_rq = backend.build_op(&rq_node.op).expect("build rq op");
    let cost_rq: BTreeMap<&'static str, u64> = op_rq.cost(&[]).into_iter().collect();

    assert_eq!(cost_rq.get("poly_evals"), Some(&1));
    let rq_depth = cost_rq.get("depth_levels").copied().unwrap_or(0);
    assert!(rq_depth >= 1, "requant depth should be >= 1");
    assert_eq!(cost_rq.get("rescales"), Some(&rq_depth));
}

#[test]
fn test_ckks_realized_depth_within_budget_all_fixtures() {
    let backend = CkksBackend::default();
    let fixtures = [
        "../../examples/mnist/phase2_fixture.json",
        "../../examples/mnist/phase4_cnn_fixture.json",
        "../../examples/mnist/phase5_digits_fixture.json",
        "../../examples/mnist/phase5_qat_fixture.json",
        "../../examples/mnist/phase6_onnx_fixture.json",
        "../../examples/mnist/phase6_sklearn_fixture.json",
        "../../examples/faces/phase7_faces_fixture.json",
    ];

    let budget = DEFAULT_PARAMS.log_budget() as u64;

    for fixture in fixtures {
        let graph = load_fixture_graph(fixture);
        let mut realized_levels = 0u64;

        for node in &graph.nodes {
            let op = backend
                .build_op(&node.op)
                .unwrap_or_else(|e| panic!("build_op failed for node '{}' in {fixture}: {e}", node.name));
            let cost: BTreeMap<&'static str, u64> = op.cost(&[]).into_iter().collect();
            let levels = cost.get("depth_levels").copied().unwrap_or(0);
            realized_levels += levels;
        }

        let realized_bits = realized_levels * (DEFAULT_PARAMS.log_delta as u64);
        assert!(
            realized_bits <= budget,
            "fixture {fixture}: realized depth {realized_bits} bits ({realized_levels} levels) exceeds budget {budget} bits"
        );
    }
}
