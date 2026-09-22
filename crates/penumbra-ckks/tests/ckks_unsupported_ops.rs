#![cfg(feature = "ckks")]

//! Asserts that unsupported ops (e.g. Pool(max)) fail loudly at load time
//! naming the op, the offending node, and the backend (`AGENTS.md` §1.4).

use penumbra_ckks::backend::check_graph_depth_budget;
use penumbra_ckks::params::DEFAULT_PARAMS;
use penumbra_ckks::CkksBackend;
use penumbra_core::ir::{Graph, Node, OpSpec};

#[test]
fn pool_max_rejected_loudly_at_load_time() {
    let graph = Graph {
        schema_version: "0.6.0".to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "max_pool_node".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Pool {
                mode: "max".to_string(),
                in_h: 4,
                in_w: 4,
                channels: 1,
                pool_h: 2,
                pool_w: 2,
                stride: 2,
            },
        }],
    };

    let backend = CkksBackend::new(DEFAULT_PARAMS);
    let err = check_graph_depth_budget(&backend, &graph)
        .expect_err("Pool(max) must be rejected on CKKS backend");

    assert!(
        err.contains("max_pool_node"),
        "error must name offending node: {err}"
    );
    assert!(
        err.contains("Pool(max)"),
        "error must name unsupported op: {err}"
    );
    assert!(err.contains("ckks"), "error must name the backend: {err}");
}
