//! Tests for TFHE named crypto-parameter profiles (ROADMAP Phase 9).
//!
//! Asserts that:
//! 1. Encrypted evaluation produces bit-for-bit identical results under both "default"
//!    and "gaussian" profiles (TFHE is exact under either noise distribution; `AGENTS.md` §1.1).
//! 2. Mismatched profile pairing produces the actionable `ensure_profile_match` error.

use std::collections::HashMap;

use penumbra_core::backend::{Backend, EvalCtx};
use penumbra_core::ir::{Graph, Node, OpSpec};
use penumbra_tfhe::keys::{ensure_profile_match, TfheProfile};
use penumbra_tfhe::TfheBackend;

#[test]
fn test_tfhe_crypto_profiles_exact_agreement() {
    let graph = Graph {
        schema_version: penumbra_core::ir::SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 2,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "lin".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Linear {
                weights: vec![vec![2, 3]],
                bias: vec![1],
                weight_bits: 3,
            },
        }],
    };

    let input_vals = vec![2i64, -1i64];
    let expected = vec![2 * 2 - 3 + 1]; // [2]

    for profile in [TfheProfile::Default, TfheProfile::Gaussian] {
        let backend = TfheBackend::new(profile);
        let (ck, sk) = backend.keygen(graph.num_blocks);
        let ctx = EvalCtx {
            sk: &sk,
            num_blocks: graph.num_blocks,
        };

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), backend.encrypt(&ck, &input_vals));

        let outputs = penumbra_core::eval::evaluate_graph(&backend, &ctx, &graph, inputs)
            .expect("evaluate_graph should succeed");
        let y_ct = outputs.get("y").expect("output y present");
        let got = backend.decrypt_vec(&ck, y_ct);
        assert_eq!(
            got, expected,
            "TFHE output must match bit-for-bit under profile {profile:?}"
        );
    }
}

#[test]
fn test_tfhe_profile_mismatch_fails_loudly() {
    let err = ensure_profile_match(TfheProfile::Default, TfheProfile::Gaussian).unwrap_err();
    assert!(
        err.contains("key/profile mismatch"),
        "error should state mismatch: {err}"
    );
    assert!(
        err.contains("this key was generated under crypto profile 'default'"),
        "error should name key profile: {err}"
    );
    assert!(
        err.contains("run uses 'gaussian'"),
        "error should name run profile: {err}"
    );
}
