#![cfg(feature = "ckks")]

//! Golden exactness for the Phase-2 logistic regression under CKKS.

use std::collections::HashMap;
use std::path::PathBuf;

use penumbra_ckks::backend::evaluate_graph;
use penumbra_ckks::bounds;
use penumbra_ckks::encrypt::{decrypt_raw_vec, decrypt_vec, encrypt};
use penumbra_ckks::keys::keygen;
use penumbra_ckks::params::DEFAULT_PARAMS;
use penumbra_core::backend::EvalCtx;
use penumbra_core::ir::Graph;
use serde_json::Value;

fn load_fixture() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/mnist/phase2_fixture.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture at {}: {e}", path.display()));
    serde_json::from_str(&text).expect("valid JSON")
}

fn as_i64_vec(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_i64().expect("int"))
        .collect()
}

#[test]
fn ckks_fhe_matches_quantized_cleartext_logreg() {
    let fx = load_fixture();
    let graph: Graph = serde_json::from_value(fx["graph"].clone()).expect("valid graph");
    let params = DEFAULT_PARAMS;

    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    let inputs = fx["test_inputs"].as_array().expect("test_inputs array");
    let want_labels = fx["expected_labels"]
        .as_array()
        .expect("expected_labels array");
    assert!(!inputs.is_empty(), "fixture has no test inputs");
    assert_eq!(
        inputs.len(),
        want_labels.len(),
        "inputs/expected_labels length mismatch"
    );

    for (s, input_val) in inputs.iter().enumerate() {
        let input = as_i64_vec(input_val);
        let want_label = want_labels[s].as_i64().expect("int label");

        let mut inputs_map = HashMap::new();
        inputs_map.insert(graph.inputs[0].clone(), encrypt(&ck, &input));

        let outputs = evaluate_graph(&ctx, &graph, inputs_map).expect("eval failed");
        let out_cts = &outputs[&graph.outputs[0]];
        let raw_floats = decrypt_raw_vec(&ck, out_cts);
        let rounded_ints = decrypt_vec(&ck, out_cts);

        let max_err = (raw_floats[0] - want_label as f64).abs();

        println!(
            "[ckks:{}] phase2_logreg sample {s}: max |err| = {max_err:.6e} (declared bound {:.3e})",
            penumbra_ckks::hal_backend_name(),
            bounds::PHASE2_LOGREG
        );

        assert!(
            max_err <= bounds::PHASE2_LOGREG,
            "sample {s}: CKKS error {max_err:.6e} exceeds declared bound {:.3e} for phase2_logreg",
            bounds::PHASE2_LOGREG
        );

        assert_eq!(
            rounded_ints[0], want_label,
            "sample {s}: decrypted label does not round to expected label"
        );
    }
}
