#![cfg(feature = "ckks")]

//! Golden exactness for the Phase-8 Tanh MLP under CKKS.

use std::collections::HashMap;
use std::path::PathBuf;

use penumbra_ckks::backend::evaluate_graph;
use penumbra_ckks::bounds;
use penumbra_ckks::encrypt::{decrypt_raw_vec, encrypt};
use penumbra_ckks::keys::keygen;
use penumbra_ckks::params::DEFAULT_PARAMS;
use penumbra_core::backend::EvalCtx;
use penumbra_core::ir::Graph;
use serde_json::Value;

fn load_fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/mnist/phase8_tanh_fixture.json");
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
fn ckks_fhe_matches_quantized_cleartext_tanh_mlp() {
    let fx = load_fixture();
    let graph: Graph = serde_json::from_value(fx["graph"].clone()).expect("valid graph");
    let params = DEFAULT_PARAMS;

    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    let inputs = fx["test_inputs"].as_array().expect("test_inputs array");
    let want_all = fx["expected_logits"]
        .as_array()
        .expect("expected_logits array");
    assert!(!inputs.is_empty(), "fixture has no test inputs");
    assert_eq!(
        inputs.len(),
        want_all.len(),
        "inputs/expected_logits length mismatch"
    );

    let mut max_err_all = 0.0f64;

    for (s, input_val) in inputs.iter().enumerate() {
        let input = as_i64_vec(input_val);
        let want_logits = as_i64_vec(&want_all[s]);

        let mut inputs_map = HashMap::new();
        inputs_map.insert(graph.inputs[0].clone(), encrypt(&ck, &input));

        let outputs = evaluate_graph(&ctx, &graph, inputs_map).expect("eval failed");
        let out_cts = &outputs[&graph.outputs[0]];
        let raw_floats = decrypt_raw_vec(&ck, out_cts);

        let max_err = raw_floats
            .iter()
            .zip(&want_logits)
            .map(|(&g, &w)| (g - w as f64).abs())
            .fold(0.0f64, f64::max);

        println!(
            "[ckks:{}] phase8_tanh sample {s}: max |err| = {max_err:.6e} (declared bound {:.3e})",
            penumbra_ckks::hal_backend_name(),
            bounds::PHASE8_TANH
        );

        if max_err > max_err_all {
            max_err_all = max_err;
        }

        assert!(
            max_err <= bounds::PHASE8_TANH,
            "sample {s}: CKKS error {max_err:.6e} exceeds declared bound {:.3e} for phase8_tanh",
            bounds::PHASE8_TANH
        );
    }

    println!(
        "[ckks:{}] phase8_tanh max error over all samples: {max_err_all:.6e} (1.5x recommendation: {:.3e})",
        penumbra_ckks::hal_backend_name(),
        max_err_all * 1.5
    );
}
