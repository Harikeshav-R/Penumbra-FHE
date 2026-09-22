#![cfg(feature = "ckks")]

//! Golden exactness for the Phase-5 QAT model under CKKS.

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
        .join("../../examples/mnist/phase5_qat_fixture.json");
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
fn ckks_fhe_matches_quantized_cleartext_qat() {
    let fx = load_fixture();
    let graph: Graph = serde_json::from_value(fx["graph"].clone()).expect("valid graph");
    let params = DEFAULT_PARAMS;

    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    let input_0 = as_i64_vec(&fx["test_inputs"][0]);
    let want_logits = as_i64_vec(&fx["expected_logits"][0]);
    let want_label = fx["expected_labels"][0].as_i64().unwrap();

    let mut inputs_map = HashMap::new();
    inputs_map.insert(graph.inputs[0].clone(), encrypt(&ck, &input_0));

    let outputs = evaluate_graph(&ctx, &graph, inputs_map).expect("eval failed");
    let out_cts = &outputs[&graph.outputs[0]];
    let raw_floats = decrypt_raw_vec(&ck, out_cts);

    let max_err = raw_floats
        .iter()
        .zip(&want_logits)
        .map(|(&g, &w)| (g - w as f64).abs())
        .fold(0.0f64, f64::max);

    println!(
        "[ckks:{}] phase5_qat: max |err| = {max_err:.6e} (declared bound {:.3e})",
        penumbra_ckks::hal_backend_name(),
        bounds::PHASE5_QAT
    );

    assert!(
        max_err <= bounds::PHASE5_QAT,
        "CKKS error {max_err:.6e} exceeds declared bound {:.3e} for phase5_qat",
        bounds::PHASE5_QAT
    );

    let pred_label = raw_floats
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as i64)
        .unwrap();
    assert_eq!(
        pred_label, want_label,
        "argmax prediction does not match expected label"
    );
}
