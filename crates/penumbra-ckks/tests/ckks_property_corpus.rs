#![cfg(feature = "ckks")]

//! Property corpus evaluation under CKKS within declared error bound.

use std::collections::HashMap;
use std::path::PathBuf;

use penumbra_ckks::backend::{check_graph_depth_budget, evaluate_graph, CkksBackend};
use penumbra_ckks::bounds;
use penumbra_ckks::encrypt::{decrypt_raw_vec, encrypt};
use penumbra_ckks::keys::keygen;
use penumbra_ckks::params::DEFAULT_PARAMS;
use penumbra_core::backend::EvalCtx;
use penumbra_core::ir::Graph;
use serde_json::Value;

mod common;
use common::as_i64_vec;

#[test]
fn ckks_within_declared_bound_on_every_accepted_property_corpus_model() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/property_corpus.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read corpus at {}: {e}", path.display()));
    let corpus: Value = serde_json::from_str(&text).expect("valid JSON corpus");

    assert_eq!(
        corpus["schema"], "penumbra-property-corpus/1",
        "corpus schema must match"
    );
    let models = corpus["models"]
        .as_array()
        .expect("models must be a JSON array");
    assert!(!models.is_empty(), "corpus must not be empty");

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);
    let backend = CkksBackend::new(params);

    let mut accepted = 0;
    let mut rejected = 0;
    let total = models.len();

    for model in models {
        let name = model["name"].as_str().unwrap_or("unknown");
        let graph: Graph = serde_json::from_value(model["graph"].clone())
            .unwrap_or_else(|e| panic!("failed to deserialize Graph for {name}: {e}"));

        match check_graph_depth_budget(&backend, &graph) {
            Err(e) => {
                assert!(
                    e.contains("ckks"),
                    "rejection error must name the backend: {e}"
                );
                rejected += 1;
            }
            Ok(_) => {
                accepted += 1;
                let test_inputs = model["test_inputs"]
                    .as_array()
                    .expect("test_inputs is array");
                let expected_outputs = model["expected_outputs"]
                    .as_array()
                    .expect("expected_outputs is array");

                for (s, (inp_val, want_val)) in test_inputs.iter().zip(expected_outputs).enumerate()
                {
                    let inp = as_i64_vec(inp_val);
                    let want = as_i64_vec(want_val);

                    let mut inputs_map = HashMap::new();
                    inputs_map.insert(graph.inputs[0].clone(), encrypt(&ck, &inp));

                    let outputs = evaluate_graph(&ctx, &graph, inputs_map)
                        .unwrap_or_else(|e| panic!("eval failed for {name} sample {s}: {e}"));
                    let raw_floats = decrypt_raw_vec(&ck, &outputs[&graph.outputs[0]]);

                    let mut sample_max_err = 0.0f64;
                    for (i, &w) in want.iter().enumerate() {
                        let got_f = raw_floats[i];
                        let err = (got_f - w as f64).abs();
                        if err > sample_max_err {
                            sample_max_err = err;
                        }
                    }

                    println!(
                        "[ckks:{}] {} sample {s}: max |err| = {sample_max_err:.6e} (declared bound {:.3e})",
                        penumbra_ckks::hal_backend_name(),
                        name,
                        bounds::PROPERTY_CORPUS
                    );

                    assert!(
                        sample_max_err <= bounds::PROPERTY_CORPUS,
                        "GOLDEN BOUND EXCEEDED on {name} sample {s}: err {sample_max_err} > bound {}",
                        bounds::PROPERTY_CORPUS
                    );
                }
            }
        }
    }

    println!(
        "Property corpus summary: {accepted}/{total} accepted, {rejected} rejected by depth budget"
    );
    assert!(
        accepted * 2 >= total,
        "at least half of the property models must be accepted (got {accepted}/{total})"
    );
}
