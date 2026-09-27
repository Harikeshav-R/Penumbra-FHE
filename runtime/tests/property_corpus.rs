//! Golden exactness replay over the 24-model property corpus under TFHE.
//!
//! Replays `tests/fixtures/property_corpus.json` against `penumbra_fhe_runtime`'s
//! TFHE backend, verifying the golden invariant holds bit-for-bit on every sample.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
};
use serde_json::Value;

#[test]
fn tfhe_matches_reference_on_every_property_corpus_model() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/property_corpus.json");
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

    // Group models by graph.num_blocks so keygen runs once per group
    let mut by_nb: BTreeMap<usize, Vec<&Value>> = BTreeMap::new();
    for model in models {
        let nb = model["graph"]["num_blocks"]
            .as_u64()
            .expect("num_blocks is integer") as usize;
        by_nb.entry(nb).or_default().push(model);
    }

    for (nb, group) in by_nb {
        let (ck, sk) = keygen(nb);
        let ctx = EvalCtx {
            sk: &sk,
            num_blocks: nb,
        };

        for model in group {
            let name = model["name"].as_str().unwrap_or("unknown");
            let graph_val = &model["graph"];
            let graph = Graph::from_json(&graph_val.to_string())
                .unwrap_or_else(|e| panic!("failed to deserialize Graph for {name}: {e}"));

            check_graph_bit_width_budget(&graph)
                .unwrap_or_else(|e| panic!("budget check failed for {name}: {e}"));

            let test_inputs = model["test_inputs"]
                .as_array()
                .expect("test_inputs is array");
            let expected_outputs = model["expected_outputs"]
                .as_array()
                .expect("expected_outputs is array");

            for s in 0..test_inputs.len() {
                let inp: Vec<i64> = test_inputs[s]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect();
                let want: Vec<i64> = expected_outputs[s]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect();

                let mut env = HashMap::new();
                env.insert(graph.inputs[0].clone(), encrypt(&ck, &inp));

                let out = evaluate_graph(&ctx, &graph, env)
                    .unwrap_or_else(|e| panic!("eval failed for {name} sample {s}: {e}"));
                let got = decrypt_vec(&ck, &out[&graph.outputs[0]]);

                assert_eq!(
                    got, want,
                    "GOLDEN VIOLATION: {name} sample {s} got {got:?} != expected {want:?}"
                );
            }
        }
    }
}
