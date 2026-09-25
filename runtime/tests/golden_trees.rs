//! Golden exactness for Phase-8 tree ensembles (AGENTS.md §1.1).
//!
//! > FHE output must equal the quantized-cleartext output, **bit-for-bit.**
//!
//! Evaluates lowered decision tree ensembles (scikit-learn RandomForestClassifier and XGBoost
//! XGBClassifier) over the Wisconsin Breast Cancer dataset via the 4-stage sum-of-comparisons
//! formulation:
//!     1. `split_cmp`:  Compare (threshold comparisons with fused gather)
//!     2. `leaf_score`: Linear (accumulates path conditions)
//!     3. `leaf_sel`:   Compare (one-hot leaf indicators)
//!     4. `logits`:     Linear (computes class logits)
//!
//! Deserializes each committed fixture, computes the cleartext integer oracle in Rust i64,
//! runs the encrypted evaluation under the TFHE backend, decrypts the logits, and asserts
//! bit-for-bit agreement with both the Rust oracle and the fixture's committed expected_logits.

use std::collections::HashMap;
use std::time::Instant;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    OpSpec,
};
use serde_json::Value;

fn load_fixture(rel_path: &str) -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel_path);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn as_i64_vec(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_i64().expect("int"))
        .collect()
}

/// Evaluates the 4-stage tree ensemble in plain Rust `i64` arithmetic (the reference oracle).
fn cleartext_trees_oracle(graph: &Graph, input: &[i64]) -> Vec<i64> {
    let mut env: HashMap<String, Vec<i64>> = HashMap::new();
    env.insert(graph.inputs[0].clone(), input.to_vec());

    for node in &graph.nodes {
        let x = env[&node.inputs[0]].clone();
        let out = match &node.op {
            OpSpec::Compare {
                indices,
                thresholds,
            } => indices
                .iter()
                .zip(thresholds)
                .map(|(&idx, &t)| if x[idx] >= t { 1 } else { 0 })
                .collect(),
            OpSpec::Linear { weights, bias, .. } => weights
                .iter()
                .zip(bias)
                .map(|(row, &b)| row.iter().zip(&x).map(|(&w, &v)| w * v).sum::<i64>() + b)
                .collect(),
            other => panic!("unexpected op in tree fixture: {}", other.op_type()),
        };
        env.insert(node.outputs[0].clone(), out);
    }

    env.remove(&graph.outputs[0])
        .expect("graph produces logits")
}

fn argmax(logits: &[i64]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(i, _)| i)
        .expect("non-empty logits")
}

fn run_tree_fixture(fx_rel_path: &str, label: &str) {
    let fx = load_fixture(fx_rel_path);
    let graph = Graph::from_json(&fx["graph"].to_string()).expect("fixture graph deserializes");

    let test_inputs: Vec<Vec<i64>> = fx["test_inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(as_i64_vec)
        .collect();
    let expected_labels = as_i64_vec(&fx["expected_labels"]);
    let expected_logits: Vec<Vec<i64>> = fx["expected_logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(as_i64_vec)
        .collect();

    check_graph_bit_width_budget(&graph).expect("tree ensemble bit-width budget must fit");

    let input_name = graph.inputs[0].clone();
    let output_name = graph.outputs[0].clone();
    let (ck, sk) = keygen(graph.num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks: graph.num_blocks,
    };

    println!(
        "\n--- Golden test: {label} ({} blocks, {} samples) ---",
        graph.num_blocks,
        test_inputs.len()
    );

    for (i, input) in test_inputs.iter().enumerate() {
        let ref_logits = cleartext_trees_oracle(&graph, input);
        assert_eq!(
            ref_logits, expected_logits[i],
            "fixture expected_logits[{i}] disagrees with Rust cleartext oracle"
        );
        assert_eq!(
            argmax(&ref_logits) as i64,
            expected_labels[i],
            "sample {i}: oracle argmax disagrees with expected_labels"
        );

        let t0 = Instant::now();
        let mut env = HashMap::new();
        env.insert(input_name.clone(), encrypt(&ck, input));

        let fhe_env = evaluate_graph(&ctx, &graph, env).expect("evaluate_graph failed");
        let fhe_logits = decrypt_vec(&ck, &fhe_env[&output_name]);
        let elapsed = t0.elapsed();

        println!(
            "  sample {i}: FHE eval in {:.2?} -> logits {:?} (label {})",
            elapsed,
            fhe_logits,
            argmax(&fhe_logits)
        );

        assert_eq!(
            fhe_logits, ref_logits,
            "sample {i}: FHE logits disagree with cleartext oracle (bit-for-bit exactness violated!)"
        );
        assert_eq!(
            fhe_logits, expected_logits[i],
            "sample {i}: FHE logits disagree with committed fixture logits"
        );
        assert_eq!(
            argmax(&fhe_logits) as i64,
            expected_labels[i],
            "sample {i}: FHE argmax label disagrees with expected_labels"
        );
    }
}

/// FHE scikit-learn RandomForestClassifier == quantized-cleartext, bit-for-bit.
#[test]
fn fhe_matches_quantized_cleartext_trees() {
    run_tree_fixture(
        "../examples/tabular/phase8_trees_fixture.json",
        "RandomForestClassifier (sklearn)",
    );
}

/// FHE XGBoost XGBClassifier == quantized-cleartext, bit-for-bit.
/// Marked `#[ignore]` so the always-on gate is one FHE tree run; run with `--release -- --ignored`.
#[test]
#[ignore = "second-framework FHE gate; run with: cargo test --release --test golden_trees -- --ignored"]
fn fhe_matches_quantized_cleartext_xgb_trees() {
    run_tree_fixture(
        "../examples/tabular/phase8_xgb_fixture.json",
        "XGBClassifier (XGBoost)",
    );
}
