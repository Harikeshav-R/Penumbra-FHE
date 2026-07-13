//! Golden exactness for the Phase-6 scikit-learn linear digit classifier (`AGENTS.md` §1.1) —
//! `#[ignore]` by default.
//!
//! > FHE output must equal the quantized-cleartext output, **bit-for-bit.**
//!
//! This is the FHE bit-for-bit gate for the **second framework** behind the ONNX front door:
//! `examples/mnist/sklearn_export.py` trains a linear classifier in scikit-learn, exports it to
//! `digit_linear_sklearn.onnx` with `skl2onnx`, then loads it *through* `penumbra.load_onnx`
//! (folding away skl2onnx's leading `Cast`) and quantizes it via the unchanged Phase-5 service. The
//! committed `phase6_sklearn_fixture.json` IR graph is a single `Linear` — proving the front door
//! lowers a real scikit-learn export to the same primitives as the torch CNN, with no `runtime/`
//! change (`ROADMAP.md` Phase 6: "load_onnx works for at least two models from two frameworks").
//!
//! It has **no PBS** (a `Linear` is plaintext-weight arithmetic), but the accurate quantization of a
//! 64→10 logit head needs a **20-bit radix** (`num_blocks = 10`): 640 per-output scalar-muls across
//! 10 blocks is minutes per sample, so like `golden_onnx`/`golden_digits` it is **`#[ignore]`d** —
//! too slow for every-commit CI. The fast Python guard (`tests/test_sklearn_fixture.py`) checks
//! fixture self-consistency on every CI run; this FHE gate is run explicitly:
//!
//! ```text
//! cargo test --release --test golden_sklearn -- --ignored --nocapture
//! ```
//!
//! It deserializes the fixture, runs the encrypted forward pass, decrypts the 10 logits, and
//! asserts they (and the argmax) equal the cleartext integer oracle bit-for-bit.

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    OpSpec,
};
use serde_json::Value;

fn load_fixture() -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../examples/mnist/phase6_sklearn_fixture.json");
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

/// The cleartext integer linear classifier, recovered from the graph and run in plain `i64`. A
/// linear head is a single `Linear`, so this mirrors exactly that op's arithmetic — the true oracle.
fn cleartext_logits(graph: &Graph, input: &[i64]) -> Vec<i64> {
    let mut env: HashMap<String, Vec<i64>> = HashMap::new();
    env.insert(graph.inputs[0].clone(), input.to_vec());

    for node in &graph.nodes {
        let x = env[&node.inputs[0]].clone();
        let out = match &node.op {
            OpSpec::Linear { weights, bias, .. } => weights
                .iter()
                .zip(bias)
                .map(|(row, &b)| row.iter().zip(&x).map(|(&w, &v)| w * v).sum::<i64>() + b)
                .collect(),
            other => panic!("unexpected op in sklearn fixture: {}", other.op_type()),
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

/// FHE sklearn-lowered linear classifier == quantized-cleartext, bit-for-bit, over the batch.
/// `#[ignore]` — minutes per sample (a 20-bit-radix Linear); run with `--release -- --ignored`.
#[test]
#[ignore = "minutes per sample (20-bit-radix Linear); run with: cargo test --release --test golden_sklearn -- --ignored"]
fn fhe_matches_quantized_cleartext_sklearn() {
    let fx = load_fixture();
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

    check_graph_bit_width_budget(&graph).expect("sklearn-lowered bit-width budget must fit");

    let input_name = graph.inputs[0].clone();
    let output_name = graph.outputs[0].clone();
    let (ck, sk) = keygen(graph.num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks: graph.num_blocks,
    };

    for (i, input) in test_inputs.iter().enumerate() {
        let ref_logits = cleartext_logits(&graph, input);
        assert_eq!(
            ref_logits, expected_logits[i],
            "fixture expected_logits[{i}] disagrees with the Rust cleartext oracle"
        );
        assert_eq!(argmax(&ref_logits) as i64, expected_labels[i]);

        let mut env = HashMap::new();
        env.insert(input_name.clone(), encrypt(&ck, input));
        let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
        let fhe_logits = decrypt_vec(&ck, &out[&output_name]);

        assert_eq!(
            fhe_logits, ref_logits,
            "GOLDEN VIOLATION at sample {i}: FHE logits {fhe_logits:?} != cleartext {ref_logits:?}"
        );
        assert_eq!(argmax(&fhe_logits) as i64, expected_labels[i]);
    }
}
