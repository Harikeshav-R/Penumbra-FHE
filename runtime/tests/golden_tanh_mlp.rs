//! Golden exactness for the Phase-8 Tanh MLP (`AGENTS.md` §1.1) — `#[ignore]` by default.
//!
//! > FHE output must equal the quantized-cleartext output, **bit-for-bit.**
//!
//! This is the FHE bit-for-bit gate for the non-ReLU activation pipeline:
//! `examples/mnist/tanh_mlp_export.py` trains an MLP, exports it to `digit_tanh_mlp.onnx`,
//! then loads it through `load_onnx` and quantizes it via the signed Requant + affine Activation path.
//! The committed `phase8_tanh_fixture.json` IR graph is `Linear → Requant → Activation → Linear`.

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    OpSpec,
};
use serde_json::Value;

fn load_fixture() -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../examples/mnist/phase8_tanh_fixture.json");
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

fn cleartext_logits(graph: &Graph, input: &[i64]) -> Vec<i64> {
    let mut env: HashMap<String, Vec<i64>> = HashMap::new();
    env.insert(graph.inputs[0].clone(), input.to_vec());

    for node in &graph.nodes {
        let x = env[&node.inputs[0]].clone();
        let out = match &node.op {
            OpSpec::Requant {
                shift,
                mult,
                round_bias,
                clamp_lo,
                zero_point,
                out_bits,
                mults,
                shifts,
                round_biases,
                channel_size,
                ..
            } => {
                let ceil = (1i64 << out_bits) - 1;
                x.iter()
                    .enumerate()
                    .map(|(idx, &v)| {
                        let (m, s, rb) = if mults.is_empty() {
                            (*mult, *shift, *round_bias)
                        } else {
                            let ch = idx / channel_size.expect("per-channel needs channel_size");
                            (mults[ch], shifts[ch], round_biases[ch])
                        };
                        let t = v.max(*clamp_lo);
                        let u = (t * m as i64 + rb as i64) >> s;
                        let w = u + *zero_point as i64;
                        w.clamp(0, ceil)
                    })
                    .collect()
            }
            OpSpec::Activation { lut, .. } => x.iter().map(|&v| lut[v as usize] as i64).collect(),
            OpSpec::Linear { weights, bias, .. } => weights
                .iter()
                .zip(bias)
                .map(|(row, &b)| row.iter().zip(&x).map(|(&w, &v)| w * v).sum::<i64>() + b)
                .collect(),
            other => panic!("unexpected op in Tanh MLP fixture: {}", other.op_type()),
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

#[test]
#[ignore = "TFHE evaluation; run with: cargo test --release --test golden_tanh_mlp -- --ignored"]
fn fhe_matches_quantized_cleartext_tanh_mlp() {
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

    check_graph_bit_width_budget(&graph).expect("bit-width budget must fit");

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
