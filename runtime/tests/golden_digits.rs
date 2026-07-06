//! Golden exactness for the Phase-5 real-digit CNN (`AGENTS.md` §1.1) — `#[ignore]` by default.
//!
//! > FHE output must equal the quantized-cleartext output, **bit-for-bit.**
//!
//! This is the first golden gate over a model trained on a **real dataset** (sklearn's 8x8
//! handwritten digits) and quantized through the Phase-5 service. The committed
//! `phase5_digits_fixture.json` IR graph is `Conv2d(stride 2) → Requant(fused ReLU) → Linear`;
//! this test deserializes it, runs the encrypted forward pass, decrypts the 10 logits, and
//! asserts they (and the argmax) equal the cleartext integer oracle bit-for-bit.
//!
//! It is **`#[ignore]`d**: at 12 conv channels × 3×3 positions = 108 requant bootstraps per
//! sample it is minutes per sample, too slow for every-commit CI. The fast Python guard
//! (`tests/test_real_digits_fixture.py`) checks fixture self-consistency on every CI run; this
//! FHE gate is run explicitly:
//!
//! ```text
//! cargo test --release --test golden_digits -- --ignored --nocapture
//! ```

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    OpSpec,
};
use serde_json::Value;

fn load_fixture() -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../examples/mnist/phase5_digits_fixture.json");
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

/// The cleartext integer CNN, recovered from the graph and run in plain `i64`. Mirrors each op's
/// exact arithmetic (the generalized Requant included), so it is the true oracle.
fn cleartext_logits(graph: &Graph, input: &[i64]) -> Vec<i64> {
    let mut env: HashMap<String, Vec<i64>> = HashMap::new();
    env.insert(graph.inputs[0].clone(), input.to_vec());

    for node in &graph.nodes {
        let x = env[&node.inputs[0]].clone();
        let out = match &node.op {
            OpSpec::Conv2d {
                weights,
                bias,
                in_h,
                in_w,
                in_channels,
                kernel_h,
                kernel_w,
                stride,
                padding,
                ..
            } => {
                let out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
                let in_hw = in_h * in_w;
                let mut o = Vec::new();
                for (kernel, &b) in weights.iter().zip(bias) {
                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let mut acc = 0i64;
                            for ic in 0..*in_channels {
                                for ky in 0..*kernel_h {
                                    let iy = (oy * stride + ky) as isize - *padding as isize;
                                    for kx in 0..*kernel_w {
                                        let ix = (ox * stride + kx) as isize - *padding as isize;
                                        if iy < 0
                                            || ix < 0
                                            || iy as usize >= *in_h
                                            || ix as usize >= *in_w
                                        {
                                            continue;
                                        }
                                        let w = kernel[(ic * kernel_h + ky) * kernel_w + kx];
                                        acc += w * x[ic * in_hw + iy as usize * in_w + ix as usize];
                                    }
                                }
                            }
                            o.push(acc + b);
                        }
                    }
                }
                o
            }
            OpSpec::Requant {
                shift,
                mult,
                round_bias,
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
                        // Per-channel overlay (0.6.0): element idx uses channel idx/channel_size.
                        // A per-tensor Requant leaves the arrays empty and uses the scalars — the
                        // `..` must NOT silently pick the scalar path for a per-channel fixture.
                        let (m, s, rb) = if mults.is_empty() {
                            (*mult, *shift, *round_bias)
                        } else {
                            let ch = idx / channel_size.expect("per-channel needs channel_size");
                            (mults[ch], shifts[ch], round_biases[ch])
                        };
                        let scaled = (v.max(0) * m as i64 + rb as i64) >> s;
                        scaled.max(0).min(ceil)
                    })
                    .collect()
            }
            OpSpec::Linear { weights, bias, .. } => weights
                .iter()
                .zip(bias)
                .map(|(row, &b)| row.iter().zip(&x).map(|(&w, &v)| w * v).sum::<i64>() + b)
                .collect(),
            other => panic!("unexpected op in digits fixture: {}", other.op_type()),
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

/// FHE real-digit CNN == quantized-cleartext, bit-for-bit, over the committed batch.
/// `#[ignore]` — minutes per sample; run with `--release -- --ignored`.
#[test]
#[ignore = "minutes per sample (108 PBS); run with: cargo test --release --test golden_digits -- --ignored"]
fn fhe_matches_quantized_cleartext_digits() {
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

    check_graph_bit_width_budget(&graph).expect("digits bit-width budget must fit");

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
