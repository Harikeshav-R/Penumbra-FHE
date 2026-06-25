//! Golden exactness for the Phase-4 CNN — the headline end-to-end gate (`AGENTS.md` §1.1).
//!
//! > FHE output must equal the quantized-cleartext output, **bit-for-bit.**
//!
//! This is the Phase-4 exit criterion (ROADMAP Phase 4): a small 10-class CNN
//! (`Conv2d → Requant+ReLU → avg Pool → Linear`) classifies synthetic MNIST-like data, and
//! the encrypted forward pass produces the identical class label as the quantized-integer
//! reference. The model is the committed `phase4_cnn_fixture.json` IR graph (with its
//! `Requant` automatically inserted by the Python compile pass); this test deserializes and
//! walks it through `evaluate_graph`, decrypts the 10 logits, and argmaxes them **on the
//! client** (`PROJECT.md` §11 — no wide-domain in-FHE argmax this phase).
//!
//! The cleartext oracle is recomputed here in plain `i64` from the *same* graph (one source
//! of model truth), and the fixture's committed `expected_labels`/`expected_logits` are
//! checked against it too, guarding against fixture drift.
//!
//! Run with `cargo test --release` — debug FHE is impractically slow (`docs/DEVELOPMENT.md`).

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    OpSpec,
};
use serde_json::Value;

fn load_fixture() -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../examples/mnist/phase4_cnn_fixture.json");
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

/// The cleartext integer CNN, recovered from the graph and run in plain `i64`. Mirrors each
/// op's exact arithmetic so it is the true oracle. Returns the 10 logits for one input.
fn cleartext_logits(graph: &Graph, input: &[i64]) -> Vec<i64> {
    // A flat, channel-major / row-major tensor flowing between layers, like the FHE `CtVec`.
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
                shift, out_bits, ..
            } => x
                .iter()
                .map(|&v| (v >> shift).max(0).min((1i64 << out_bits) - 1))
                .collect(),
            OpSpec::Pool {
                mode,
                in_h,
                in_w,
                channels,
                pool_h,
                pool_w,
                stride,
            } => {
                let out_h = (in_h - pool_h) / stride + 1;
                let out_w = (in_w - pool_w) / stride + 1;
                let mut o = Vec::new();
                for c in 0..*channels {
                    let base = c * in_h * in_w;
                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let mut vals = Vec::new();
                            for ky in 0..*pool_h {
                                for kx in 0..*pool_w {
                                    let y = oy * stride + ky;
                                    let xx = ox * stride + kx;
                                    vals.push(x[base + y * in_w + xx]);
                                }
                            }
                            o.push(match mode.as_str() {
                                "avg" => vals.iter().sum(),
                                "max" => *vals.iter().max().unwrap(),
                                m => panic!("unknown pool mode {m}"),
                            });
                        }
                    }
                }
                o
            }
            OpSpec::Linear { weights, bias, .. } => weights
                .iter()
                .zip(bias)
                .map(|(row, &b)| row.iter().zip(&x).map(|(&w, &v)| w * v).sum::<i64>() + b)
                .collect(),
            other => panic!("unexpected op in CNN fixture: {}", other.op_type()),
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

/// The headline gate: the encrypted CNN's per-sample argmax equals the quantized-cleartext
/// argmax, bit-for-bit, over the committed test batch — run entirely from the serialized IR.
#[test]
fn fhe_matches_quantized_cleartext_cnn() {
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

    // Fail loudly before any crypto if the budget can't hold an accumulator.
    check_graph_bit_width_budget(&graph).expect("CNN bit-width budget must fit");

    let input_name = graph.inputs[0].clone();
    let output_name = graph.outputs[0].clone();

    let (ck, sk) = keygen(graph.num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks: graph.num_blocks,
    };

    for (i, input) in test_inputs.iter().enumerate() {
        // Oracle: recompute the integer CNN in Rust, and confirm the fixture agrees (no drift).
        let ref_logits = cleartext_logits(&graph, input);
        assert_eq!(
            ref_logits, expected_logits[i],
            "fixture expected_logits[{i}] disagrees with the Rust cleartext CNN oracle"
        );
        let ref_label = argmax(&ref_logits) as i64;
        assert_eq!(
            ref_label, expected_labels[i],
            "fixture expected_labels[{i}] disagrees with the cleartext argmax"
        );

        // FHE path: encrypt -> walk the IR graph -> decrypt the 10 logits -> client argmax.
        let mut env = HashMap::new();
        env.insert(input_name.clone(), encrypt(&ck, input));
        let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
        let fhe_logits = decrypt_vec(&ck, &out[&output_name]);

        assert_eq!(
            fhe_logits, ref_logits,
            "GOLDEN VIOLATION at sample {i}: FHE logits {fhe_logits:?} != cleartext {ref_logits:?}"
        );
        let fhe_label = argmax(&fhe_logits) as i64;
        assert_eq!(
            fhe_label, ref_label,
            "GOLDEN VIOLATION at sample {i}: FHE label {fhe_label} != cleartext {ref_label}"
        );
    }
}
