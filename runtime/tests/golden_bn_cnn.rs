//! Golden exactness for the Phase-8 BN CNN (`AGENTS.md` §1.1) — `#[ignore]` by default.
//!
//! > FHE output must equal the quantized-cleartext output, **bit-for-bit.**
//!
//! This is the FHE bit-for-bit gate for the folded BatchNorm CNN pipeline:
//! `examples/mnist/bn_cnn_export.py` trains a CNN with BatchNorm and AveragePool,
//! exports it to `digit_bn_cnn.onnx`, loads it through `load_onnx` (folding BatchNorm into Conv2d)
//! and quantizes it. The committed `phase8_bn_cnn_fixture.json` IR graph is:
//! Conv2d -> Requant -> Pool(avg) -> Linear.

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    OpSpec,
};
use serde_json::Value;

fn load_fixture() -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../examples/mnist/phase8_bn_cnn_fixture.json");
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
                clamp_lo,
                zero_point,
                out_bits,
                ..
            } => {
                let ceil = (1i64 << out_bits) - 1;
                x.iter()
                    .map(|&v| {
                        let t = v.max(*clamp_lo);
                        let u = (t * *mult as i64 + *round_bias as i64) >> shift;
                        let w = u + *zero_point as i64;
                        w.clamp(0, ceil)
                    })
                    .collect()
            }
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
            other => panic!("unexpected op in BN CNN fixture: {}", other.op_type()),
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
#[ignore = "TFHE evaluation; run with: cargo test --release --test golden_bn_cnn -- --ignored"]
fn fhe_matches_quantized_cleartext_bn_cnn() {
    let fx = load_fixture();
    let graph = Graph::from_json(&fx["graph"].to_string()).expect("fixture graph deserializes");

    let test_inputs: Vec<Vec<i64>> = fx["test_inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(as_i64_vec)
        .collect();
    let expected_labels: Vec<i64> = fx["expected_labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect();
    let expected_logits: Vec<Vec<i64>> = fx["expected_logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(as_i64_vec)
        .collect();

    check_graph_bit_width_budget(&graph).expect("graph fits budget");

    println!("Keygen: num_blocks = {}", graph.num_blocks);
    let (client_key, server_key) = keygen(graph.num_blocks);
    let ctx = EvalCtx::new(&server_key, graph.num_blocks);

    for (i, ((input, &exp_label), exp_logits)) in test_inputs
        .iter()
        .zip(&expected_labels)
        .zip(&expected_logits)
        .enumerate()
    {
        println!("Sample {i}: encrypting input of length {}", input.len());
        let ct_in = encrypt(&client_key, input);
        let mut in_map = HashMap::new();
        in_map.insert(graph.inputs[0].clone(), ct_in);

        println!(
            "Sample {i}: evaluating graph with {} nodes",
            graph.nodes.len()
        );
        let mut out_map = evaluate_graph(&ctx, &graph, in_map).expect("graph evaluation succeeds");
        let ct_out = out_map
            .remove(&graph.outputs[0])
            .expect("graph produces output");

        let fhe_logits = decrypt_vec(&client_key, &ct_out);
        let fhe_label = argmax(&fhe_logits) as i64;

        assert_eq!(
            fhe_logits, *exp_logits,
            "Sample {i}: FHE logits {fhe_logits:?} != expected cleartext logits {exp_logits:?} (GOLDEN VIOLATION)"
        );
        assert_eq!(
            fhe_label, exp_label,
            "Sample {i}: FHE label {fhe_label} != expected label {exp_label}"
        );

        let cleartext_run = cleartext_logits(&graph, input);
        assert_eq!(
            cleartext_run, *exp_logits,
            "Sample {i}: in-test cleartext logits != fixture expected logits"
        );
        println!("Sample {i}: PASS (exact match)");
    }
}
