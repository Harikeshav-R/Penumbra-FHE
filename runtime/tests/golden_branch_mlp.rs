//! Golden exactness for the Phase-8 branching MLP (`AGENTS.md` §1.1) — `#[ignore]` by default.
//!
//! > FHE output must equal the quantized-cleartext output, **bit-for-bit.**
//!
//! This is the FHE bit-for-bit gate for branching graphs (Split, Concat, residual Add):
//! `examples/mnist/branch_mlp_export.py` trains an MLP with Split, Concat, and Add,
//! exports it to `digit_branch_mlp.onnx`, loads it through `load_onnx` and quantizes it.
//! The committed `phase8_branch_fixture.json` IR graph is:
//! Linear -> Requant -> Split -> [Linear -> Requant, Linear -> Requant] -> Concat -> Add -> Linear.

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    OpSpec,
};
use serde_json::Value;

fn load_fixture() -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../examples/mnist/phase8_branch_fixture.json");
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
        match &node.op {
            OpSpec::Split { sizes } => {
                let x = env[&node.inputs[0]].clone();
                let mut offset = 0;
                for (sz, out_name) in sizes.iter().zip(&node.outputs) {
                    env.insert(out_name.clone(), x[offset..offset + sz].to_vec());
                    offset += sz;
                }
            }
            OpSpec::Concat { sizes: _ } => {
                let mut cat = Vec::new();
                for inp_name in &node.inputs {
                    cat.extend_from_slice(&env[inp_name]);
                }
                env.insert(node.outputs[0].clone(), cat);
            }
            OpSpec::Add {} => {
                let a = &env[&node.inputs[0]];
                let b = &env[&node.inputs[1]];
                let sum: Vec<i64> = a.iter().zip(b).map(|(&x, &y)| x + y).collect();
                env.insert(node.outputs[0].clone(), sum);
            }
            OpSpec::Linear { weights, bias, .. } => {
                let x = &env[&node.inputs[0]];
                let out: Vec<i64> = weights
                    .iter()
                    .zip(bias)
                    .map(|(row, &b)| row.iter().zip(x).map(|(&w, &v)| w * v).sum::<i64>() + b)
                    .collect();
                env.insert(node.outputs[0].clone(), out);
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
                let x = &env[&node.inputs[0]];
                let ceil = (1i64 << out_bits) - 1;
                let out: Vec<i64> = x
                    .iter()
                    .map(|&v| {
                        let t = v.max(*clamp_lo);
                        let u = (t * *mult as i64 + *round_bias as i64) >> shift;
                        let w = u + *zero_point as i64;
                        w.clamp(0, ceil)
                    })
                    .collect();
                env.insert(node.outputs[0].clone(), out);
            }
            other => panic!("unexpected op in branch fixture: {}", other.op_type()),
        }
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
#[ignore = "TFHE evaluation; run with: cargo test --release --test golden_branch_mlp -- --ignored"]
fn fhe_matches_quantized_cleartext_branch_mlp() {
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
