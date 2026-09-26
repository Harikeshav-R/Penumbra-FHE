//! Golden exactness for signed Requant + affine Activation under TFHE (`AGENTS.md` §1.1).
//!
//! Asserts that a two-node graph `Requant(clamp_lo=-N, zero_point=Z) → Activation(lut)`
//! evaluated under TFHE matches the cleartext quantized formula bit-for-bit over inputs
//! spanning negative, zero, positive, and saturation values.

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    Node, OpSpec, SCHEMA_VERSION,
};

fn cleartext_oracle(
    x: &[i64],
    clamp_lo: i64,
    zero_point: i64,
    shift: u32,
    mult: u64,
    round_bias: u64,
    act_lut: &[u64],
) -> Vec<i64> {
    x.iter()
        .map(|&v| {
            let t = v.max(clamp_lo);
            let u = (t * mult as i64 + round_bias as i64) >> shift;
            let w = (u + zero_point).clamp(0, 3) as usize;
            act_lut[w] as i64
        })
        .collect()
}

#[test]
#[ignore = "TFHE evaluation; run with: cargo test --release --test golden_requant_signed -- --ignored"]
fn test_tfhe_signed_requant_matches_cleartext() {
    let num_blocks = 6;
    let input_bits = 8;
    let shift = 3;
    let mult = 1;
    let round_bias = 4;
    let clamp_lo = -16;
    let zero_point = 2;
    let out_bits = 2;
    let clamp_lut = vec![0u64, 1, 2, 3];
    let act_lut = vec![0u64, 2, 1, 3];

    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks,
        input_bits,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![
            Node {
                name: "rq".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["rq_out".to_string()],
                op: OpSpec::Requant {
                    shift,
                    mult,
                    round_bias,
                    clamp_lo,
                    zero_point,
                    out_bits,
                    clamp_lut,
                    mults: vec![],
                    shifts: vec![],
                    round_biases: vec![],
                    channel_size: None,
                },
            },
            Node {
                name: "act".to_string(),
                inputs: vec!["rq_out".to_string()],
                outputs: vec!["y".to_string()],
                op: OpSpec::Activation {
                    lut: act_lut.clone(),
                    output_bits: 2,
                },
            },
        ],
    };

    check_graph_bit_width_budget(&graph).expect("budget must fit");

    let inputs = vec![-50i64, -16, -10, -4, 0, 4, 12, 100];
    let expected = cleartext_oracle(
        &inputs,
        clamp_lo,
        zero_point as i64,
        shift,
        mult,
        round_bias,
        &act_lut,
    );

    let (ck, sk) = keygen(num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks,
    };

    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &inputs));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let got = decrypt_vec(&ck, &out["y"]);

    assert_eq!(
        got, expected,
        "TFHE signed requant + activation must equal quantized cleartext bit-for-bit"
    );
}
