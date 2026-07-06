//! Golden exactness for the **per-channel** `Requant` overlay (`AGENTS.md` §1.1, ROADMAP Phase 5).
//!
//! 0.6.0 added a per-channel overlay to `Requant`: each output channel gets its own fixed-point
//! multiplier, so per-channel weight quantization rescales each channel by its true ratio. The
//! flat element `idx` maps to channel `idx / channel_size`:
//!
//! ```text
//! requant(x[idx]) = clamp( (max(x,0) * mults[ch] + round_biases[ch]) >> shifts[ch],
//!                          0, 2^out_bits - 1 ),   ch = idx / channel_size
//! ```
//!
//! Every step (ReLU, per-channel scalar-mul, add, arithmetic shift, saturate, single-block PBS)
//! has an exact integer counterpart, so TFHE must equal the cleartext function **bit-for-bit**.
//! This test uses a `channel_size > 1` layout (like a Conv2d's `[out_ch][out_h][out_w]`), distinct
//! per-channel `(mult, shift, round_bias)` — including a `mult = 1` channel that exercises the
//! scalar-mul skip alongside a `mult != 1` channel — and asserts the decrypted FHE output matches
//! the i64 oracle over an adversarial input sweep spanning channels.
//!
//! Run with `cargo test --release` — debug FHE is impractically slow.

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    Node, OpSpec, SCHEMA_VERSION,
};

const MESSAGE_BITS: usize = 2;

/// The per-channel cleartext oracle: for each flat element, pick its channel's rescale.
fn requant_per_channel_cleartext(
    x: &[i64],
    mults: &[u64],
    shifts: &[u32],
    round_biases: &[u64],
    channel_size: usize,
    out_bits: usize,
) -> Vec<i64> {
    let ceil = (1i64 << out_bits) - 1;
    x.iter()
        .enumerate()
        .map(|(idx, &v)| {
            let ch = idx / channel_size;
            let scaled = (v.max(0) * mults[ch] as i64 + round_biases[ch] as i64) >> shifts[ch];
            scaled.max(0).min(ceil)
        })
        .collect()
}

fn clamp_lut(out_bits: usize) -> Vec<u64> {
    let ceil = (1u64 << out_bits) - 1;
    (0..(1u64 << MESSAGE_BITS)).map(|v| v.min(ceil)).collect()
}

/// FHE per-channel Requant == cleartext oracle, bit-for-bit.
#[test]
fn fhe_per_channel_requant_matches_cleartext() {
    let out_bits = 2;
    // 3 elements per channel (e.g. a 1x3 Conv2d output row). Two channels with deliberately
    // different rescales:
    //  - channel 0: mult=1 (the scalar-mul skip path), shift=4, truncating (round_bias=0)
    //  - channel 1: mult=3, shift=5, round-half-up (round_bias=2^(shift-1)=16)
    let channel_size = 3;
    let mults = vec![1u64, 3];
    let shifts = vec![4u32, 5];
    let round_biases = vec![0u64, 1u64 << (5 - 1)];

    // input_bits=10 (|x| <= ~511 here); the internal peak max(x,0)*3 + 16 = 1549 (~11 magnitude
    // bits, +1 sign = 12) -> 7 blocks (14-bit) is comfortable.
    let num_blocks = 7;
    let input_bits = 10;

    // Flat inputs: 3 for channel 0, then 3 for channel 1 (channel-major, matching Conv2d layout).
    let inputs: Vec<i64> = vec![-100, 63, 511, -1, 47, 200];
    let expected = requant_per_channel_cleartext(
        &inputs,
        &mults,
        &shifts,
        &round_biases,
        channel_size,
        out_bits,
    );

    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks,
        input_bits,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "rq".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Requant {
                // Scalar fields are neutral and ignored when the per-channel arrays are present.
                shift: 0,
                mult: 1,
                round_bias: 0,
                out_bits,
                clamp_lut: clamp_lut(out_bits),
                mults: mults.clone(),
                shifts: shifts.clone(),
                round_biases: round_biases.clone(),
                channel_size: Some(channel_size),
            },
        }],
    };
    check_graph_bit_width_budget(&graph).expect("per-channel Requant budget must fit");

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
        "GOLDEN VIOLATION: per-channel Requant (mults={mults:?}, shifts={shifts:?}, \
         round_biases={round_biases:?}, channel_size={channel_size}) FHE {got:?} != cleartext \
         {expected:?} (inputs {inputs:?})"
    );
}
