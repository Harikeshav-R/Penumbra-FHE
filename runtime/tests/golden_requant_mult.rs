//! Golden exactness for the **generalized** `Requant` — fixed-point multiply-then-round-shift
//! (`AGENTS.md` §1.1, ROADMAP Phase 5).
//!
//! Phase 5 generalized `Requant` from a power-of-two-only shift to the standard
//! integer-quantization rescale
//!
//! ```text
//! requant(x) = clamp( (max(x, 0) * mult + round_bias) >> shift, 0, 2^out_bits - 1 )
//! ```
//!
//! so an arbitrary real scale ratio is approximated by `mult / 2^shift` with round-to-nearest.
//! Every step (ReLU, scalar-mul, add, arithmetic shift, saturate, single-block PBS) has an
//! exact integer counterpart, so TFHE must equal the cleartext function **bit-for-bit**. This
//! test sweeps an adversarial input set with a non-trivial multiplier and a round-half-up bias,
//! and asserts the decrypted FHE output matches the i64 oracle.
//!
//! It also pins **backward compatibility**: with `mult = 1, round_bias = 0` the generalized op
//! reduces to the Phase-4 `clamp(max(x >> shift, 0), …)` value-for-value (covered by the
//! `golden_requant.rs` legacy sweep; re-asserted here against the oracle for the mult=1 case).
//!
//! Run with `cargo test --release` — debug FHE is impractically slow.

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    Node, OpSpec, SCHEMA_VERSION,
};

const MESSAGE_BITS: usize = 2;

/// The cleartext oracle: ReLU, fixed-point multiply, round bias, arithmetic shift, clamp.
fn requant_cleartext(x: i64, mult: u64, round_bias: u64, shift: u32, out_bits: usize) -> i64 {
    let nonneg = x.max(0);
    let scaled = nonneg * mult as i64 + round_bias as i64;
    let shifted = scaled >> shift; // arithmetic (floor) shift; value is non-negative here
    shifted.min((1i64 << out_bits) - 1)
}

fn clamp_lut(out_bits: usize) -> Vec<u64> {
    let ceil = (1u64 << out_bits) - 1;
    (0..(1u64 << MESSAGE_BITS)).map(|v| v.min(ceil)).collect()
}

#[allow(clippy::too_many_arguments)]
fn requant_graph(
    num_blocks: usize,
    input_bits: usize,
    shift: u32,
    mult: u64,
    round_bias: u64,
    out_bits: usize,
) -> Graph {
    Graph {
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
                shift,
                mult,
                round_bias,
                out_bits,
                clamp_lut: clamp_lut(out_bits),
                // Per-tensor scalar rescale — the per-channel overlay is exercised by
                // golden_requant_per_channel.rs.
                mults: vec![],
                shifts: vec![],
                round_biases: vec![],
                channel_size: None,
            },
        }],
    }
}

/// Run one `(mult, round_bias, shift)` config over an input sweep and assert FHE == oracle.
fn assert_matches(num_blocks: usize, input_bits: usize, mult: u64, round_bias: u64, shift: u32) {
    let out_bits = 2;
    let graph = requant_graph(num_blocks, input_bits, shift, mult, round_bias, out_bits);
    check_graph_bit_width_budget(&graph).expect("generalized Requant budget must fit");

    let inputs: Vec<i64> = vec![-100, -1, 0, 1, 7, 16, 31, 47, 63, 64, 200, 511];
    let expected: Vec<i64> = inputs
        .iter()
        .map(|&x| requant_cleartext(x, mult, round_bias, shift, out_bits))
        .collect();

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
        "GOLDEN VIOLATION: generalized Requant (mult={mult}, round_bias={round_bias}, \
         shift={shift}) FHE {got:?} != cleartext {expected:?} (inputs {inputs:?})"
    );
}

/// A non-trivial multiplier with round-to-nearest: `mult=3, round_bias=2^(shift-1)`.
#[test]
fn fhe_generalized_requant_matches_cleartext() {
    // input_bits=10 (|x| <= ~511 here); the internal peak max(x,0)*3 + bias must fit the radix.
    // 511*3 + 16 = 1549 (~11 magnitude bits, +1 sign = 12) -> 7 blocks (14-bit) is comfortable.
    let shift = 5; // /32
    assert_matches(7, 10, 3, 1u64 << (shift - 1), shift);
}

/// Backward-compat: `mult=1, round_bias=0` reproduces the legacy pure-shift semantics exactly.
#[test]
fn fhe_legacy_mult_one_matches_cleartext() {
    assert_matches(6, 10, 1, 0, 4);
}
