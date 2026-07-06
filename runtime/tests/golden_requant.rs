//! Golden exactness for the `Requant` op — the Phase-4 crux (`AGENTS.md` §1.1).
//!
//! `Requant` is the new primitive that makes multi-layer models feasible: it rescales a wide
//! signed accumulator down to a narrow non-negative value the next activation/layer can
//! consume. Its exact semantics are
//!
//! ```text
//! requant(x) = clamp(max(x >> shift, 0), 0, 2^out_bits - 1)   // arithmetic >>
//! ```
//!
//! and TFHE is exact, so the FHE op must equal that integer function **bit-for-bit**. This
//! test sweeps a deliberately adversarial set of inputs — negatives (ReLU'd to 0), values
//! that saturate at the clamp ceiling, and crucially values whose post-shift magnitude
//! exceeds one message block (the "value mod 2^MESSAGE_BITS" trap that radix-level saturation
//! must defeat) — and asserts the decrypted output matches the cleartext oracle.
//!
//! Run with `cargo test --release` — debug FHE is impractically slow.

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    Node, OpSpec, SCHEMA_VERSION,
};

/// MESSAGE_BITS under the default profile (2). Kept local to avoid leaking a crypto constant
/// into the test surface; the runtime is the source of truth.
const MESSAGE_BITS: usize = 2;

/// The cleartext oracle: arithmetic shift, ReLU, clamp to `2^out_bits - 1`.
fn requant_cleartext(x: i64, shift: u32, out_bits: usize) -> i64 {
    let shifted = x >> shift; // arithmetic (floor) shift for signed i64
    let nonneg = shifted.max(0);
    nonneg.min((1i64 << out_bits) - 1)
}

/// Build the identity-over-range clamp LUT the way the quantization service will: index `v`
/// (an already-saturated block value) maps to `min(v, 2^out_bits - 1)`.
fn clamp_lut(out_bits: usize) -> Vec<u64> {
    let ceil = (1u64 << out_bits) - 1;
    (0..(1u64 << MESSAGE_BITS)).map(|v| v.min(ceil)).collect()
}

fn requant_graph(num_blocks: usize, input_bits: usize, shift: u32, out_bits: usize) -> Graph {
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
                // This legacy sweep pins the pure power-of-two-shift semantics: mult=1,
                // round_bias=0 makes the generalized op reduce to clamp(max(x>>shift,0),…).
                mult: 1,
                round_bias: 0,
                out_bits,
                clamp_lut: clamp_lut(out_bits),
                // Per-tensor: the 0.6.0 per-channel overlay is unused (omitted from the JSON).
                mults: vec![],
                shifts: vec![],
                round_biases: vec![],
                channel_size: None,
            },
        }],
    }
}

/// FHE `Requant` equals `clamp(max(x>>shift,0),0,2^out-1)` over an adversarial input sweep.
#[test]
fn fhe_requant_matches_cleartext() {
    // 6 blocks = 12-bit signed radix: holds the wide inputs below comfortably while staying
    // cheaper than the 8-block logreg fixture.
    let num_blocks = 6;
    let input_bits = 10; // inputs up to ~±1000
    let shift = 4; // divide by 16
    let out_bits = 2; // narrow to a single 2-bit block, ceiling 3

    let graph = requant_graph(num_blocks, input_bits, shift, out_bits);
    check_graph_bit_width_budget(&graph).expect("Requant budget must fit");

    // Adversarial inputs (all fit input_bits=10, |x| <= 1023):
    //   -100 -> -7 -> ReLU 0;  0 -> 0;  16 -> 1;  47 -> 2;  64 -> 4 -> clamp 3;
    //   1000 -> 62 -> clamp 3 (the "62 mod 4 = 2" trap: must saturate to 3, not read 2);
    //   -1 -> -1 -> ReLU 0;  31 -> 1;  48 -> 3.
    let inputs: Vec<i64> = vec![-100, 0, 16, 47, 64, 1000, -1, 31, 48];
    let expected: Vec<i64> = inputs
        .iter()
        .map(|&x| requant_cleartext(x, shift, out_bits))
        .collect();
    // Sanity: the trap cases are actually present (post-shift >= 4 that must clamp to 3).
    assert_eq!(requant_cleartext(64, shift, out_bits), 3);
    assert_eq!(requant_cleartext(1000, shift, out_bits), 3);

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
        "GOLDEN VIOLATION: FHE Requant {got:?} != cleartext {expected:?} (inputs {inputs:?})"
    );
}
