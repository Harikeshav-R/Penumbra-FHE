//! `Activation` — apply an arbitrary single-input function via a lookup table (LUT).
//!
//! Covers ReLU, sigmoid, GELU, any 1-input function (`PROJECT.md` §6). This is the
//! *expensive* regime (`PROJECT.md` §5): a **programmable bootstrap (PBS)** applies the
//! LUT to a ciphertext. Runtime ≈ number of bootstraps, so activations dominate cost.
//!
//! ## Domain (Phase 2)
//!
//! A LUT lives in a small integer domain — at most one radix block wide
//! ([`crate::keys::MESSAGE_BITS`] bits). That is exactly the budget reasoning of
//! `PROJECT.md` §9: a PBS over a wide value is infeasible, so a wide accumulator must be
//! `Requant`-ed down first (Phase 4) before an activation. Phase 2 applies the activation
//! on a deliberately narrow value, the same primitive proven in `tests/hello_fhe.rs`,
//! lifted into the op interface.
//!
//! The LUT is given as an explicit table indexed by the (non-negative) input integer — it
//! must be generated in the quantized-integer domain consistent with the chosen scales, or
//! accuracy silently breaks (`PROJECT.md` §8). The library's quantization service owns
//! producing it; the runtime just applies it bit-exactly.

use tfhe::integer::{IntegerCiphertext, SignedRadixCiphertext};
use tfhe::shortint::Ciphertext;

use super::{CtVec, EvalCtx, Op};

/// Single-input activation realized as a LUT over a narrow integer domain.
///
/// `lut[v]` is the output for input value `v`. The table must cover every value the input
/// block can hold; the input is treated as a small non-negative integer in `[0, lut.len())`.
pub struct Activation {
    /// Lookup table indexed by the input integer value.
    pub lut: Vec<u64>,
    /// Bit-width of the table's output values (its bit-width growth declaration).
    pub output_bits: usize,
}

impl Op for Activation {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        let sk = ctx.sk;
        // The inner `shortint` key is where PBS / LUTs live (the integer API is built on
        // top of it). Building the LUT once and reusing it across elements is cheaper.
        let shortint_sk = sk.as_ref();
        let table = self.lut.clone();
        let lut = shortint_sk.generate_lookup_table(move |v| {
            // Clamp the index defensively: the input block can only hold message-space
            // values, and the table is sized to cover them. Out-of-range is a build bug.
            *table.get(v as usize).unwrap_or(&0)
        });

        inputs
            .iter()
            .map(|ct| {
                // Apply the LUT to the least-significant block (the narrow value) via PBS,
                // then rebuild a signed radix whose remaining blocks are trivial zeros.
                let mapped: Ciphertext = shortint_sk.apply_lookup_table(&ct.blocks()[0], &lut);
                let mut blocks = Vec::with_capacity(ctx.num_blocks);
                blocks.push(mapped);
                for _ in 1..ctx.num_blocks {
                    blocks.push(shortint_sk.create_trivial(0));
                }
                SignedRadixCiphertext::from(blocks)
            })
            .collect()
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        // An activation's output width is set by its table, independent of input width —
        // and must stay small (≤ MESSAGE_BITS) to remain a single-block, LUT-able value.
        self.output_bits
    }
}
