//! `Activation` — apply a single-input function via a lookup table (LUT).
//!
//! In principle a LUT can realize *any* 1-input function — ReLU, sigmoid, GELU (`PROJECT.md`
//! §6) — and this is the *expensive* regime (`PROJECT.md` §5): a **programmable bootstrap
//! (PBS)** applies the LUT to a ciphertext, so runtime ≈ number of bootstraps and
//! activations dominate cost.
//!
//! ## Domain (Phase 2 — deliberately narrow)
//!
//! The current representation (`lut: Vec<u64>` indexed by the input value) only models a
//! **non-negative** message-space domain with **non-negative** integer outputs: the table
//! is indexed by a single radix block's value `[0, 2^MESSAGE_BITS)` and its entries are
//! `u64`. Signed inputs/outputs and wider domains (the full sigmoid/GELU story) need a
//! Requant + a signed/offset LUT encoding and are a later phase — do not read the "any
//! 1-input function" framing above as a claim about *this* op's present reach.
//!
//! This is exactly the budget reasoning of `PROJECT.md` §9: a PBS over a wide value is
//! infeasible, so a wide accumulator must be `Requant`-ed down first (Phase 4) before an
//! activation. Phase 2 applies the activation on a deliberately narrow value, the same
//! primitive proven in `tests/hello_fhe.rs`, lifted into the op interface.
//!
//! The LUT is given as an explicit table indexed by the (non-negative) input integer — it
//! must be generated in the quantized-integer domain consistent with the chosen scales, or
//! accuracy silently breaks (`PROJECT.md` §8). The library's quantization service owns
//! producing it; the runtime just applies it bit-exactly.

use tfhe::integer::{IntegerCiphertext, SignedRadixCiphertext};
use tfhe::shortint::Ciphertext;

use super::{CtVec, EvalCtx, Op};
use crate::keys::MESSAGE_BITS;

/// Minimum bits needed to represent every entry of a LUT (its true output width).
fn lut_output_bits(lut: &[u64]) -> usize {
    let max = lut.iter().copied().max().unwrap_or(0);
    // `0` still occupies a 1-bit value; otherwise it's the position of the top set bit.
    ((u64::BITS - max.leading_zeros()) as usize).max(1)
}

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

        // The table must cover the input block's *entire* message space. A short table would
        // otherwise silently map the uncovered values to 0 (masking a fixture/export bug and
        // producing wrong-but-confident ciphertext); an oversized one signals a domain
        // mismatch. Fail loudly at build time instead (`AGENTS.md` §1.4).
        let domain = 1usize << MESSAGE_BITS;
        assert_eq!(
            self.lut.len(),
            domain,
            "Activation LUT must cover the full {MESSAGE_BITS}-bit message space \
             ({domain} entries); got {}",
            self.lut.len()
        );

        let table = self.lut.clone();
        let lut = shortint_sk.generate_lookup_table(move |v| {
            // Indexing is total over the message space by the assert above; the fallback is
            // unreachable and exists only to keep the closure infallible.
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
        //
        // Derive the true width from the table itself rather than trusting the declared
        // field blindly: if `output_bits` drifted *below* the real LUT range (a stale or
        // hand-edited fixture), the budget check would under-count downstream widths and
        // could miss an overflow (`AGENTS.md` §1.3). Fail loudly on that drift; a declared
        // value larger than necessary is allowed (conservative headroom).
        let derived = lut_output_bits(&self.lut);
        assert!(
            self.output_bits >= derived,
            "Activation output_bits ({}) is smaller than the LUT's actual range ({} bits); \
             the declared width must not under-count the table",
            self.output_bits,
            derived
        );
        self.output_bits
    }
}
