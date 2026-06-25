//! `Requant` — rescale a wide accumulator down to a narrow, LUT-able value.
//!
//! This is the primitive that makes **multi-layer** models feasible (`PROJECT.md` §9,
//! ROADMAP Phase 4). A `Linear`/`Conv2d` accumulator grows ~`log2(N)` bits per layer, but a
//! programmable bootstrap (the only way to apply an activation) is feasible only over a
//! *narrow* value — the existing `Activation` consumes a single `MESSAGE_BITS`-wide block.
//! `Requant` bridges the two: it takes the wide signed accumulator and produces a small
//! non-negative integer the next activation / layer can consume.
//!
//! ## Exact semantics (the cleartext oracle, matched bit-for-bit — `AGENTS.md` §1.1)
//!
//! ```text
//! requant(x) = clamp( max(x >> shift, 0), 0, 2^out_bits - 1 )
//! ```
//!
//! `shift` is a non-negative **power-of-two rescale** (chosen by the quantization service
//! from the layer scales). We deliberately use an **arithmetic right shift** (floor-divide by
//! `2^shift`), not `round(x / scale)`: the FHE shift and an integer `>>` agree exactly, while
//! rounding-division has no clean exact FHE counterpart. The op is a **fused ReLU+requant** —
//! the `max(·, 0)` is the ReLU CNNs apply after conv, and it also guarantees the
//! non-negative output the single-block trick requires (`PROJECT.md` §11 / Phase-4 decision).
//! Arbitrary (non-power-of-two) scales are a Phase-5 concern.
//!
//! ## How it maps onto `tfhe-rs` (verified against `tfhe-1.6.2`)
//!
//! 1. `scalar_right_shift_parallelized` — arithmetic on a *signed* radix (sign-bit padded),
//!    so it equals `x >> shift` bit-for-bit.
//! 2. `scalar_max_parallelized(·, 0)` — ReLU at the radix level (kills negatives).
//! 3. `scalar_min_parallelized(·, 2^out_bits - 1)` — **saturate at the radix level** so the
//!    value genuinely fits one block. This step is essential: extracting `blocks()[0]` from an
//!    un-saturated value would read `value mod 2^MESSAGE_BITS` and silently clamp the *wrong*
//!    number (e.g. `clamp(5,0,3)` must be 3, but `5 mod 4 = 1`).
//! 4. A single-block PBS applying `clamp_lut` to `blocks()[0]` — the same proven path as
//!    [`crate::ops::activation::Activation`]. With the value already saturated this LUT is the
//!    identity over its in-range domain, but the PBS resets noise to a clean low level for the
//!    next layer (the whole reason activations bootstrap). The result is rebuilt as a radix
//!    whose higher blocks are trivial zeros.

use tfhe::integer::{IntegerCiphertext, SignedRadixCiphertext};
use tfhe::shortint::Ciphertext;

use super::{CtVec, EvalCtx, Op};
use crate::keys::MESSAGE_BITS;

/// Minimum bits needed to represent every entry of a LUT (its true output width).
fn lut_output_bits(lut: &[u64]) -> usize {
    let max = lut.iter().copied().max().unwrap_or(0);
    crate::keys::magnitude_bits(max).max(1)
}

/// Rescale a wide accumulator to a narrow non-negative value via shift + ReLU + clamp.
///
/// `clamp_lut[v]` is the output for the (already-saturated) input block value `v`; it must
/// cover the whole `MESSAGE_BITS`-bit message space and saturate at `2^out_bits - 1`.
pub struct Requant {
    /// Power-of-two right-shift amount (the rescale).
    pub shift: u32,
    /// Bit-width of the narrowed output (`≤ MESSAGE_BITS`); its bit-width growth declaration.
    pub out_bits: usize,
    /// Clamp lookup table indexed by the saturated input block value.
    pub clamp_lut: Vec<u64>,
}

impl Op for Requant {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        let sk = ctx.sk;
        let shortint_sk = sk.as_ref();

        // Mirror the LUT validation of `Activation` (`AGENTS.md` §1.4): the table must cover
        // the block's whole message space, and no entry may exceed it (a PBS reduces modulo
        // the message modulus, so a larger entry would silently wrap).
        let domain = 1usize << MESSAGE_BITS;
        assert_eq!(
            self.clamp_lut.len(),
            domain,
            "Requant clamp_lut must cover the full {MESSAGE_BITS}-bit message space \
             ({domain} entries); got {}",
            self.clamp_lut.len()
        );
        if let Some((idx, &bad)) = self
            .clamp_lut
            .iter()
            .enumerate()
            .find(|&(_, &e)| e >= (1u64 << MESSAGE_BITS))
        {
            panic!(
                "Requant clamp_lut[{idx}] = {bad} does not fit one shortint block: every output \
                 must be < {} (the {MESSAGE_BITS}-bit message space).",
                1u64 << MESSAGE_BITS
            );
        }
        // The output block must not be the radix sign block, or a value of 2/3 would decrypt
        // negative under the 2-bit signed top block (same constraint as `Activation`).
        assert!(
            ctx.num_blocks > 1,
            "Requant requires num_blocks > 1 so the narrowed value lands in a value block, not \
             the radix sign block (got num_blocks = {})",
            ctx.num_blocks
        );

        let max_val = (1i64 << self.out_bits) - 1;

        let table = self.clamp_lut.clone();
        let lut = shortint_sk.generate_lookup_table(move |v| *table.get(v as usize).unwrap_or(&0));

        inputs
            .iter()
            .map(|ct| {
                // 1. Arithmetic right shift: floor-divide by 2^shift (== cleartext `x >> shift`).
                let shifted: SignedRadixCiphertext =
                    sk.scalar_right_shift_parallelized(ct, self.shift);
                // 2. ReLU at the radix level: max(shifted, 0).
                let nonneg = sk.scalar_max_parallelized(&shifted, 0i64);
                // 3. Saturate at the radix level so the value fits one block BEFORE we read it.
                let saturated = sk.scalar_min_parallelized(&nonneg, max_val);
                // 4. Single-block PBS (clamp LUT) — identity over the in-range value, but resets
                //    noise to a clean level; rebuild a radix with trivial-zero high blocks.
                let mapped: Ciphertext =
                    shortint_sk.apply_lookup_table(&saturated.blocks()[0], &lut);
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
        // Requant's whole purpose is to consume a *wide* input, so — unlike `Activation` — it
        // asserts nothing about `input_bits`. Its output width is set by `out_bits`, which must
        // stay single-block (≤ MESSAGE_BITS) and must not under-count the table's true range
        // (the drift guard from `Activation`).
        let derived = lut_output_bits(&self.clamp_lut);
        assert!(
            self.out_bits >= derived,
            "Requant out_bits ({}) is smaller than the clamp_lut's actual range ({} bits); the \
             declared width must not under-count the table",
            self.out_bits,
            derived
        );
        assert!(
            self.out_bits <= MESSAGE_BITS,
            "Requant out_bits ({}) exceeds MESSAGE_BITS ({MESSAGE_BITS}); the narrowed value \
             must fit a single shortint block",
            self.out_bits
        );
        self.out_bits
    }
}
