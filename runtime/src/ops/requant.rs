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
//! requant(x) = clamp( (max(x, 0) * mult + round_bias) >> shift, 0, 2^out_bits - 1 )
//! ```
//!
//! This is the standard integer-quantization rescale (gemmlowp/TFLite "fixed-point
//! multiplier"): an arbitrary real rescale `M = (s_in · s_w) / s_out` is approximated as the
//! ratio `mult / 2^shift`. The quantization service (Phase 5) chooses `(mult, shift)`; the
//! whole expression is **bit-exact in both the FHE and cleartext domains** because every step
//! — `scalar_mul`, `scalar_add`, arithmetic right shift — has an exact integer counterpart
//! (we never introduce true division, which has no clean exact FHE form).
//!
//! - `mult` is a small **non-negative plaintext multiplier** (a *plaintext-weight* scalar mul
//!   — cheap, **no PBS**, so PBS count is unchanged from the old power-of-two-only path). It
//!   widens the value by `magnitude_bits(mult)` bits before the shift narrows it back; the
//!   bit-width tracker checks that this *internal* peak still fits the radix
//!   ([`Requant::internal_bits_n`], `AGENTS.md` §1.3).
//! - `round_bias` implements **round-to-nearest** (`round_bias = 2^(shift-1)` for round-half-
//!   up); `0` gives truncation. Rounding is unambiguous here because the value is non-negative
//!   after the ReLU.
//! - `shift` is the **arithmetic right shift** (floor-divide by `2^shift`). The op is a
//!   **fused ReLU+requant** — the `max(·, 0)` is the ReLU CNNs apply after conv, and it also
//!   guarantees the non-negative output the single-block trick requires.
//!
//! Backward compatibility: `mult = 1, round_bias = 0` reduces to the Phase-4 semantics
//! `clamp(max(x >> shift, 0), 0, 2^out_bits - 1)` value-for-value (`max(x,0) >> shift` and
//! `max(x >> shift, 0)` agree for all `x` under a non-negative arithmetic shift), so legacy
//! fixtures regenerate byte-identically (only the new defaulted fields appear in the JSON).
//!
//! ## How it maps onto `tfhe-rs` (verified against `tfhe-1.6.2`)
//!
//! 1. `scalar_max_parallelized(·, 0)` — ReLU at the radix level (kills negatives) FIRST, so
//!    the multiply and rounding operate on a non-negative value.
//! 2. `scalar_mul_parallelized(·, mult)` — the fixed-point multiplier (plaintext scalar mul).
//! 3. `scalar_add_parallelized(·, round_bias)` — the round-to-nearest bias.
//! 4. `scalar_right_shift_parallelized(·, shift)` — arithmetic shift; equals `>>` bit-for-bit.
//! 5. `scalar_min_parallelized(·, 2^out_bits - 1)` — **saturate at the radix level** so the
//!    value genuinely fits one block. This step is essential: extracting `blocks()[0]` from an
//!    un-saturated value would read `value mod 2^MESSAGE_BITS` and silently clamp the *wrong*
//!    number (e.g. `clamp(5,0,3)` must be 3, but `5 mod 4 = 1`).
//! 6. A single-block PBS applying `clamp_lut` to `blocks()[0]` — the same proven path as
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

/// Peak *internal* bit-width the rescale needs before the shift narrows it, given a signed
/// `input_bits`-wide accumulator. The intermediate `max(x,0) * mult + round_bias` is the
/// widest value `Requant` materializes; it must fit the radix even though the op's *output*
/// is tiny (`out_bits`). For the legacy `mult = 1, round_bias = 0` path this equals
/// `input_bits` exactly (no growth), so existing models keep fitting their radix.
pub fn requant_internal_bits(input_bits: usize, mult: u64, round_bias: u64) -> usize {
    // Largest positive value a signed `input_bits`-wide accumulator holds, post-ReLU.
    let relu_max: u64 = if input_bits >= 1 {
        (1u64 << (input_bits - 1)).saturating_sub(1)
    } else {
        0
    };
    let intermediate_max = relu_max.saturating_mul(mult).saturating_add(round_bias);
    // +1 for the sign bit: the intermediate lives in the signed radix even though it is
    // non-negative. Never report less than the input width itself.
    (crate::keys::magnitude_bits(intermediate_max) + 1).max(input_bits)
}

/// Rescale a wide accumulator to a narrow non-negative value via ReLU + fixed-point multiply
/// + round + shift + clamp.
///
/// `clamp_lut[v]` is the output for the (already-saturated) input block value `v`; it must
/// cover the whole `MESSAGE_BITS`-bit message space and saturate at `2^out_bits - 1`.
pub struct Requant {
    /// Power-of-two right-shift amount (the denominator of the rescale `mult / 2^shift`).
    pub shift: u32,
    /// Fixed-point multiplier (the numerator of the rescale); `1` is the legacy pure-shift.
    pub mult: u64,
    /// Round-to-nearest bias added before the shift (`2^(shift-1)` rounds half-up; `0`
    /// truncates).
    pub round_bias: u64,
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
                // 1. ReLU at the radix level FIRST: max(x, 0). Doing the ReLU before the
                //    multiply/round keeps the value non-negative, which the rounding and the
                //    single-block trick both rely on.
                let nonneg = sk.scalar_max_parallelized(ct, 0i64);
                // 2. Fixed-point multiplier (plaintext scalar mul, NO PBS). Skip the op when
                //    mult == 1 so the legacy path is bit-identical and adds no noise.
                let scaled = if self.mult == 1 {
                    nonneg
                } else {
                    sk.scalar_mul_parallelized(&nonneg, self.mult)
                };
                // 3. Round-to-nearest bias (skip when 0 — the truncating/legacy path).
                let biased = if self.round_bias == 0 {
                    scaled
                } else {
                    sk.scalar_add_parallelized(&scaled, self.round_bias)
                };
                // 4. Arithmetic right shift: floor-divide by 2^shift (the value is non-negative
                //    here, so this equals the cleartext `>>` bit-for-bit).
                let shifted: SignedRadixCiphertext =
                    sk.scalar_right_shift_parallelized(&biased, self.shift);
                // 5. Saturate at the radix level so the value fits one block BEFORE we read it.
                let saturated = sk.scalar_min_parallelized(&shifted, max_val);
                // 6. Single-block PBS (clamp LUT) — identity over the in-range value, but resets
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

    fn internal_bits_n(&self, input_bits: &[usize]) -> usize {
        assert_eq!(
            input_bits.len(),
            1,
            "Requant is single-input; got {} input widths",
            input_bits.len()
        );
        // The widest value the rescale materializes (max(x,0)*mult + round_bias) before the
        // shift narrows it. The budget check verifies THIS fits the radix, not just the tiny
        // output (`AGENTS.md` §1.3): a too-large multiplier overflows here even when both the
        // input accumulator and the narrowed output individually fit.
        requant_internal_bits(input_bits[0], self.mult, self.round_bias)
    }
}
