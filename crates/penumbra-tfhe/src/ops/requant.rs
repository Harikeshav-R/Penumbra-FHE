//! `Requant` — rescale a wide accumulator down to a narrow, LUT-able value.

use tfhe::integer::{IntegerCiphertext, SignedRadixCiphertext};
use tfhe::shortint::Ciphertext;

use penumbra_core::ops::Op;

use super::{CtVec, EvalCtx};
use crate::backend::TfheBackend;
use crate::keys::MESSAGE_BITS;

/// Minimum bits needed to represent every entry of a LUT (its true output width).
fn lut_output_bits(lut: &[u64]) -> usize {
    let max = lut.iter().copied().max().unwrap_or(0);
    crate::keys::magnitude_bits(max as i64).max(1)
}

/// Peak internal bit-width the rescale needs before the shift narrows it.
pub fn requant_internal_bits(input_bits: usize, mult: u64, round_bias: u64) -> usize {
    let relu_max: u128 = if input_bits >= 1 {
        (1u128 << (input_bits - 1)) - 1
    } else {
        0
    };
    let intermediate_max = relu_max * mult as u128 + round_bias as u128;
    let magnitude = (u128::BITS - intermediate_max.leading_zeros()) as usize;
    (magnitude + 1).max(input_bits)
}

/// Rescale a wide accumulator to a narrow non-negative value via ReLU + fixed-point multiply
/// + round + shift + clamp.
pub struct Requant {
    pub shift: u32,
    pub mult: u64,
    pub round_bias: u64,
    pub out_bits: usize,
    pub clamp_lut: Vec<u64>,
    pub mults: Vec<u64>,
    pub shifts: Vec<u32>,
    pub round_biases: Vec<u64>,
    pub channel_size: usize,
}

impl Op<TfheBackend> for Requant {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        let sk = ctx.sk;
        let shortint_sk = sk.as_ref();

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
        assert!(
            ctx.num_blocks > 1,
            "Requant requires num_blocks > 1 so the narrowed value lands in a value block, not \
             the radix sign block (got num_blocks = {})",
            ctx.num_blocks
        );

        let max_val = (1i64 << self.out_bits) - 1;

        let per_channel = !self.mults.is_empty();
        if per_channel {
            assert!(
                self.channel_size >= 1,
                "Requant per-channel channel_size must be >= 1"
            );
            assert_eq!(
                inputs.len() % self.channel_size,
                0,
                "Requant per-channel: {} elements not divisible by channel_size {}",
                inputs.len(),
                self.channel_size
            );
            assert_eq!(
                inputs.len() / self.channel_size,
                self.mults.len(),
                "Requant per-channel: {} channels (len {} / channel_size {}) != {} multipliers",
                inputs.len() / self.channel_size,
                inputs.len(),
                self.channel_size,
                self.mults.len()
            );
        }

        let table = self.clamp_lut.clone();
        let lut = shortint_sk.generate_lookup_table(move |v| *table.get(v as usize).unwrap_or(&0));

        inputs
            .iter()
            .enumerate()
            .map(|(idx, ct)| {
                let (mult, shift, round_bias) = if per_channel {
                    let ch = idx / self.channel_size;
                    (
                        self.mults[ch],
                        self.shifts[ch],
                        self.round_biases[ch] as i64,
                    )
                } else {
                    (self.mult, self.shift, self.round_bias as i64)
                };
                let nonneg = sk.scalar_max_parallelized(ct, 0i64);
                let scaled = if mult == 1 {
                    nonneg
                } else {
                    sk.scalar_mul_parallelized(&nonneg, mult as i64)
                };
                let biased = if round_bias == 0 {
                    scaled
                } else {
                    sk.scalar_add_parallelized(&scaled, round_bias)
                };
                let shifted: SignedRadixCiphertext =
                    sk.scalar_right_shift_parallelized(&biased, shift);
                let saturated = sk.scalar_min_parallelized(&shifted, max_val);
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
        if self.mults.is_empty() {
            requant_internal_bits(input_bits[0], self.mult, self.round_bias)
        } else {
            self.mults
                .iter()
                .zip(&self.round_biases)
                .map(|(&m, &rb)| requant_internal_bits(input_bits[0], m, rb))
                .max()
                .expect("per-channel Requant has at least one channel")
        }
    }
}
