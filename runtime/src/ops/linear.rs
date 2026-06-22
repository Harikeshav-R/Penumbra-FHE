//! `Linear` — matrix-vector product against **plaintext** weights, plus bias.
//!
//! Covers dense layers and logistic/linear regression (`PROJECT.md` §6). This is the
//! *cheap* regime (`PROJECT.md` §5): the data is encrypted but the weights are plaintext,
//! so each output is `sum_i (ciphertext_i * plaintext_weight) + plaintext_bias` —
//! scalar-multiplies and additions only, **no programmable bootstrap**.
//!
//! ## Bit-width growth rule (`PROJECT.md` §9)
//!
//! A dot product over `N` inputs of `b`-bit values against `w`-bit weights produces, per
//! output, an accumulator of about `b + w + ceil(log2(N))` bits (the `+ceil(log2 N)` is
//! the carry from summing `N` terms). The bias adds at most one more bit. This growth is
//! why a later `Requant` (Phase 4) is needed before feeding a narrow LUT — in Phase 2 we
//! simply size the radix (`num_blocks`) to hold the result.

use tfhe::integer::SignedRadixCiphertext;

use super::{CtVec, EvalCtx, Op};

/// Dense layer / logistic-regression head with plaintext quantized weights.
///
/// `weights` is row-major `[out][in]` and `bias` has one entry per output. Both are
/// quantized signed integers in the same integer domain as the encrypted inputs (the
/// quantization service owns choosing them; `PROJECT.md` §8).
pub struct Linear {
    /// Quantized weight matrix, row-major `[n_out][n_in]`.
    pub weights: Vec<Vec<i64>>,
    /// Quantized bias, one per output row.
    pub bias: Vec<i64>,
    /// Quantized weight bit-width (magnitude+sign), used by the bit-width growth rule.
    pub weight_bits: usize,
}

impl Op for Linear {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        let sk = ctx.sk;

        self.weights
            .iter()
            .zip(&self.bias)
            .map(|(row, &b)| {
                assert_eq!(
                    row.len(),
                    inputs.len(),
                    "Linear weight row width ({}) must match input length ({})",
                    row.len(),
                    inputs.len()
                );

                // Accumulate sum_i (input_i * w_i) against plaintext weights — no PBS.
                // Start from a trivial encryption of zero sized to the model's radix.
                let mut acc: SignedRadixCiphertext = sk.create_trivial_zero_radix(ctx.num_blocks);
                for (ct, &w) in inputs.iter().zip(row) {
                    let term = sk.scalar_mul_parallelized(ct, w);
                    acc = sk.add_parallelized(&acc, &term);
                }

                // Plaintext bias add (still cheap, no PBS).
                sk.scalar_add_parallelized(&acc, b)
            })
            .collect()
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        // Accumulator width: per-term product is input_bits + weight_bits; summing N terms
        // adds ceil(log2 N); the bias adds at most one guard bit. (`PROJECT.md` §9.)
        let n = self.weights.first().map_or(0, Vec::len);
        let sum_growth = if n <= 1 {
            0
        } else {
            usize::BITS as usize - (n - 1).leading_zeros() as usize
        };
        input_bits + self.weight_bits + sum_growth + 1
    }
}
