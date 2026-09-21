//! `Linear` — matrix-vector product against **plaintext** weights, plus bias.
//!
//! Covers dense layers and logistic/linear regression (`PROJECT.md` §6). This is the
//! *cheap* regime (`PROJECT.md` §5): the data is encrypted but the weights are plaintext,
//! so each output is `sum_i (ciphertext_i * plaintext_weight) + plaintext_bias` —
//! scalar-multiplies and additions only, **no programmable bootstrap**.

use tfhe::integer::SignedRadixCiphertext;

use penumbra_core::ops::Op;

use super::{CtVec, EvalCtx};
use crate::backend::TfheBackend;

/// Dense layer / logistic-regression head with plaintext quantized weights.
pub struct Linear {
    /// Quantized weight matrix, row-major `[n_out][n_in]`.
    pub weights: Vec<Vec<i64>>,
    /// Quantized bias, one per output row.
    pub bias: Vec<i64>,
    /// Quantized weight bit-width (magnitude+sign), used by the bit-width growth rule.
    pub weight_bits: usize,
}

impl Op<TfheBackend> for Linear {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        let sk = ctx.sk;

        assert_eq!(
            self.weights.len(),
            self.bias.len(),
            "Linear must have one bias per weight row: {} rows vs {} biases",
            self.weights.len(),
            self.bias.len()
        );

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

                let mut acc: SignedRadixCiphertext = sk.create_trivial_zero_radix(ctx.num_blocks);
                for (ct, &w) in inputs.iter().zip(row) {
                    let term = sk.scalar_mul_parallelized(ct, w);
                    acc = sk.add_parallelized(&acc, &term);
                }

                sk.scalar_add_parallelized(&acc, b)
            })
            .collect()
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        let n = self.weights.first().map_or(0, Vec::len);
        let sum_growth = if n <= 1 {
            0
        } else {
            usize::BITS as usize - (n - 1).leading_zeros() as usize
        };
        let sum_bits = input_bits + self.weight_bits + sum_growth;

        let max_bias = self
            .bias
            .iter()
            .map(|b| b.unsigned_abs())
            .max()
            .unwrap_or(0);
        let bias_bits = crate::keys::magnitude_bits(max_bias as i64);

        sum_bits.max(bias_bits) + 2
    }
}
