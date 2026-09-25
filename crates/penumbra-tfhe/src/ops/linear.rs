//! `Linear` — matrix-vector product against **plaintext** weights, plus bias.
//!
//! Covers dense layers and logistic/linear regression (`PROJECT.md` §6). This is the
//! *cheap* regime (`PROJECT.md` §5): the data is encrypted but the weights are plaintext,
//! so each output is `sum_i (ciphertext_i * plaintext_weight) + plaintext_bias` —
//! scalar-multiplies and additions only, **no programmable bootstrap**.

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;
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
        assert_eq!(
            self.weights.len(),
            self.bias.len(),
            "Linear must have one bias per weight row: {} rows vs {} biases",
            self.weights.len(),
            self.bias.len()
        );

        self.weights
            .par_iter()
            .zip(self.bias.par_iter())
            .map(|(row, &b)| {
                assert_eq!(
                    row.len(),
                    inputs.len(),
                    "Linear weight row width ({}) must match input length ({})",
                    row.len(),
                    inputs.len()
                );

                // Group inputs by non-zero weight into a BTreeMap for deterministic order.
                let mut groups: BTreeMap<i64, Vec<&SignedRadixCiphertext>> = BTreeMap::new();
                for (ct, &w) in inputs.iter().zip(row) {
                    if w == 0 {
                        continue;
                    }
                    groups.entry(w).or_default().push(ct);
                }

                super::evaluate_weighted_mac(ctx, groups, b)
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

    fn cost(&self, _input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        let mut scalar_mul = 0u64;
        let mut ct_add = 0u64;
        let rows = self.weights.len() as u64;

        for row in &self.weights {
            let mut distinct = BTreeSet::new();
            let mut nonzero_count = 0u64;
            for &w in row {
                if w != 0 {
                    distinct.insert(w);
                    nonzero_count += 1;
                }
            }
            scalar_mul += distinct.len() as u64;
            if nonzero_count > 0 {
                ct_add += nonzero_count - 1;
            }
        }

        let mut counters = Vec::new();
        if scalar_mul > 0 {
            counters.push(("scalar_mul", scalar_mul));
        }
        if ct_add > 0 {
            counters.push(("ct_add", ct_add));
        }
        if rows > 0 {
            counters.push(("scalar_add", rows));
        }
        counters
    }
}
