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
//! the carry from summing `N` terms). The plaintext bias is a *separate* contributor sized
//! from its own magnitude — it can exceed the summed products after rescaling — so the
//! accumulator width is `max(sum_bits, bias_bits) + 2`, not `sum_bits + 1`: adding two
//! magnitude-sized signed quantities (the summed products and the bias) can produce one
//! carry bit, and the result is signed so it needs a sign bit on top — two distinct `+1`s.
//! This growth is why a later `Requant` (Phase 4) is needed before feeding a narrow LUT —
//! in Phase 2 we simply size the radix (`num_blocks`) to hold the result.

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

        // Fail loudly on a malformed layer: `zip` would otherwise silently truncate to the
        // shorter of the two, dropping outputs or biases without any error (`AGENTS.md` §1.4).
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
        // Accumulator width has two contributors that we must take the *max* of, not assume
        // the dot-product dominates (`PROJECT.md` §9):
        //   - the summed products: per-term width is input_bits + weight_bits, and summing
        //     N terms adds ceil(log2 N) of carry;
        //   - the plaintext bias: a quantized bias can be larger than the summed products
        //     (especially after rescaling), so it must be sized from its actual magnitude —
        //     assuming "+1 guard bit" under-counts and would let the budget check miss a
        //     real overflow (`AGENTS.md` §1.3).
        let n = self.weights.first().map_or(0, Vec::len);
        let sum_growth = if n <= 1 {
            0
        } else {
            usize::BITS as usize - (n - 1).leading_zeros() as usize
        };
        let sum_bits = input_bits + self.weight_bits + sum_growth;

        // A zero bias contributes 0 magnitude bits, which is correct here — it adds nothing
        // to the accumulator, so we want it to vanish from the `max` (the `.max(1)` clamp
        // used for LUT entries would be wrong in this arithmetic context).
        let max_bias = self
            .bias
            .iter()
            .map(|b| b.unsigned_abs())
            .max()
            .unwrap_or(0);
        let bias_bits = crate::keys::magnitude_bits(max_bias);

        // The accumulator is `sum_of_products + bias`, a *signed* quantity. `sum_bits` and
        // `bias_bits` are magnitude widths; adding two magnitude-sized values can produce a
        // carry into one extra bit (`+1`), and a signed radix needs a dedicated sign bit on
        // top (`+1`). These are two independent guard bits — collapsing them into a single
        // `+1` under-counts by exactly one bit whenever `sum_bits` and `bias_bits` are
        // comparable, letting a model pass the budget check yet silently wrap the signed
        // radix at eval time (a §1.1/§1.3 violation). Hence `+ 2`.
        sum_bits.max(bias_bits) + 2
    }
}
