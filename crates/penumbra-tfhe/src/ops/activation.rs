//! `Activation` — apply a single-input function via a lookup table (LUT).

use tfhe::integer::{IntegerCiphertext, SignedRadixCiphertext};
use tfhe::shortint::Ciphertext;

use penumbra_core::ops::Op;
use rayon::prelude::*;

use super::{CtVec, EvalCtx};
use crate::backend::TfheBackend;
use crate::keys::MESSAGE_BITS;

/// Minimum bits needed to represent every entry of a LUT (its true output width).
fn lut_output_bits(lut: &[u64]) -> usize {
    let max = lut.iter().copied().max().unwrap_or(0);
    crate::keys::magnitude_bits(max as i64).max(1)
}

/// Single-input activation realized as a LUT over a narrow integer domain.
pub struct Activation {
    /// Lookup table indexed by the input integer value.
    pub lut: Vec<u64>,
    /// Bit-width of the table's output values (its bit-width growth declaration).
    pub output_bits: usize,
}

impl Op<TfheBackend> for Activation {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        let sk = ctx.sk;
        let shortint_sk = sk.as_ref();

        let domain = 1usize << MESSAGE_BITS;
        assert_eq!(
            self.lut.len(),
            domain,
            "Activation LUT must cover the full {MESSAGE_BITS}-bit message space \
             ({domain} entries); got {}",
            self.lut.len()
        );

        if let Some((idx, &bad)) = self
            .lut
            .iter()
            .enumerate()
            .find(|&(_, &e)| e >= (1u64 << MESSAGE_BITS))
        {
            panic!(
                "Activation LUT entry lut[{idx}] = {bad} does not fit one shortint block: \
                 every output must be < {} (the {MESSAGE_BITS}-bit message space). \
                 apply_lookup_table reduces modulo the message modulus, so a larger value \
                 would silently wrap.",
                1u64 << MESSAGE_BITS
            );
        }

        assert!(
            ctx.num_blocks > 1,
            "Phase-2 Activation requires num_blocks > 1 so the LUT output lands in a value \
             block, not the radix sign block (got num_blocks = {})",
            ctx.num_blocks
        );

        let table = self.lut.clone();
        let lut = shortint_sk.generate_lookup_table(move |v| *table.get(v as usize).unwrap_or(&0));

        inputs
            .par_iter()
            .map(|ct| {
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

    fn output_bits(&self, input_bits: usize) -> usize {
        assert!(
            input_bits <= MESSAGE_BITS,
            "Activation input is wider ({input_bits} bits) than the single {MESSAGE_BITS}-bit \
             block it consumes; Phase-2 activations need a Requant (Phase 4) in front to narrow \
             the value first"
        );

        let derived = lut_output_bits(&self.lut);
        assert!(
            self.output_bits >= derived,
            "Activation output_bits ({}) is smaller than the LUT's actual range ({} bits); \
             the declared width must not under-count the table",
            self.output_bits,
            derived
        );

        assert!(
            self.output_bits <= MESSAGE_BITS,
            "Activation output_bits ({}) exceeds MESSAGE_BITS ({MESSAGE_BITS}); the eval writes \
             a single shortint block, which cannot hold a wider value",
            self.output_bits
        );
        self.output_bits
    }

    fn cost(&self, input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        let n = input_lens.first().copied().unwrap_or(0) as u64;
        let mut counters = Vec::new();
        if n > 0 {
            counters.push(("bootstraps", n));
        }
        counters
    }
}
