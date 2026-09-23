//! `Argmax` — pick the predicted class from an encrypted logit/score vector.

use penumbra_core::ops::Op;

use super::{CtVec, EvalCtx};
use crate::backend::TfheBackend;

/// 2-class argmax: threshold a single encrypted logit into an encrypted `0`/`1` label.
pub struct Argmax {
    /// Decision threshold in the quantized accumulator domain. Class `1` iff `z >= threshold`.
    pub threshold: i64,
}

impl Op<TfheBackend> for Argmax {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        assert_eq!(
            inputs.len(),
            1,
            "Phase-2 Argmax handles the 2-class single-logit case; got {} inputs",
            inputs.len()
        );
        let sk = ctx.sk;

        let ge = sk.scalar_ge_parallelized(&inputs[0], self.threshold);
        vec![ge.into_radix(ctx.num_blocks, sk)]
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        1
    }

    fn cost(&self, _input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        vec![("cmp_pbs_ops", 1)]
    }
}
