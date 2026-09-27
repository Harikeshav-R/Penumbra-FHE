//! `Argmax` — pick the predicted class from an encrypted logit/score vector.

use penumbra_core::ops::Op;

use super::{CtVec, EvalCtx, WidthAwareOp};
use crate::backend::TfheBackend;
use crate::width::{resize, scalar_blocks, tensor_blocks, value_blocks, NodeWidths};

/// 2-class argmax: threshold a single encrypted logit into an encrypted `0`/`1` label.
pub struct Argmax {
    /// Decision threshold in the quantized accumulator domain. Class `1` iff `z >= threshold`.
    pub threshold: i64,
}

impl WidthAwareOp for Argmax {
    fn eval_with_widths(
        &self,
        ctx: &EvalCtx,
        inputs: &[&CtVec],
        widths: &NodeWidths,
    ) -> Vec<CtVec> {
        let in_cts = inputs[0];
        assert_eq!(
            in_cts.len(),
            1,
            "Phase-2 Argmax handles the 2-class single-logit case; got {} inputs",
            in_cts.len()
        );
        let sk = ctx.sk;
        let nb = ctx.num_blocks;
        let ib = widths.input_bits(0, nb);
        let ob = widths.output_bits(0, nb);

        let c = tensor_blocks(in_cts, ib, nb);
        let w = c.max(scalar_blocks(self.threshold, nb));
        let resized = resize(sk, &in_cts[0], w);
        let ge = sk.scalar_ge_parallelized(resized.as_ref(), self.threshold);
        let out = ge.into_radix(value_blocks(ob, nb), sk);
        vec![vec![out]]
    }
}

impl Op<TfheBackend> for Argmax {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        super::single(self.eval_with_widths(ctx, &[inputs], &NodeWidths::Uniform))
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        1
    }

    fn cost(&self, _input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        vec![("cmp_pbs_ops", 1)]
    }
}
