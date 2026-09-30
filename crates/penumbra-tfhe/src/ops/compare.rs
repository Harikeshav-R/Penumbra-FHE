//! `Compare` — element-wise threshold comparison with a fused gather.

use penumbra_core::ops::Op;
use rayon::prelude::*;

use super::{CtVec, EvalCtx, WidthAwareOp};
use crate::backend::TfheBackend;
use crate::width::{resize, scalar_blocks, tensor_blocks, value_blocks, NodeWidths};

/// `out[i] = (x[indices[i]] >= thresholds[i]) ? 1 : 0` — one comparison PBS per entry.
pub struct Compare {
    pub indices: Vec<usize>,
    pub thresholds: Vec<i64>,
}

impl WidthAwareOp for Compare {
    fn eval_with_widths(
        &self,
        ctx: &EvalCtx,
        inputs: &[&CtVec],
        widths: &NodeWidths,
    ) -> Vec<CtVec> {
        let in_cts = inputs[0];
        if let Some((i, &idx)) = self
            .indices
            .iter()
            .enumerate()
            .find(|&(_, &idx)| idx >= in_cts.len())
        {
            panic!(
                "Compare indices[{i}] = {idx} is out of range for an input tensor of length {}; \
                 the graph wiring feeding this Compare is wrong",
                in_cts.len()
            );
        }
        let sk = ctx.sk;
        let nb = ctx.num_blocks;
        let ib = widths.input_bits(0, nb);
        let ob = widths.output_bits(0, nb);
        let c = tensor_blocks(in_cts, ib, nb);

        let out: CtVec = self
            .indices
            .par_iter()
            .zip(self.thresholds.par_iter())
            .map(|(&idx, &t)| {
                let w = c.max(scalar_blocks(t, nb));
                let resized = resize(sk, &in_cts[idx], w);
                let ge = sk.scalar_ge_parallelized(resized.as_ref(), t);
                ge.into_radix(value_blocks(ob, nb), sk)
            })
            .collect();

        vec![out]
    }
}

impl Op<TfheBackend> for Compare {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        super::single(self.eval_with_widths(ctx, &[inputs], &NodeWidths::Uniform))
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        1
    }

    fn cost(&self, _input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        vec![("cmp_pbs_ops", self.thresholds.len() as u64)]
    }
}
