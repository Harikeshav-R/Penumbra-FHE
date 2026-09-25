//! `Compare` — element-wise threshold comparison with a fused gather.

use penumbra_core::ops::Op;
use rayon::prelude::*;

use super::{CtVec, EvalCtx};
use crate::backend::TfheBackend;

/// `out[i] = (x[indices[i]] >= thresholds[i]) ? 1 : 0` — one comparison PBS per entry.
pub struct Compare {
    pub indices: Vec<usize>,
    pub thresholds: Vec<i64>,
}

impl Op<TfheBackend> for Compare {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        if let Some((i, &idx)) = self
            .indices
            .iter()
            .enumerate()
            .find(|&(_, &idx)| idx >= inputs.len())
        {
            panic!(
                "Compare indices[{i}] = {idx} is out of range for an input tensor of length {}; \
                 the graph wiring feeding this Compare is wrong",
                inputs.len()
            );
        }
        let sk = ctx.sk;
        self.indices
            .par_iter()
            .zip(self.thresholds.par_iter())
            .map(|(&idx, &t)| {
                let ge = sk.scalar_ge_parallelized(&inputs[idx], t);
                ge.into_radix(ctx.num_blocks, sk)
            })
            .collect()
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        1
    }

    fn cost(&self, _input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        vec![("cmp_pbs_ops", self.thresholds.len() as u64)]
    }
}
