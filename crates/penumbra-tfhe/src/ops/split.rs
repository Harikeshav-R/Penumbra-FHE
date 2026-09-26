//! `Split` — contiguous segmentation of a flat encrypted tensor into N output tensors.

use penumbra_core::ops::Op;

use super::{CtVec, EvalCtx};
use crate::backend::TfheBackend;

/// Contiguous segmentation of a flat wire into N output tensors.
pub struct Split {
    pub sizes: Vec<usize>,
}

impl Op<TfheBackend> for Split {
    fn eval(&self, _ctx: &EvalCtx, _inputs: &CtVec) -> CtVec {
        panic!("Split is a multi-output op; call eval_multi, not eval");
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        input_bits
    }

    fn eval_multi(&self, _ctx: &EvalCtx, inputs: &[&CtVec]) -> Vec<CtVec> {
        assert_eq!(
            inputs.len(),
            1,
            "Split takes exactly one input tensor; got {}",
            inputs.len()
        );
        let total: usize = self.sizes.iter().sum();
        assert_eq!(
            inputs[0].len(),
            total,
            "Split input length {} does not match sum of declared sizes {}",
            inputs[0].len(),
            total
        );

        let mut results = Vec::with_capacity(self.sizes.len());
        let mut offset = 0;
        for &sz in &self.sizes {
            results.push(inputs[0][offset..offset + sz].to_vec());
            offset += sz;
        }
        results
    }

    fn output_bits_multi(&self, input_bits: &[usize]) -> Vec<usize> {
        assert_eq!(
            input_bits.len(),
            1,
            "Split takes exactly one input; got {}",
            input_bits.len()
        );
        vec![input_bits[0]; self.sizes.len()]
    }

    fn cost(&self, _input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        Vec::new()
    }
}
