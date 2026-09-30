//! `Split` — contiguous segmentation of a flat encrypted tensor into N output tensors.

use penumbra_core::ops::Op;

use super::{CtVec, EvalCtx, WidthAwareOp};
use crate::backend::TfheBackend;
use crate::width::{resize_tensor, tensor_blocks, NodeWidths};

/// Contiguous segmentation of a flat wire into N output tensors.
pub struct Split {
    pub sizes: Vec<usize>,
}

impl WidthAwareOp for Split {
    fn eval_with_widths(
        &self,
        ctx: &EvalCtx,
        inputs: &[&CtVec],
        widths: &NodeWidths,
    ) -> Vec<CtVec> {
        assert_eq!(
            inputs.len(),
            1,
            "Split takes exactly one input tensor; got {}",
            inputs.len()
        );
        let in_cts = inputs[0];
        let total: usize = self.sizes.iter().sum();
        assert_eq!(
            in_cts.len(),
            total,
            "Split input length {} does not match sum of declared sizes {}",
            in_cts.len(),
            total
        );

        let nb = ctx.num_blocks;
        let ib = widths.input_bits(0, nb);
        let c = tensor_blocks(in_cts, ib, nb);
        let sk = ctx.sk;

        let resized = resize_tensor(sk, in_cts, c);

        let mut results = Vec::with_capacity(self.sizes.len());
        let mut offset = 0;
        for &sz in &self.sizes {
            let seg: CtVec = resized[offset..offset + sz]
                .iter()
                .map(|item| item.clone().into_owned())
                .collect();
            results.push(seg);
            offset += sz;
        }
        results
    }
}

impl Op<TfheBackend> for Split {
    fn eval(&self, _ctx: &EvalCtx, _inputs: &CtVec) -> CtVec {
        panic!("Split is a multi-output op; call eval_multi, not eval");
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        input_bits
    }

    fn eval_multi(&self, ctx: &EvalCtx, inputs: &[&CtVec]) -> Vec<CtVec> {
        self.eval_with_widths(ctx, inputs, &NodeWidths::Uniform)
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
