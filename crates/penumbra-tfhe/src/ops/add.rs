//! `Add` — element-wise ciphertext addition of two tensors (residuals / skip connections).

use super::{CtVec, EvalCtx, WidthAwareOp};
use crate::backend::TfheBackend;
use crate::width::{resize_tensor, value_blocks, NodeWidths};
use penumbra_core::ops::Op;
use rayon::prelude::*;

/// Element-wise addition of two encrypted tensors.
pub struct Add;

impl WidthAwareOp for Add {
    fn eval_with_widths(
        &self,
        ctx: &EvalCtx,
        inputs: &[&CtVec],
        widths: &NodeWidths,
    ) -> Vec<CtVec> {
        assert_eq!(
            inputs.len(),
            2,
            "Add takes exactly two input tensors; got {}",
            inputs.len()
        );
        let (lhs, rhs) = (inputs[0], inputs[1]);
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "Add operands must have equal length: {} vs {}",
            lhs.len(),
            rhs.len()
        );

        let nb = ctx.num_blocks;
        let ob = widths.output_bits(0, nb);
        let o = value_blocks(ob, nb);
        let sk = ctx.sk;

        let lhs_resized = resize_tensor(sk, lhs, o);
        let rhs_resized = resize_tensor(sk, rhs, o);

        let out: CtVec = lhs_resized
            .par_iter()
            .zip(rhs_resized.par_iter())
            .map(|(a, b)| sk.add_parallelized(a.as_ref(), b.as_ref()))
            .collect();

        vec![out]
    }
}

impl Op<TfheBackend> for Add {
    fn eval(&self, _ctx: &EvalCtx, _inputs: &CtVec) -> CtVec {
        panic!("Add is a multi-input op; call eval_n, not eval");
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        panic!("Add is a multi-input op; call output_bits_n, not output_bits");
    }

    fn eval_n(&self, ctx: &EvalCtx, inputs: &[&CtVec]) -> CtVec {
        super::single(self.eval_with_widths(ctx, inputs, &NodeWidths::Uniform))
    }

    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        assert_eq!(
            input_bits.len(),
            2,
            "Add takes exactly two inputs; got {}",
            input_bits.len()
        );
        input_bits[0].max(input_bits[1]) + 1
    }

    fn cost(&self, input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        let n = input_lens.first().copied().unwrap_or(0) as u64;
        let mut counters = Vec::new();
        if n > 0 {
            counters.push(("ct_add", n));
        }
        counters
    }
}
