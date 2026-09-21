//! `Add` — element-wise ciphertext addition of two tensors (residuals / skip connections).

use penumbra_core::ops::Op;

use super::{CtVec, EvalCtx};
use crate::backend::TfheBackend;

/// Element-wise addition of two encrypted tensors.
pub struct Add;

impl Op<TfheBackend> for Add {
    fn eval(&self, _ctx: &EvalCtx, _inputs: &CtVec) -> CtVec {
        panic!("Add is a multi-input op; call eval_n, not eval");
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        panic!("Add is a multi-input op; call output_bits_n, not output_bits");
    }

    fn eval_n(&self, ctx: &EvalCtx, inputs: &[&CtVec]) -> CtVec {
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

        let sk = ctx.sk;
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| sk.add_parallelized(a, b))
            .collect()
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
}
