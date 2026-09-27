//! `Concat` — channel-axis concatenation of N tensors on the flat wire.

use penumbra_core::ops::Op;

use super::{CtVec, EvalCtx};
use crate::backend::TfheBackend;

/// Channel-axis concatenation of multiple encrypted tensors.
pub struct Concat {
    pub sizes: Vec<usize>,
}

impl Op<TfheBackend> for Concat {
    fn eval(&self, _ctx: &EvalCtx, _inputs: &CtVec) -> CtVec {
        panic!("Concat is a multi-input op; call eval_n, not eval");
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        panic!("Concat is a multi-input op; call output_bits_n, not output_bits");
    }

    fn eval_n(&self, _ctx: &EvalCtx, inputs: &[&CtVec]) -> CtVec {
        assert_eq!(
            inputs.len(),
            self.sizes.len(),
            "Concat expects {} input tensors; got {}",
            self.sizes.len(),
            inputs.len()
        );
        for (i, (&input, &expected_size)) in inputs.iter().zip(&self.sizes).enumerate() {
            assert_eq!(
                input.len(),
                expected_size,
                "Concat segment {} expects length {}; got {}",
                i,
                expected_size,
                input.len()
            );
        }
        inputs.iter().flat_map(|t| t.iter().cloned()).collect()
    }

    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        assert_eq!(
            input_bits.len(),
            self.sizes.len(),
            "Concat takes one input per declared segment; got {}",
            input_bits.len()
        );
        input_bits.iter().copied().max().unwrap_or(0)
    }

    fn cost(&self, _input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        Vec::new()
    }
}
