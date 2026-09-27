//! Op implementations (Layer 1) for the TFHE backend.
//!
//! Each ML op implemented against `tfhe-rs` primitives and conforming to the
//! backend-neutral [`penumbra_core::ops::Op`] contract.

/// The stable op-eval interface specialized for the TFHE backend.
pub trait Op: penumbra_core::ops::Op<crate::backend::TfheBackend> + Send + Sync {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        penumbra_core::ops::Op::eval(self, ctx, inputs)
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        penumbra_core::ops::Op::output_bits(self, input_bits)
    }

    fn eval_n(&self, ctx: &EvalCtx, inputs: &[&CtVec]) -> CtVec {
        penumbra_core::ops::Op::eval_n(self, ctx, inputs)
    }

    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        penumbra_core::ops::Op::output_bits_n(self, input_bits)
    }

    fn internal_bits_n(&self, input_bits: &[usize]) -> usize {
        penumbra_core::ops::Op::internal_bits_n(self, input_bits)
    }
}
impl<T: penumbra_core::ops::Op<crate::backend::TfheBackend> + Send + Sync + ?Sized> Op for T {}

use crate::encrypt::CtVec;

pub mod activation;
pub mod add;
pub mod argmax;
pub mod compare;
pub mod concat;
pub mod conv2d;
pub mod linear;
pub(crate) mod mac;
pub mod pool;
pub mod requant;
pub mod split;

pub use activation::Activation;
pub use add::Add;
pub use argmax::Argmax;
pub use compare::Compare;
pub use concat::Concat;
pub use conv2d::Conv2d;
pub use linear::Linear;
pub use pool::{Pool, PoolMode};
pub use requant::Requant;
pub use split::Split;

/// Evaluation context specialized for the TFHE backend.
pub type EvalCtx<'a> = penumbra_core::backend::EvalCtx<'a, tfhe::integer::ServerKey>;
use crate::backend::TfheBackend;
use crate::width::NodeWidths;
use penumbra_core::ops::Op as CoreOp;

pub(crate) trait WidthAwareOp: CoreOp<TfheBackend> + Send + Sync {
    fn eval_with_widths(&self, ctx: &EvalCtx, inputs: &[&CtVec], widths: &NodeWidths)
        -> Vec<CtVec>;
}

pub(crate) struct WithWidths<O> {
    pub(crate) op: O,
    pub(crate) widths: NodeWidths,
}

pub(crate) fn single(mut v: Vec<CtVec>) -> CtVec {
    assert_eq!(v.len(), 1, "expected 1 output tensor, got {}", v.len());
    v.pop().unwrap()
}

impl<O: WidthAwareOp> CoreOp<TfheBackend> for WithWidths<O> {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        single(self.op.eval_with_widths(ctx, &[inputs], &self.widths))
    }

    fn eval_n(&self, ctx: &EvalCtx, inputs: &[&CtVec]) -> CtVec {
        single(self.op.eval_with_widths(ctx, inputs, &self.widths))
    }

    fn eval_multi(&self, ctx: &EvalCtx, inputs: &[&CtVec]) -> Vec<CtVec> {
        self.op.eval_with_widths(ctx, inputs, &self.widths)
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        self.op.output_bits(input_bits)
    }

    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        self.op.output_bits_n(input_bits)
    }

    fn output_bits_multi(&self, input_bits: &[usize]) -> Vec<usize> {
        self.op.output_bits_multi(input_bits)
    }

    fn internal_bits_n(&self, input_bits: &[usize]) -> usize {
        self.op.internal_bits_n(input_bits)
    }

    fn cost(&self, input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        self.op.cost(input_lens)
    }
}
