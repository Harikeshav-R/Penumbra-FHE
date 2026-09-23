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
pub mod conv2d;
pub mod linear;
pub mod pool;
pub mod requant;

pub use activation::Activation;
pub use add::Add;
pub use argmax::Argmax;
pub use conv2d::Conv2d;
pub use linear::Linear;
pub use pool::{Pool, PoolMode};
pub use requant::Requant;

/// Evaluation context specialized for the TFHE backend.
pub type EvalCtx<'a> = penumbra_core::backend::EvalCtx<'a, tfhe::integer::ServerKey>;
