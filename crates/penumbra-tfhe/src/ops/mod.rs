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

/// Evaluates a weighted multiply-accumulate (MAC) over grouped ciphertexts.
///
/// Inputs sharing the same non-zero weight are pre-grouped in `groups`.
/// Each group is summed via `sk.sum_ciphertexts_parallelized` before a single
/// scalar multiplication by `w`, reducing the total number of scalar multiplications.
/// The resulting group terms are then summed via a sum-tree and the bias is added.
pub(crate) fn evaluate_weighted_mac(
    ctx: &EvalCtx,
    groups: std::collections::BTreeMap<i64, Vec<&tfhe::integer::SignedRadixCiphertext>>,
    bias: i64,
) -> tfhe::integer::SignedRadixCiphertext {
    let sk = ctx.sk;
    let mut group_terms: Vec<tfhe::integer::SignedRadixCiphertext> =
        Vec::with_capacity(groups.len());
    for (w, cts) in groups {
        let s = if cts.len() == 1 {
            cts[0].clone()
        } else {
            sk.sum_ciphertexts_parallelized(cts.iter().copied())
                .expect("non-empty cts group")
        };
        let term = if w == 1 {
            s
        } else {
            sk.scalar_mul_parallelized(&s, w)
        };
        group_terms.push(term);
    }

    let acc = if group_terms.is_empty() {
        sk.create_trivial_zero_radix(ctx.num_blocks)
    } else if group_terms.len() == 1 {
        group_terms.pop().unwrap()
    } else {
        sk.sum_ciphertexts_parallelized(group_terms.iter())
            .expect("non-empty group_terms")
    };

    sk.scalar_add_parallelized(&acc, bias)
}
