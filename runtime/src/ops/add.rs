//! `Add` — element-wise ciphertext addition of two tensors (residuals / skip connections).
//!
//! Covers residual and skip connections (`PROJECT.md` §6). This is the *cheap* regime
//! (`PROJECT.md` §5): adding two ciphertexts is a plain homomorphic add, **no programmable
//! bootstrap**. `Add` is the project's first **multi-input** op — it takes two input tensors
//! of equal length and adds them position-wise.
//!
//! ## Bit-width growth rule (`PROJECT.md` §9)
//!
//! Adding two signed values of `a` and `b` magnitude bits yields at most
//! `max(a, b) + 1` bits — a single carry. The result stays signed (both inputs are), so no
//! extra sign bit is needed beyond what the wider input already carries.
//!
//! ## Multi-input dispatch
//!
//! `Add` overrides [`Op::eval_n`]/[`Op::output_bits_n`] (the two-input path) rather than the
//! single-input [`Op::eval`]/[`Op::output_bits`]. Inputs arrive in the node's declared
//! `inputs` order; addition is commutative so the order does not affect the result, but the
//! contract (declared order = merge order) is the same one every multi-input op follows.

use super::{CtVec, EvalCtx, Op};

/// Element-wise addition of two encrypted tensors.
pub struct Add;

impl Op for Add {
    fn eval(&self, _ctx: &EvalCtx, _inputs: &CtVec) -> CtVec {
        // `Add` is multi-input; the graph walker calls `eval_n`. The single-input `eval` is
        // unreachable for this op, but the trait requires it — fail loudly if ever reached.
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
        // Summing two signed magnitudes adds at most one carry bit; the wider operand's sign
        // bit covers the result's sign, so no second guard bit is needed (unlike `Linear`,
        // which adds two *separately sized* contributors — products and bias).
        input_bits[0].max(input_bits[1]) + 1
    }
}
