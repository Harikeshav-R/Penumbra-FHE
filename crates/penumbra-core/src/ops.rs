//! Op trait definition (Layer 2).
//!
//! The stable op-eval interface (`PROJECT.md` §4, ROADMAP Phase 2).
//!
//! Every op takes encrypted inputs + the evaluation context and returns encrypted outputs.
//! No plaintext data ever flows through `eval` — that is the privacy boundary.

use crate::backend::{Backend, CtVec, EvalCtx};

/// Op interface for evaluating a node under a specific backend.
pub trait Op<B: Backend + ?Sized>: Send + Sync {
    /// Evaluate this op over a single input tensor, returning the encrypted outputs.
    ///
    /// This is the common case (every op except `Add`). Multi-input ops implement
    /// [`Op::eval_n`] instead; the default `eval` for those is unreachable.
    fn eval(&self, ctx: &EvalCtx<B::ServerKey>, inputs: &CtVec<B>) -> CtVec<B>;

    /// Declare how this op grows the bit-width budget (`PROJECT.md` §9).
    ///
    /// Given the bit-width of its (single) input, return the bit-width of its outputs.
    fn output_bits(&self, input_bits: usize) -> usize;

    /// Evaluate over an ordered slice of input tensors — the multi-input generalization of
    /// [`Op::eval`] that the graph walker dispatches.
    fn eval_n(&self, ctx: &EvalCtx<B::ServerKey>, inputs: &[&CtVec<B>]) -> CtVec<B> {
        assert_eq!(
            inputs.len(),
            1,
            "this op is single-input; override eval_n for a multi-input op"
        );
        self.eval(ctx, inputs[0])
    }

    /// Bit-width growth for the multi-input case — the companion to [`Op::eval_n`].
    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        assert_eq!(
            input_bits.len(),
            1,
            "this op is single-input; override output_bits_n for a multi-input op"
        );
        self.output_bits(input_bits[0])
    }

    /// Peak internal bit-width this op materializes while computing, given its input widths.
    fn internal_bits_n(&self, input_bits: &[usize]) -> usize {
        self.output_bits_n(input_bits)
    }

    /// Declare this op's scheme-specific cost, given its input tensor lengths.
    ///
    /// Analytic, not instrumented: a backend returns `(counter_name, count)` pairs it can
    /// derive from its own prepared state (`docs/BACKENDS.md`, "Cost models"). Layer 2 never
    /// interprets the names. The default is "this backend declares no cost proxy".
    fn cost(&self, _input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        Vec::new()
    }
}

/// Summary interface for bit-width tracking and conformance checking.
pub trait OpSummary: Send + Sync {
    fn output_bits(&self, input_bits: usize) -> usize;
    fn output_bits_n(&self, input_bits: &[usize]) -> usize;
    fn internal_bits_n(&self, input_bits: &[usize]) -> usize;
}
