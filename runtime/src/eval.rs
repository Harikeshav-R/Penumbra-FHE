//! The graph walker / evaluation loop.
//!
//! Evaluates a model under FHE by dispatching each op to its implementation in
//! [`crate::ops`]. This loop is written **once** and never changes per use case
//! (`PROJECT.md` §4, §7): adding an op never edits the loop, and a new use case never
//! edits any op (`AGENTS.md` §1.2).
//!
//! Runtime ≈ number of bootstraps (`PROJECT.md` §5): linear ops with plaintext weights are
//! cheap; activations/requant are programmable bootstraps and dominate cost.
//!
//! Phase 2 walks a flat `Vec` of ops in order (a linear chain: `Linear → Argmax`, with
//! `Activation` exercised as a standalone op in the golden test). Phase 3 swaps the
//! hardcoded `Vec` for an IR graph deserialized from a file; Phase 8 generalizes the walk
//! to true topological ordering with intermediate-result storage for branching graphs.
//! Neither changes this loop's body — that is the point of the [`Op`] trait.

use crate::ops::{CtVec, EvalCtx, Op};

/// Evaluate a linear chain of ops over an encrypted input, threading each op's output into
/// the next. Returns the final encrypted output (e.g. a one-element class-index vector).
pub fn evaluate(ctx: &EvalCtx, ops: &[Box<dyn Op>], input: &CtVec) -> CtVec {
    // We clone the input once so the borrowed `&CtVec` API stays ergonomic for callers.
    // Threading ownership through the loop to elide this clone is a possible optimization,
    // but it's negligible against PBS cost (runtime ≈ number of bootstraps) and not worth an
    // API change here — perf tuning is deferred to Phase 10 (`PROJECT.md` §10).
    let mut acc = input.clone();
    for op in ops {
        acc = op.eval(ctx, &acc);
    }
    acc
}

/// Verify a model's declared bit-width budget fits the radix capacity, failing **loudly**
/// before evaluation if not (`AGENTS.md` §1.3, §1.4).
///
/// Walks the op chain propagating bit-widths from `input_bits`; if any op's output width
/// exceeds what `num_blocks` radix blocks can hold, returns an `Err` naming the offending
/// layer index and the required-vs-available bits. Automatic `Requant` insertion that
/// would *prevent* such overflow is Phase 4; Phase 2 at least refuses to run silently.
pub fn check_bit_width_budget(
    ops: &[Box<dyn Op>],
    input_bits: usize,
    num_blocks: usize,
) -> Result<(), String> {
    let capacity = crate::keys::radix_capacity_bits(num_blocks);
    let mut bits = input_bits;
    for (i, op) in ops.iter().enumerate() {
        let out = op.output_bits(bits);
        if out > capacity {
            return Err(format!(
                "bit-width budget exceeded at op #{i}: requires {out} bits but the radix \
                 holds only {capacity} ({num_blocks} blocks × {} bits). Reduce precision or \
                 widen num_blocks (a Requant here is Phase 4).",
                crate::keys::MESSAGE_BITS
            ));
        }
        bits = out;
    }
    Ok(())
}
