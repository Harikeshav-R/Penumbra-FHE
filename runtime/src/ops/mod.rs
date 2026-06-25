//! Op implementations (Layer 1) — one module per op, implemented **once** against
//! `tfhe-rs` primitives.
//!
//! The narrow-waist vocabulary (`PROJECT.md` §6) is ~8 ops that cover an enormous range
//! of models:
//!
//! | Op            | Covers                               | TFHE realization              |
//! |---------------|--------------------------------------|-------------------------------|
//! | `Linear`      | dense layers, logistic/linear reg.   | ct × plaintext weights (cheap)|
//! | `Conv2d`      | CNNs                                 | MACs vs plaintext weights     |
//! | `Activation`  | ReLU, sigmoid, GELU, any 1-input fn  | programmable bootstrap (LUT)  |
//! | `Requant`     | rescale wide accumulator → small int | LUT                           |
//! | `Pool`/`Sum`  | avg/max pool, reductions             | adds (+ LUT for max)          |
//! | `Compare`/`Argmax` | classification head, trees      | LUT                           |
//! | `Add`/`Concat`| residuals, skip connections          | adds                          |
//!
//! ## The canonical "add an op" path (`AGENTS.md` §4)
//!
//! 1. registry entry (ONNX → internal op, Python side)
//! 2. Rust implementation here, against `tfhe-rs` primitives
//! 3. bit-width growth rule (`PROJECT.md` §9)
//! 4. golden test: FHE == quantized-cleartext, bit-for-bit
//! 5. docs update (`docs/SUPPORTED-OPS.md`)
//!
//! ## Representation (Phase 2)
//!
//! Every value flows as a [`CtVec`] — a `Vec` of **signed** radix ciphertexts. A `Vec`
//! (rather than a shaped tensor type) is deliberate: it lets the Phase-3 IR walker store
//! named intermediate results in a `HashMap<String, CtVec>` and dispatch each node through
//! the same [`Op::eval`] signature without changing this trait.
//!
//! Phase 2 implements `Linear`, `Activation`, `Argmax`. Phase 4 adds `Conv2d`/`Pool`/
//! `Requant` (all single-input) and `Add` (the first **multi-input** op) against this same
//! trait — see [`Op::eval_n`]/[`Op::output_bits_n`] for how multi-input ops slot in without
//! the eval loop ever special-casing an op.

use tfhe::integer::{ServerKey, SignedRadixCiphertext};

pub mod activation;
pub mod add;
pub mod argmax;
pub mod linear;

pub use activation::Activation;
pub use add::Add;
pub use argmax::Argmax;
pub use linear::Linear;

/// An encrypted tensor flowing between ops. Phase 2 is a flat vector of signed radix
/// integers; shaped tensors (for conv) are a later concern layered on top, not a change
/// to this type's role as the inter-op currency.
pub type CtVec = Vec<SignedRadixCiphertext>;

/// Shared, read-only evaluation context handed to every op.
///
/// Carries the server key (the public evaluation key — the only key the server holds,
/// `PROJECT.md` §11) and `num_blocks`, the central bit-width budget (`keys::keygen`).
pub struct EvalCtx<'a> {
    /// Public evaluation key enabling plaintext-weight arithmetic and bootstrapping.
    pub sk: &'a ServerKey,
    /// Radix width shared by every ciphertext in the model (the bit-width budget).
    pub num_blocks: usize,
}

/// The stable op-eval interface (`PROJECT.md` §4, ROADMAP Phase 2).
///
/// Every op takes encrypted inputs + the server key and returns encrypted outputs. No
/// plaintext data ever flows through `eval` — that is the privacy boundary. The eval loop
/// ([`crate::eval::evaluate`]) dispatches uniformly through this trait, so adding an op
/// never edits the loop, and a new *use case* never edits any op (`AGENTS.md` §1.2).
pub trait Op {
    /// Evaluate this op over a single input tensor, returning the encrypted outputs.
    ///
    /// This is the common case (every op except `Add`). Multi-input ops implement
    /// [`Op::eval_n`] instead; the default `eval` for those is unreachable.
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec;

    /// Declare how this op grows the bit-width budget (`PROJECT.md` §9).
    ///
    /// Given the bit-width of its (single) input, return the bit-width of its outputs. This
    /// is the seed of the Phase-4 central bit-width tracker: implementing it now means the
    /// contract exists before automatic `Requant` insertion makes it load-bearing. The
    /// returned width must never exceed the radix capacity, or the model fails loudly
    /// before evaluation (`AGENTS.md` §1.3, §1.4).
    fn output_bits(&self, input_bits: usize) -> usize;

    /// Evaluate over an ordered slice of input tensors — the multi-input generalization of
    /// [`Op::eval`] that the graph walker ([`crate::eval::evaluate_graph`]) dispatches.
    ///
    /// The default asserts a **single** input and delegates to [`Op::eval`], so every
    /// existing single-input op works unchanged and the eval loop never special-cases an op
    /// (`PROJECT.md` §4, `AGENTS.md` §1.2). A genuinely multi-input op (Phase-4 `Add`)
    /// overrides this; inputs are passed in the node's declared `inputs` order, which *is*
    /// the defined merge order.
    fn eval_n(&self, ctx: &EvalCtx, inputs: &[&CtVec]) -> CtVec {
        assert_eq!(
            inputs.len(),
            1,
            "this op is single-input; override eval_n for a multi-input op"
        );
        self.eval(ctx, inputs[0])
    }

    /// Bit-width growth for the multi-input case — the companion to [`Op::eval_n`].
    ///
    /// The default asserts a single input and delegates to [`Op::output_bits`]. `Add`
    /// overrides it (`max(input widths) + 1`). Kept in lockstep with `eval_n` so the
    /// bit-width tracker ([`crate::eval::propagate_bit_widths`]) and the evaluator agree on
    /// how many inputs an op consumes.
    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        assert_eq!(
            input_bits.len(),
            1,
            "this op is single-input; override output_bits_n for a multi-input op"
        );
        self.output_bits(input_bits[0])
    }
}
