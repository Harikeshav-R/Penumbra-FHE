//! # Penumbra-FHE Runtime (Layers 1 + 2)
//!
//! The Rust TFHE backend for Penumbra-FHE. This crate is the *stable* core of the
//! project's "narrow waist" architecture (see `PROJECT.md` §4):
//!
//! - **Layer 1 — TFHE backend:** each ML op implemented once against `tfhe-rs`
//!   primitives ([`ops`]), plus key management ([`keys`]) and encrypt/decrypt
//!   helpers ([`encrypt`]).
//! - **Layer 2 — IR + eval:** deserialize the Intermediate Representation ([`ir`])
//!   emitted by the Python front end and walk the op graph ([`eval`]).
//!
//! ## Invariant (non-negotiable)
//!
//! TFHE is *exact*. The encrypted forward pass must equal the quantized-cleartext
//! forward pass **bit-for-bit**. Any discrepancy is a quantization or implementation
//! bug, never crypto noise. See `AGENTS.md` §1 and `ROADMAP.md`'s golden invariant.
//!
//! ## Discipline
//!
//! A new *use case* never edits this crate — it only adds a Python-side graph/adapter.
//! Backend edits are reserved for genuinely new primitive ops (`AGENTS.md` §1.2).
//!
//! The minimal narrow waist — `Linear`, `Activation(LUT)`, `Argmax` — runs end to end
//! (keygen → encrypt → evaluate → decrypt), gated by the golden exactness test in
//! `tests/golden_logreg.rs`. As of Phase 3 the model is no longer hardcoded: it is a
//! serializable IR graph ([`ir`]) that Python emits and [`eval::evaluate_graph`] walks.

pub mod encrypt;
pub mod eval;
pub mod ir;
pub mod keys;
pub mod ops;

// Public API surface (`PROJECT.md` §12): keys, the client-side encrypt/decrypt boundary,
// the op-eval interface, the serializable IR, and the graph walker.
pub use encrypt::{decrypt_label, decrypt_vec, encrypt};
pub use eval::{
    check_bit_width_budget, check_graph_bit_width_budget, evaluate, evaluate_graph,
    propagate_bit_widths,
};
pub use ir::{Graph, Node, OpSpec, SCHEMA_VERSION};
pub use keys::keygen;
pub use ops::{Activation, Add, Argmax, CtVec, EvalCtx, Linear, Op, Requant};
