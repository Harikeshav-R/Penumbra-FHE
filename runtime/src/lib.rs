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
//! This crate is currently a scaffold (ROADMAP.md Phase 0). Modules are stubs to be
//! filled in by later phases; the `hello_fhe` round-trip in `tests/` proves the
//! `tfhe-rs` toolchain works.

pub mod encrypt;
pub mod eval;
pub mod ir;
pub mod keys;
pub mod ops;
