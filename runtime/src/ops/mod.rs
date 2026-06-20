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
//! TODO(phase-2): `linear`, `activation`, `argmax`.
//! TODO(phase-4): `conv2d`, `pool`, `requant`, `add`.
