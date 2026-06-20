//! Key generation and crypto-parameter profiles.
//!
//! The client holds the secret key (encrypt/decrypt); the server holds the public
//! evaluation/server key (enables bootstrapping) plus the plaintext model weights.
//! See `PROJECT.md` §11.
//!
//! Policy (`PROJECT.md` §12, `AGENTS.md` §7): ship a single secure default parameter
//! profile and expose at most **one** override knob. Never surface raw `tfhe-rs`
//! parameters to end users.
//!
//! TODO(phase-1): implement `keygen()` over the `tfhe-rs` default secure parameter
//! profile and expose `(client_key, server_key)`.
