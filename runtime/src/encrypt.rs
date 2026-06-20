//! Encrypt / decrypt helpers.
//!
//! Thin client-side helpers over `tfhe-rs`: turn a quantized integer input vector into
//! ciphertexts (with the client key) and turn the encrypted output back into a
//! prediction. The server never sees plaintext (`PROJECT.md` §11).
//!
//! TODO(phase-1): implement `encrypt(&client_key, &input)` / `decrypt(&client_key, &out)`
//! over the small quantized-integer domain.
