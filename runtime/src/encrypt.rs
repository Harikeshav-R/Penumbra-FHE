//! Encrypt / decrypt helpers.
//!
//! Thin client-side helpers over `tfhe-rs`: turn a quantized-integer input vector into
//! ciphertexts (with the client key) and turn the encrypted output back into a prediction.
//! The server never sees plaintext (`PROJECT.md` §11).
//!
//! Penumbra carries every value as a **signed** radix integer ([`crate::ops::CtVec`]).
//! Quantized activations are small signed integers; weights and the resulting logit are
//! naturally signed, so a signed representation avoids a zero-point-offset dance and
//! generalizes cleanly to the conv accumulators of later phases.

use tfhe::integer::RadixClientKey;

use crate::ops::CtVec;

/// Encrypt a quantized-integer input vector into a [`CtVec`] of signed radix ciphertexts.
///
/// Each element becomes one radix integer sized by the client key's `num_blocks`. This is
/// the client-side boundary: plaintext goes in here and only ciphertext leaves.
pub fn encrypt(ck: &RadixClientKey, input: &[i64]) -> CtVec {
    input.iter().map(|&v| ck.encrypt_signed(v)).collect()
}

/// Decrypt a single-element output [`CtVec`] (e.g. an `Argmax` class index) to an integer.
///
/// Panics if `out` is not exactly one element — the 2-class `Argmax` funnels to a single
/// scalar output, and a shape mismatch is a bug worth surfacing loudly (`AGENTS.md` §1.4).
pub fn decrypt_label(ck: &RadixClientKey, out: &CtVec) -> i64 {
    assert_eq!(
        out.len(),
        1,
        "decrypt_label expects a single output ciphertext, got {}",
        out.len()
    );
    ck.decrypt_signed(&out[0])
}

/// Decrypt every element of an output [`CtVec`] to a vector of integers.
///
/// The client-side companion to a multi-element output — e.g. a multi-class logit vector the
/// client then argmaxes locally (the Phase-4 10-class head: the graph emits the logits and the
/// client picks the max, so no wide-domain in-FHE argmax is needed; `PROJECT.md` §11).
pub fn decrypt_vec(ck: &RadixClientKey, out: &CtVec) -> Vec<i64> {
    out.iter().map(|ct| ck.decrypt_signed(ct)).collect()
}
