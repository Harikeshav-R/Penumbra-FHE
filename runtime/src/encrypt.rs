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

/// Serialize an encrypted tensor to bytes for transport (the client/server wire, `PROJECT.md`
/// §11) or on-disk staging in the split round trip (ROADMAP Phase 9).
///
/// `SignedRadixCiphertext` derives serde, so a `CtVec` is a plain serde value; we use bincode
/// (compact, and the format `tfhe-rs`'s own integer docs use). This is what crosses the boundary
/// between the client (encrypt/decrypt) and the server (evaluate) — ciphertext only, never
/// plaintext or the secret key.
pub fn serialize_cts(cts: &CtVec) -> Result<Vec<u8>, String> {
    bincode::serialize(cts).map_err(|e| format!("cannot serialize ciphertext: {e}"))
}

/// Deserialize an encrypted tensor produced by [`serialize_cts`].
///
/// Fails loudly (`AGENTS.md` §1.4) on malformed bytes rather than surfacing an opaque error
/// deep in evaluation.
pub fn deserialize_cts(bytes: &[u8]) -> Result<CtVec, String> {
    bincode::deserialize(bytes)
        .map_err(|e| format!("cannot deserialize ciphertext (is it a Penumbra ciphertext?): {e}"))
}
