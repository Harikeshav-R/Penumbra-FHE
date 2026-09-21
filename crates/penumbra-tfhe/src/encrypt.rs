//! Encrypt / decrypt helpers for the TFHE backend.
//!
//! Client-side helpers over `tfhe-rs`: turn a quantized-integer input vector into
//! ciphertexts (with the client key) and turn the encrypted output back into a prediction.
//! The server never sees plaintext (`PROJECT.md` §11).

use serde::{Deserialize, Serialize};
use tfhe::integer::{RadixClientKey, SignedRadixCiphertext};

use crate::keys::SCHEME_TFHE;

/// An encrypted tensor of signed radix integers.
pub type CtVec = Vec<SignedRadixCiphertext>;

/// Envelope tagging ciphertext material with its backend scheme identifier.
#[derive(Serialize, Deserialize)]
pub struct TaggedCts<T> {
    pub scheme: String,
    pub payload: T,
}

/// Encrypt a quantized-integer input vector into a [`CtVec`] of signed radix ciphertexts.
pub fn encrypt(ck: &RadixClientKey, input: &[i64]) -> CtVec {
    input.iter().map(|&v| ck.encrypt_signed(v)).collect()
}

/// Decrypt a single-element output [`CtVec`] to an integer label.
pub fn decrypt_label(ck: &RadixClientKey, out: &[SignedRadixCiphertext]) -> i64 {
    assert_eq!(
        out.len(),
        1,
        "decrypt_label expects a single output ciphertext, got {}",
        out.len()
    );
    ck.decrypt_signed(&out[0])
}

/// Decrypt every element of an output [`CtVec`] to a vector of integers.
pub fn decrypt_vec(ck: &RadixClientKey, out: &[SignedRadixCiphertext]) -> Vec<i64> {
    out.iter().map(|ct| ck.decrypt_signed(ct)).collect()
}

/// Serialize an encrypted tensor to bytes tagged with the `"tfhe"` backend identifier.
pub fn serialize_cts(cts: &[SignedRadixCiphertext]) -> Result<Vec<u8>, String> {
    let tagged = TaggedCts {
        scheme: SCHEME_TFHE.to_string(),
        payload: cts,
    };
    bincode::serialize(&tagged).map_err(|e| format!("cannot serialize ciphertext: {e}"))
}

#[derive(Deserialize)]
struct SchemeHeader {
    scheme: String,
}

/// Deserialize an encrypted tensor from bytes, verifying the scheme tag matches `"tfhe"`.
pub fn deserialize_cts(bytes: &[u8]) -> Result<CtVec, String> {
    let header: SchemeHeader = bincode::deserialize(bytes).map_err(|e| {
        format!("cannot deserialize ciphertext (is it a Penumbra ciphertext?): {e}")
    })?;
    if header.scheme != SCHEME_TFHE {
        return Err(format!(
            "backend/scheme mismatch for ciphertext: expected '{SCHEME_TFHE}', but found '{}' \
             (ciphertext material is not portable across backends; see docs/BACKENDS.md)",
            header.scheme
        ));
    }
    let tagged: TaggedCts<CtVec> = bincode::deserialize(bytes).map_err(|e| {
        format!("cannot deserialize ciphertext (is it a Penumbra ciphertext?): {e}")
    })?;
    Ok(tagged.payload)
}

/// Serialize a batch of encrypted tensors (`Vec<CtVec>`) tagged with the `"tfhe"` backend identifier.
pub fn serialize_cts_batch(batch: &[CtVec]) -> Result<Vec<u8>, String> {
    let tagged = TaggedCts {
        scheme: SCHEME_TFHE.to_string(),
        payload: batch,
    };
    bincode::serialize(&tagged).map_err(|e| format!("cannot serialize ciphertext batch: {e}"))
}

/// Deserialize a batch of encrypted tensors (`Vec<CtVec>`) from bytes, verifying the scheme tag matches `"tfhe"`.
pub fn deserialize_cts_batch(bytes: &[u8]) -> Result<Vec<CtVec>, String> {
    let header: SchemeHeader = bincode::deserialize(bytes).map_err(|e| {
        format!("cannot deserialize ciphertext batch (is it a Penumbra ciphertext?): {e}")
    })?;
    if header.scheme != SCHEME_TFHE {
        return Err(format!(
            "backend/scheme mismatch for ciphertext batch: expected '{SCHEME_TFHE}', but found '{}' \
             (ciphertext material is not portable across backends; see docs/BACKENDS.md)",
            header.scheme
        ));
    }
    let tagged: TaggedCts<Vec<CtVec>> = bincode::deserialize(bytes).map_err(|e| {
        format!("cannot deserialize ciphertext batch (is it a Penumbra ciphertext?): {e}")
    })?;
    Ok(tagged.payload)
}
