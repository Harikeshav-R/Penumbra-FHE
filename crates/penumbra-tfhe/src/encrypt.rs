//! Encrypt / decrypt helpers for the TFHE backend.
//!
//! Client-side helpers over `tfhe-rs`: turn a quantized-integer input vector into
//! ciphertexts (with the client key) and turn the encrypted output back into a prediction.
//! The server never sees plaintext (`PROJECT.md` §11).

use tfhe::integer::{RadixClientKey, SignedRadixCiphertext};

use crate::keys::SCHEME_TFHE;
use penumbra_core::wire::{decode_tagged, encode_tagged};

/// An encrypted tensor of signed radix integers.
pub type CtVec = Vec<SignedRadixCiphertext>;

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
    encode_tagged(cts, SCHEME_TFHE, "ciphertext")
}

/// Deserialize an encrypted tensor from bytes, verifying the scheme tag matches `"tfhe"`.
pub fn deserialize_cts(bytes: &[u8]) -> Result<CtVec, String> {
    decode_tagged(bytes, SCHEME_TFHE, "ciphertext")
}

/// Serialize a batch of encrypted tensors (`Vec<CtVec>`) tagged with the `"tfhe"` backend identifier.
pub fn serialize_cts_batch(batch: &[CtVec]) -> Result<Vec<u8>, String> {
    encode_tagged(batch, SCHEME_TFHE, "ciphertext batch")
}

/// Deserialize a batch of encrypted tensors (`Vec<CtVec>`) from bytes, verifying the scheme tag matches `"tfhe"`.
pub fn deserialize_cts_batch(bytes: &[u8]) -> Result<Vec<CtVec>, String> {
    decode_tagged(bytes, SCHEME_TFHE, "ciphertext batch")
}
