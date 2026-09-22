//! Encrypt / decrypt helpers and wire format for the CKKS backend.
//!
//! Client-side helpers: turn a quantized-integer input vector into packed
//! ciphertexts (with the client key) and turn the encrypted output back into a
//! prediction. One tensor lives in the slots of ONE packed ciphertext (Option B,
//! settled in `docs/BACKENDS.md`).

use std::io::Cursor;

use poulpy_ckks::layouts::{CKKSCiphertextOwned, CKKSModuleAlloc};
use poulpy_ckks::{CKKSInfos, CKKSMeta, SlotsKind};
use poulpy_core::layouts::{Base2K, Degree, GLWEInfos, GLWELayout, LWEInfos, Rank, TorusPrecision};
use poulpy_core::EncryptionLayout;
use poulpy_hal::layouts::{Module, ReaderFrom, WriterTo};
use serde::{Deserialize, Serialize};

use crate::hal::ActiveBackend;
use crate::keys::{decrypt_raw, encrypt_raw, CkksClientKey, SCHEME_CKKS};

/// One packed CKKS tensor: the whole tensor lives in the slots of ONE ciphertext
/// (`docs/BACKENDS.md` open fork 1, option B).
#[derive(Clone)]
pub struct CkksCt {
    pub(crate) ct: CKKSCiphertextOwned<ActiveBackend>,
    /// Live elements in slots `[0, len)`; `[len, lt_slots)` is zero padding.
    pub(crate) len: usize,
}
impl CkksCt {
    pub fn new(ct: CKKSCiphertextOwned<ActiveBackend>, len: usize) -> Self {
        Self { ct, len }
    }
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn ct(&self) -> &CKKSCiphertextOwned<ActiveBackend> {
        &self.ct
    }
}

impl std::fmt::Debug for CkksCt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CkksCt").field("len", &self.len).finish()
    }
}
/// An encrypted tensor: packed into ONE ciphertext.
pub type CtVec = Vec<CkksCt>;

/// Envelope tagging ciphertext material with its backend scheme identifier.
#[derive(Serialize, Deserialize)]
pub struct TaggedCts<T> {
    pub scheme: String,
    pub payload: T,
}

#[derive(Deserialize)]
struct SchemeHeader {
    scheme: String,
}

/// A serialized CKKS ciphertext: the layout needed to re-allocate the receiver, the CKKS
/// metadata (which `WriterTo` does not carry), and the `GLWE` body bytes.
#[derive(Serialize, Deserialize)]
pub struct CkksCtBytes {
    pub n: u32,
    pub base2k: u32,
    pub k: u32,
    pub rank: u32,
    pub log_delta: u32,
    pub log_sparsity: u32,
    /// `SlotsKind` as a stable string, "real" | "complex".
    pub slots_kind: String,
    pub len: u32,
    pub body: Vec<u8>,
}

fn ct_to_bytes(c: &CkksCt) -> Result<CkksCtBytes, String> {
    let host_ct = c.ct.to_host_owned::<ActiveBackend>();
    let mut body = Vec::new();
    host_ct
        .write_to(&mut body)
        .map_err(|e| format!("cannot serialize GLWE ciphertext: {e}"))?;

    let meta = c.ct.meta();
    let slots_kind = match meta.slots {
        SlotsKind::Real => "real",
        SlotsKind::Complex => "complex",
    }
    .to_string();

    Ok(CkksCtBytes {
        n: c.ct.n().as_usize() as u32,
        base2k: c.ct.base2k().as_usize() as u32,
        k: c.ct.k().as_usize() as u32,
        rank: c.ct.rank().as_usize() as u32,
        log_delta: meta.log_delta as u32,
        log_sparsity: meta.log_sparsity as u32,
        slots_kind,
        len: c.len as u32,
        body,
    })
}

fn ct_from_bytes(bytes: &CkksCtBytes) -> Result<CkksCt, String> {
    let module = Module::<ActiveBackend>::new(bytes.n as u64);
    let glwe_layout = GLWELayout {
        n: Degree(bytes.n),
        base2k: Base2K(bytes.base2k),
        k: TorusPrecision(bytes.k),
        rank: Rank(bytes.rank),
    };
    let enc_layout = EncryptionLayout::new_from_default_sigma(glwe_layout)
        .map_err(|e| format!("invalid encryption layout: {e:?}"))?;

    let mut ct = module.ckks_ciphertext_alloc_from_glwe_infos(&enc_layout);
    let mut cursor = Cursor::new(&bytes.body);
    ct.read_from(&mut cursor)
        .map_err(|e| format!("cannot read GLWE ciphertext body: {e}"))?;

    let slots = if bytes.slots_kind == "real" {
        SlotsKind::Real
    } else {
        SlotsKind::Complex
    };
    ct.set_meta_checked(CKKSMeta {
        log_delta: bytes.log_delta as usize,
        log_sparsity: bytes.log_sparsity as usize,
        slots,
    })
    .map_err(|e| format!("cannot set CKKS metadata on deserialized ciphertext: {e}"))?;

    Ok(CkksCt {
        ct,
        len: bytes.len as usize,
    })
}

/// Encrypt a quantized-integer input vector into a [`CtVec`] containing ONE packed CKKS ciphertext.
pub fn encrypt(ck: &CkksClientKey, input: &[i64]) -> CtVec {
    let floats: Vec<f64> = input.iter().map(|&v| v as f64).collect();
    let ct = encrypt_raw(ck, &floats).expect("CKKS encrypt failed");
    vec![CkksCt {
        ct,
        len: input.len(),
    }]
}

/// Decrypt every element of an output [`CtVec`] to a vector of integers.
pub fn decrypt_vec(ck: &CkksClientKey, out: &[CkksCt]) -> Vec<i64> {
    assert!(!out.is_empty(), "cannot decrypt empty ciphertext vector");
    let c = &out[0];
    let decoded = decrypt_raw(ck, &c.ct).expect("CKKS decrypt failed");
    decoded[..c.len].iter().map(|&v| v.round() as i64).collect()
}

/// Decrypt the raw float values of an output [`CtVec`].
pub fn decrypt_raw_vec(ck: &CkksClientKey, out: &[CkksCt]) -> Vec<f64> {
    assert!(!out.is_empty(), "cannot decrypt empty ciphertext vector");
    let c = &out[0];
    let decoded = decrypt_raw(ck, &c.ct).expect("CKKS decrypt failed");
    decoded[..c.len].to_vec()
}

/// Decrypt a single-element output [`CtVec`] to an integer label.
pub fn decrypt_label(ck: &CkksClientKey, out: &[CkksCt]) -> i64 {
    let vals = decrypt_vec(ck, out);
    assert!(!vals.is_empty(), "empty output vector");
    vals[0]
}

/// Serialize an encrypted tensor to bytes tagged with the `"ckks"` backend identifier.
pub fn serialize_cts(cts: &[CkksCt]) -> Result<Vec<u8>, String> {
    let mut payload = Vec::with_capacity(cts.len());
    for c in cts {
        payload.push(ct_to_bytes(c)?);
    }
    let tagged = TaggedCts {
        scheme: SCHEME_CKKS.to_string(),
        payload,
    };
    bincode::serialize(&tagged).map_err(|e| format!("cannot serialize ciphertext: {e}"))
}

/// Deserialize an encrypted tensor from bytes, verifying the scheme tag matches `"ckks"`.
pub fn deserialize_cts(bytes: &[u8]) -> Result<CtVec, String> {
    let header: SchemeHeader = bincode::deserialize(bytes).map_err(|e| {
        format!("cannot deserialize ciphertext (is it a Penumbra ciphertext?): {e}")
    })?;
    if header.scheme != SCHEME_CKKS {
        return Err(format!(
            "backend/scheme mismatch for ciphertext: expected '{SCHEME_CKKS}', but found '{}' \
             (ciphertext material is not portable across backends; see docs/BACKENDS.md)",
            header.scheme
        ));
    }
    let tagged: TaggedCts<Vec<CkksCtBytes>> = bincode::deserialize(bytes).map_err(|e| {
        format!("cannot deserialize ciphertext (is it a Penumbra ciphertext?): {e}")
    })?;
    let mut cts = Vec::with_capacity(tagged.payload.len());
    for b in &tagged.payload {
        cts.push(ct_from_bytes(b)?);
    }
    Ok(cts)
}

/// Serialize a batch of encrypted tensors (`Vec<CtVec>`) tagged with the `"ckks"` backend identifier.
pub fn serialize_cts_batch(batch: &[CtVec]) -> Result<Vec<u8>, String> {
    let mut payload_batch = Vec::with_capacity(batch.len());
    for vec in batch {
        let mut cts_bytes = Vec::with_capacity(vec.len());
        for c in vec {
            cts_bytes.push(ct_to_bytes(c)?);
        }
        payload_batch.push(cts_bytes);
    }
    let tagged = TaggedCts {
        scheme: SCHEME_CKKS.to_string(),
        payload: payload_batch,
    };
    bincode::serialize(&tagged).map_err(|e| format!("cannot serialize ciphertext batch: {e}"))
}

/// Deserialize a batch of encrypted tensors (`Vec<CtVec>`) from bytes, verifying the scheme tag matches `"ckks"`.
pub fn deserialize_cts_batch(bytes: &[u8]) -> Result<Vec<CtVec>, String> {
    let header: SchemeHeader = bincode::deserialize(bytes).map_err(|e| {
        format!("cannot deserialize ciphertext batch (is it a Penumbra ciphertext?): {e}")
    })?;
    if header.scheme != SCHEME_CKKS {
        return Err(format!(
            "backend/scheme mismatch for ciphertext batch: expected '{SCHEME_CKKS}', but found '{}' \
             (ciphertext material is not portable across backends; see docs/BACKENDS.md)",
            header.scheme
        ));
    }
    let tagged: TaggedCts<Vec<Vec<CkksCtBytes>>> = bincode::deserialize(bytes).map_err(|e| {
        format!("cannot deserialize ciphertext batch (is it a Penumbra ciphertext?): {e}")
    })?;
    let mut batch = Vec::with_capacity(tagged.payload.len());
    for vec_bytes in &tagged.payload {
        let mut cts = Vec::with_capacity(vec_bytes.len());
        for b in vec_bytes {
            cts.push(ct_from_bytes(b)?);
        }
        batch.push(cts);
    }
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::keygen;
    use crate::params::CkksParams;

    #[test]
    fn test_encrypt_decrypt_roundtrip_and_serialization() {
        let params = CkksParams {
            n: 512,
            base2k: 19,
            k: 100,
            log_delta: 25,
            dsize: 2,
            rank: 1,
            lt_slots: 64,
            giant_step: 8,
            secret_ternary_prob: 2.0 / 3.0,
            max_poly_degree: 7,
        };

        let (ck, _sk) = keygen(&params).expect("keygen failed");
        let input = vec![10, -20, 30, 42];
        let ct_vec = encrypt(&ck, &input);
        assert_eq!(ct_vec.len(), 1);
        assert_eq!(ct_vec[0].len(), 4);

        let decrypted = decrypt_vec(&ck, &ct_vec);
        assert_eq!(decrypted, input);

        // Test serialization round-trip
        let bytes = serialize_cts(&ct_vec).expect("serialize_cts failed");
        let deserialized = deserialize_cts(&bytes).expect("deserialize_cts failed");
        assert_eq!(deserialized.len(), 1);
        assert_eq!(deserialized[0].len(), 4);
        let dec_after = decrypt_vec(&ck, &deserialized);
        assert_eq!(dec_after, input);

        // Test batch serialization
        let batch = vec![ct_vec.clone(), ct_vec];
        let batch_bytes = serialize_cts_batch(&batch).expect("serialize_cts_batch failed");
        let deserialized_batch =
            deserialize_cts_batch(&batch_bytes).expect("deserialize_cts_batch failed");
        assert_eq!(deserialized_batch.len(), 2);
        assert_eq!(decrypt_vec(&ck, &deserialized_batch[0]), input);
        assert_eq!(decrypt_vec(&ck, &deserialized_batch[1]), input);
    }
}
