//! Scheme-tagged wire envelopes shared by every backend (Layer 2).
//!
//! Keys and ciphertext are **not** portable across backends (`AGENTS.md` §1.4), so every
//! serialized blob carries a leading `scheme` string. This module owns that envelope once,
//! so the backends cannot drift apart on either the byte layout or the rejection message.
//!
//! Nothing here is scheme-specific: the expected scheme is a parameter, never a branch
//! (`AGENTS.md` §1.2).

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Envelope tagging ciphertext material with its backend scheme identifier.
#[derive(Serialize, Deserialize)]
pub struct TaggedCts<T> {
    pub scheme: String,
    pub payload: T,
}

/// Prefix-compatible view of any tagged envelope. `bincode` writes fields in declaration
/// order, so deserializing this reads the leading `scheme` without touching the payload —
/// which is what lets a mismatch be reported instead of panicking on a foreign payload.
#[derive(Serialize, Deserialize)]
pub struct SchemeHeader {
    pub scheme: String,
}

/// Serialize `payload` into a [`TaggedCts`] envelope tagged with `scheme`.
///
/// `kind` names the material in the error message: `"ciphertext"` or `"ciphertext batch"`.
pub fn encode_tagged<T: Serialize>(
    payload: T,
    scheme: &str,
    kind: &str,
) -> Result<Vec<u8>, String> {
    let tagged = TaggedCts {
        scheme: scheme.to_string(),
        payload,
    };
    bincode::serialize(&tagged).map_err(|e| format!("cannot serialize {kind}: {e}"))
}

/// Read a [`TaggedCts`] envelope, rejecting material tagged with a different scheme before
/// attempting to decode the payload (`AGENTS.md` §1.4).
pub fn decode_tagged<T: DeserializeOwned>(
    bytes: &[u8],
    expected_scheme: &str,
    kind: &str,
) -> Result<T, String> {
    let header: SchemeHeader = bincode::deserialize(bytes)
        .map_err(|e| format!("cannot deserialize {kind} (is it a Penumbra ciphertext?): {e}"))?;
    if header.scheme != expected_scheme {
        return Err(format!(
            "backend/scheme mismatch for {kind}: expected '{expected_scheme}', but found '{}' \
             (ciphertext material is not portable across backends; see docs/BACKENDS.md)",
            header.scheme
        ));
    }
    let tagged: TaggedCts<T> = bincode::deserialize(bytes)
        .map_err(|e| format!("cannot deserialize {kind} (is it a Penumbra ciphertext?): {e}"))?;
    Ok(tagged.payload)
}
