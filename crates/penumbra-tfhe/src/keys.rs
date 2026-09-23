//! Key generation and crypto-parameter profiles for the TFHE backend.
//!
//! The client holds the secret key (encrypt/decrypt); the server holds the public
//! evaluation/server key (enables bootstrapping) plus the plaintext model weights.
//! See `PROJECT.md` §11.
//!
//! Policy (`PROJECT.md` §12, `AGENTS.md` §7): ship a single secure default parameter
//! profile and expose at most **one** override knob. Never surface raw `tfhe-rs`
//! parameters to end users. Phase 2 exposes none — everything runs on [`DEFAULT_PARAMS`].

use std::path::Path;

use serde::{Deserialize, Serialize};
use tfhe::integer::{gen_keys_radix, RadixClientKey, ServerKey};
use tfhe::shortint::parameters::{
    PARAM_MESSAGE_2_CARRY_2_KS_PBS, PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128,
};
use tfhe::shortint::ClassicPBSParameters;

pub use penumbra_core::bitwidth::{magnitude_bits, radix_capacity_bits, MESSAGE_BITS};
use penumbra_core::wire::SchemeHeader;

/// Identifier for this backend's scheme.
pub const SCHEME_TFHE: &str = "tfhe";

/// The single secure default parameter profile (`PROJECT.md` §12, `AGENTS.md` §7).
pub const DEFAULT_PARAMS: ClassicPBSParameters = PARAM_MESSAGE_2_CARRY_2_KS_PBS;

/// Named crypto-parameter profiles — the single TFHE override knob (`PROJECT.md` §12).
///
/// Both are `tfhe-rs`'s own vetted 128-bit sets at message=2/carry=2, so `MESSAGE_BITS`
/// (`penumbra_core::bitwidth::MESSAGE_BITS` = 2) and every model's `num_blocks` are identical
/// across profiles — only the LWE noise distribution differs. Tuning message precision is
/// Phase-10 work (`ROADMAP.md` Phase 10, "Parameter tuning").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TfheProfile {
    /// TUniform noise — `tfhe-rs`'s own default (`PARAM_MESSAGE_2_CARRY_2_KS_PBS`).
    #[default]
    Default,
    /// Discrete-Gaussian noise at the same message/carry width.
    Gaussian,
}

impl TfheProfile {
    pub const NAMES: [&'static str; 2] = ["default", "gaussian"];

    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "default" => Ok(Self::Default),
            "gaussian" => Ok(Self::Gaussian),
            other => Err(format!(
                "unknown TFHE crypto profile '{other}'; available profiles: default, gaussian"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Gaussian => "gaussian",
        }
    }

    pub fn params(self) -> ClassicPBSParameters {
        match self {
            Self::Default => PARAM_MESSAGE_2_CARRY_2_KS_PBS,
            Self::Gaussian => PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128,
        }
    }
}

impl std::fmt::Display for TfheProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Generate a `(client_key, server_key)` pair over `profile` for radix integers
/// of `num_blocks` blocks.
pub fn keygen_with_profile(num_blocks: usize, profile: TfheProfile) -> (RadixClientKey, ServerKey) {
    assert!(
        num_blocks > 0,
        "keygen requires num_blocks > 0 (radix integers need at least one block)"
    );
    gen_keys_radix(profile.params(), num_blocks)
}

/// Generate a `(client_key, server_key)` pair over [`DEFAULT_PARAMS`] for radix integers
/// of `num_blocks` blocks.
pub fn keygen(num_blocks: usize) -> (RadixClientKey, ServerKey) {
    keygen_with_profile(num_blocks, TfheProfile::default())
}

/// A key plus its backend scheme, `num_blocks`, and `profile` — the on-disk envelope for both key kinds.
#[derive(Serialize, Deserialize)]
pub struct TaggedKey<K> {
    pub scheme: String,
    pub num_blocks: usize,
    pub profile: String,
    pub key: K,
}

#[derive(Serialize, Deserialize)]
struct TfheKeyHeader {
    scheme: String,
    num_blocks: usize,
    profile: String,
}

/// Serialize the client secret key to its tagged wire bytes (for size accounting and persistence).
pub fn client_key_bytes(
    ck: &RadixClientKey,
    num_blocks: usize,
    profile: TfheProfile,
) -> Result<Vec<u8>, String> {
    let tagged = TaggedKey {
        scheme: SCHEME_TFHE.to_string(),
        num_blocks,
        profile: profile.name().to_string(),
        key: ck,
    };
    bincode::serialize(&tagged).map_err(|e| format!("cannot serialize client key: {e}"))
}

/// Decode a client secret key from tagged wire bytes, validating the scheme and returning `(key, num_blocks, profile)`.
pub fn client_key_from_bytes(bytes: &[u8]) -> Result<(RadixClientKey, usize, TfheProfile), String> {
    let header: SchemeHeader = bincode::deserialize(bytes).map_err(|e| {
        format!(
            "cannot deserialize client key (is it a Penumbra client key file? keys generated before the crypto-profile field was added must be regenerated): {e}"
        )
    })?;
    if header.scheme != SCHEME_TFHE {
        return Err(format!(
            "backend/scheme mismatch for client key: expected '{SCHEME_TFHE}', found '{}' \
             (key material is not portable across backends; see docs/BACKENDS.md)",
            header.scheme
        ));
    }
    let key_header: TfheKeyHeader = bincode::deserialize(bytes).map_err(|e| {
        format!(
            "cannot deserialize client key (is it a Penumbra client key file? keys generated before the crypto-profile field was added must be regenerated): {e}"
        )
    })?;
    let profile = TfheProfile::from_name(&key_header.profile)?;
    let tagged: TaggedKey<RadixClientKey> = bincode::deserialize(bytes).map_err(|e| {
        format!(
            "cannot deserialize client key (is it a Penumbra client key file? keys generated before the crypto-profile field was added must be regenerated): {e}"
        )
    })?;
    Ok((tagged.key, tagged.num_blocks, profile))
}

/// Persist the client secret key to `path`, tagged with scheme, `num_blocks`, and `profile`.
pub fn save_client_key(
    ck: &RadixClientKey,
    num_blocks: usize,
    profile: TfheProfile,
    path: &Path,
) -> Result<(), String> {
    let bytes = client_key_bytes(ck, num_blocks, profile)?;
    std::fs::write(path, bytes)
        .map_err(|e| format!("cannot write client key to {}: {e}", path.display()))
}

/// Load a client secret key from `path`, validating the scheme and returning `(key, num_blocks, profile)`.
pub fn load_client_key(path: &Path) -> Result<(RadixClientKey, usize, TfheProfile), String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read client key from {}: {e}", path.display()))?;
    client_key_from_bytes(&bytes).map_err(|e| format!("{e} (file: {})", path.display()))
}

/// Serialize the public server/evaluation key to its tagged wire bytes (for size accounting and persistence).
pub fn server_key_bytes(
    sk: &ServerKey,
    num_blocks: usize,
    profile: TfheProfile,
) -> Result<Vec<u8>, String> {
    let tagged = TaggedKey {
        scheme: SCHEME_TFHE.to_string(),
        num_blocks,
        profile: profile.name().to_string(),
        key: sk,
    };
    bincode::serialize(&tagged).map_err(|e| format!("cannot serialize server key: {e}"))
}

/// Decode a server key from tagged wire bytes, validating the scheme and returning `(key, num_blocks, profile)`.
pub fn server_key_from_bytes(bytes: &[u8]) -> Result<(ServerKey, usize, TfheProfile), String> {
    let header: SchemeHeader = bincode::deserialize(bytes).map_err(|e| {
        format!(
            "cannot deserialize server key (is it a Penumbra server key file? keys generated before the crypto-profile field was added must be regenerated): {e}"
        )
    })?;
    if header.scheme != SCHEME_TFHE {
        return Err(format!(
            "backend/scheme mismatch for server key: expected '{SCHEME_TFHE}', found '{}' \
             (key material is not portable across backends; see docs/BACKENDS.md)",
            header.scheme
        ));
    }
    let key_header: TfheKeyHeader = bincode::deserialize(bytes).map_err(|e| {
        format!(
            "cannot deserialize server key (is it a Penumbra server key file? keys generated before the crypto-profile field was added must be regenerated): {e}"
        )
    })?;
    let profile = TfheProfile::from_name(&key_header.profile)?;
    let tagged: TaggedKey<ServerKey> = bincode::deserialize(bytes).map_err(|e| {
        format!(
            "cannot deserialize server key (is it a Penumbra server key file? keys generated before the crypto-profile field was added must be regenerated): {e}"
        )
    })?;
    Ok((tagged.key, tagged.num_blocks, profile))
}

/// Persist the public server/evaluation key to `path`, tagged with scheme, `num_blocks`, and `profile`.
pub fn save_server_key(
    sk: &ServerKey,
    num_blocks: usize,
    profile: TfheProfile,
    path: &Path,
) -> Result<(), String> {
    let bytes = server_key_bytes(sk, num_blocks, profile)?;
    std::fs::write(path, bytes)
        .map_err(|e| format!("cannot write server key to {}: {e}", path.display()))
}

/// Load a server key from `path`, validating the scheme and returning `(key, num_blocks, profile)`.
pub fn load_server_key(path: &Path) -> Result<(ServerKey, usize, TfheProfile), String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read server key from {}: {e}", path.display()))?;
    server_key_from_bytes(&bytes).map_err(|e| format!("{e} (file: {})", path.display()))
}

/// Fail loudly if a key's `num_blocks` does not match the model it is being used with.
pub fn ensure_num_blocks_match(
    key_num_blocks: usize,
    graph_num_blocks: usize,
) -> Result<(), String> {
    if key_num_blocks != graph_num_blocks {
        return Err(format!(
            "key/model mismatch: this key was generated for num_blocks={key_num_blocks}, but the \
             model needs num_blocks={graph_num_blocks}. Regenerate keys for this model \
             (keys are tied to a model's radix width)."
        ));
    }
    Ok(())
}

/// Fail loudly if a key was generated under a different parameter profile than the running backend.
pub fn ensure_profile_match(key: TfheProfile, run: TfheProfile) -> Result<(), String> {
    if key != run {
        return Err(format!(
            "key/profile mismatch: this key was generated under crypto profile '{}', but this \
             run uses '{}'. Keys are tied to their parameter profile — regenerate keys for this \
             profile, or select profile '{}'.",
            key.name(),
            run.name(),
            key.name()
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keygen_signed_roundtrip() {
        let num_blocks = 8;
        let (ck, _sk) = keygen(num_blocks);

        assert_eq!(radix_capacity_bits(num_blocks), 16);

        for v in [-1234i64, -1, 0, 1, 4321] {
            let ct = ck.encrypt_signed(v);
            let got: i64 = ck.decrypt_signed(&ct);
            assert_eq!(got, v, "signed radix round-trip must be exact");
        }
    }

    #[test]
    fn key_save_load_roundtrip() {
        let num_blocks = 4;
        let (ck, sk) = keygen(num_blocks);

        let dir = std::env::temp_dir().join(format!("penumbra_keytest_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ck_path = dir.join("client.key");
        let sk_path = dir.join("server.key");

        save_client_key(&ck, num_blocks, TfheProfile::default(), &ck_path)
            .expect("save client key");
        save_server_key(&sk, num_blocks, TfheProfile::default(), &sk_path)
            .expect("save server key");

        let (ck2, ck_nb, profile) = load_client_key(&ck_path).expect("load client key");
        let (_sk2, sk_nb, sk_profile) = load_server_key(&sk_path).expect("load server key");
        assert_eq!(ck_nb, num_blocks, "client key num_blocks tag must survive");
        assert_eq!(sk_nb, num_blocks, "server key num_blocks tag must survive");
        assert_eq!(profile, TfheProfile::Default);
        assert_eq!(sk_profile, TfheProfile::Default);

        let ct = ck.encrypt_signed(7i64);
        let got: i64 = ck2.decrypt_signed(&ct);
        assert_eq!(got, 7, "reloaded client key must decrypt exactly");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn num_blocks_mismatch_fails_loudly() {
        assert!(ensure_num_blocks_match(8, 8).is_ok());
        let err = ensure_num_blocks_match(8, 7).unwrap_err();
        assert!(
            err.contains("num_blocks=8"),
            "message names the key width: {err}"
        );
        assert!(
            err.contains("num_blocks=7"),
            "message names the model width: {err}"
        );
    }
}
