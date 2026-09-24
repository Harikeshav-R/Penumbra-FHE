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
use tfhe::shortint::parameters::current_params::{
    V1_8_PARAM_MULTI_BIT_GROUP_2_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
    V1_8_PARAM_MULTI_BIT_GROUP_3_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
    V1_8_PARAM_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
};
use tfhe::shortint::parameters::{
    PBSParameters, PARAM_MESSAGE_2_CARRY_2_KS_PBS, PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128,
};

pub use penumbra_core::bitwidth::{magnitude_bits, radix_capacity_bits, MESSAGE_BITS};
use penumbra_core::wire::SchemeHeader;

/// Identifier for this backend's scheme.
pub const SCHEME_TFHE: &str = "tfhe";

/// The single secure default parameter profile (`PROJECT.md` §12, `AGENTS.md` §7).
pub const DEFAULT_PARAMS: PBSParameters = TfheProfile::MultiBit3.params();

/// Named crypto-parameter profiles — the single TFHE override knob (`PROJECT.md` §12).
///
/// All five are 128-bit-secure `tfhe-rs` parameter sets at message=2/carry=2, so `MESSAGE_BITS`
/// (`penumbra_core::bitwidth::MESSAGE_BITS` = 2) and every model's `num_blocks` are identical
/// across profiles. Multi-bit PBS sets use deterministic execution to ensure reproducible
/// ciphertext representations across threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TfheProfile {
    /// Classic PBS, TUniform noise — `tfhe-rs`'s own `PARAM_MESSAGE_2_CARRY_2_KS_PBS`.
    /// p-fail = 2^-129.581, algorithmic cost ~ 113.
    Classic,
    /// Classic PBS, discrete-Gaussian noise at the same message/carry width.
    Gaussian,
    /// Multi-bit PBS, grouping factor 2. p-fail = 2^-140.341, algorithmic cost ~ 188.
    MultiBit2,
    /// Multi-bit PBS, grouping factor 3. p-fail = 2^-128.235, algorithmic cost ~ 143.
    #[default]
    MultiBit3,
    /// Multi-bit PBS, grouping factor 4. p-fail = 2^-134.345, algorithmic cost ~ 100.
    MultiBit4,
}

impl TfheProfile {
    pub const NAMES: [&'static str; 5] =
        ["classic", "gaussian", "multibit2", "multibit3", "multibit4"];

    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "classic" => Ok(Self::Classic),
            "gaussian" => Ok(Self::Gaussian),
            "multibit2" => Ok(Self::MultiBit2),
            "multibit3" => Ok(Self::MultiBit3),
            "multibit4" => Ok(Self::MultiBit4),
            other => Err(format!(
                "unknown TFHE crypto profile '{other}'; available profiles: classic, gaussian, \
                 multibit2, multibit3, multibit4"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::Gaussian => "gaussian",
            Self::MultiBit2 => "multibit2",
            Self::MultiBit3 => "multibit3",
            Self::MultiBit4 => "multibit4",
        }
    }

    pub const fn params(self) -> PBSParameters {
        match self {
            Self::Classic => PBSParameters::PBS(PARAM_MESSAGE_2_CARRY_2_KS_PBS),
            Self::Gaussian => PBSParameters::PBS(PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128),
            Self::MultiBit2 => PBSParameters::MultiBitPBS(
                V1_8_PARAM_MULTI_BIT_GROUP_2_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128
                    .with_deterministic_execution(),
            ),
            Self::MultiBit3 => PBSParameters::MultiBitPBS(
                V1_8_PARAM_MULTI_BIT_GROUP_3_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128
                    .with_deterministic_execution(),
            ),
            Self::MultiBit4 => PBSParameters::MultiBitPBS(
                V1_8_PARAM_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128
                    .with_deterministic_execution(),
            ),
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
        assert_eq!(profile, TfheProfile::default());
        assert_eq!(sk_profile, TfheProfile::default());

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

    #[test]
    fn default_params_tracks_default_profile() {
        assert_eq!(DEFAULT_PARAMS, TfheProfile::default().params());
    }

    #[test]
    fn retired_default_profile_name_fails_loudly() {
        let err = TfheProfile::from_name("default").unwrap_err();
        assert!(
            err.contains("available profiles: classic, gaussian, multibit2, multibit3, multibit4")
        );
    }
}
