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
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
use tfhe::shortint::ClassicPBSParameters;

pub use penumbra_core::bitwidth::{magnitude_bits, radix_capacity_bits, MESSAGE_BITS};

/// Identifier for this backend's scheme.
pub const SCHEME_TFHE: &str = "tfhe";

/// The single secure default parameter profile (`PROJECT.md` §12, `AGENTS.md` §7).
pub const DEFAULT_PARAMS: ClassicPBSParameters = PARAM_MESSAGE_2_CARRY_2_KS_PBS;

/// Generate a `(client_key, server_key)` pair over [`DEFAULT_PARAMS`] for radix integers
/// of `num_blocks` blocks.
pub fn keygen(num_blocks: usize) -> (RadixClientKey, ServerKey) {
    assert!(
        num_blocks > 0,
        "keygen requires num_blocks > 0 (radix integers need at least one block)"
    );
    gen_keys_radix(DEFAULT_PARAMS, num_blocks)
}

/// A key plus its backend scheme and `num_blocks` — the on-disk envelope for both key kinds.
#[derive(Serialize, Deserialize)]
pub struct TaggedKey<K> {
    pub scheme: String,
    pub num_blocks: usize,
    pub key: K,
}

fn write_bincode<T: Serialize>(value: &T, path: &Path, what: &str) -> Result<(), String> {
    let bytes = bincode::serialize(value).map_err(|e| format!("cannot serialize {what}: {e}"))?;
    std::fs::write(path, bytes)
        .map_err(|e| format!("cannot write {what} to {}: {e}", path.display()))
}

/// Persist the client secret key to `path`, tagged with scheme and `num_blocks`.
pub fn save_client_key(ck: &RadixClientKey, num_blocks: usize, path: &Path) -> Result<(), String> {
    write_bincode(
        &TaggedKey {
            scheme: SCHEME_TFHE.to_string(),
            num_blocks,
            key: ck.clone(),
        },
        path,
        "client key",
    )
}

#[derive(Deserialize)]
struct SchemeHeader {
    scheme: String,
}

/// Load a client secret key from `path`, validating the scheme and returning `num_blocks`.
pub fn load_client_key(path: &Path) -> Result<(RadixClientKey, usize), String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read client key from {}: {e}", path.display()))?;
    let header: SchemeHeader = bincode::deserialize(&bytes).map_err(|e| {
        format!(
            "cannot deserialize client key from {} (is it a Penumbra client key file?): {e}",
            path.display()
        )
    })?;
    if header.scheme != SCHEME_TFHE {
        return Err(format!(
            "backend/scheme mismatch for client key at {}: expected '{}', found '{}' \
             (key material is not portable across backends; see docs/BACKENDS.md)",
            path.display(),
            SCHEME_TFHE,
            header.scheme
        ));
    }
    let tagged: TaggedKey<RadixClientKey> = bincode::deserialize(&bytes).map_err(|e| {
        format!(
            "cannot deserialize client key from {} (is it a Penumbra client key file?): {e}",
            path.display()
        )
    })?;
    Ok((tagged.key, tagged.num_blocks))
}

/// Persist the public server/evaluation key to `path`, tagged with scheme and `num_blocks`.
pub fn save_server_key(sk: &ServerKey, num_blocks: usize, path: &Path) -> Result<(), String> {
    write_bincode(
        &TaggedKey {
            scheme: SCHEME_TFHE.to_string(),
            num_blocks,
            key: sk.clone(),
        },
        path,
        "server key",
    )
}

/// Load a server key from `path`, validating the scheme and returning `num_blocks`.
pub fn load_server_key(path: &Path) -> Result<(ServerKey, usize), String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read server key from {}: {e}", path.display()))?;
    let header: SchemeHeader = bincode::deserialize(&bytes).map_err(|e| {
        format!(
            "cannot deserialize server key from {} (is it a Penumbra server key file?): {e}",
            path.display()
        )
    })?;
    if header.scheme != SCHEME_TFHE {
        return Err(format!(
            "backend/scheme mismatch for server key at {}: expected '{}', found '{}' \
             (key material is not portable across backends; see docs/BACKENDS.md)",
            path.display(),
            SCHEME_TFHE,
            header.scheme
        ));
    }
    let tagged: TaggedKey<ServerKey> = bincode::deserialize(&bytes).map_err(|e| {
        format!(
            "cannot deserialize server key from {} (is it a Penumbra server key file?): {e}",
            path.display()
        )
    })?;
    Ok((tagged.key, tagged.num_blocks))
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

        save_client_key(&ck, num_blocks, &ck_path).expect("save client key");
        save_server_key(&sk, num_blocks, &sk_path).expect("save server key");

        let (ck2, ck_nb) = load_client_key(&ck_path).expect("load client key");
        let (_sk2, sk_nb) = load_server_key(&sk_path).expect("load server key");
        assert_eq!(ck_nb, num_blocks, "client key num_blocks tag must survive");
        assert_eq!(sk_nb, num_blocks, "server key num_blocks tag must survive");

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
