//! Key generation and crypto-parameter profiles.
//!
//! The client holds the secret key (encrypt/decrypt); the server holds the public
//! evaluation/server key (enables bootstrapping) plus the plaintext model weights.
//! See `PROJECT.md` §11.
//!
//! Policy (`PROJECT.md` §12, `AGENTS.md` §7): ship a single secure default parameter
//! profile and expose at most **one** override knob. Never surface raw `tfhe-rs`
//! parameters to end users. Phase 2 exposes none — everything runs on [`DEFAULT_PARAMS`].
//!
//! ## The `integer` API and the bit-width budget
//!
//! Penumbra builds on the `tfhe-rs` high-level `integer` API: a value is a **radix**
//! ciphertext made of `num_blocks` small `shortint` blocks. Under [`DEFAULT_PARAMS`] each
//! block carries [`MESSAGE_BITS`] bits of message, so a radix integer holds
//! `num_blocks * MESSAGE_BITS` bits. `num_blocks` is therefore the single, central
//! **bit-width budget** handle (`PROJECT.md` §9): the library derives it from a model's
//! widest accumulator; it is never a user-facing crypto parameter.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tfhe::integer::{gen_keys_radix, RadixClientKey, ServerKey};
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
use tfhe::shortint::ClassicPBSParameters;

/// The single secure default parameter profile (`PROJECT.md` §12, `AGENTS.md` §7).
///
/// `PARAM_MESSAGE_2_CARRY_2_KS_PBS` is the `tfhe-rs` default secure profile: a 2-bit
/// message space with a 2-bit carry buffer per block. Small integers are the working
/// range (`PROJECT.md` §9); we never hand-roll crypto parameters.
pub const DEFAULT_PARAMS: ClassicPBSParameters = PARAM_MESSAGE_2_CARRY_2_KS_PBS;

/// Message bits carried by a single radix block under [`DEFAULT_PARAMS`].
///
/// `message_modulus` is 2^2 = 4 for this profile, i.e. 2 usable message bits per block.
/// A radix integer of `n` blocks thus holds `n * MESSAGE_BITS` bits of value. This is the
/// arithmetic behind every bit-width-budget check (`PROJECT.md` §9, `AGENTS.md` §1.3).
pub const MESSAGE_BITS: usize = 2;

/// Generate a `(client_key, server_key)` pair over [`DEFAULT_PARAMS`] for radix integers
/// of `num_blocks` blocks.
///
/// - The **client** keeps `RadixClientKey` (the secret key) for encrypt/decrypt.
/// - The **server** holds `ServerKey` (the public evaluation key) and runs the encrypted
///   forward pass; it never sees plaintext (`PROJECT.md` §11).
///
/// `num_blocks` fixes the radix width — the central bit-width budget (`PROJECT.md` §9).
/// All ciphertexts in one model must share it so they are arithmetic-compatible.
pub fn keygen(num_blocks: usize) -> (RadixClientKey, ServerKey) {
    // Zero blocks is a nonsensical radix width: it carries no value and would otherwise
    // surface as an opaque panic deep inside `tfhe-rs` (or in `blocks()[0]` accesses).
    // Reject it here with an actionable message (`AGENTS.md` §1.4).
    assert!(
        num_blocks > 0,
        "keygen requires num_blocks > 0 (radix integers need at least one block)"
    );
    gen_keys_radix(DEFAULT_PARAMS, num_blocks)
}

/// Bits of value a radix integer of `num_blocks` blocks can hold under [`DEFAULT_PARAMS`].
///
/// Used by the bit-width-budget check to fail loudly *before* an accumulator would
/// overflow the chosen radix (`AGENTS.md` §1.3, §1.4).
pub fn radix_capacity_bits(num_blocks: usize) -> usize {
    num_blocks * MESSAGE_BITS
}

/// Minimum number of bits to represent the magnitude of `x` (0 for `x == 0`).
///
/// This is the position of the top set bit — the unsigned/magnitude width every bit-width
/// growth rule is built on (`PROJECT.md` §9, `AGENTS.md` §1.3). It is the single source of
/// truth for "how many bits does this nonnegative integer occupy"; callers add their own
/// sign/carry headroom and decide how to treat the zero case (a zero bias contributes 0
/// magnitude bits, whereas a LUT entry of 0 still occupies a 1-bit representable value).
pub fn magnitude_bits(x: u64) -> usize {
    (u64::BITS - x.leading_zeros()) as usize
}

// ---------------------------------------------------------------------------------------------
// Key persistence (ROADMAP Phase 9: "keygen, save/load keys, reuse keys across calls").
//
// The client/server split (`PROJECT.md` §11) needs keys to outlive one process: the client
// generates a pair once, persists it, and reuses it across inferences; the server loads *only*
// the public server key. Both `RadixClientKey` and `ServerKey` derive serde, so we persist them
// with bincode (compact — a server key is large). Each key is tagged with the `num_blocks` it
// was generated for: a key is only arithmetic-compatible with a model of the *same* radix width
// (`DEFAULT_PARAMS` is fixed, so `num_blocks` is the only degree of freedom). Tagging lets a
// load fail loudly on a mismatch (`AGENTS.md` §1.4) instead of surfacing as an opaque panic deep
// inside `tfhe-rs` when a wrong-width ciphertext meets the server key.
// ---------------------------------------------------------------------------------------------

/// A key plus the `num_blocks` it was generated for — the on-disk envelope for both key kinds.
///
/// `num_blocks` is the compatibility tag: a persisted key only works with a model whose graph
/// declares the same radix width. Storing it lets [`load_client_key`]/[`load_server_key`] and
/// the callers that pair a key with a graph detect a mismatch up front.
#[derive(Serialize, Deserialize)]
struct TaggedKey<K> {
    num_blocks: usize,
    key: K,
}

fn write_bincode<T: Serialize>(value: &T, path: &Path, what: &str) -> Result<(), String> {
    let bytes = bincode::serialize(value).map_err(|e| format!("cannot serialize {what}: {e}"))?;
    std::fs::write(path, bytes)
        .map_err(|e| format!("cannot write {what} to {}: {e}", path.display()))
}

fn read_bincode<T: for<'de> Deserialize<'de>>(path: &Path, what: &str) -> Result<T, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read {what} from {}: {e}", path.display()))?;
    bincode::deserialize(&bytes).map_err(|e| {
        format!(
            "cannot deserialize {what} from {} (is it a Penumbra {what} file?): {e}",
            path.display()
        )
    })
}

/// Persist the client secret key to `path`, tagged with `num_blocks`.
///
/// The client key is **secret** (it encrypts and decrypts) — never hand it to the server
/// (`PROJECT.md` §11). Written in bincode; pair with [`load_client_key`].
pub fn save_client_key(ck: &RadixClientKey, num_blocks: usize, path: &Path) -> Result<(), String> {
    write_bincode(
        &TaggedKey {
            num_blocks,
            key: ck.clone(),
        },
        path,
        "client key",
    )
}

/// Load a client secret key from `path`, returning it with the `num_blocks` it was generated for.
///
/// Callers pairing the key with a model must check the returned `num_blocks` equals the graph's
/// (see [`ensure_num_blocks_match`]) — a wrong-width key produces wrong-width ciphertext.
pub fn load_client_key(path: &Path) -> Result<(RadixClientKey, usize), String> {
    let tagged: TaggedKey<RadixClientKey> = read_bincode(path, "client key")?;
    Ok((tagged.key, tagged.num_blocks))
}

/// Persist the public server/evaluation key to `path`, tagged with `num_blocks`.
///
/// The server key is **public** — it enables bootstrapping but cannot decrypt, so it is the
/// only key material the server holds (`PROJECT.md` §11). Written in bincode (server keys are
/// large); pair with [`load_server_key`].
pub fn save_server_key(sk: &ServerKey, num_blocks: usize, path: &Path) -> Result<(), String> {
    write_bincode(
        &TaggedKey {
            num_blocks,
            key: sk.clone(),
        },
        path,
        "server key",
    )
}

/// Load a server key from `path`, returning it with the `num_blocks` it was generated for.
pub fn load_server_key(path: &Path) -> Result<(ServerKey, usize), String> {
    let tagged: TaggedKey<ServerKey> = read_bincode(path, "server key")?;
    Ok((tagged.key, tagged.num_blocks))
}

/// Fail loudly if a key's `num_blocks` does not match the model it is being used with.
///
/// A key generated for one radix width cannot evaluate/decrypt a model of another
/// (`AGENTS.md` §1.4). This is the "key mismatch" failure mode called out in ROADMAP Phase 9;
/// catching it here gives an actionable message instead of a wrong result or a deep panic.
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

    /// Keygen + a signed radix round-trip over the default profile. Confirms the chosen
    /// `num_blocks` actually carries the values we claim (`radix_capacity_bits`).
    #[test]
    fn keygen_signed_roundtrip() {
        let num_blocks = 8; // 16-bit signed range under DEFAULT_PARAMS
        let (ck, _sk) = keygen(num_blocks);

        assert_eq!(radix_capacity_bits(num_blocks), 16);

        for v in [-1234i64, -1, 0, 1, 4321] {
            let ct = ck.encrypt_signed(v);
            let got: i64 = ck.decrypt_signed(&ct);
            assert_eq!(got, v, "signed radix round-trip must be exact");
        }
    }

    /// Persist both keys to disk, reload them, and confirm the reloaded pair still does an exact
    /// signed round trip — and that the `num_blocks` tag survives (the key-reuse story).
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

        // The reloaded client key must decrypt what the original encrypts (same secret key).
        let ct = ck.encrypt_signed(7i64);
        let got: i64 = ck2.decrypt_signed(&ct);
        assert_eq!(got, 7, "reloaded client key must decrypt exactly");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `ensure_num_blocks_match` is the loud "key mismatch" gate (ROADMAP Phase 9).
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
