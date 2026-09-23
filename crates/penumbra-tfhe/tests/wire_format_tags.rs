//! Tests for backend/scheme wire-format tags on keys and ciphertexts (ROADMAP Phase 12.1).
//!
//! Asserts that feeding mismatched key or ciphertext material (e.g. tagged with "ckks")
//! to the TFHE backend fails loudly with an actionable message naming both backends
//! (`AGENTS.md` §1.4), never panics.

use penumbra_core::wire::TaggedCts;
use penumbra_tfhe::encrypt::{deserialize_cts, deserialize_cts_batch};
use penumbra_tfhe::keys::{load_client_key, load_server_key, TaggedKey};

#[test]
fn client_key_scheme_mismatch_fails_loudly() {
    let dir = std::env::temp_dir().join(format!("penumbra_wire_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let key_path = dir.join("fake_ckks_client.key");

    // Persist a key tagged with "ckks"
    let fake_tagged = TaggedKey {
        scheme: "ckks".to_string(),
        num_blocks: 4,
        key: (),
    };
    let bytes = bincode::serialize(&fake_tagged).unwrap();
    std::fs::write(&key_path, bytes).unwrap();

    let err = load_client_key(&key_path).expect_err("mismatched client key must fail");
    assert!(
        err.contains("backend/scheme mismatch"),
        "expected error to mention scheme mismatch: {err}"
    );
    assert!(
        err.contains("expected 'tfhe', found 'ckks'"),
        "expected error to name both schemes: {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn server_key_scheme_mismatch_fails_loudly() {
    let dir = std::env::temp_dir().join(format!("penumbra_wire_test_sk_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let key_path = dir.join("fake_ckks_server.key");

    let fake_tagged = TaggedKey {
        scheme: "ckks".to_string(),
        num_blocks: 4,
        key: (),
    };
    let bytes = bincode::serialize(&fake_tagged).unwrap();
    std::fs::write(&key_path, bytes).unwrap();
    let err = match load_server_key(&key_path) {
        Ok(_) => panic!("mismatched server key must fail"),
        Err(e) => e,
    };
    assert!(
        err.contains("backend/scheme mismatch"),
        "expected error to mention scheme mismatch: {err}"
    );
    assert!(
        err.contains("expected 'tfhe', found 'ckks'"),
        "expected error to name both schemes: {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn ciphertext_scheme_mismatch_fails_loudly() {
    let fake_tagged = TaggedCts {
        scheme: "ckks".to_string(),
        payload: vec![123u8],
    };
    let bytes = bincode::serialize(&fake_tagged).unwrap();

    let err = deserialize_cts(&bytes).expect_err("mismatched single ciphertext must fail");
    assert!(
        err.contains("backend/scheme mismatch"),
        "expected error to mention scheme mismatch: {err}"
    );
    assert!(
        err.contains("expected 'tfhe', but found 'ckks'"),
        "expected error to name both schemes: {err}"
    );

    let err_batch =
        deserialize_cts_batch(&bytes).expect_err("mismatched ciphertext batch must fail");
    assert!(
        err_batch.contains("backend/scheme mismatch"),
        "expected error to mention scheme mismatch: {err_batch}"
    );
    assert!(
        err_batch.contains("expected 'tfhe', but found 'ckks'"),
        "expected error to name both schemes: {err_batch}"
    );
}
