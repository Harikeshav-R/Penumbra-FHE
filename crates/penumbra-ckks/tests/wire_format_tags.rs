#![cfg(feature = "ckks")]

//! Tests for backend/scheme wire-format tags on keys and ciphertexts.
//!
//! Asserts that feeding mismatched key or ciphertext material (e.g. tagged with "tfhe")
//! to the CKKS backend fails loudly with an actionable message naming both backends
//! (`AGENTS.md` §1.4), never panics.

use penumbra_ckks::encrypt::{
    decrypt_vec, deserialize_cts, deserialize_cts_batch, encrypt, serialize_cts,
    serialize_cts_batch, TaggedCts,
};
use penumbra_ckks::keys::{
    keygen, load_client_key, load_server_key, rotate_raw, save_client_key, save_server_key,
    TaggedKey,
};
use penumbra_ckks::params::CkksParams;

#[test]
fn client_key_scheme_mismatch_fails_loudly() {
    let dir = std::env::temp_dir().join(format!("penumbra_ckks_wire_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let key_path = dir.join("fake_tfhe_client.key");

    // Persist a key tagged with "tfhe"
    let fake_tagged = TaggedKey {
        scheme: "tfhe".to_string(),
        payload: (),
    };
    let bytes = bincode::serialize(&fake_tagged).unwrap();
    std::fs::write(&key_path, bytes).unwrap();

    let err = load_client_key(&key_path).expect_err("mismatched client key must fail");
    assert!(
        err.contains("backend/scheme mismatch"),
        "expected error to mention scheme mismatch: {err}"
    );
    assert!(
        err.contains("expected 'ckks', found 'tfhe'"),
        "expected error to name both schemes: {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn server_key_scheme_mismatch_fails_loudly() {
    let dir =
        std::env::temp_dir().join(format!("penumbra_ckks_wire_test_sk_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let key_path = dir.join("fake_tfhe_server.key");

    let fake_tagged = TaggedKey {
        scheme: "tfhe".to_string(),
        payload: (),
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
        err.contains("expected 'ckks', found 'tfhe'"),
        "expected error to name both schemes: {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn ciphertext_scheme_mismatch_fails_loudly() {
    let fake_tagged = TaggedCts {
        scheme: "tfhe".to_string(),
        payload: vec![123u8],
    };
    let bytes = bincode::serialize(&fake_tagged).unwrap();

    let err = deserialize_cts(&bytes).expect_err("mismatched single ciphertext must fail");
    assert!(
        err.contains("backend/scheme mismatch"),
        "expected error to mention scheme mismatch: {err}"
    );
    assert!(
        err.contains("expected 'ckks', but found 'tfhe'"),
        "expected error to name both schemes: {err}"
    );

    let err_batch =
        deserialize_cts_batch(&bytes).expect_err("mismatched ciphertext batch must fail");
    assert!(
        err_batch.contains("backend/scheme mismatch"),
        "expected error to mention scheme mismatch: {err_batch}"
    );
    assert!(
        err_batch.contains("expected 'ckks', but found 'tfhe'"),
        "expected error to name both schemes: {err_batch}"
    );
}

#[test]
fn test_key_save_load_roundtrip() {
    let dir = std::env::temp_dir().join(format!("penumbra_ckks_key_rt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ck_path = dir.join("client.key");
    let sk_path = dir.join("server.key");

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

    let (ck, sk) = keygen(&params).expect("keygen failed");
    save_client_key(&ck, &ck_path).expect("save_client_key failed");
    save_server_key(&sk, &sk_path).expect("save_server_key failed");

    let loaded_ck = load_client_key(&ck_path).expect("load_client_key failed");
    let loaded_sk = load_server_key(&sk_path).expect("load_server_key failed");

    let input = vec![10, 20, -30, 40];
    let ct = encrypt(&loaded_ck, &input);
    let dec = decrypt_vec(&loaded_ck, &ct);
    assert_eq!(dec, input);

    // Test rotating with loaded server key
    let rotated = rotate_raw(&loaded_sk, ct[0].ct(), 1).expect("rotate failed");
    let rotated_ct = vec![penumbra_ckks::encrypt::CkksCt::new(rotated, input.len())];
    let rotated_dec = decrypt_vec(&loaded_ck, &rotated_ct);
    assert_eq!(rotated_dec[0], 20);
    assert_eq!(rotated_dec[1], -30);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_ciphertext_serialize_deserialize_roundtrip() {
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
    let input = vec![5, -15, 25, -35, 45];
    let ct = encrypt(&ck, &input);

    let serialized = serialize_cts(&ct).expect("serialize_cts failed");
    let deserialized = deserialize_cts(&serialized).expect("deserialize_cts failed");
    assert_eq!(deserialized.len(), 1);
    assert_eq!(deserialized[0].len(), input.len());

    let dec = decrypt_vec(&ck, &deserialized);
    assert_eq!(dec, input);

    let batch = vec![ct.clone(), ct];
    let batch_serialized = serialize_cts_batch(&batch).expect("serialize_cts_batch failed");
    let batch_deserialized =
        deserialize_cts_batch(&batch_serialized).expect("deserialize_cts_batch failed");
    assert_eq!(batch_deserialized.len(), 2);
    assert_eq!(decrypt_vec(&ck, &batch_deserialized[0]), input);
    assert_eq!(decrypt_vec(&ck, &batch_deserialized[1]), input);
}
