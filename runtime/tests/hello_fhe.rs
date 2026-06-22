//! Hello-FHE: the Phase 0 toolchain proof (ROADMAP.md Phase 0 / Phase 1 spike).
//!
//! Exercises, at the `shortint` layer Penumbra-FHE actually builds on, the two primitives
//! everything is composed from:
//!   - **plaintext-scalar arithmetic** on a ciphertext (the `Linear` core — cheap, no PBS)
//!   - **programmable bootstrapping / LUT** applied to a ciphertext (the
//!     `Activation`/`Requant` core — expensive, dominates runtime)
//!
//! Demonstrates the golden discipline in miniature: the encrypted result must equal the
//! cleartext-integer result **exactly** (`AGENTS.md` §1.1). TFHE is exact — any mismatch
//! would be a bug, never noise.
//!
//! Uses the default secure parameter profile `PARAM_MESSAGE_2_CARRY_2_KS_PBS` (a 2-bit
//! message space — tiny on purpose; small integers are the working range, `PROJECT.md` §9).
//!
//! NOTE: FHE is dramatically faster in `--release`. Run with `cargo test --release` for
//! anything timing-sensitive (`docs/DEVELOPMENT.md`).

use tfhe::shortint::gen_keys;
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;

/// Encrypt → plaintext-scalar mul + add → decrypt; assert it matches cleartext integers.
/// This is the cheap `Linear` pattern: ciphertext combined with *plaintext* weights.
#[test]
fn encrypt_scalar_arithmetic_decrypt_roundtrip() {
    let (client_key, server_key) = gen_keys(PARAM_MESSAGE_2_CARRY_2_KS_PBS);

    // Cleartext reference (the oracle). Kept within the 2-bit message space (0..=3).
    let x: u64 = 1;
    let w: u64 = 2;
    let b: u64 = 1;
    let expected = x * w + b; // 3

    // Encrypted path: scalar mul + add against plaintext constants (no bootstrap).
    let ct = client_key.encrypt(x);
    let prod = server_key.scalar_mul(&ct, w as u8);
    let out = server_key.scalar_add(&prod, b as u8);

    let decrypted = client_key.decrypt(&out);
    assert_eq!(
        decrypted, expected,
        "FHE plaintext-weight arithmetic must equal cleartext integers exactly"
    );
}

/// Build a lookup table and apply it via programmable bootstrapping; assert the decrypted
/// result matches the cleartext table over the whole message space. This is the
/// `Activation`/`Requant` core — how any single-input function is realized under TFHE.
#[test]
fn lookup_table_via_pbs_matches_table() {
    let (client_key, server_key) = gen_keys(PARAM_MESSAGE_2_CARRY_2_KS_PBS);

    // A toy activation-like LUT over the 2-bit message space: f(v) = (v + 1) mod 4.
    let f = |v: u64| (v + 1) % 4;
    let lut = server_key.generate_lookup_table(f);

    for v in 0u64..4 {
        let ct = client_key.encrypt(v);
        let mapped = server_key.apply_lookup_table(&ct, &lut);
        let decrypted = client_key.decrypt(&mapped);
        assert_eq!(
            decrypted,
            f(v),
            "LUT-via-PBS output must match the cleartext table for input {v}"
        );
    }
}
