use penumbra_runtime::{
    add, decrypt, encrypt, keygen, scalar_mul, sub, ClientKey, SecurityParams, ServerKey,
};
use proptest::prelude::*;
use std::cell::Cell;
use std::sync::OnceLock;

static KEYS: OnceLock<(ClientKey, ServerKey)> = OnceLock::new();

fn get_keys() -> &'static (ClientKey, ServerKey) {
    KEYS.get_or_init(|| {
        let params = SecurityParams { rng_seed: 42 };
        keygen(params).unwrap()
    })
}

fn ensure_server_key() {
    thread_local! {
        static IS_SET: Cell<bool> = const { Cell::new(false) };
    }
    IS_SET.with(|set| {
        if !set.get() {
            let (_, server_key) = get_keys();
            penumbra_runtime::set_server_key(server_key);
            set.set(true);
        }
    });
}

// We use 100 cases as the baseline for CI as specified by the DoD
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_encrypted_add(x in 0..1000u32, y in 0..1000u32) {
        ensure_server_key();
        let (client_key, _) = get_keys();

        let ct_x = encrypt(x, client_key);
        let ct_y = encrypt(y, client_key);

        let ct_res = add(&ct_x, &ct_y).unwrap();
        let res = decrypt(&ct_res, client_key);

        assert_eq!(res, x + y);
    }

    #[test]
    fn test_encrypted_sub(x in 1000..2000u32, y in 0..1000u32) {
        ensure_server_key();
        let (client_key, _) = get_keys();

        let ct_x = encrypt(x, client_key);
        let ct_y = encrypt(y, client_key);

        let ct_res = sub(&ct_x, &ct_y).unwrap();
        let res = decrypt(&ct_res, client_key);

        assert_eq!(res, x - y);
    }

    #[test]
    fn test_encrypted_scalar_mul(x in 0..100u32, k in 0..10u32) {
        ensure_server_key();
        let (client_key, _) = get_keys();

        let ct_x = encrypt(x, client_key);

        let ct_res = scalar_mul(&ct_x, k).unwrap();
        let res = decrypt(&ct_res, client_key);

        assert_eq!(res, x * k);
    }
}
