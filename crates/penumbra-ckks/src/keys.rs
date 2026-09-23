use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::path::Path;
use std::sync::Mutex;

use crate::hal::ActiveBackend;
use crate::params::CkksParams;
use penumbra_core::wire::SchemeHeader;
use poulpy_ckks::api::{
    CKKSAllOpsTmpBytes, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncryptOps, CKKSRotateOps,
};
use poulpy_ckks::layouts::{CKKSCiphertextOwned, CKKSModuleAlloc};
use poulpy_ckks::{CKKSInfos, SetCKKSInfos};
use poulpy_core::api::{GLWEAutomorphismKeyEncryptSk, GLWETensorKeyEncryptSk};
use poulpy_core::layouts::prepared::{
    GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWESecretPrepared,
    GLWESecretPreparedFactory, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory,
};
use poulpy_core::layouts::{
    GLWEAutomorphismKey, GLWESecret, GLWESecretSampling, GLWETensorKey, LWEInfos, ModuleCoreAlloc,
};
use poulpy_core::{Distribution, GetDistributionMut};
use poulpy_hal::api::{ScratchOwnedAlloc, ScratchOwnedBorrow};
use poulpy_hal::layouts::{
    Backend, CyclotomicOrder, HostBytesBackend, Module, ReaderFrom, ScratchOwned, WriterTo,
};
use poulpy_hal::source::Source;
use serde::{Deserialize, Serialize};

pub const SCHEME_CKKS: &str = "ckks";

pub struct CkksClientKey {
    pub(crate) sk_raw:
        GLWESecret<<ActiveBackend as Backend>::OwnedBuf, <ActiveBackend as Backend>::ZnxWord>,
    pub(crate) sk_prepared: GLWESecretPrepared<<ActiveBackend as Backend>::OwnedBuf, ActiveBackend>,
    pub(crate) module: Module<ActiveBackend>,
    pub(crate) scratch: Mutex<ScratchOwned<ActiveBackend>>,
    pub(crate) params: CkksParams,
}

impl std::fmt::Debug for CkksClientKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CkksClientKey")
            .field("params", &self.params)
            .finish()
    }
}

impl CkksClientKey {
    pub fn params(&self) -> CkksParams {
        self.params
    }
}

type RawAtkVec = Vec<(
    i64,
    GLWEAutomorphismKey<<ActiveBackend as Backend>::OwnedBuf, <ActiveBackend as Backend>::ZnxWord>,
)>;

pub struct CkksServerKey {
    pub(crate) module: Module<ActiveBackend>,
    pub(crate) host_module: Module<HostBytesBackend>,
    pub(crate) tsk_raw:
        GLWETensorKey<<ActiveBackend as Backend>::OwnedBuf, <ActiveBackend as Backend>::ZnxWord>,
    pub(crate) tsk: GLWETensorKeyPrepared<<ActiveBackend as Backend>::OwnedBuf, ActiveBackend>,
    pub(crate) atks_raw: RawAtkVec,
    pub(crate) atks: HashMap<
        i64,
        GLWEAutomorphismKeyPrepared<<ActiveBackend as Backend>::OwnedBuf, ActiveBackend>,
    >,
    pub(crate) scratch: Mutex<ScratchOwned<ActiveBackend>>,
    pub(crate) params: CkksParams,
}

impl std::fmt::Debug for CkksServerKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CkksServerKey")
            .field("params", &self.params)
            .finish()
    }
}

impl CkksServerKey {
    pub fn params(&self) -> CkksParams {
        self.params
    }
}

/// Every slot rotation the fixed BSGS schedule can ask for, at any diagonal pattern:
/// the baby steps `1..giant_step` and the giant steps `giant_step, 2*giant_step, …`.
/// Rotation 0 is the identity and needs no key.
pub fn schedule_rotations(p: &CkksParams) -> Vec<i64> {
    (1..p.giant_step as i64)
        .chain((1..(p.lt_slots / p.giant_step) as i64).map(|j| j * p.giant_step as i64))
        .collect()
}

pub(crate) fn alloc_scratch(
    params: &CkksParams,
    module: &Module<ActiveBackend>,
) -> ScratchOwned<ActiveBackend> {
    let mut ct = module.ckks_ciphertext_alloc_from_glwe_infos(&params.glwe_layout());
    ct.set_meta(params.prec().meta);
    let tsk_infos = params.tsk_layout();
    let atk_infos = params.atk_layout();
    let scratch_size =
        module.ckks_all_ops_with_atk_tmp_bytes(&ct, &tsk_infos, &atk_infos, &params.prec());
    let padded_size = scratch_size.max(32 * 1024 * 1024);
    ScratchOwned::<ActiveBackend>::alloc(padded_size)
}

/// Generates a (client_key, server_key) pair for the given CKKS parameter profile.
pub fn keygen(params: &CkksParams) -> Result<(CkksClientKey, CkksServerKey), String> {
    let module = Module::<ActiveBackend>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let mut scratch = alloc_scratch(params, &module);

    let mut seed_sk = [0u8; 32];
    let mut seed_xa = [0u8; 32];
    let mut seed_xe = [0u8; 32];
    getrandom::getrandom(&mut seed_sk)
        .map_err(|e| format!("failed to sample OS randomness for sk: {e}"))?;
    getrandom::getrandom(&mut seed_xa)
        .map_err(|e| format!("failed to sample OS randomness for xa: {e}"))?;
    getrandom::getrandom(&mut seed_xe)
        .map_err(|e| format!("failed to sample OS randomness for xe: {e}"))?;

    let mut src_sk = Source::new(seed_sk);
    let mut src_xa = Source::new(seed_xa);
    let mut src_xe = Source::new(seed_xe);

    let mut sk_raw = module.glwe_secret_alloc_from_infos(&params.glwe_layout());
    module.glwe_secret_fill_ternary_prob(&mut sk_raw, params.secret_ternary_prob, &mut src_sk);

    let mut sk_prepared = module.glwe_secret_prepared_alloc_from_infos(&params.glwe_layout());
    module.glwe_secret_prepare(&mut sk_prepared, &sk_raw);

    let tsk_infos = params.tsk_layout();
    let mut tsk = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
    module.glwe_tensor_key_encrypt_sk(
        &mut tsk,
        &sk_raw,
        &tsk_infos,
        &mut src_xe,
        &mut src_xa,
        &mut scratch.borrow(),
    );

    let mut tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
    module.prepare_tensor_key(&mut tsk_prepared, &tsk, &mut scratch.borrow());

    let atk_infos = params.atk_layout();
    let mut atks_raw = Vec::new();
    let mut atks = HashMap::new();
    for rot in schedule_rotations(params) {
        let galois_el = poulpy_hal::layouts::galois_element(rot, module.cyclotomic_order());
        let mut atk = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
        module.glwe_automorphism_key_encrypt_sk(
            &mut atk,
            galois_el,
            &sk_raw,
            &atk_infos,
            &mut src_xe,
            &mut src_xa,
            &mut scratch.borrow(),
        );

        let mut atk_prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_infos);
        module.glwe_automorphism_key_prepare(&mut atk_prepared, &atk, &mut scratch.borrow());
        atks.insert(galois_el, atk_prepared);
        atks_raw.push((galois_el, atk));
    }

    let client_scratch = alloc_scratch(params, &module);

    Ok((
        CkksClientKey {
            sk_raw,
            sk_prepared,
            module: Module::<ActiveBackend>::new(params.n as u64),
            scratch: Mutex::new(client_scratch),
            params: *params,
        },
        CkksServerKey {
            module,
            host_module,
            tsk_raw: tsk,
            tsk: tsk_prepared,
            atks_raw,
            atks,
            scratch: Mutex::new(scratch),
            params: *params,
        },
    ))
}

pub fn encrypt_raw(
    ck: &CkksClientKey,
    values: &[f64],
) -> Result<CKKSCiphertextOwned<ActiveBackend>, String> {
    let total_slots = ck.params.lt_slots;
    if values.len() > total_slots {
        return Err(format!(
            "input length {} exceeds lt_slots {}",
            values.len(),
            total_slots
        ));
    }
    let mut re = vec![0.0f64; total_slots];
    re[..values.len()].copy_from_slice(values);
    let im = vec![0.0f64; total_slots];

    let mut pt = ck
        .module
        .ckks_pt_vec_alloc(ck.params.base2k.into(), ck.params.k.into());
    pt.set_meta(ck.params.prec().meta);

    let mut scratch_guard = ck
        .scratch
        .lock()
        .map_err(|e| format!("mutex poisoned: {e}"))?;

    ck.module
        .ckks_encode_reim_into(&mut pt, &re, &im, &mut scratch_guard.borrow())
        .map_err(|e| format!("CKKS encode failed: {e}"))?;

    let mut ct = ck
        .module
        .ckks_ciphertext_alloc_from_glwe_infos(&ck.params.glwe_layout());

    let mut seed_xa = [0u8; 32];
    let mut seed_xe = [0u8; 32];
    getrandom::getrandom(&mut seed_xa)
        .map_err(|e| format!("failed to sample OS randomness for xa: {e}"))?;
    getrandom::getrandom(&mut seed_xe)
        .map_err(|e| format!("failed to sample OS randomness for xe: {e}"))?;
    let mut src_xa = Source::new(seed_xa);
    let mut src_xe = Source::new(seed_xe);

    ck.module
        .ckks_encrypt_sk(
            &mut ct,
            &pt,
            &ck.sk_prepared,
            &ck.params.glwe_layout(),
            &mut src_xe,
            &mut src_xa,
            &mut scratch_guard.borrow(),
        )
        .map_err(|e| format!("CKKS encrypt failed: {e}"))?;

    Ok(ct)
}

pub fn decrypt_raw(
    ck: &CkksClientKey,
    ct: &CKKSCiphertextOwned<ActiveBackend>,
) -> Result<Vec<f64>, String> {
    let total_slots = ck.params.lt_slots;
    let log_budget = ct.log_budget().min(127usize.saturating_sub(ct.log_delta()));
    let dec_k = (ct.log_delta() + log_budget).min(127);
    let mut pt = ck.module.ckks_pt_vec_alloc(ct.base2k(), dec_k.into());
    pt.set_meta(ct.meta());

    let mut scratch_guard = ck
        .scratch
        .lock()
        .map_err(|e| format!("mutex poisoned: {e}"))?;

    ck.module
        .ckks_decrypt(&mut pt, ct, &ck.sk_prepared, &mut scratch_guard.borrow())
        .map_err(|e| format!("CKKS decrypt failed: {e}"))?;

    let mut re = vec![0.0f64; total_slots];
    let mut im = vec![0.0f64; total_slots];

    ck.module
        .ckks_decode_reim_into(&pt, &mut re, &mut im, &mut scratch_guard.borrow())
        .map_err(|e| format!("CKKS decode failed: {e}"))?;

    Ok(re)
}

pub fn rotate_raw(
    sk: &CkksServerKey,
    ct: &CKKSCiphertextOwned<ActiveBackend>,
    rot: i64,
) -> Result<CKKSCiphertextOwned<ActiveBackend>, String> {
    let mut out = sk
        .module
        .ckks_ciphertext_alloc_from_glwe_infos(&sk.params.glwe_layout());
    let mut scratch_guard = sk
        .scratch
        .lock()
        .map_err(|e| format!("mutex poisoned: {e}"))?;
    sk.module
        .ckks_rotate_into(&mut out, ct, rot, &sk.atks, &mut scratch_guard.borrow())
        .map_err(|e| format!("CKKS rotate failed: {e}"))?;
    Ok(out)
}

// ─── Key Serialization & Wire Format ──────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct TaggedKey<K> {
    pub scheme: String,
    pub payload: K,
}

#[derive(Serialize, Deserialize)]
pub struct CkksClientKeyPayload {
    pub params: CkksParams,
    pub sk_bytes: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
pub struct CkksServerKeyPayload {
    pub params: CkksParams,
    pub tsk_bytes: Vec<u8>,
    pub atks: Vec<(i64, Vec<u8>)>,
}

/// Serialize the client secret key to its tagged wire bytes (for size accounting and persistence).
pub fn client_key_bytes(ck: &CkksClientKey) -> Result<Vec<u8>, String> {
    let mut sk_bytes = Vec::new();
    ck.sk_raw
        .data()
        .write_to(&mut sk_bytes)
        .map_err(|e| format!("failed to serialize secret key: {e}"))?;

    let payload = CkksClientKeyPayload {
        params: ck.params,
        sk_bytes,
    };
    let tagged = TaggedKey {
        scheme: SCHEME_CKKS.to_string(),
        payload,
    };
    bincode::serialize(&tagged).map_err(|e| format!("bincode encode failed: {e}"))
}

pub fn save_client_key(ck: &CkksClientKey, path: impl AsRef<Path>) -> Result<(), String> {
    let encoded = client_key_bytes(ck)?;
    let mut file = File::create(path).map_err(|e| format!("cannot create file: {e}"))?;
    file.write_all(&encoded)
        .map_err(|e| format!("cannot write file: {e}"))?;
    Ok(())
}

pub fn client_key_from_bytes(bytes: &[u8]) -> Result<CkksClientKey, String> {
    let header: SchemeHeader =
        bincode::deserialize(bytes).map_err(|e| format!("failed to parse key header: {e}"))?;
    if header.scheme != SCHEME_CKKS {
        return Err(format!(
            "backend/scheme mismatch for client key: expected '{SCHEME_CKKS}', found '{}' (key material is not portable across backends; see docs/BACKENDS.md)",
            header.scheme
        ));
    }

    let tagged: TaggedKey<CkksClientKeyPayload> = bincode::deserialize(bytes)
        .map_err(|e| format!("failed to deserialize client key: {e}"))?;

    let params = tagged.payload.params;
    let module = Module::<ActiveBackend>::new(params.n as u64);
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&params.glwe_layout());
    let mut cursor = Cursor::new(tagged.payload.sk_bytes);
    sk_raw
        .data_mut()
        .read_from(&mut cursor)
        .map_err(|e| format!("failed to read secret key: {e}"))?;
    *sk_raw.dist_mut() = Distribution::TernaryProb(params.secret_ternary_prob);

    let mut sk_prepared = module.glwe_secret_prepared_alloc_from_infos(&params.glwe_layout());
    module.glwe_secret_prepare(&mut sk_prepared, &sk_raw);

    let scratch = alloc_scratch(&params, &module);

    Ok(CkksClientKey {
        sk_raw,
        sk_prepared,
        module,
        scratch: Mutex::new(scratch),
        params,
    })
}

pub fn load_client_key(path: impl AsRef<Path>) -> Result<CkksClientKey, String> {
    let mut file = File::open(&path).map_err(|e| format!("cannot open key file: {e}"))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read key file: {e}"))?;
    client_key_from_bytes(&bytes).map_err(|e| format!("{e} (at {})", path.as_ref().display()))
}

/// Serialize the public server/evaluation key to its tagged wire bytes (for size accounting and persistence).
pub fn server_key_bytes(sk: &CkksServerKey) -> Result<Vec<u8>, String> {
    let mut tsk_bytes = Vec::new();
    sk.tsk_raw
        .write_to(&mut tsk_bytes)
        .map_err(|e| format!("failed to serialize tensor key: {e}"))?;

    let mut atks_serialized = Vec::new();
    for (p, atk_raw) in &sk.atks_raw {
        let mut atk_bytes = Vec::new();
        atk_raw
            .write_to(&mut atk_bytes)
            .map_err(|e| format!("failed to serialize automorphism key p={p}: {e}"))?;
        atks_serialized.push((*p, atk_bytes));
    }

    let payload = CkksServerKeyPayload {
        params: sk.params,
        tsk_bytes,
        atks: atks_serialized,
    };
    let tagged = TaggedKey {
        scheme: SCHEME_CKKS.to_string(),
        payload,
    };
    bincode::serialize(&tagged).map_err(|e| format!("bincode encode failed: {e}"))
}

pub fn save_server_key(sk: &CkksServerKey, path: impl AsRef<Path>) -> Result<(), String> {
    let encoded = server_key_bytes(sk)?;
    let mut file = File::create(path).map_err(|e| format!("cannot create file: {e}"))?;
    file.write_all(&encoded)
        .map_err(|e| format!("cannot write file: {e}"))?;
    Ok(())
}

pub fn server_key_from_bytes(bytes: &[u8]) -> Result<CkksServerKey, String> {
    let header: SchemeHeader =
        bincode::deserialize(bytes).map_err(|e| format!("failed to parse key header: {e}"))?;
    if header.scheme != SCHEME_CKKS {
        return Err(format!(
            "backend/scheme mismatch for server key: expected '{SCHEME_CKKS}', found '{}' (key material is not portable across backends; see docs/BACKENDS.md)",
            header.scheme
        ));
    }

    let tagged: TaggedKey<CkksServerKeyPayload> = bincode::deserialize(bytes)
        .map_err(|e| format!("failed to deserialize server key: {e}"))?;

    let params = tagged.payload.params;
    let module = Module::<ActiveBackend>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let mut scratch = alloc_scratch(&params, &module);

    let tsk_infos = params.tsk_layout();
    let mut tsk_raw = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
    let mut tsk_cursor = Cursor::new(tagged.payload.tsk_bytes);
    tsk_raw
        .read_from(&mut tsk_cursor)
        .map_err(|e| format!("failed to read tensor key: {e}"))?;

    let mut tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
    module.prepare_tensor_key(&mut tsk_prepared, &tsk_raw, &mut scratch.borrow());

    let atk_infos = params.atk_layout();
    let mut atks = HashMap::new();
    let mut atks_raw = Vec::new();
    for (p, atk_bytes) in tagged.payload.atks {
        let mut atk = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
        let mut atk_cursor = Cursor::new(atk_bytes);
        atk.read_from(&mut atk_cursor)
            .map_err(|e| format!("failed to read automorphism key p={p}: {e}"))?;

        let mut atk_prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_infos);
        module.glwe_automorphism_key_prepare(&mut atk_prepared, &atk, &mut scratch.borrow());
        atks.insert(p, atk_prepared);
        atks_raw.push((p, atk));
    }

    Ok(CkksServerKey {
        module,
        host_module,
        tsk_raw,
        tsk: tsk_prepared,
        atks_raw,
        atks,
        scratch: Mutex::new(scratch),
        params,
    })
}

pub fn load_server_key(path: impl AsRef<Path>) -> Result<CkksServerKey, String> {
    let mut file = File::open(&path).map_err(|e| format!("cannot open key file: {e}"))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read key file: {e}"))?;
    server_key_from_bytes(&bytes).map_err(|e| format!("{e} (at {})", path.as_ref().display()))
}

pub fn ensure_params_match(key: &CkksParams, run: &CkksParams) -> Result<(), String> {
    if key != run {
        if key.max_poly_degree != run.max_poly_degree
            && key.n == run.n
            && key.base2k == run.base2k
            && key.k == run.k
            && key.log_delta == run.log_delta
            && key.dsize == run.dsize
            && key.rank == run.rank
            && key.lt_slots == run.lt_slots
            && key.giant_step == run.giant_step
            && (key.secret_ternary_prob - run.secret_ternary_prob).abs() < 1e-9
        {
            return Err(format!(
                "key/profile mismatch: this key was generated under CKKS max_poly_degree={}, but this run uses {}. Keys are tied to their parameter profile — regenerate keys for this profile.",
                key.max_poly_degree, run.max_poly_degree
            ));
        }
        return Err(format!(
            "key/profile mismatch: this key was generated under CKKS params {key:?}, but this run uses {run:?}. Keys are tied to their parameter profile — regenerate keys for this profile."
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::DEFAULT_PARAMS;
    #[test]
    fn test_keygen_encrypt_decrypt_rotate() {
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
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let ct = encrypt_raw(&ck, &input).expect("encrypt failed");
        let decrypted = decrypt_raw(&ck, &ct).expect("decrypt failed");

        for (i, (&want, &got)) in input.iter().zip(&decrypted).enumerate() {
            assert!(
                (want - got).abs() < 1e-3,
                "slot {i}: want {want}, got {got}"
            );
        }

        // Test rotation by 1
        let rotated_ct = rotate_raw(&sk, &ct, 1).expect("rotate failed");
        let rotated_dec = decrypt_raw(&ck, &rotated_ct).expect("decrypt rotated failed");
        // In CKKS rotation by 1 shifts slots: slot 0 receives slot 1 (2.0), slot 1 receives slot 2 (3.0), etc.
        assert!(
            (rotated_dec[0] - 2.0).abs() < 1e-3,
            "rotated slot 0: got {}, want 2.0",
            rotated_dec[0]
        );
        assert!(
            (rotated_dec[1] - 3.0).abs() < 1e-3,
            "rotated slot 1: got {}, want 3.0",
            rotated_dec[1]
        );
    }

    #[test]
    fn test_client_and_server_key_bytes_roundtrip() {
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
        assert_eq!(ck.params(), params);
        assert_eq!(sk.params(), params);

        let ck_bytes = client_key_bytes(&ck).expect("client_key_bytes failed");
        let sk_bytes = server_key_bytes(&sk).expect("server_key_bytes failed");

        let loaded_ck = client_key_from_bytes(&ck_bytes).expect("client_key_from_bytes failed");
        let loaded_sk = server_key_from_bytes(&sk_bytes).expect("server_key_from_bytes failed");

        assert_eq!(loaded_ck.params(), params);
        assert_eq!(loaded_sk.params(), params);

        let input = vec![10.0, -20.0, 30.0];
        let ct = encrypt_raw(&loaded_ck, &input).expect("encrypt failed");
        let rotated = rotate_raw(&loaded_sk, &ct, 1).expect("rotate failed");
        let dec = decrypt_raw(&loaded_ck, &rotated).expect("decrypt failed");
        assert!((dec[0] - (-20.0)).abs() < 1e-3);
    }

    #[test]
    fn test_ensure_params_match() {
        let p1 = CkksParams {
            max_poly_degree: 7,
            ..DEFAULT_PARAMS
        };
        let p2 = CkksParams {
            max_poly_degree: 15,
            ..DEFAULT_PARAMS
        };
        assert!(ensure_params_match(&p1, &p1).is_ok());
        let err = ensure_params_match(&p1, &p2).unwrap_err();
        assert!(
            err.contains("key/profile mismatch: this key was generated under CKKS max_poly_degree=7, but this run uses 15"),
            "unexpected error message: {err}"
        );

        let mut p3 = p1;
        p3.n = 8192;
        let err2 = ensure_params_match(&p1, &p3).unwrap_err();
        assert!(
            err2.contains("under CKKS params"),
            "unexpected error message: {err2}"
        );

        assert!(p1.with_max_poly_degree(0).is_err());
        assert_eq!(p1.with_max_poly_degree(9).unwrap().max_poly_degree, 9);
    }
}
