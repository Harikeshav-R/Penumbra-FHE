use poulpy_ckks::api::CKKSAddOps;
use poulpy_ckks::layouts::CKKSModuleAlloc;
use poulpy_hal::api::ScratchOwnedBorrow;

use crate::encrypt::CkksCt;
use crate::keys::CkksServerKey;

pub fn eval_add(sk: &CkksServerKey, a: &CkksCt, b: &CkksCt) -> Result<CkksCt, String> {
    if a.len != b.len {
        return Err(format!(
            "cannot add ciphertexts with different lengths: {} vs {}",
            a.len, b.len
        ));
    }
    let mut out = sk
        .module
        .ckks_ciphertext_alloc_from_glwe_infos(&sk.params.glwe_layout());
    let mut scratch_guard = sk
        .scratch
        .lock()
        .map_err(|e| format!("mutex poisoned: {e}"))?;

    sk.module
        .ckks_add_into(&mut out, &a.ct, &b.ct, &mut scratch_guard.borrow())
        .map_err(|e| format!("add failed: {e}"))?;

    Ok(CkksCt {
        ct: out,
        len: a.len,
    })
}
