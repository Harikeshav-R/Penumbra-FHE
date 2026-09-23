use poulpy_ckks::approximation::{precision_at_depth, Parity};
use poulpy_ckks::polynomial::SplitStrategy;
use poulpy_core::layouts::bsgs_eval_depth;
use poulpy_hal::layouts::{HostBytesBackend, Module};

use crate::encrypt::CkksCt;
use crate::keys::CkksServerKey;
use crate::ops::polymap::{eval_polymap, PolyMap};
use crate::params::CkksParams;

pub struct PreparedArgmax {
    pub(crate) pm: PolyMap,
    pub threshold: i64,
}

impl PreparedArgmax {
    pub fn depth(&self) -> usize {
        self.pm.depth
    }
}
pub fn prepare_argmax(
    _host_module: &Module<HostBytesBackend>,
    params: &CkksParams,
    threshold: i64,
    input_bits: usize,
    max_depth: usize,
) -> Result<PreparedArgmax, String> {
    let x_max = (1i64 << (input_bits.saturating_sub(1).max(1))) as f64;
    let lo = -x_max;
    let hi = x_max;
    let tau = (x_max * 0.05).max(1.0);
    let thresh = threshold as f64;

    let f = move |t: f64| -> f64 {
        let diff = t - thresh;
        if diff < -tau {
            0.0
        } else if diff > tau {
            1.0
        } else {
            0.5 + 0.5 * (diff / tau)
        }
    };

    let choice = precision_at_depth(
        f,
        lo,
        hi,
        Parity::Full,
        max_depth,
        params.max_poly_degree,
        SplitStrategy::MinDepth,
    )
    .map_err(|e| format!("cannot fit argmax step polynomial: {e}"))?;

    let depth = bsgs_eval_depth(choice.degree, SplitStrategy::MinDepth);

    Ok(PreparedArgmax {
        pm: PolyMap {
            poly: choice.minimax.poly,
            depth,
            fit_error: choice.minimax.error,
        },
        threshold,
    })
}

pub fn eval_argmax(
    sk: &CkksServerKey,
    x: &CkksCt,
    prepared: &PreparedArgmax,
) -> Result<CkksCt, String> {
    if x.len != 1 {
        return Err(format!(
            "Argmax currently supports single-element inputs, got len {}",
            x.len
        ));
    }
    let out = eval_polymap(sk, x, &prepared.pm)?;
    Ok(CkksCt { ct: out.ct, len: 1 })
}
