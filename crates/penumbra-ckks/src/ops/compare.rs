use poulpy_ckks::approximation::{precision_at_depth, Parity};
use poulpy_ckks::polynomial::SplitStrategy;
use poulpy_core::layouts::bsgs_eval_depth;
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::encrypt::CkksCt;
use crate::hal::ActiveBackend;
use crate::keys::CkksServerKey;
use crate::ops::matvec::{eval_matvec, prepare_linear_map, PlainMatrix, PreparedLinearMap};
use crate::ops::polymap::{eval_polymap, PolyMap};
use crate::params::CkksParams;

pub struct PreparedCompare {
    pub(crate) prepared: PreparedLinearMap,
    pub(crate) pm: PolyMap,
    pub(crate) len: usize,
}

impl PreparedCompare {
    pub fn depth(&self) -> usize {
        1 + self.pm.depth
    }

    pub fn rotation_count(&self) -> usize {
        self.prepared.rotation_count()
    }
}

pub fn compare_matrix(indices: &[usize], thresholds: &[i64]) -> PlainMatrix {
    let rows = indices.len();
    let cols = indices.iter().max().map(|&m| m + 1).unwrap_or(0);
    let mut data = vec![0.0f64; rows * cols];
    for (i, &idx) in indices.iter().enumerate() {
        data[i * cols + idx] = 1.0;
    }
    let bias = thresholds.iter().map(|&t| -(t as f64)).collect();
    PlainMatrix {
        rows,
        cols,
        data,
        bias,
    }
}

pub fn prepare_compare(
    params: &CkksParams,
    module: &Module<ActiveBackend>,
    scratch: &mut ScratchArena<'_, ActiveBackend>,
    indices: &[usize],
    thresholds: &[i64],
    input_bits: usize,
    max_depth: usize,
) -> Result<PreparedCompare, String> {
    let mat = compare_matrix(indices, thresholds);
    let prepared = prepare_linear_map(&mat, params, module, scratch)?;

    let tau = 0.45f64;
    let f = move |t: f64| -> f64 {
        let d = t + 0.5;
        if d < -tau {
            0.0
        } else if d > tau {
            1.0
        } else {
            0.5 + 0.5 * (d / tau)
        }
    };
    let x_max = (1i64 << input_bits) as f64;
    let t_max = thresholds.iter().map(|&t| t.abs()).max().unwrap_or(0) as f64;
    let (lo, hi) = (-(x_max + t_max), x_max + t_max);
    let choice = precision_at_depth(
        f,
        lo,
        hi,
        Parity::Full,
        max_depth,
        params.max_poly_degree,
        SplitStrategy::MinDepth,
    )
    .map_err(|e| format!("cannot fit compare step polynomial: {e}"))?;

    let depth = bsgs_eval_depth(choice.degree, SplitStrategy::MinDepth);

    Ok(PreparedCompare {
        prepared,
        pm: PolyMap {
            poly: choice.minimax.poly,
            depth,
            fit_error: choice.minimax.error,
        },
        len: indices.len(),
    })
}

pub fn eval_compare(
    sk: &CkksServerKey,
    x: &CkksCt,
    prepared: &PreparedCompare,
) -> Result<CkksCt, String> {
    let diff = eval_matvec(sk, x, &prepared.prepared)?;
    let out = eval_polymap(sk, &diff, &prepared.pm)?;
    Ok(CkksCt {
        ct: out.ct,
        len: prepared.len,
    })
}
