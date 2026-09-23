use poulpy_ckks::api::{
    CKKSAddOps, CKKSEncodingHostOps, CKKSLinearTransformationOps, LinearTransformationStrategy,
};
use poulpy_ckks::default::ckks_encode_linear_transformation_from_diagonals;
use poulpy_ckks::layouts::{CKKSModuleAlloc, ComplexDiagonals};
use poulpy_ckks::{CKKSInfos, SetCKKSInfos};
use poulpy_core::layouts::Diagonals;
use poulpy_core::layouts::{Base2K, LWEInfos};
use poulpy_hal::api::ScratchOwnedBorrow;
use poulpy_hal::layouts::{CyclotomicOrder, Module, ScratchArena};

use crate::encrypt::CkksCt;
use crate::hal::ActiveBackend;
use crate::keys::CkksServerKey;
use crate::params::CkksParams;

/// A plaintext linear map `y[i] = Σ_j m[i][j] · x[j] + bias[i]`, in raw integer units.
pub struct PlainMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
    pub bias: Vec<f64>,
}

pub fn linear_matrix(weights: &[Vec<i64>], bias: &[i64]) -> PlainMatrix {
    let rows = weights.len();
    let cols = if rows > 0 { weights[0].len() } else { 0 };
    let mut data = Vec::with_capacity(rows * cols);
    for row in weights {
        for &w in row {
            data.push(w as f64);
        }
    }
    let bias_f64 = bias.iter().map(|&b| b as f64).collect();
    PlainMatrix {
        rows,
        cols,
        data,
        bias: bias_f64,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_matrix(
    weights: &[Vec<i64>],
    bias: &[i64],
    in_h: usize,
    in_w: usize,
    in_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
) -> Result<PlainMatrix, String> {
    let out_channels = weights.len();
    let out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    let out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
    let rows = out_channels * out_h * out_w;
    let cols = in_channels * in_h * in_w;
    let in_hw = in_h * in_w;

    let mut data = vec![0.0f64; rows * cols];
    let mut bias_f64 = vec![0.0f64; rows];

    for (oc, (kernel, &b)) in weights.iter().zip(bias).enumerate() {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let row_idx = oc * (out_h * out_w) + oy * out_w + ox;
                bias_f64[row_idx] = b as f64;
                for ic in 0..in_channels {
                    for ky in 0..kernel_h {
                        let iy = (oy * stride + ky) as isize - padding as isize;
                        for kx in 0..kernel_w {
                            let ix = (ox * stride + kx) as isize - padding as isize;
                            if iy < 0 || ix < 0 || iy as usize >= in_h || ix as usize >= in_w {
                                continue;
                            }
                            let w = kernel[(ic * kernel_h + ky) * kernel_w + kx];
                            let col_idx = ic * in_hw + (iy as usize) * in_w + (ix as usize);
                            data[row_idx * cols + col_idx] += w as f64;
                        }
                    }
                }
            }
        }
    }

    Ok(PlainMatrix {
        rows,
        cols,
        data,
        bias: bias_f64,
    })
}

pub fn avg_pool_matrix(
    channels: usize,
    in_h: usize,
    in_w: usize,
    pool_h: usize,
    pool_w: usize,
    stride: usize,
) -> Result<PlainMatrix, String> {
    let out_h = (in_h - pool_h) / stride + 1;
    let out_w = (in_w - pool_w) / stride + 1;
    let rows = channels * out_h * out_w;
    let cols = channels * in_h * in_w;

    let mut data = vec![0.0f64; rows * cols];
    let bias_f64 = vec![0.0f64; rows];

    for c in 0..channels {
        let base_in = c * in_h * in_w;
        let base_out = c * out_h * out_w;
        for oy in 0..out_h {
            for ox in 0..out_w {
                let row_idx = base_out + oy * out_w + ox;
                for ky in 0..pool_h {
                    for kx in 0..pool_w {
                        let y = oy * stride + ky;
                        let xx = ox * stride + kx;
                        let col_idx = base_in + y * in_w + xx;
                        data[row_idx * cols + col_idx] += 1.0;
                    }
                }
            }
        }
    }

    Ok(PlainMatrix {
        rows,
        cols,
        data,
        bias: bias_f64,
    })
}

pub struct PreparedLinearMap {
    pub rows: usize,
    pub cols: usize,
    pub bias: Vec<f64>,
    pub cd: ComplexDiagonals<f64>,
    pub rotation_count: usize,
}

impl PreparedLinearMap {
    pub fn rotation_count(&self) -> usize {
        self.rotation_count
    }
}

pub fn prepare_linear_map(
    m: &PlainMatrix,
    params: &CkksParams,
    module: &Module<ActiveBackend>,
    _scratch: &mut ScratchArena<'_, ActiveBackend>,
) -> Result<PreparedLinearMap, String> {
    if m.rows > params.lt_slots || m.cols > params.lt_slots {
        return Err(format!(
            "matrix dimension ({}, {}) exceeds backend lt_slots {}",
            m.rows, m.cols, params.lt_slots
        ));
    }

    let mut re = Diagonals::new(params.lt_slots);
    for d in 0..params.lt_slots {
        let mut has_nonzero = false;
        let mut v = vec![0.0f64; params.lt_slots];
        for i in 0..m.rows {
            let col = (i + d) % params.lt_slots;
            if col < m.cols {
                let val = m.data[i * m.cols + col];
                if val != 0.0 {
                    has_nonzero = true;
                    v[i] = val;
                }
            }
        }
        if has_nonzero {
            re.set(d as i64, v);
        }
    }

    let im = Diagonals::new(params.lt_slots);
    let cd = ComplexDiagonals::new(re, im);

    let plan = poulpy_core::layouts::LinearTransformationLayout {
        indexes: cd.indexes(),
        slots: params.lt_slots,
        strategy: LinearTransformationStrategy::Bsgs {
            giant_step: params.giant_step,
        },
    }
    .index();
    let rotation_count = plan.galois_elements(module.cyclotomic_order()).len();

    Ok(PreparedLinearMap {
        rows: m.rows,
        cols: m.cols,
        bias: m.bias.clone(),
        cd,
        rotation_count,
    })
}

pub fn eval_matvec(
    sk: &CkksServerKey,
    x: &CkksCt,
    prepared: &PreparedLinearMap,
) -> Result<CkksCt, String> {
    let mut out = sk
        .module
        .ckks_ciphertext_alloc_from_glwe_infos(&sk.params.glwe_layout());
    let mut scratch_guard = sk
        .scratch
        .lock()
        .map_err(|e| format!("mutex poisoned: {e}"))?;

    let log_delta = x.ct.log_delta();
    let coeff_budget = x.ct.log_budget().saturating_sub(log_delta);
    let coeff_meta = poulpy_ckks::CoeffsMeta::from_delta_budget(log_delta, coeff_budget);

    let lt = ckks_encode_linear_transformation_from_diagonals(
        &sk.module,
        Base2K(sk.params.base2k as u32),
        coeff_meta,
        &prepared.cd,
        LinearTransformationStrategy::Bsgs {
            giant_step: sk.params.giant_step,
        },
        false,
        &mut scratch_guard.borrow(),
    )
    .map_err(|e| format!("cannot compile linear transformation: {e}"))?;

    sk.module
        .ckks_eval_linear_transformation_self_into(
            &mut out,
            &x.ct,
            &lt,
            &sk.atks,
            &mut scratch_guard.borrow(),
        )
        .map_err(|e| format!("BSGS linear transformation failed: {e}"))?;

    // Add bias if any entry is non-zero
    let has_bias = prepared.bias.iter().any(|&b| b != 0.0);
    if has_bias {
        let total_slots = sk.params.lt_slots;
        let mut bias_re = vec![0.0f64; total_slots];
        bias_re[..prepared.rows].copy_from_slice(&prepared.bias);
        let bias_im = vec![0.0f64; total_slots];

        let mut pt_bias = sk.module.ckks_pt_vec_alloc(out.base2k(), out.k());
        pt_bias.set_meta(out.meta());

        sk.module
            .ckks_encode_reim_into(
                &mut pt_bias,
                &bias_re,
                &bias_im,
                &mut scratch_guard.borrow(),
            )
            .map_err(|e| format!("bias encode failed: {e}"))?;

        sk.module
            .ckks_add_pt_vec_assign(&mut out, &pt_bias, &mut scratch_guard.borrow())
            .map_err(|e| format!("bias add failed: {e}"))?;
    }

    Ok(CkksCt {
        ct: out,
        len: prepared.rows,
    })
}
