use std::collections::HashMap;

use poulpy_ckks::api::{CKKSAddOps, CKKSApproximationOps, CKKSEncodingHostOps, CKKSMulOps};
use poulpy_ckks::approximation::{precision_at_depth, Parity};
use poulpy_ckks::layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, PolynomialApproximation};
use poulpy_ckks::{CKKSInfos, SetCKKSInfos};
use poulpy_core::layouts::{bsgs_eval_depth, Base2K, LWEInfos};
use poulpy_hal::api::ScratchOwnedBorrow;
use poulpy_hal::layouts::{HostBytesBackend, Module};

use crate::encrypt::CkksCt;
use crate::hal::ActiveBackend;
use crate::keys::CkksServerKey;
use crate::params::CkksParams;

pub struct PolyMap {
    pub(crate) poly: poulpy_ckks::polynomial::Polynomial<f64>,
    /// Analytic multiplicative depth, from `DegreeChoice::depth`.
    pub depth: usize,
    /// The minimax sup-norm error of the fit.
    pub fit_error: f64,
}

impl PolyMap {
    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn fit_error(&self) -> f64 {
        self.fit_error
    }
}

#[allow(clippy::too_many_arguments)]
pub fn fit_requant(
    _host_module: &Module<HostBytesBackend>,
    params: &CkksParams,
    mult: i64,
    shift: usize,
    round_bias: i64,
    out_bits: usize,
    input_bits: usize,
    max_depth: usize,
) -> Result<PolyMap, String> {
    let t_sat = (((1usize << out_bits) * (1usize << shift)) / (mult as usize).max(1))
        .next_power_of_two()
        .max(512);
    let cap = (1i64 << (input_bits.saturating_sub(1).max(1))) as f64;
    let x_max = (t_sat as f64).min(cap).max(512.0);
    let lo = -x_max;
    let hi = x_max;
    let divisor = (1u64 << shift) as f64;
    let max_val = ((1i64 << out_bits) - 1) as f64;
    // The integer op floors: clamp((relu*mult + round_bias) >> shift, 0, max_val)
    // (`penumbra_tfhe::ops::requant`, `python/penumbra/reference.py::_requant`). Fitting the
    // pre-floor value overshoots every integer answer by the floor residual, whose mean over the
    // 2^-shift grid is (1 - 2^-shift)/2 — a systematic +0.5-LSB bias that a following Linear
    // amplifies by its L1 weight norm (~200 integer units on phase7_faces). Fit the midpoint of
    // each floor step instead, so the residual is zero-mean. The (1 - 1/divisor) factor makes the
    // correction vanish at shift = 0, where the shift is exact and no residual exists — that is
    // the identity/ReLU fit `Backend::scalar_max`/`scalar_min` rely on.
    let floor_midpoint = 0.5 * (1.0 - 1.0 / divisor);
    let f = move |t: f64| -> f64 {
        let relu = t.max(0.0);
        let scaled = (relu * (mult as f64) + (round_bias as f64)) / divisor - floor_midpoint;
        scaled.clamp(0.0, max_val)
    };

    let choice = precision_at_depth(
        f,
        lo,
        hi,
        Parity::Full,
        max_depth,
        params.max_poly_degree,
        poulpy_ckks::polynomial::SplitStrategy::MinDepth,
    )
    .map_err(|e| format!("cannot fit requant polynomial: {e}"))?;

    Ok(PolyMap {
        poly: choice.minimax.poly,
        depth: choice.depth,
        fit_error: choice.minimax.error,
    })
}

fn lagrange_eval(lut: &[i64], x: f64) -> f64 {
    let mut sum = 0.0;
    for (i, &y_i) in lut.iter().enumerate() {
        let mut term = y_i as f64;
        let x_i = i as f64;
        for (j, _) in lut.iter().enumerate() {
            if i != j {
                let x_j = j as f64;
                term *= (x - x_j) / (x_i - x_j);
            }
        }
        sum += term;
    }
    sum
}

pub fn fit_activation(
    _host_module: &Module<HostBytesBackend>,
    _params: &CkksParams,
    lut: &[i64],
) -> Result<PolyMap, String> {
    if lut.len() < 2 {
        return Err("activation LUT length must be at least 2".to_string());
    }
    if !lut.len().is_power_of_two() {
        return Err(format!(
            "activation LUT length {} must be a power of two",
            lut.len()
        ));
    }

    let deg = lut.len() - 1;
    let poly =
        poulpy_ckks::polynomial::Polynomial::chebyshev_interpolate(deg, 0.0, deg as f64, |x| {
            lagrange_eval(lut, x)
        })
        .map_err(|e| format!("cannot interpolate activation LUT: {e}"))?;
    let depth = bsgs_eval_depth(deg, poulpy_ckks::polynomial::SplitStrategy::MinDepth);
    Ok(PolyMap {
        poly,
        depth,
        fit_error: 0.0,
    })
}

pub fn eval_polymap(sk: &CkksServerKey, x: &CkksCt, pm: &PolyMap) -> Result<CkksCt, String> {
    let log_delta = x.ct.log_delta();
    let (scale, _offset) = pm.poly.change_of_basis();
    let affine_shift = if scale > 0.0 && scale.is_finite() && scale < 1.0 {
        (-scale.log2()).round() as usize
    } else {
        0
    };
    let poly_consumed_bits = pm.depth * log_delta;
    let coeff_budget =
        x.ct.log_budget()
            .saturating_sub(affine_shift)
            .saturating_sub(poly_consumed_bits);
    let coeff_meta = poulpy_ckks::CoeffsMeta::from_delta_budget(log_delta, coeff_budget);

    let approx = PolynomialApproximation::from_polynomial(
        &pm.poly,
        Base2K(sk.params.base2k as u32),
        coeff_meta,
        poulpy_ckks::polynomial::SplitStrategy::MinDepth,
        &sk.host_module,
    )
    .map_err(|e| format!("cannot prepare polynomial approximation: {e}"))?;

    let mut out = sk
        .module
        .ckks_ciphertext_alloc_from_glwe_infos(&sk.params.glwe_layout());
    let mut scratch_guard = sk
        .scratch
        .lock()
        .map_err(|e| format!("mutex poisoned: {e}"))?;

    sk.module
        .ckks_eval_approximation(
            &mut out,
            &x.ct,
            &approx,
            &sk.tsk,
            &mut scratch_guard.borrow(),
        )
        .map_err(|e| format!("polynomial evaluation failed: {e}"))?;

    Ok(CkksCt {
        ct: out,
        len: x.len,
    })
}

/// Per-channel Requant map: one `PolyMap` per distinct parameter triple,
/// plus a channel mask vector selecting where each triple applies.
pub struct PerChannelRequantMap {
    pub(crate) branches: Vec<(PolyMap, Vec<f64>)>,
    pub depth: usize,
    pub max_fit_error: f64,
}

impl PerChannelRequantMap {
    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn max_fit_error(&self) -> f64 {
        self.max_fit_error
    }
}

#[allow(clippy::too_many_arguments)]
pub fn fit_per_channel_requant(
    host_module: &Module<HostBytesBackend>,
    params: &CkksParams,
    mults: &[i64],
    shifts: &[usize],
    round_biases: &[i64],
    channel_size: usize,
    out_bits: usize,
    input_bits: usize,
    total_len: usize,
    max_depth: usize,
) -> Result<PerChannelRequantMap, String> {
    let num_channels = mults.len();
    if shifts.len() != num_channels || round_biases.len() != num_channels {
        return Err("per-channel requant parameters length mismatch".to_string());
    }

    // Group channels by distinct triple (mult, shift, round_bias)
    let mut distinct_triples: HashMap<(i64, usize, i64), Vec<usize>> = HashMap::new();
    for (c, ((&m, &s), &rb)) in mults.iter().zip(shifts).zip(round_biases).enumerate() {
        distinct_triples.entry((m, s, rb)).or_default().push(c);
    }

    let total_slots = params.lt_slots;
    let mut branches = Vec::new();
    let mut max_depth_found = 0;
    let mut max_fit_error: f64 = 0.0;

    for ((m, s, rb), channels) in distinct_triples {
        let pm = fit_requant(
            host_module,
            params,
            m,
            s,
            rb,
            out_bits,
            input_bits,
            max_depth,
        )?;
        if pm.depth > max_depth_found {
            max_depth_found = pm.depth;
        }
        if pm.fit_error > max_fit_error {
            max_fit_error = pm.fit_error;
        }

        // Build 0/1 mask for this branch
        let mut mask = vec![0.0f64; total_slots];
        for c in channels {
            let start = c * channel_size;
            let end = (start + channel_size).min(total_len);
            for slot in &mut mask[start..end] {
                *slot = 1.0;
            }
        }
        branches.push((pm, mask));
    }

    Ok(PerChannelRequantMap {
        branches,
        // One ct × pt multiply level added for the channel mask
        depth: max_depth_found + 1,
        max_fit_error,
    })
}

pub fn eval_per_channel_requant(
    sk: &CkksServerKey,
    x: &CkksCt,
    map: &PerChannelRequantMap,
) -> Result<CkksCt, String> {
    if map.branches.is_empty() {
        return Err("empty per-channel requant branches".to_string());
    }

    let mut acc: Option<CKKSCiphertextOwned<ActiveBackend>> = None;
    let total_slots = sk.params.lt_slots;
    for (pm, mask) in &map.branches {
        let eval_ct = eval_polymap(sk, x, pm)?;

        let mut masked = sk
            .module
            .ckks_ciphertext_alloc_from_glwe_infos(&sk.params.glwe_layout());
        let mut scratch_guard = sk
            .scratch
            .lock()
            .map_err(|e| format!("mutex poisoned: {e}"))?;

        // Prepare mask plaintext at matching precision and scale
        let mut pt_mask = sk
            .module
            .ckks_pt_vec_alloc(eval_ct.ct.base2k(), eval_ct.ct.k());
        pt_mask.set_meta(eval_ct.ct.meta());

        let im = vec![0.0f64; total_slots];
        sk.module
            .ckks_encode_reim_into(&mut pt_mask, mask, &im, &mut scratch_guard.borrow())
            .map_err(|e| format!("channel mask encode failed: {e}"))?;

        sk.module
            .ckks_mul_pt_vec_into(
                &mut masked,
                &eval_ct.ct,
                &pt_mask,
                &mut scratch_guard.borrow(),
            )
            .map_err(|e| format!("channel mask multiply failed: {e}"))?;
        if let Some(accumulator) = &mut acc {
            let mut sum = sk
                .module
                .ckks_ciphertext_alloc_from_glwe_infos(&sk.params.glwe_layout());
            sk.module
                .ckks_add_into(&mut sum, accumulator, &masked, &mut scratch_guard.borrow())
                .map_err(|e| format!("channel branch add failed: {e}"))?;
            *accumulator = sum;
        } else {
            acc = Some(masked);
        }
    }

    Ok(CkksCt {
        ct: acc.expect("branches was not empty"),
        len: x.len,
    })
}
