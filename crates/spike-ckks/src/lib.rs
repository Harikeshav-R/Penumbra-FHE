//! Standalone CKKS spike for Penumbra-FHE (ROADMAP Phase 12.0).
//!
//! Validates:
//! 1. Toolchain & dependencies (poulpy-ckks 0.8.3, poulpy-hal, and hardware acceleration).
//! 2. Standalone packed dot product / plaintext-scalar linear layer (`mul_pt` + `add_pt`).
//! 3. Standalone polynomial non-linearity (`ReLU`) via minimax Chebyshev approximation
//!    and Baby-Step Giant-Step (BSGS) evaluation.
//! 4. Separation of polynomial approximation error from cryptographic noise.
//! 5. Dependency coexistence with `tfhe-rs` in the same workspace.

use anyhow::Result;
use poulpy_ckks::approximation::minimax;
use poulpy_ckks::layouts::CKKSCiphertextOwned;
use poulpy_ckks::polynomial::{Basis, EncodeBSGS, Parity};
use poulpy_ckks::power_basis::{PowerBasis, PowerBasisGen};
use poulpy_ckks::prelude::*;
pub use poulpy_ckks::test_suite::FFT64_PARAMS_F64;
use poulpy_ckks::test_suite::helpers::*;
use poulpy_ckks::test_suite::reference_encoder::ReferenceEncoder;
use poulpy_core::layouts::{GLWETensorKeyPrepared, LWEInfos, prepared::GLWESecretPrepared};
use poulpy_hal::api::ScratchOwnedBorrow;
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchOwned};

#[cfg(target_arch = "aarch64")]
pub type ActiveBackend = poulpy_cpu_arm::FFT64Neon;
#[cfg(target_arch = "aarch64")]
pub type ActiveEncoderTable = poulpy_cpu_arm::FFT64NeonReimTable;

#[cfg(not(target_arch = "aarch64"))]
pub type ActiveBackend = poulpy_cpu_ref::FFT64Ref;
#[cfg(not(target_arch = "aarch64"))]
pub type ActiveEncoderTable = poulpy_cpu_ref::FFT64ReimTable<f64>;

pub type SecretKey = GLWESecretPrepared<<ActiveBackend as Backend>::OwnedBuf, ActiveBackend>;
pub type TensorKey = GLWETensorKeyPrepared<<ActiveBackend as Backend>::OwnedBuf, ActiveBackend>;

/// Context bundle managing modules, scratch arena, and slot encoder.
pub struct SpikeContext {
    pub module: Module<ActiveBackend>,
    pub host_module: Module<HostBytesBackend>,
    pub encoder: ReferenceEncoder<ActiveEncoderTable>,
    pub scratch: ScratchOwned<ActiveBackend>,
    pub slots: usize,
}

impl SpikeContext {
    /// Creates a new spike execution context with the default FFT64 parameter profile.
    pub fn new() -> Result<Self> {
        let params = FFT64_PARAMS_F64;
        let n = params.n as u64;
        let slots = params.n / 2;

        let module = Module::<ActiveBackend>::new(n);
        let host_module = Module::<HostBytesBackend>::new(n);
        let encoder = ReferenceEncoder::<ActiveEncoderTable>::new(slots)?;
        let scratch = alloc_scratch(&params, &module);

        Ok(Self {
            module,
            host_module,
            encoder,
            scratch,
            slots,
        })
    }

    /// Generates a secret key and a tensor/relinearization key.
    pub fn keygen(&mut self, seed: [u8; 32]) -> (SecretKey, TensorKey) {
        let params = FFT64_PARAMS_F64;
        let (sk_raw, sk) = gen_sk_with_raw(&params, &self.module, &self.host_module, seed);
        let tsk = gen_tsk(&params, &self.module, &sk_raw, &mut self.scratch.borrow());
        (sk, tsk)
    }

    /// Encrypts a real slice into a packed CKKS ciphertext.
    pub fn encrypt(
        &mut self,
        values: &[f64],
        sk: &SecretKey,
    ) -> CKKSCiphertextOwned<ActiveBackend> {
        assert_eq!(
            values.len(),
            self.slots,
            "value length must match slot count"
        );
        let params = FFT64_PARAMS_F64;
        let zeros = vec![0.0f64; self.slots];
        ckks_encrypt(
            &params,
            &self.module,
            &self.host_module,
            &self.encoder,
            sk,
            params.k,
            values,
            &zeros,
            &mut self.scratch.borrow(),
        )
    }

    /// Decrypts and decodes a ciphertext back into a vector of real numbers.
    pub fn decrypt(&mut self, ct: &CKKSCiphertextOwned<ActiveBackend>, sk: &SecretKey) -> Vec<f64> {
        let params = FFT64_PARAMS_F64;
        let (re, _im): (Vec<f64>, Vec<f64>) = ckks_decrypt_decode(
            &params,
            &self.module,
            &self.encoder,
            ct,
            sk,
            &mut self.scratch.borrow(),
        );
        re
    }

    /// Evaluates a packed linear layer `y = weight * x + bias` using plaintext scalar weights.
    pub fn eval_linear(
        &mut self,
        ct: &CKKSCiphertextOwned<ActiveBackend>,
        weight: f64,
        bias: f64,
    ) -> Result<CKKSCiphertextOwned<ActiveBackend>> {
        let params = FFT64_PARAMS_F64;
        let pt_weight = ckks_pt_cst_full::<ActiveBackend, f64>(
            &self.host_module,
            &self.module,
            params.base2k.into(),
            params.prec(),
            self.slots,
            Some(weight),
            None,
        );
        let pt_bias = ckks_pt_cst_full::<ActiveBackend, f64>(
            &self.host_module,
            &self.module,
            params.base2k.into(),
            params.prec(),
            self.slots,
            Some(bias),
            None,
        );

        let mut prod_ct = alloc_ct(&params, &self.module, ct.k().as_usize());
        self.module.ckks_mul_pt_vec_into(
            &mut prod_ct,
            ct,
            &pt_weight,
            &mut self.scratch.borrow(),
        )?;

        let mut out_ct = alloc_ct(&params, &self.module, prod_ct.k().as_usize());
        self.module.ckks_add_pt_vec_into(
            &mut out_ct,
            &prod_ct,
            &pt_bias,
            &mut self.scratch.borrow(),
        )?;

        Ok(out_ct)
    }

    /// Evaluates a polynomial approximation of ReLU over `[-1, 1]` using the Remez algorithm.
    pub fn eval_relu(
        &mut self,
        ct: &CKKSCiphertextOwned<ActiveBackend>,
        tsk: &TensorKey,
        degree: usize,
    ) -> Result<(CKKSCiphertextOwned<ActiveBackend>, f64)> {
        let params = FFT64_PARAMS_F64;
        let relu = |x: f64| if x > 0.0 { x } else { 0.0 };
        let minimax_fit = minimax(relu, -1.0, 1.0, degree, Parity::Full)?;

        let coeff_meta = poulpy_ckks::CoeffsMeta {
            k: poulpy_core::layouts::TorusPrecision(params.prec_meta.log_delta as u32 + 1),
            meta: params.prec_meta,
        };
        let bsgs =
            minimax_fit
                .poly
                .encode_bsgs(&self.host_module, params.base2k.into(), coeff_meta)?;

        let mut pb = PowerBasis::new(Basis::Chebyshev, ct.clone());
        pb.populate(
            degree,
            bsgs.log_split(),
            bsgs.parity(),
            &self.module,
            tsk,
            &mut self.scratch.borrow(),
        )?;

        let mut relu_ct = alloc_ct(&params, &self.module, params.k);
        use poulpy_ckks::api::CKKSPolynomialEvaluationOps;
        self.module
            .ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<Vec<u8>, i64>, _, _>(
                &mut relu_ct,
                &bsgs,
                &pb,
                tsk,
                &mut self.scratch.borrow(),
            )?;

        Ok((relu_ct, minimax_fit.error))
    }
}

/// Breakdown of errors isolating mathematical polynomial fit error from FHE evaluation noise.
#[derive(Debug, Clone, Copy)]
pub struct ErrorBreakdown {
    /// Mathematical approximation error: max |p(x) - true_fn(x)|.
    pub poly_approx_error: f64,
    /// Total error of the homomorphic pipeline: max |decrypted(x) - true_fn(x)|.
    pub total_homomorphic_error: f64,
    /// Crypto noise contribution: |total_error - approx_error|.
    pub fhe_noise_contribution: f64,
}

/// Computes the empirical error breakdown for a decrypted vector against a target scalar function.
pub fn compute_error_breakdown<F1, F2>(
    inputs: &[f64],
    decrypted: &[f64],
    target_fn: F1,
    poly_eval_fn: F2,
) -> ErrorBreakdown
where
    F1: Fn(f64) -> f64,
    F2: Fn(f64) -> f64,
{
    let mut max_poly_approx_err = 0.0f64;
    let mut max_total_err = 0.0f64;

    for (&x, &d) in inputs.iter().zip(decrypted.iter()) {
        let true_val = target_fn(x);
        let poly_val = poly_eval_fn(x);

        let approx_err = (poly_val - true_val).abs();
        let total_err = (d - true_val).abs();

        if approx_err > max_poly_approx_err {
            max_poly_approx_err = approx_err;
        }
        if total_err > max_total_err {
            max_total_err = total_err;
        }
    }

    let fhe_noise = (max_total_err - max_poly_approx_err).abs();
    ErrorBreakdown {
        poly_approx_error: max_poly_approx_err,
        total_homomorphic_error: max_total_err,
        fhe_noise_contribution: fhe_noise,
    }
}
