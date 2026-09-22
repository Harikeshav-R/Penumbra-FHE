use poulpy_ckks::{CKKSLayout, CKKSMeta, CoeffsMeta, SlotsKind};
use poulpy_core::{
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWELayout, GLWEAutomorphismKeyLayout, GLWELayout,
        GLWETensorKeyLayout, Rank, TorusPrecision,
    },
    EncryptionLayout,
};
use serde::{Deserialize, Serialize};

/// Penumbra's CKKS parameter profile — the analogue of `penumbra_tfhe::keys::DEFAULT_PARAMS`.
///
/// `poulpy-ckks` 0.8.3 ships no general parameter preset (`presets` holds only bootstrapping
/// plans), and `CKKSTestParams`/`FFT64_PARAMS_F64` are `test-utils`-gated test sets at n = 256
/// — too few slots for Penumbra's tensors and not 128-bit secure. See `docs/NOTES-ckks.md`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CkksParams {
    pub n: usize,
    pub base2k: usize,
    pub k: usize,
    pub log_delta: usize,
    pub dsize: usize,
    pub rank: usize,
    /// Dimension of the homomorphic linear transform, and therefore of every packed tensor.
    /// Fixed (not `n / 2`) to bound the automorphism key set.
    pub lt_slots: usize,
    /// BSGS giant step. Fixed so the automorphism key set does not depend on the model.
    pub giant_step: usize,
    /// Secret-key distribution: uniform ternary, `prob` for `glwe_secret_fill_ternary_prob`.
    pub secret_ternary_prob: f64,
    /// Max polynomial degree any op may fit. The single crypto override knob
    /// (`PROJECT.md` §12, `docs/BACKENDS.md` fork 3).
    pub max_poly_degree: usize,
}

/// The single secure default profile calibrated in Phase 12.2.
pub const DEFAULT_PARAMS: CkksParams = CkksParams {
    n: 16384,
    base2k: 19,
    k: 360,
    log_delta: 30,
    dsize: 2,
    rank: 1,
    lt_slots: 256,
    giant_step: 16,
    secret_ternary_prob: 2.0 / 3.0,
    max_poly_degree: 15,
};

pub fn slots(p: &CkksParams) -> usize {
    p.n / 2
}

/// `log_sparsity` for a packed value: `log2(n/2) - log2(lt_slots)`.
pub fn log_sparsity(p: &CkksParams) -> usize {
    (p.n / 2).ilog2() as usize - p.lt_slots.ilog2() as usize
}

impl CkksParams {
    pub fn log_n(&self) -> usize {
        self.n.ilog2() as usize
    }

    pub fn prec(&self) -> CKKSLayout {
        CKKSLayout {
            glwe_layout: GLWELayout {
                n: Degree(self.n as u32),
                base2k: Base2K(self.base2k as u32),
                k: TorusPrecision(self.k as u32),
                rank: Rank(self.rank as u32),
            },
            meta: CKKSMeta {
                log_delta: self.log_delta,
                log_sparsity: log_sparsity(self),
                slots: SlotsKind::Complex,
            },
        }
    }

    pub fn glwe_layout(&self) -> EncryptionLayout<GLWELayout> {
        EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: Degree(self.n as u32),
            base2k: Base2K(self.base2k as u32),
            k: TorusPrecision(self.k as u32),
            rank: Rank(self.rank as u32),
        })
        .unwrap()
    }

    /// `(GGLWELayout::dnum_for_input(base2k, k_in, dsize), TorusPrecision(dsize*base2k + log_n))`
    fn key_shape(&self, k_in: usize) -> (Dnum, TorusPrecision) {
        let dnum = GGLWELayout::dnum_for_input(
            Base2K(self.base2k as u32),
            TorusPrecision(k_in as u32),
            Dsize(self.dsize as u32),
        );
        (
            dnum,
            TorusPrecision((self.dsize * self.base2k + self.log_n()) as u32),
        )
    }

    pub fn tsk_layout(&self) -> EncryptionLayout<GLWETensorKeyLayout> {
        let (dnum, k_aux) = self.key_shape(self.k);
        EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
            n: Degree(self.n as u32),
            base2k: Base2K(self.base2k as u32),
            dnum,
            k_aux,
            rank: Rank(self.rank as u32),
            dsize: Dsize(self.dsize as u32),
        })
        .unwrap()
    }

    pub fn atk_layout(&self) -> EncryptionLayout<GLWEAutomorphismKeyLayout> {
        let (dnum, k_aux) = self.key_shape(self.k);
        EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
            n: Degree(self.n as u32),
            base2k: Base2K(self.base2k as u32),
            dnum,
            k_aux,
            rank: Rank(self.rank as u32),
            dsize: Dsize(self.dsize as u32),
        })
        .unwrap()
    }

    /// Multiplicative budget in bits: `k - log_delta`.
    pub fn log_budget(&self) -> usize {
        self.k.saturating_sub(self.log_delta)
    }

    pub fn coeffs_meta(&self) -> CoeffsMeta {
        CoeffsMeta::from_delta_budget(self.log_delta, self.log_budget())
    }
}
