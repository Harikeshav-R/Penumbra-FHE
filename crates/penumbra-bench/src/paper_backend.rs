//! Benchmark-only paper backend extension trait and scheme policies.

use std::path::Path;

use penumbra_core::backend::Backend;

use crate::models::ModelFixture;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowSelection {
    FullTest,
    SeededSpots,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Comparator {
    ExactInteger,
    AbsoluteBound(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PaperRunPolicy {
    pub selection: RowSelection,
    pub comparator: Comparator,
    pub has_pbs_counters: bool,
}

/// Benchmark harness extension trait for backend key management and decoding.
pub trait PaperBackend: Backend {
    fn load_client_key(&self, path: &Path, num_blocks: usize) -> Result<Self::ClientKey, String>;
    fn load_server_key(&self, path: &Path, num_blocks: usize) -> Result<Self::ServerKey, String>;
    fn decode_raw(&self, ck: &Self::ClientKey, cts: &[Self::Ciphertext]) -> Vec<f64>;
    fn paper_policy(&self, model: &ModelFixture) -> Result<PaperRunPolicy, String>;
}

// ---------------------------------------------------------------------------
// TFHE Implementation
// ---------------------------------------------------------------------------

impl PaperBackend for penumbra_tfhe::TfheBackend {
    fn load_client_key(&self, path: &Path, num_blocks: usize) -> Result<Self::ClientKey, String> {
        let (ck, loaded_nb, profile) = penumbra_tfhe::keys::load_client_key(path)?;
        if loaded_nb != num_blocks {
            return Err(format!(
                "client key at {} has num_blocks {loaded_nb}, expected {num_blocks}",
                path.display()
            ));
        }
        if profile != self.profile {
            return Err(format!(
                "client key at {} has profile {profile:?}, expected {:?}",
                path.display(),
                self.profile
            ));
        }
        Ok(ck)
    }

    fn load_server_key(&self, path: &Path, num_blocks: usize) -> Result<Self::ServerKey, String> {
        let (sk, loaded_nb, profile) = penumbra_tfhe::keys::load_server_key(path)?;
        if loaded_nb != num_blocks {
            return Err(format!(
                "server key at {} has num_blocks {loaded_nb}, expected {num_blocks}",
                path.display()
            ));
        }
        if profile != self.profile {
            return Err(format!(
                "server key at {} has profile {profile:?}, expected {:?}",
                path.display(),
                self.profile
            ));
        }
        Ok(sk)
    }

    fn decode_raw(&self, ck: &Self::ClientKey, cts: &[Self::Ciphertext]) -> Vec<f64> {
        let ints = self.decrypt_vec(ck, cts);
        ints.into_iter().map(|v| v as f64).collect()
    }

    fn paper_policy(&self, _model: &ModelFixture) -> Result<PaperRunPolicy, String> {
        Ok(PaperRunPolicy {
            selection: RowSelection::SeededSpots,
            comparator: Comparator::ExactInteger,
            has_pbs_counters: true,
        })
    }
}

// ---------------------------------------------------------------------------
// CKKS Implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "ckks")]
impl PaperBackend for penumbra_ckks::CkksBackend {
    fn load_client_key(&self, path: &Path, _num_blocks: usize) -> Result<Self::ClientKey, String> {
        penumbra_ckks::load_client_key(path)
    }

    fn load_server_key(&self, path: &Path, _num_blocks: usize) -> Result<Self::ServerKey, String> {
        penumbra_ckks::load_server_key(path)
    }

    fn decode_raw(&self, ck: &Self::ClientKey, cts: &[Self::Ciphertext]) -> Vec<f64> {
        penumbra_ckks::decrypt_raw_vec(ck, cts)
    }

    fn paper_policy(&self, model: &ModelFixture) -> Result<PaperRunPolicy, String> {
        let bound = match model.key {
            "phase2_logreg" => penumbra_ckks::bounds::PHASE2_LOGREG,
            "phase4_cnn" => penumbra_ckks::bounds::PHASE4_CNN,
            "phase5_digits" => penumbra_ckks::bounds::PHASE5_DIGITS,
            "phase5_qat" => penumbra_ckks::bounds::PHASE5_QAT,
            "phase6_onnx" => penumbra_ckks::bounds::PHASE6_ONNX,
            "phase6_sklearn" => penumbra_ckks::bounds::PHASE6_SKLEARN,
            "phase7_faces" => penumbra_ckks::bounds::PHASE7_FACES,
            "phase8_branch" => penumbra_ckks::bounds::PHASE8_BRANCH,
            "phase8_bn_cnn" => penumbra_ckks::bounds::PHASE8_BN_CNN,
            "phase8_gap_cnn" => penumbra_ckks::bounds::PHASE8_GAP_CNN,
            "phase8_tanh" => penumbra_ckks::bounds::PHASE8_TANH,
            "phase11_tabular_mlp" => penumbra_ckks::bounds::PHASE11_TABULAR_MLP,
            "phase8_trees" | "phase8_xgb" => {
                return Err(format!(
                    "operator Compare (node in '{}') not supported on CKKS backend: depth budget exceeded",
                    model.key
                ));
            }
            other => return Err(format!("unknown model '{other}' for CKKS paper policy")),
        };

        Ok(PaperRunPolicy {
            selection: RowSelection::FullTest,
            comparator: Comparator::AbsoluteBound(bound),
            has_pbs_counters: false,
        })
    }
}
