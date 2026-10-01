//! Security parameter extraction and input JSON generation for the lattice estimator.

use std::path::Path;

use penumbra_tfhe::keys::TfheProfile;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LweSecurityInput {
    pub name: String,
    pub n: usize,
    pub q: String,
    pub q_bits: usize,
    pub secret_distribution: String,
    pub error_distribution: String,
    pub error_param: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SecurityInputsReport {
    pub tfhe_classic_lwe: LweSecurityInput,
    pub tfhe_classic_glwe: LweSecurityInput,
    pub ckks_ciphertext: LweSecurityInput,
    pub ckks_evaluation_key: LweSecurityInput,
}

/// Extract security inputs from live crate parameters and verify against pinned defaults.
pub fn extract_security_inputs() -> Result<SecurityInputsReport, String> {
    // 1. TFHE Classic LWE & GLWE parameters
    let tfhe_params = TfheProfile::Classic.params();
    let tfhe_lwe_dim = tfhe_params.lwe_dimension().0;
    let tfhe_glwe_n = tfhe_params.glwe_dimension().0 * tfhe_params.polynomial_size().0;

    let tfhe_classic_lwe = LweSecurityInput {
        name: "tfhe_classic_lwe".to_string(),
        n: tfhe_lwe_dim,
        q: "2^64".to_string(),
        q_bits: 64,
        secret_distribution: "Binary".to_string(),
        error_distribution: "TUniform(45)".to_string(),
        error_param: 45.0,
    };

    let tfhe_classic_glwe = LweSecurityInput {
        name: "tfhe_classic_glwe".to_string(),
        n: tfhe_glwe_n,
        q: "2^64".to_string(),
        q_bits: 64,
        secret_distribution: "Binary".to_string(),
        error_distribution: "TUniform(17)".to_string(),
        error_param: 17.0,
    };

    // Verify TFHE dimensions match pinned specification
    if tfhe_classic_lwe.n != 918 {
        return Err(format!(
            "TFHE LWE dimension mismatch: got {}, expected 918",
            tfhe_classic_lwe.n
        ));
    }
    if tfhe_classic_glwe.n != 2048 {
        return Err(format!(
            "TFHE GLWE dimension mismatch: got {}, expected 2048",
            tfhe_classic_glwe.n
        ));
    }

    // 2. CKKS parameters
    #[cfg(feature = "ckks")]
    let (ckks_ciphertext, ckks_evaluation_key) = {
        use penumbra_ckks::params::DEFAULT_PARAMS;
        use poulpy_core::layouts::LWEInfos;

        let glwe_layout = DEFAULT_PARAMS.glwe_layout();
        let tsk_layout = DEFAULT_PARAMS.tsk_layout();
        let atk_layout = DEFAULT_PARAMS.atk_layout();

        let ct_q_bits: usize = glwe_layout.k().into();
        let tsk_q_bits: usize = tsk_layout.k().into();
        let atk_q_bits: usize = atk_layout.k().into();
        if tsk_q_bits != atk_q_bits {
            return Err(format!(
                "CKKS key layout precision mismatch: tsk={tsk_q_bits}, atk={atk_q_bits}"
            ));
        }
        let key_q_bits = tsk_q_bits;
        let n = DEFAULT_PARAMS.n;

        let ct = LweSecurityInput {
            name: "ckks_ciphertext".to_string(),
            n,
            q: format!("2^{ct_q_bits}"),
            q_bits: ct_q_bits,
            secret_distribution: "Ternary".to_string(),
            error_distribution: "DiscreteGaussian(3.2)".to_string(),
            error_param: 3.2,
        };

        let evk = LweSecurityInput {
            name: "ckks_evaluation_key".to_string(),
            n,
            q: format!("2^{key_q_bits}"),
            q_bits: key_q_bits,
            secret_distribution: "Ternary".to_string(),
            error_distribution: "DiscreteGaussian(3.2)".to_string(),
            error_param: 3.2,
        };

        if ct.q_bits != 360 {
            return Err(format!(
                "CKKS ct q_bits mismatch: got {}, expected 360",
                ct.q_bits
            ));
        }
        if evk.q_bits != 432 {
            return Err(format!(
                "CKKS evk q_bits mismatch: got {}, expected 432",
                evk.q_bits
            ));
        }

        (ct, evk)
    };

    #[cfg(not(feature = "ckks"))]
    let (ckks_ciphertext, ckks_evaluation_key) = {
        let ct = LweSecurityInput {
            name: "ckks_ciphertext".to_string(),
            n: 16384,
            q: "2^360".to_string(),
            q_bits: 360,
            secret_distribution: "Ternary".to_string(),
            error_distribution: "DiscreteGaussian(3.2)".to_string(),
            error_param: 3.2,
        };
        let evk = LweSecurityInput {
            name: "ckks_evaluation_key".to_string(),
            n: 16384,
            q: "2^432".to_string(),
            q_bits: 432,
            secret_distribution: "Ternary".to_string(),
            error_distribution: "DiscreteGaussian(3.2)".to_string(),
            error_param: 3.2,
        };
        (ct, evk)
    };

    Ok(SecurityInputsReport {
        tfhe_classic_lwe,
        tfhe_classic_glwe,
        ckks_ciphertext,
        ckks_evaluation_key,
    })
}

/// Write security inputs JSON to destination path.
pub fn generate_security_inputs_file(out_path: &Path) -> Result<(), String> {
    let report = extract_security_inputs()?;
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create dir {}: {e}", parent.display()))?;
    }
    let json_text = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("failed to format security inputs json: {e}"))?;
    std::fs::write(out_path, json_text + "\n")
        .map_err(|e| format!("cannot write {}: {e}", out_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_security_inputs() {
        let inputs = extract_security_inputs().unwrap();
        assert_eq!(inputs.tfhe_classic_lwe.n, 918);
        assert_eq!(inputs.tfhe_classic_lwe.q_bits, 64);
        assert_eq!(inputs.tfhe_classic_glwe.n, 2048);
        assert_eq!(inputs.tfhe_classic_glwe.q_bits, 64);
        assert_eq!(inputs.ckks_ciphertext.n, 16384);
        assert_eq!(inputs.ckks_ciphertext.q_bits, 360);
        assert_eq!(inputs.ckks_evaluation_key.n, 16384);
        assert_eq!(inputs.ckks_evaluation_key.q_bits, 432);
    }
}
