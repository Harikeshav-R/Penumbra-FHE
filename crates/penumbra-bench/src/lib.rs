//! Penumbra-FHE Shared Comparison Harness.
//!
//! Provides the generic evaluation session, model fixtures, reporting, and criterion
//! benchmarks parameterized over backend x model (ROADMAP Phase 12.3).

pub mod baseline;
pub mod latency;
pub mod memory;
pub mod metrics;
pub mod models;
pub mod paper;
pub mod paper_backend;
pub mod protocol;
pub mod report;
pub mod security;
pub mod session;
pub use baseline::{baseline_from_runs, check_against, Baseline, BaselineEntry};

pub use models::{find, load, selection_from_env, LoadedModel, ModelFixture, MODELS};
pub use report::{
    run_model, to_json, to_markdown, ModelRun, NodeReport, Report, ReportMeta, SampleReport,
};
pub use session::Session;

/// Return a configured instance of the TFHE backend.
pub fn tfhe_backend() -> penumbra_tfhe::TfheBackend {
    penumbra_tfhe::TfheBackend::default()
}

/// Return a configured instance of the CKKS backend with default parameters.
#[cfg(feature = "ckks")]
pub fn ckks_backend() -> penumbra_ckks::CkksBackend {
    penumbra_ckks::CkksBackend::default()
}

/// Return a configured instance of the CKKS backend tailored for the given model fixture key.
///
/// For `phase8_branch`, overrides `max_poly_degree` to 3 to satisfy depth requirements
/// while maintaining standard default parameters for all other models.
#[cfg(feature = "ckks")]
pub fn ckks_backend_for_model(model_key: &str) -> penumbra_ckks::CkksBackend {
    if model_key == "phase8_branch" {
        let params = penumbra_ckks::params::DEFAULT_PARAMS
            .with_max_poly_degree(3)
            .expect("phase8_branch requires valid degree-3 polynomial parameters");
        penumbra_ckks::CkksBackend::new(params)
    } else {
        ckks_backend()
    }
}

/// Backend names compiled into this build, in report order.
pub fn available_backends() -> Vec<&'static str> {
    vec![
        "tfhe",
        #[cfg(feature = "ckks")]
        "ckks",
    ]
}
