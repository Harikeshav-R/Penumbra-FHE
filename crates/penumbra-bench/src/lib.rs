//! Penumbra-FHE Shared Comparison Harness.
//!
//! Provides the generic evaluation session, model fixtures, reporting, and criterion
//! benchmarks parameterized over backend x model (ROADMAP Phase 12.3).

pub mod models;
pub mod report;
pub mod session;

pub use models::{find, load, selection_from_env, LoadedModel, ModelFixture, MODELS};
pub use report::{run_model, to_json, to_markdown, ModelRun, NodeReport, SampleReport};
pub use session::Session;

/// Return a configured instance of the TFHE backend.
pub fn tfhe_backend() -> penumbra_tfhe::TfheBackend {
    penumbra_tfhe::TfheBackend
}

/// Return a configured instance of the CKKS backend with default parameters.
#[cfg(feature = "ckks")]
pub fn ckks_backend() -> penumbra_ckks::CkksBackend {
    penumbra_ckks::CkksBackend::default()
}

/// Backend names compiled into this build, in report order.
pub fn available_backends() -> Vec<&'static str> {
    vec![
        "tfhe",
        #[cfg(feature = "ckks")]
        "ckks",
    ]
}
