//! Shared benchmark and comparison harness for Penumbra-FHE backends.
//!
//! This crate will house the Criterion-based comparison benchmarks parameterized
//! over backend x model, enabling fair, identical-workload latency, memory,
//! and noise-growth profiling across TFHE and CKKS backends (ROADMAP Phase 12.3).

pub use penumbra_core as core;
pub use penumbra_tfhe as tfhe;

/// Stub benchmark runner identifier for Phase 12.1 workspace scaffolding.
#[must_use]
pub fn harness_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harness_scaffolding() {
        assert_eq!(harness_version(), "0.0.0");
    }
}
