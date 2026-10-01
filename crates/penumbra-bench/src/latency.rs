//! Criterion latency benchmarking and artifact parsing.

use std::path::Path;

use criterion::{BenchmarkId, Criterion, SamplingMode};
use penumbra_core::backend::Backend;
use serde::Deserialize;

use crate::models::LoadedModel;
use crate::protocol::CriterionLatency;
use crate::session::Session;

#[derive(Debug, Deserialize)]
struct CriterionEstimatesJson {
    median: CriterionEstimateEntry,
}

#[derive(Debug, Deserialize)]
struct CriterionEstimateEntry {
    point_estimate: f64,
    confidence_interval: CriterionConfidenceInterval,
}

#[derive(Debug, Deserialize)]
struct CriterionConfidenceInterval {
    confidence_level: f64,
    lower_bound: f64,
    upper_bound: f64,
}

#[derive(Debug, Deserialize)]
struct CriterionSampleJson {
    sampling_mode: String,
    iters: Vec<f64>,
    times: Vec<f64>,
}

/// Parse Criterion 0.7 `estimates.json` and `sample.json` from a benchmark run output directory.
pub fn parse_criterion_latency(
    benchmark_dir: &Path,
    sample_id: &str,
) -> Result<CriterionLatency, String> {
    let estimates_path = benchmark_dir.join("new").join("estimates.json");
    let sample_path = benchmark_dir.join("new").join("sample.json");

    if !estimates_path.exists() {
        return Err(format!(
            "missing Criterion estimates.json at {}",
            estimates_path.display()
        ));
    }
    if !sample_path.exists() {
        return Err(format!(
            "missing Criterion sample.json at {}",
            sample_path.display()
        ));
    }

    let estimates_text = std::fs::read_to_string(&estimates_path)
        .map_err(|e| format!("cannot read {}: {e}", estimates_path.display()))?;
    let estimates: CriterionEstimatesJson = serde_json::from_str(&estimates_text)
        .map_err(|e| format!("cannot parse {}: {e}", estimates_path.display()))?;

    let sample_text = std::fs::read_to_string(&sample_path)
        .map_err(|e| format!("cannot read {}: {e}", sample_path.display()))?;
    let sample: CriterionSampleJson = serde_json::from_str(&sample_text)
        .map_err(|e| format!("cannot parse {}: {e}", sample_path.display()))?;

    // Validate sample.json
    if sample.sampling_mode != "Flat" {
        return Err(format!(
            "unexpected Criterion sampling_mode {:?}, expected 'Flat'",
            sample.sampling_mode
        ));
    }
    if sample.iters.len() < 10 {
        return Err(format!(
            "insufficient Criterion samples: got {}, expected >= 10",
            sample.iters.len()
        ));
    }
    if sample.iters.len() != sample.times.len() {
        return Err(format!(
            "mismatched Criterion iters ({}) and times ({})",
            sample.iters.len(),
            sample.times.len()
        ));
    }
    for (iter, time) in sample.iters.iter().zip(sample.times.iter()) {
        if !iter.is_finite() || *iter <= 0.0 || !time.is_finite() || *time <= 0.0 {
            return Err(format!(
                "non-finite or non-positive sample iter/time ({iter}, {time})"
            ));
        }
    }

    // Validate estimates.json
    let ci = &estimates.median.confidence_interval;
    if (ci.confidence_level - 0.95).abs() > 1e-4 {
        return Err(format!(
            "unexpected confidence_level {}, expected 0.95",
            ci.confidence_level
        ));
    }

    let pt = estimates.median.point_estimate;
    let lo = ci.lower_bound;
    let hi = ci.upper_bound;

    if !pt.is_finite() || pt <= 0.0 || !lo.is_finite() || lo <= 0.0 || !hi.is_finite() || hi <= 0.0
    {
        return Err(format!(
            "non-finite or non-positive median estimates (pt={pt}, lo={lo}, hi={hi})"
        ));
    }
    if lo > pt || pt > hi {
        return Err(format!(
            "median point estimate ({pt}) outside CI [{lo}, {hi}]"
        ));
    }

    const NS_PER_SEC: f64 = 1_000_000_000.0;
    Ok(CriterionLatency {
        source: "criterion".to_string(),
        median_secs: pt / NS_PER_SEC,
        ci95_lower_secs: lo / NS_PER_SEC,
        ci95_upper_secs: hi / NS_PER_SEC,
        sample_count: sample.iters.len(),
        representative_sample_id: sample_id.to_string(),
    })
}

/// Execute Criterion timing closure for one session and input.
pub fn bench_session<B: Backend, M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<M>,
    backend_name: &str,
    model_key: &str,
    session: &Session<B>,
    graph: &penumbra_core::ir::Graph,
    input_cts: &penumbra_core::backend::CtVec<B>,
) {
    group.bench_function(BenchmarkId::new(backend_name, model_key), |b| {
        b.iter(|| {
            let res = session
                .eval(graph, input_cts)
                .unwrap_or_else(|e| panic!("session evaluation failed: {e}"));
            std::hint::black_box(res);
        });
    });
}

/// Parse backend selection from `PENUMBRA_BENCH_BACKENDS` environment variable.
pub fn backend_selection_from_env() -> Vec<String> {
    match std::env::var("PENUMBRA_BENCH_BACKENDS") {
        Ok(val) => {
            let list: Vec<String> = val
                .split(',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            if list.is_empty() {
                crate::available_backends()
                    .iter()
                    .map(|&s| s.to_string())
                    .collect()
            } else {
                list
            }
        }
        Err(_) => crate::available_backends()
            .iter()
            .map(|&s| s.to_string())
            .collect(),
    }
}

/// Run a canonical Criterion benchmark for a single session and model.
pub fn measure_latency<B: Backend>(
    session: &Session<B>,
    model: &LoadedModel,
    output_dir: &Path,
    sample_id: &str,
) -> Result<CriterionLatency, String> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("cannot create output dir {}: {e}", output_dir.display()))?;

    let mut criterion = Criterion::default()
        .output_directory(output_dir)
        .confidence_level(0.95)
        .without_plots();

    let backend_name = session.backend.name();
    let model_key = model.fixture.key;
    let group_name = format!("inference/{model_key}");

    let input_cts = session.encrypt(&model.inputs[0]);
    let mut group = criterion.benchmark_group(&group_name);
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(60));

    bench_session(
        &mut group,
        backend_name,
        model_key,
        session,
        &model.graph,
        &input_cts,
    );
    group.finish();

    // In Criterion 0.7, slashes in group names are replaced with underscores in path names
    let sanitized_group = group_name.replace('/', "_");
    let bench_dir = output_dir
        .join(sanitized_group)
        .join(backend_name)
        .join(model_key);

    parse_criterion_latency(&bench_dir, sample_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parse_valid_criterion_artifacts() {
        let dir = tempdir().unwrap();
        let new_dir = dir.path().join("new");
        std::fs::create_dir_all(&new_dir).unwrap();

        let estimates_json = r#"{
            "median": {
                "point_estimate": 1000000000.0,
                "confidence_interval": {
                    "confidence_level": 0.95,
                    "lower_bound": 950000000.0,
                    "upper_bound": 1050000000.0
                }
            }
        }"#;
        std::fs::write(new_dir.join("estimates.json"), estimates_json).unwrap();

        let sample_json = r#"{
            "sampling_mode": "Flat",
            "iters": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "times": [1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9]
        }"#;
        std::fs::write(new_dir.join("sample.json"), sample_json).unwrap();

        let lat = parse_criterion_latency(dir.path(), "test_0").unwrap();
        assert_eq!(lat.source, "criterion");
        assert_eq!(lat.sample_count, 10);
        assert_eq!(lat.representative_sample_id, "test_0");
        assert_eq!(lat.median_secs, 1.0);
        assert_eq!(lat.ci95_lower_secs, 0.95);
        assert_eq!(lat.ci95_upper_secs, 1.05);
    }

    #[test]
    fn test_parse_invalid_confidence_or_samples() {
        let dir = tempdir().unwrap();
        let new_dir = dir.path().join("new");
        std::fs::create_dir_all(&new_dir).unwrap();

        let estimates_bad_ci = r#"{
            "median": {
                "point_estimate": 1000000000.0,
                "confidence_interval": {
                    "confidence_level": 0.90,
                    "lower_bound": 950000000.0,
                    "upper_bound": 1050000000.0
                }
            }
        }"#;
        std::fs::write(new_dir.join("estimates.json"), estimates_bad_ci).unwrap();

        let sample_few = r#"{
            "sampling_mode": "Flat",
            "iters": [1.0, 1.0],
            "times": [1.0e9, 1.0e9]
        }"#;
        std::fs::write(new_dir.join("sample.json"), sample_few).unwrap();

        assert!(parse_criterion_latency(dir.path(), "test_0").is_err());
    }
}
