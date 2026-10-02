//! Paper evaluation protocol schemas, worker contracts, and validation.

use crate::report::NodeReport;
use penumbra_core::ir::Graph;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;

pub const CALIBRATION_SEED: u64 = 1507;
pub const SPOT_CHECK_SEED: u64 = 1503;
pub const CALIBRATION_LIMIT: usize = 128;
pub const TFHE_SPOT_CHECK_COUNT: usize = 30;
pub const CKKS_BOUND_MARGIN: f64 = 2.0;

/// Reference to a sample in either the test or calibration split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpotCheckRef {
    pub split: String,
    pub index: usize,
}

/// A single sample in the paper evaluation dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sample {
    pub id: String,
    pub inputs: Vec<i64>,
    pub expected_output: Vec<i64>,
    pub expected_label: i64,
    pub target: i64,
    pub expected_scores: Option<Vec<i64>>,
}

/// Baseline float accuracy provenance and metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloatAccuracy {
    pub value: f64,
    pub source: String,
    pub source_path: Option<String>,
    pub source_sha256: Option<String>,
    pub sample_count: usize,
    pub recomputed: bool,
}

/// Root `paper` evaluation metadata and data arrays in committed fixtures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaperData {
    pub schema_version: usize,
    pub dataset: String,
    pub graph_sha256: String,
    pub output_kind: String,
    pub score_tensor: String,
    pub decision_threshold: Option<i64>,
    pub calibration: Vec<Sample>,
    pub test: Vec<Sample>,
    pub tfhe_spot_check: Vec<SpotCheckRef>,
    pub float_accuracy: FloatAccuracy,
}

impl PaperData {
    /// Validate all paper protocol constraints against the graph and model key.
    pub fn validate(&self, model_key: &str, graph: &Graph) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "model {model_key}: unsupported paper schema_version {}",
                self.schema_version
            ));
        }

        if self.output_kind != "logits" && self.output_kind != "label" {
            return Err(format!(
                "model {model_key}: invalid output_kind {:?}",
                self.output_kind
            ));
        }

        if self.output_kind == "label" && self.decision_threshold.is_none() {
            return Err(format!(
                "model {model_key}: output_kind 'label' requires decision_threshold"
            ));
        }

        // Canonical graph hash check
        let expected_hash = canonical_graph_hash(graph)?;
        if self.graph_sha256 != expected_hash {
            return Err(format!(
                "model {model_key}: graph_sha256 mismatch: recorded {}, computed {}",
                self.graph_sha256, expected_hash
            ));
        }

        if self.calibration.is_empty() {
            return Err(format!("model {model_key}: calibration split is empty"));
        }
        if self.test.is_empty() {
            return Err(format!("model {model_key}: test split is empty"));
        }

        let is_faces = model_key == "phase7_faces";
        let expected_cal_count = if is_faces { 60 } else { CALIBRATION_LIMIT };
        if self.calibration.len() != expected_cal_count {
            return Err(format!(
                "model {model_key}: expected {expected_cal_count} calibration samples, got {}",
                self.calibration.len()
            ));
        }

        let mut cal_ids = HashSet::new();
        for s in &self.calibration {
            if !cal_ids.insert(&s.id) {
                return Err(format!(
                    "model {model_key}: duplicate ID {:?} in calibration split",
                    s.id
                ));
            }
            if self.output_kind == "label" && s.expected_scores.is_none() {
                return Err(format!(
                    "model {model_key}: sample {:?} missing expected_scores for label model",
                    s.id
                ));
            }
        }

        let mut test_ids = HashSet::new();
        for s in &self.test {
            if !test_ids.insert(&s.id) {
                return Err(format!(
                    "model {model_key}: duplicate ID {:?} in test split",
                    s.id
                ));
            }
            if cal_ids.contains(&s.id) {
                return Err(format!(
                    "model {model_key}: sample ID {:?} appears in both calibration and test splits",
                    s.id
                ));
            }
            if self.output_kind == "label" && s.expected_scores.is_none() {
                return Err(format!(
                    "model {model_key}: sample {:?} missing expected_scores for label model",
                    s.id
                ));
            }
        }

        if self.tfhe_spot_check.len() != TFHE_SPOT_CHECK_COUNT {
            return Err(format!(
                "model {model_key}: expected {TFHE_SPOT_CHECK_COUNT} tfhe_spot_checks, got {}",
                self.tfhe_spot_check.len()
            ));
        }

        let mut spot_check_ids = HashSet::new();
        for (i, sc) in self.tfhe_spot_check.iter().enumerate() {
            let sample_id = match sc.split.as_str() {
                "test" => {
                    if sc.index >= self.test.len() {
                        return Err(format!(
                            "model {model_key}: spot check {i} index {} out of bounds for test ({})",
                            sc.index, self.test.len()
                        ));
                    }
                    &self.test[sc.index].id
                }
                "calibration" => {
                    if !is_faces {
                        return Err(format!(
                            "model {model_key}: calibration spot checks are only allowed for faces"
                        ));
                    }
                    if sc.index >= self.calibration.len() {
                        return Err(format!(
                            "model {model_key}: spot check {i} index {} out of bounds for calibration ({})",
                            sc.index, self.calibration.len()
                        ));
                    }
                    &self.calibration[sc.index].id
                }
                other => {
                    return Err(format!(
                        "model {model_key}: invalid spot check split {other:?}"
                    ));
                }
            };
            if !spot_check_ids.insert(sample_id) {
                return Err(format!(
                    "model {model_key}: duplicate sample ID {sample_id:?} in tfhe_spot_check"
                ));
            }
        }

        Ok(())
    }
}

/// Compute canonical SHA-256 of an IR Graph.
pub fn canonical_graph_hash(graph: &Graph) -> Result<String, String> {
    let json_val = serde_json::to_value(graph)
        .map_err(|e| format!("cannot serialize graph to json value: {e}"))?;
    let canonical = canonical_json_string(&json_val);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(hex::encode(hasher.finalize()))
}

/// Produce canonical JSON string (sorted object keys, compact separators).
fn canonical_json_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            let parts: Vec<String> = entries
                .into_iter()
                .map(|(k, v)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap(),
                        canonical_json_string(v)
                    )
                })
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        serde_json::Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(canonical_json_string).collect();
            format!("[{}]", parts.join(","))
        }
        _ => serde_json::to_string(value).unwrap(),
    }
}

mod hex {
    pub fn encode<T: AsRef<[u8]>>(data: T) -> String {
        data.as_ref().iter().map(|b| format!("{b:02x}")).collect()
    }
}

// ---------------------------------------------------------------------------
// Worker configs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperConfig {
    pub threads: usize,
    pub output_dir: PathBuf,
    pub protocol_version: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareWorkerConfig {
    pub model_key: String,
    pub backend: String,
    pub profile: Option<String>,
    pub artifact_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerRssWorkerConfig {
    pub model_key: String,
    pub backend: String,
    pub profile: Option<String>,
    pub graph_path: PathBuf,
    pub server_key_path: PathBuf,
    pub input_path: PathBuf,
    pub output_meta_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsWorkerConfig {
    pub model_key: String,
    pub backend: String,
    pub profile: Option<String>,
    pub client_key_path: PathBuf,
    pub server_key_path: PathBuf,
    pub input_path: PathBuf,
    pub criterion_dir: PathBuf,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerConfig {
    Prepare(PrepareWorkerConfig),
    ServerRss(ServerRssWorkerConfig),
    Metrics(MetricsWorkerConfig),
}

// ---------------------------------------------------------------------------
// Paper output report schema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionLatency {
    pub source: String,
    pub median_secs: f64,
    pub ci95_lower_secs: f64,
    pub ci95_upper_secs: f64,
    pub sample_count: usize,
    pub representative_sample_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerMemoryMetrics {
    pub peak_server_rss_bytes: u64,
    pub method: String,
    pub scope: String,
    pub worker_pid: u32,
    pub sample_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireSizes {
    pub client_key_bytes: usize,
    pub server_key_bytes: usize,
    pub input_ciphertext_bytes: usize,
    pub output_ciphertext_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PbsSplit {
    pub total_pbs: u64,
    pub lookup_pbs: u64,
    pub carry_pbs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbsoluteDistribution {
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginRelativeMetrics {
    pub median: Option<f64>,
    pub p95: Option<f64>,
    pub max: Option<f64>,
    pub finite_count: usize,
    pub zero_margin_samples: usize,
    pub zero_margin_with_nonzero_error: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub sample_count: usize,
    pub encrypted_task_accuracy: f64,
    pub full_test_quantized_accuracy: f64,
    pub float_accuracy: FloatAccuracy,
    pub label_flips: usize,
    pub label_flip_rate: f64,
    pub absolute_output_error: Option<AbsoluteDistribution>,
    pub margin_relative_score_error: Option<MarginRelativeMetrics>,
    pub check_method: String,
    pub exact_checks_passed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperModelRun {
    pub status: String, // "ok" | "unsupported"
    pub model: String,
    pub backend: String,
    pub profile: Option<String>,
    pub graph_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejection: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency: Option<CriterionLatency>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<ServerMemoryMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sizes: Option<WireSizes>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pbs_split: Option<PbsSplit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<AccuracyMetrics>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<NodeReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_sample_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperReportMeta {
    pub machine_model: String,
    pub os_product_version: String,
    pub kernel_version: String,
    pub runtime_commit: String,
    pub build_commit: String,
    pub dirty: bool,
    pub rustc_version: String,
    pub requested_threads: usize,
    pub actual_threads: usize,
    pub hal_name: String,
    pub protocol_version: usize,
    pub mode: String,
}

impl PaperReportMeta {
    pub fn capture(mode: &str, requested_threads: usize) -> Result<Self, String> {
        let machine_model = if cfg!(target_os = "macos") {
            let out = Command::new("sysctl")
                .args(["-n", "hw.model"])
                .output()
                .map_err(|e| format!("sysctl failed: {e}"))?;
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        } else {
            std::fs::read_to_string("/proc/cpuinfo")
                .ok()
                .and_then(|text| {
                    for line in text.lines() {
                        if line.starts_with("model name") {
                            return line.split(':').nth(1).map(|s| s.trim().to_string());
                        }
                    }
                    None
                })
                .unwrap_or_else(|| "linux-x86_64".to_string())
        };

        let os_product_version = if cfg!(target_os = "macos") {
            let out = Command::new("sw_vers")
                .args(["-productVersion"])
                .output()
                .map_err(|e| format!("sw_vers failed: {e}"))?;
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        } else {
            std::fs::read_to_string("/etc/os-release")
                .ok()
                .and_then(|text| {
                    for line in text.lines() {
                        if line.starts_with("PRETTY_NAME=") {
                            return Some(
                                line.trim_start_matches("PRETTY_NAME=")
                                    .trim_matches('"')
                                    .to_string(),
                            );
                        }
                    }
                    None
                })
                .unwrap_or_else(|| std::env::consts::OS.to_string())
        };

        let kernel_version = Command::new("uname")
            .args(["-r"])
            .output()
            .map_err(|e| format!("uname -r failed: {e}"))
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())?;

        let runtime_commit = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .map_err(|e| format!("git rev-parse HEAD failed: {e}"))
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())?;

        let build_commit = env!("PENUMBRA_BUILD_COMMIT").to_string();

        let dirty = Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .map_err(|e| format!("git status failed: {e}"))
            .map(|out| !out.stdout.is_empty())?;

        let rustc_version = env!("PENUMBRA_BUILD_RUSTC").to_string();

        #[cfg(feature = "ckks")]
        let hal_name = penumbra_ckks::hal_backend_name().to_string();
        #[cfg(not(feature = "ckks"))]
        let hal_name = "none".to_string();

        let actual_threads = rayon::current_num_threads();

        if mode == "paper" || mode == "calibrate" || mode == "probe" {
            if dirty {
                return Err(format!(
                    "working tree has uncommitted changes; {mode} mode requires a clean working tree"
                ));
            }
            if runtime_commit != build_commit {
                return Err(format!(
                    "runtime commit ({runtime_commit}) does not match build commit ({build_commit}); binary must be rebuilt at HEAD"
                ));
            }
        }

        Ok(Self {
            machine_model,
            os_product_version,
            kernel_version,
            runtime_commit,
            build_commit,
            dirty,
            rustc_version,
            requested_threads,
            actual_threads,
            hal_name,
            protocol_version: 1,
            mode: mode.to_string(),
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperReport {
    pub schema_version: usize,
    pub meta: PaperReportMeta,
    pub runs: Vec<PaperModelRun>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paper_report_meta_capture_structure() {
        let meta = PaperReportMeta::capture("test", 4).expect("capture should succeed");
        assert!(!meta.machine_model.is_empty());
        assert!(!meta.os_product_version.is_empty());
        assert!(!meta.kernel_version.is_empty());
        assert_eq!(meta.requested_threads, 4);
        assert_eq!(meta.mode, "test");
        assert_eq!(meta.protocol_version, 1);
    }
}
