//! Benchmark report generation and serialization (JSON / Markdown).

use std::collections::BTreeMap;
use std::time::Instant;

use penumbra_core::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::models::LoadedModel;
use crate::session::Session;

/// Measured execution metrics for a single graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeReport {
    pub name: String,
    pub op_type: String,
    pub build_secs: f64,
    pub eval_secs: f64,
    pub input_lens: Vec<usize>,
    pub output_len: usize,
    pub counters: BTreeMap<String, u64>,
    #[serde(default)]
    pub measured: BTreeMap<String, u64>,
}

/// Timing and verification report for a single input sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleReport {
    pub index: usize,
    pub encrypt_secs: f64,
    pub eval_secs: f64,
    pub decrypt_secs: f64,
    pub nodes: Vec<NodeReport>,
    /// Raw decrypted outputs — keeps any error statistic re-derivable from the artifact.
    pub decrypted: Vec<i64>,
    pub max_abs_err: Option<f64>,
    pub mean_abs_err: Option<f64>,
    pub label_matches: Option<bool>,
}

/// Aggregated report for one (backend, model) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRun {
    pub backend: String,
    /// The backend parameter profile this run used; `None` when the backend has no named profile.
    #[serde(default)]
    pub profile: Option<String>,
    pub model: String,
    pub fixture: String,
    pub num_blocks: usize,
    pub keygen_secs: f64,
    pub client_key_bytes: usize,
    pub server_key_bytes: usize,
    pub input_ct_bytes: usize,
    pub output_ct_bytes: usize,
    pub samples: Vec<SampleReport>,
    /// Mean per-sample seconds by op type — the "why one scheme wins" breakdown.
    pub op_type_secs: BTreeMap<String, f64>,
    /// Mean per-sample seconds spent in `Backend::build_op` (plaintext weight prep), by op type.
    pub op_type_build_secs: BTreeMap<String, f64>,
    /// Summed counters for one sample — this backend's cost proxy.
    pub cost_proxy: BTreeMap<String, u64>,
    /// Measured counters for one sample — ground truth, unlike `cost_proxy`.
    #[serde(default)]
    pub measured_totals: BTreeMap<String, u64>,
    #[serde(default)]
    pub accuracy: Option<crate::models::FixtureAccuracy>,
    #[serde(default)]
    pub bit_plan: Option<serde_json::Value>,
    #[serde(default)]
    pub ir_bytes: usize,
    #[serde(default)]
    pub ir_load_secs: f64,
}

/// Where a report came from. A committed result file that cannot be attributed to a machine,
/// build profile, and HAL backend is not a comparison result (`docs/COMPARISON.md`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMeta {
    pub os: String,
    pub arch: String,
    /// `false` means debug-build numbers — never publishable.
    pub release: bool,
    pub backends_compiled: Vec<String>,
    /// The `poulpy` HAL backend linked into this build; `None` without the `ckks` feature.
    pub ckks_hal: Option<String>,
    pub samples_requested: usize,
}

impl ReportMeta {
    pub fn capture(samples_requested: usize) -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            release: !cfg!(debug_assertions),
            backends_compiled: crate::available_backends()
                .into_iter()
                .map(str::to_string)
                .collect(),
            #[cfg(feature = "ckks")]
            ckks_hal: Some(penumbra_ckks::hal_backend_name().to_string()),
            #[cfg(not(feature = "ckks"))]
            ckks_hal: None,
            samples_requested,
        }
    }
}

/// One report: provenance plus every (backend, model) run in it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub meta: ReportMeta,
    pub runs: Vec<ModelRun>,
}

/// Execute a benchmark session over `model` using `backend` for `samples` inputs.
pub fn run_model<B: Backend>(
    backend: B,
    model: &LoadedModel,
    samples: usize,
) -> Result<ModelRun, String> {
    let session = Session::new(backend, &model.graph)?;
    let (client_key_bytes, server_key_bytes) = session.key_bytes()?;

    let num_samples = samples.min(model.inputs.len()).max(1);
    let mut sample_reports = Vec::with_capacity(num_samples);
    let mut first_measured_totals: BTreeMap<String, u64> = BTreeMap::new();

    let mut input_ct_bytes = 0;
    let mut output_ct_bytes = 0;
    let mut first_cost_proxy = BTreeMap::new();

    for i in 0..num_samples {
        let input = &model.inputs[i % model.inputs.len()];

        let t_enc = Instant::now();
        let input_cts = session.encrypt(input);
        let encrypt_secs = t_enc.elapsed().as_secs_f64();

        if i == 0 {
            input_ct_bytes = session.ct_bytes(&input_cts)?;
        }

        let (output_cts, profile) = session.eval(&model.graph, &input_cts)?;
        let eval_secs = profile.total.as_secs_f64();

        if i == 0 {
            output_ct_bytes = session.ct_bytes(&output_cts)?;
            first_cost_proxy = profile
                .counter_totals()
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            first_measured_totals = profile
                .measured_totals()
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
        }

        let t_dec = Instant::now();
        let decrypted = session.decrypt(&output_cts);
        let decrypt_secs = t_dec.elapsed().as_secs_f64();

        let mut max_abs_err = None;
        let mut mean_abs_err = None;
        if let Some(logits) = &model.expected_logits {
            if i < logits.len() {
                let exp = &logits[i];
                let errors: Vec<f64> = decrypted
                    .iter()
                    .zip(exp.iter())
                    .map(|(&a, &b)| (a - b).abs() as f64)
                    .collect();
                if !errors.is_empty() {
                    let max_err = errors.iter().copied().fold(0.0, f64::max);
                    let mean_err = errors.iter().copied().sum::<f64>() / errors.len() as f64;
                    max_abs_err = Some(max_err);
                    mean_abs_err = Some(mean_err);
                }
            }
        }

        let mut label_matches = None;
        if let Some(labels) = &model.expected_labels {
            if i < labels.len() {
                let expected_lbl = labels[i];
                if decrypted.len() == 1 {
                    let pred = session.decrypt_label(&output_cts);
                    label_matches = Some(pred == expected_lbl);
                } else if !decrypted.is_empty() {
                    let max_idx = decrypted
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, &v)| v)
                        .map(|(idx, _)| idx as i64)
                        .unwrap_or(-1);
                    label_matches = Some(max_idx == expected_lbl);
                }
            }
        }

        let nodes = profile
            .nodes
            .into_iter()
            .map(|n| NodeReport {
                name: n.name,
                op_type: n.op_type.to_string(),
                build_secs: n.build.as_secs_f64(),
                eval_secs: n.eval.as_secs_f64(),
                input_lens: n.input_lens,
                output_len: n.output_len,
                counters: n
                    .counters
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
                measured: n
                    .measured
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            })
            .collect();

        sample_reports.push(SampleReport {
            index: i,
            encrypt_secs,
            eval_secs,
            decrypt_secs,
            nodes,
            decrypted,
            max_abs_err,
            mean_abs_err,
            label_matches,
        });
    }

    // Mean per-sample seconds by op type (both eval and op-build)
    let mut op_type_eval_totals: BTreeMap<String, f64> = BTreeMap::new();
    let mut op_type_build_totals: BTreeMap<String, f64> = BTreeMap::new();
    for sample in &sample_reports {
        for node in &sample.nodes {
            *op_type_eval_totals.entry(node.op_type.clone()).or_default() += node.eval_secs;
            *op_type_build_totals
                .entry(node.op_type.clone())
                .or_default() += node.build_secs;
        }
    }
    let n_f64 = num_samples as f64;
    let op_type_secs: BTreeMap<String, f64> = op_type_eval_totals
        .into_iter()
        .map(|(k, sum)| (k, sum / n_f64))
        .collect();
    let op_type_build_secs: BTreeMap<String, f64> = op_type_build_totals
        .into_iter()
        .map(|(k, sum)| (k, sum / n_f64))
        .collect();

    Ok(ModelRun {
        backend: session.backend.name().to_string(),
        profile: None,
        model: model.fixture.key.to_string(),
        fixture: model.fixture.path.to_string(),
        num_blocks: session.num_blocks,
        keygen_secs: session.keygen.as_secs_f64(),
        client_key_bytes,
        server_key_bytes,
        input_ct_bytes,
        output_ct_bytes,
        samples: sample_reports,
        op_type_secs,
        op_type_build_secs,
        cost_proxy: first_cost_proxy,
        measured_totals: first_measured_totals,
        accuracy: model.accuracy,
        bit_plan: model.bit_plan.clone(),
        ir_bytes: model.ir_bytes,
        ir_load_secs: model.ir_load_secs,
    })
}

pub fn to_json(report: &Report) -> Result<String, String> {
    serde_json::to_string_pretty(report).map_err(|e| format!("cannot serialize JSON report: {e}"))
}

fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

pub fn to_markdown(report: &Report) -> String {
    let mut out = String::new();

    // Table 0: Provenance
    out.push_str("### 0. Provenance\n\n");
    out.push_str("| Property | Value |\n");
    out.push_str("|---|---|\n");
    out.push_str(&format!(
        "| OS / arch | {} / {} |\n",
        report.meta.os, report.meta.arch
    ));
    out.push_str(&format!(
        "| Build profile | {} |\n",
        if report.meta.release {
            "release"
        } else {
            "debug"
        }
    ));
    out.push_str(&format!(
        "| Backends compiled | {} |\n",
        report.meta.backends_compiled.join(", ")
    ));
    out.push_str(&format!(
        "| CKKS HAL backend | {} |\n",
        report.meta.ckks_hal.as_deref().unwrap_or("n/a")
    ));
    out.push_str(&format!(
        "| Samples requested | {} |\n\n",
        report.meta.samples_requested
    ));

    if !report.meta.release {
        out.push_str("> **DEBUG BUILD — these numbers are not comparison-grade.**\n\n");
    }

    // Table 1: Latency
    out.push_str("### 1. Latency (Wall-Clock)\n\n");
    out.push_str("| Model | Backend | Profile | Keygen (s) | Encrypt (s) | Eval total (s) | of which op-build (s) | Decrypt (s) | Accuracy / Error |\n");
    out.push_str("|---|---|---|---:|---:|---:|---:|---:|---|\n");
    for run in &report.runs {
        let n = run.samples.len() as f64;
        let avg_enc = run.samples.iter().map(|s| s.encrypt_secs).sum::<f64>() / n;
        let avg_eval = run.samples.iter().map(|s| s.eval_secs).sum::<f64>() / n;
        let avg_build = run
            .samples
            .iter()
            .map(|s| s.nodes.iter().map(|node| node.build_secs).sum::<f64>())
            .sum::<f64>()
            / n;
        let avg_dec = run.samples.iter().map(|s| s.decrypt_secs).sum::<f64>() / n;

        let has_err = run.samples.iter().any(|s| s.max_abs_err.is_some());
        let has_lbl = run.samples.iter().any(|s| s.label_matches.is_some());

        let mut acc_parts = Vec::new();
        if has_err {
            let max_err = run
                .samples
                .iter()
                .filter_map(|s| s.max_abs_err)
                .fold(0.0, f64::max);
            let mean_errs: Vec<f64> = run.samples.iter().filter_map(|s| s.mean_abs_err).collect();
            let mean_err = if !mean_errs.is_empty() {
                mean_errs.iter().sum::<f64>() / mean_errs.len() as f64
            } else {
                0.0
            };
            acc_parts.push(format!("max |err| = {max_err:.3}, mean = {mean_err:.3}"));
        }
        if has_lbl {
            let k = run
                .samples
                .iter()
                .filter(|s| s.label_matches == Some(true))
                .count();
            let n = run
                .samples
                .iter()
                .filter(|s| s.label_matches.is_some())
                .count();
            acc_parts.push(format!("labels {k}/{n}"));
        }
        let acc_str = if acc_parts.is_empty() {
            "n/a".to_string()
        } else {
            acc_parts.join("; ")
        };

        let prof_str = run.profile.as_deref().unwrap_or("-");
        out.push_str(&format!(
            "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} |\n",
            run.model,
            run.backend,
            prof_str,
            run.keygen_secs,
            avg_enc,
            avg_eval,
            avg_build,
            avg_dec,
            acc_str
        ));
    }
    out.push('\n');

    // Table 2: Per-Op-Type Eval Breakdown
    out.push_str("### 2. Per-Op-Type Eval Breakdown (Mean Seconds per Sample)\n\n");
    out.push_str("| Model | Backend | Op Type | Calls | Build (s) | Eval (s) | PBS (measured) |\n");
    out.push_str("|---|---|---|---:|---:|---:|---:|\n");
    for run in &report.runs {
        for (op, &eval_secs) in &run.op_type_secs {
            let calls = run
                .samples
                .first()
                .map(|s| s.nodes.iter().filter(|n| &n.op_type == op).count())
                .unwrap_or(0);
            let build_secs = run.op_type_build_secs.get(op).copied().unwrap_or(0.0);
            let pbs_str = run
                .samples
                .first()
                .and_then(|s| {
                    let nodes: Vec<_> = s.nodes.iter().filter(|n| &n.op_type == op).collect();
                    let has_pbs = nodes.iter().any(|n| n.measured.contains_key("pbs"));
                    if has_pbs {
                        let sum: u64 = nodes
                            .iter()
                            .map(|n| n.measured.get("pbs").copied().unwrap_or(0))
                            .sum();
                        Some(sum.to_string())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| "-".to_string());
            out.push_str(&format!(
                "| {} | {} | {} | {} | {:.4} | {:.4} | {} |\n",
                run.model, run.backend, op, calls, build_secs, eval_secs, pbs_str
            ));
        }
    }
    out.push('\n');

    // Table 3: Sizes & Scheme Cost Proxies
    out.push_str("### 3. Sizes & Scheme Cost Proxies\n\n");
    out.push_str("| Model | Backend | Input CT | Output CT | Client Key | Server Key | Float acc | Quantized acc | Cost Proxy Counters |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---|\n");
    for run in &report.runs {
        let in_ct = format_bytes(run.input_ct_bytes);
        let out_ct = format_bytes(run.output_ct_bytes);
        let ck_sz = format_bytes(run.client_key_bytes);
        let sk_sz = format_bytes(run.server_key_bytes);

        let mut proxy_str = if run.cost_proxy.is_empty() {
            "none".to_string()
        } else {
            run.cost_proxy
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        if !run.measured_totals.is_empty() {
            let measured_parts = run
                .measured_totals
                .iter()
                .map(|(k, v)| format!("measured {k}: {v}"))
                .collect::<Vec<_>>()
                .join(", ");
            if proxy_str == "none" {
                proxy_str = measured_parts;
            } else {
                proxy_str.push_str(&format!(", {measured_parts}"));
            }
        }

        let (float_acc_str, quant_acc_str) = match run.accuracy {
            Some(acc) => (
                format!("{:.4}", acc.float_accuracy),
                format!("{:.4}", acc.quantized),
            ),
            None => ("-".to_string(), "-".to_string()),
        };

        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            run.model,
            run.backend,
            in_ct,
            out_ct,
            ck_sz,
            sk_sz,
            float_acc_str,
            quant_acc_str,
            proxy_str
        ));
    }
    out.push('\n');

    // Table 4: IR Load Cost & Binary-Format Evidence
    out.push_str("### 4. IR Load Cost & Binary-Format Evidence\n\n");
    out.push_str(
        "| Model | Backend | IR bytes | IR load (ms) | Eval total (s) | IR load as % of eval |\n",
    );
    out.push_str("|---|---|---:|---:|---:|---:|\n");
    for run in &report.runs {
        let n = run.samples.len() as f64;
        let avg_eval = run.samples.iter().map(|s| s.eval_secs).sum::<f64>() / n;
        let ir_load_ms = run.ir_load_secs * 1000.0;
        let pct = if avg_eval > 0.0 {
            (run.ir_load_secs / avg_eval) * 100.0
        } else {
            0.0
        };
        out.push_str(&format!(
            "| {} | {} | {} | {:.2} | {:.4} | {:.4}% |\n",
            run.model,
            run.backend,
            format_bytes(run.ir_bytes),
            ir_load_ms,
            avg_eval,
            pct
        ));
    }
    out.push('\n');

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_committed_comparison_json_deserialization() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let json_path = manifest_dir.join("../../docs/results/phase12-4-comparison.json");
        if json_path.exists() {
            let data = std::fs::read_to_string(&json_path).expect("read comparison json");
            let report: Report = serde_json::from_str(&data).expect("deserialize Report");
            assert!(!report.runs.is_empty());
            let md = to_markdown(&report);
            assert!(md.contains("Per-Op-Type Eval Breakdown"));
            assert!(md.contains("PBS (measured)"));
        }
    }

    #[test]
    fn test_committed_phase10_sweep_json_deserialization() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let json_path = manifest_dir.join("../../docs/results/phase10-tfhe-sweep.json");
        if json_path.exists() {
            let data = std::fs::read_to_string(&json_path).expect("read phase10 sweep json");
            let report: Report = serde_json::from_str(&data).expect("deserialize Report");
            assert_eq!(report.runs.len(), 7);
            let md = to_markdown(&report);
            assert!(md.contains("Per-Op-Type Eval Breakdown"));
            assert!(md.contains("PBS (measured)"));
            assert!(md.contains("measured pbs:"));
        }
    }
}
