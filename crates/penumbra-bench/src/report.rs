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
}

/// Timing and verification report for a single input sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleReport {
    pub index: usize,
    pub encrypt_secs: f64,
    pub eval_secs: f64,
    pub decrypt_secs: f64,
    pub nodes: Vec<NodeReport>,
    pub max_abs_err: Option<f64>,
    pub label_matches: Option<bool>,
}

/// Aggregated report for one (backend, model) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRun {
    pub backend: String,
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
    /// Summed counters for one sample — this backend's cost proxy.
    pub cost_proxy: BTreeMap<String, u64>,
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
        }

        let t_dec = Instant::now();
        let decrypted = session.decrypt(&output_cts);
        let decrypt_secs = t_dec.elapsed().as_secs_f64();

        let mut max_abs_err = None;
        if let Some(logits) = &model.expected_logits {
            if i < logits.len() {
                let exp = &logits[i];
                let err = decrypted
                    .iter()
                    .zip(exp.iter())
                    .map(|(&a, &b)| (a - b).abs() as f64)
                    .fold(0.0, f64::max);
                max_abs_err = Some(err);
            }
        }

        let mut label_matches = None;
        if let Some(labels) = &model.expected_labels {
            if i < labels.len() {
                let expected_lbl = labels[i];
                if output_cts.len() == 1 {
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
            })
            .collect();

        sample_reports.push(SampleReport {
            index: i,
            encrypt_secs,
            eval_secs,
            decrypt_secs,
            nodes,
            max_abs_err,
            label_matches,
        });
    }

    // Mean per-sample seconds by op type
    let mut op_type_totals: BTreeMap<String, f64> = BTreeMap::new();
    for sample in &sample_reports {
        for node in &sample.nodes {
            *op_type_totals.entry(node.op_type.clone()).or_default() += node.eval_secs;
        }
    }
    let n_f64 = num_samples as f64;
    let op_type_secs: BTreeMap<String, f64> = op_type_totals
        .into_iter()
        .map(|(k, sum)| (k, sum / n_f64))
        .collect();

    Ok(ModelRun {
        backend: session.backend.name().to_string(),
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
        cost_proxy: first_cost_proxy,
    })
}

pub fn to_json(runs: &[ModelRun]) -> Result<String, String> {
    serde_json::to_string_pretty(runs).map_err(|e| format!("cannot serialize JSON report: {e}"))
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

pub fn to_markdown(runs: &[ModelRun]) -> String {
    let mut out = String::new();

    // Table 1: Latency
    out.push_str("### 1. Latency (Wall-Clock)\n\n");
    out.push_str("| Model | Backend | Keygen (s) | Encrypt (s) | Eval (s) | Decrypt (s) | Accuracy / Error |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|\n");
    for run in runs {
        let n = run.samples.len() as f64;
        let avg_enc = run.samples.iter().map(|s| s.encrypt_secs).sum::<f64>() / n;
        let avg_eval = run.samples.iter().map(|s| s.eval_secs).sum::<f64>() / n;
        let avg_dec = run.samples.iter().map(|s| s.decrypt_secs).sum::<f64>() / n;

        let acc_str = if let Some(s0) = run.samples.first() {
            if let Some(err) = s0.max_abs_err {
                format!("max |err| = {err:.2}")
            } else if let Some(m) = s0.label_matches {
                if m {
                    "exact match (pass)".to_string()
                } else {
                    "mismatch (fail)".to_string()
                }
            } else {
                "n/a".to_string()
            }
        } else {
            "n/a".to_string()
        };

        out.push_str(&format!(
            "| {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {} |\n",
            run.model, run.backend, run.keygen_secs, avg_enc, avg_eval, avg_dec, acc_str
        ));
    }
    out.push('\n');

    // Table 2: Per-Op-Type Eval Breakdown
    out.push_str("### 2. Per-Op-Type Eval Breakdown (Mean Seconds per Sample)\n\n");
    out.push_str("| Model | Backend | Op Type | Eval (s) |\n");
    out.push_str("|---|---|---|---:|\n");
    for run in runs {
        for (op, &secs) in &run.op_type_secs {
            out.push_str(&format!(
                "| {} | {} | {} | {:.4} |\n",
                run.model, run.backend, op, secs
            ));
        }
    }
    out.push('\n');

    // Table 3: Sizes & Scheme Cost Proxies
    out.push_str("### 3. Sizes & Scheme Cost Proxies\n\n");
    out.push_str("| Model | Backend | Input CT | Output CT | Client Key | Server Key | Cost Proxy Counters |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---|\n");
    for run in runs {
        let in_ct = format_bytes(run.input_ct_bytes);
        let out_ct = format_bytes(run.output_ct_bytes);
        let ck_sz = format_bytes(run.client_key_bytes);
        let sk_sz = format_bytes(run.server_key_bytes);

        let proxy_str = if run.cost_proxy.is_empty() {
            "none".to_string()
        } else {
            run.cost_proxy
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join(", ")
        };

        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            run.model, run.backend, in_ct, out_ct, ck_sz, sk_sz, proxy_str
        ));
    }
    out.push('\n');

    out
}
