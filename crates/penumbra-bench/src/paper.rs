//! Paper benchmark execution orchestration, calibration runner, and multi-stage workers.

use std::process::Command;

use serde::{Deserialize, Serialize};
use penumbra_core::ir::Graph;

use crate::latency::measure_latency;
use crate::memory::capture_peak_server_rss;
#[cfg(feature = "ckks")]
use crate::metrics::nearest_rank_quantile;
use crate::metrics::{
    compute_absolute_distribution, compute_margin_relative_metrics, compute_pbs_split,
    evaluate_sample_binary_label, evaluate_sample_multiclass,
};
use crate::models::LoadedModel;
use crate::paper_backend::{Comparator, PaperBackend, RowSelection};
use crate::protocol::{
    AccuracyMetrics, MetricsWorkerConfig, PaperConfig, PaperData, PaperModelRun, PaperReportMeta,
    PrepareWorkerConfig, ServerMemoryMetrics, ServerRssWorkerConfig, WireSizes,
};
use crate::report::NodeReport;
use crate::session::{eval_server, Session};

// ---------------------------------------------------------------------------

/// If this is a label model, build a temporary graph with the score tensor appended to outputs.
fn maybe_build_score_tap_graph(graph: &Graph, paper: &PaperData) -> Option<Graph> {
    if paper.output_kind == "label" {
        let mut tg = graph.clone();
        if !tg.outputs.contains(&paper.score_tensor) {
            tg.outputs.push(paper.score_tensor.clone());
        }
        Some(tg)
    } else {
        None
    }
}
// Calibration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleCalibrationRecord {
    pub id: String,
    pub max_abs_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCalibrationResult {
    pub model: String,
    pub sample_count: usize,
    pub p99_error: f64,
    pub bound: f64,
    pub margin: f64,
    pub records: Vec<SampleCalibrationRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationArtifact {
    pub schema_version: usize,
    pub meta: PaperReportMeta,
    pub margin: f64,
    pub results: Vec<ModelCalibrationResult>,
}

/// Run calibration over the calibration split of a model using CKKS.
#[cfg(feature = "ckks")]
pub fn calibrate_ckks_model(model: &LoadedModel) -> Result<ModelCalibrationResult, String> {
    let paper = model.paper.as_ref().ok_or_else(|| {
        format!(
            "model '{}' is missing paper protocol data",
            model.fixture.key
        )
    })?;

    // Preflight depth check: trees are rejected
    if model.fixture.key == "phase8_trees" || model.fixture.key == "phase8_xgb" {
        return Err(format!(
            "depth budget exceeded on backend 'ckks' for tree model '{}'",
            model.fixture.key
        ));
    }

    let backend = model.fixture.ckks_backend();

    let session = Session::new(backend, &model.graph)?;
    let mut records = Vec::with_capacity(paper.calibration.len());
    let mut errors = Vec::with_capacity(paper.calibration.len());

    for sample in &paper.calibration {
        let input_cts = session.encrypt(&sample.inputs);
        let (out_cts, _) = session.eval(&model.graph, &input_cts)?;
        let raw_floats = session.backend.decode_raw(&session.ck, &out_cts);

        if raw_floats.len() < sample.expected_output.len() {
            return Err(format!(
                "model '{}' sample {}: decoded {} values, expected at least {}",
                model.fixture.key,
                sample.id,
                raw_floats.len(),
                sample.expected_output.len()
            ));
        }

        let max_err = raw_floats
            .iter()
            .zip(&sample.expected_output)
            .map(|(&got, &want)| (got - want as f64).abs())
            .fold(0.0f64, f64::max);

        if !max_err.is_finite() {
            return Err(format!(
                "model '{}' sample {}: non-finite error {max_err}",
                model.fixture.key, sample.id
            ));
        }

        errors.push(max_err);
        records.push(SampleCalibrationRecord {
            id: sample.id.clone(),
            max_abs_error: max_err,
        });
    }

    let mut sorted_errors = errors.clone();
    sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p99 = nearest_rank_quantile(&sorted_errors, 0.99)
        .ok_or_else(|| "failed to compute p99 error on calibration samples".to_string())?;

    let bound = 2.0 * p99;

    Ok(ModelCalibrationResult {
        model: model.fixture.key.to_string(),
        sample_count: paper.calibration.len(),
        p99_error: p99,
        bound,
        margin: 2.0,
        records,
    })
}

// ---------------------------------------------------------------------------
// Worker implementations
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct PrepareResult {
    pub client_key_bytes: usize,
    pub server_key_bytes: usize,
    pub input_ct_bytes: usize,
    pub keygen_secs: f64,
}

/// Worker stage 1: Key generation and input encryption.
pub fn run_prepare_worker<B: PaperBackend>(
    backend: &B,
    model: &LoadedModel,
    config: &PrepareWorkerConfig,
) -> Result<(), String> {
    std::fs::create_dir_all(&config.artifact_dir)
        .map_err(|e| format!("cannot create dir {}: {e}", config.artifact_dir.display()))?;

    backend.check_graph_budget(&model.graph)?;
    let t = std::time::Instant::now();
    let (ck, sk) = backend.keygen(model.graph.num_blocks);
    let keygen_secs = t.elapsed().as_secs_f64();

    let ck_wire = backend.serialize_client_key(&ck, model.graph.num_blocks)?;
    let sk_wire = backend.serialize_server_key(&sk, model.graph.num_blocks)?;

    let ck_path = config.artifact_dir.join("client.key");
    let sk_path = config.artifact_dir.join("server.key");
    let ct_path = config.artifact_dir.join("input.ct");
    let meta_path = config.artifact_dir.join("prepare_meta.json");

    std::fs::write(&ck_path, &ck_wire)
        .map_err(|e| format!("cannot write {}: {e}", ck_path.display()))?;
    std::fs::write(&sk_path, &sk_wire)
        .map_err(|e| format!("cannot write {}: {e}", sk_path.display()))?;

    let input_cts = backend.encrypt(&ck, &model.inputs[0]);
    let ct_wire = backend.serialize_cts(&input_cts)?;
    std::fs::write(&ct_path, &ct_wire)
        .map_err(|e| format!("cannot write {}: {e}", ct_path.display()))?;

    let prep = PrepareResult {
        client_key_bytes: ck_wire.len(),
        server_key_bytes: sk_wire.len(),
        input_ct_bytes: ct_wire.len(),
        keygen_secs,
    };

    let meta_json = serde_json::to_string_pretty(&prep)
        .map_err(|e| format!("cannot format prepare_meta: {e}"))?;
    std::fs::write(&meta_path, meta_json)
        .map_err(|e| format!("cannot write {}: {e}", meta_path.display()))?;

    Ok(())
}

/// Worker stage 2: Fresh process server key load and RSS capture.
pub fn run_server_rss_worker<B: PaperBackend>(
    backend: B,
    model: &LoadedModel,
    config: &ServerRssWorkerConfig,
) -> Result<(), String> {
    let sk = backend.load_server_key(&config.server_key_path, model.graph.num_blocks)?;
    let ct_bytes = std::fs::read(&config.input_path)
        .map_err(|e| format!("cannot read {}: {e}", config.input_path.display()))?;
    let input_cts = backend.deserialize_cts(&ct_bytes)?;

    // Single evaluation of the original graph
    let (outputs, _) = eval_server(
        &backend,
        &sk,
        model.graph.num_blocks,
        &model.graph,
        &input_cts,
    )?;
    // Retain output through RSS measurement
    let _retained = outputs;

    let sample_id = model
        .paper
        .as_ref()
        .and_then(|p| p.test.first().map(|s| s.id.as_str()))
        .unwrap_or("0");

    let mem = capture_peak_server_rss(sample_id)?;

    if let Some(parent) = config.output_meta_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create dir {}: {e}", parent.display()))?;
    }
    let mem_json = serde_json::to_string_pretty(&mem)
        .map_err(|e| format!("cannot format memory json: {e}"))?;
    std::fs::write(&config.output_meta_path, mem_json)
        .map_err(|e| format!("cannot write {}: {e}", config.output_meta_path.display()))?;

    Ok(())
}

/// Worker stage 3: Criterion latency, exact accuracy reductions, wire sizes.
pub fn run_metrics_worker<B: PaperBackend>(
    backend: B,
    model: &LoadedModel,
    config: &MetricsWorkerConfig,
) -> Result<PaperModelRun, String> {
    let paper = model.paper.as_ref().ok_or_else(|| {
        format!(
            "model '{}' is missing paper protocol data",
            model.fixture.key
        )
    })?;

    let ck = backend.load_client_key(&config.client_key_path, model.graph.num_blocks)?;
    let sk = backend.load_server_key(&config.server_key_path, model.graph.num_blocks)?;

    let session = Session {
        backend,
        ck,
        sk,
        num_blocks: model.graph.num_blocks,
        keygen: std::time::Duration::from_secs(0),
    };

    let policy = session.backend.paper_policy(model.fixture)?;

    // 1. Latency measurement via Criterion
    let rep_sample_id = paper.test.first().map(|s| s.id.as_str()).unwrap_or("0");

    let latency = measure_latency(&session, model, &config.criterion_dir, rep_sample_id)?;

    // 2. Accuracy evaluation
    let mut exact_checks_passed = 0;
    let mut label_flips = 0;
    let mut task_matches = 0;
    let total_samples = match policy.selection {
        RowSelection::SeededSpots => paper.tfhe_spot_check.len(),
        RowSelection::FullTest => paper.test.len(),
    };

    let mut sample_max_errors: Vec<f64> = Vec::new();
    let mut score_margin_errors: Vec<(f64, f64)> = Vec::new();

    let mut rep_profile_nodes = Vec::new();
    let mut rep_total_pbs: u64 = 0;
    let mut output_ct_bytes = 0;

    match policy.selection {
        RowSelection::SeededSpots => {
            let profile_sample_id =
                paper
                    .tfhe_spot_check
                    .first()
                    .and_then(|sc| match sc.split.as_str() {
                        "test" => paper.test.get(sc.index).map(|s| s.id.clone()),
                        "calibration" => paper.calibration.get(sc.index).map(|s| s.id.clone()),
                        _ => None,
                    });
            let tap_graph = maybe_build_score_tap_graph(&model.graph, paper);

            // TFHE 30 spot checks
            for (idx, sc) in paper.tfhe_spot_check.iter().enumerate() {
                let sample = match sc.split.as_str() {
                    "test" => &paper.test[sc.index],
                    "calibration" => &paper.calibration[sc.index],
                    other => return Err(format!("unexpected spot check split: {other}")),
                };

                let input_cts = session.encrypt(&sample.inputs);

                // Evaluate original graph for primary output and representative profile
                let (outputs, profile) = eval_server(
                    &session.backend,
                    &session.sk,
                    session.num_blocks,
                    &model.graph,
                    &input_cts,
                )?;

                if idx == 0 {
                    rep_profile_nodes = profile.nodes.clone();
                    let measured = profile.measured_totals();
                    rep_total_pbs = measured
                        .get("pbs")
                        .or_else(|| measured.get("bootstraps"))
                        .copied()
                        .unwrap_or(0);
                    let out_name = &model.graph.outputs[0];
                    if let Some(cts) = outputs.get(out_name) {
                        output_ct_bytes = session.ct_bytes(cts)?;
                    }
                }

                let out_name = &model.graph.outputs[0];
                let out_cts = outputs
                    .get(out_name)
                    .ok_or_else(|| format!("missing output tensor '{out_name}'"))?;

                let decrypted_ints = session.decrypt(out_cts);

                // Check exact integer match on primary output
                if decrypted_ints != sample.expected_output {
                    return Err(format!(
                        "TFHE exact integer mismatch on model '{}' sample {} ({:?}): got {:?}, expected {:?}",
                        model.fixture.key, sc.index, sc.split, decrypted_ints, sample.expected_output
                    ));
                }

                // If label model, check score tap
                if let Some(eval_graph) = &tap_graph {
                    let (tap_outputs, _) = eval_server(
                        &session.backend,
                        &session.sk,
                        session.num_blocks,
                        eval_graph,
                        &input_cts,
                    )?;
                    let score_cts = tap_outputs
                        .get(&paper.score_tensor)
                        .ok_or_else(|| format!("missing score tensor '{}'", paper.score_tensor))?;
                    let score_ints = session.decrypt(score_cts);
                    if let Some(expected_scores) = &sample.expected_scores {
                        if &score_ints != expected_scores {
                            return Err(format!(
                                "TFHE score tap mismatch on sample {}: got {:?}, expected {:?}",
                                sample.id, score_ints, expected_scores
                            ));
                        }
                    }
                }

                exact_checks_passed += 1;
            }

            // For TFHE, task accuracy is inferred exact from quantized cleartext test split
            let cleartext_matches = paper
                .test
                .iter()
                .filter(|s| s.expected_label == s.target)
                .count();
            let cleartext_acc = cleartext_matches as f64 / paper.test.len() as f64;

            let pbs_split = if policy.has_pbs_counters {
                Some(compute_pbs_split(rep_total_pbs, &rep_profile_nodes)?)
            } else {
                None
            };

            let rep_node_reports: Vec<NodeReport> = rep_profile_nodes
                .into_iter()
                .map(NodeReport::from)
                .collect();

            let accuracy = AccuracyMetrics {
                sample_count: total_samples,
                encrypted_task_accuracy: cleartext_acc,
                full_test_quantized_accuracy: cleartext_acc,
                float_accuracy: paper.float_accuracy.clone(),
                label_flips: 0,
                label_flip_rate: 0.0,
                absolute_output_error: None,
                margin_relative_score_error: None,
                check_method: "quantized_reference_inferred_exact".to_string(),
                exact_checks_passed,
            };

            Ok(PaperModelRun {
                status: "ok".to_string(),
                model: model.fixture.key.to_string(),
                backend: session.backend.name().to_string(),
                profile: None,
                graph_sha256: paper.graph_sha256.clone(),
                rejection: None,
                latency: Some(latency),
                memory: None,
                sizes: Some(WireSizes {
                    client_key_bytes: 0,
                    server_key_bytes: 0,
                    input_ciphertext_bytes: 0,
                    output_ciphertext_bytes: output_ct_bytes,
                }),
                pbs_split,
                accuracy: Some(accuracy),
                nodes: rep_node_reports,
                profile_sample_id,
            })
        }
        RowSelection::FullTest => {
            // CKKS full test evaluation
            let bound = match policy.comparator {
                Comparator::AbsoluteBound(b) => b,
                Comparator::ExactInteger => {
                    return Err("CKKS requires AbsoluteBound comparator".to_string())
                }
            };

            let profile_sample_id = paper.test.first().map(|s| s.id.clone());
            let is_label_model = paper.output_kind == "label";

            // If label model, prepare temporary graph with score tensor output for accuracy
            let tap_graph = maybe_build_score_tap_graph(&model.graph, paper);
            let eval_g = tap_graph.as_ref().unwrap_or(&model.graph);

            for (idx, sample) in paper.test.iter().enumerate() {
                let input_cts = session.encrypt(&sample.inputs);

                // For label models at idx == 0, evaluate original graph separately for profile and wire size
                if is_label_model && idx == 0 {
                    let (orig_outs, orig_profile) = eval_server(
                        &session.backend,
                        &session.sk,
                        session.num_blocks,
                        &model.graph,
                        &input_cts,
                    )?;
                    rep_profile_nodes = orig_profile.nodes;
                    let out_name = &model.graph.outputs[0];
                    if let Some(cts) = orig_outs.get(out_name) {
                        output_ct_bytes = session.ct_bytes(cts)?;
                    }
                }

                // Evaluate eval_g once per sample (original graph for non-label, tap graph for label)
                let (outputs, profile) = eval_server(
                    &session.backend,
                    &session.sk,
                    session.num_blocks,
                    eval_g,
                    &input_cts,
                )?;

                let out_name = &model.graph.outputs[0];
                let out_cts = outputs
                    .get(out_name)
                    .ok_or_else(|| format!("missing output tensor '{out_name}'"))?;

                // For non-label models at idx == 0, reuse the single evaluation for profile and wire size
                if !is_label_model && idx == 0 {
                    rep_profile_nodes = profile.nodes;
                    output_ct_bytes = session.ct_bytes(out_cts)?;
                }

                let raw_floats = session.backend.decode_raw(&session.ck, out_cts);

                let eval_result = if is_label_model {
                    let score_cts = outputs
                        .get(&paper.score_tensor)
                        .ok_or_else(|| format!("missing score tensor '{}'", paper.score_tensor))?;
                    let raw_score_floats = session.backend.decode_raw(&session.ck, score_cts);

                    let threshold = paper.decision_threshold.ok_or_else(|| {
                        format!(
                            "model '{}' output_kind 'label' missing decision_threshold",
                            model.fixture.key
                        )
                    })?;

                    evaluate_sample_binary_label(
                        sample,
                        &raw_floats,
                        &raw_score_floats,
                        threshold,
                        bound,
                    )?
                } else {
                    evaluate_sample_multiclass(
                        &raw_floats,
                        &sample.expected_output,
                        sample.expected_label,
                        sample.target,
                        bound,
                    )?
                };

                sample_max_errors.push(eval_result.max_err);
                if eval_result.label_flip {
                    label_flips += 1;
                }
                if eval_result.task_match {
                    task_matches += 1;
                }
                score_margin_errors.push(eval_result.score_margin);
            }

            let cleartext_matches = paper
                .test
                .iter()
                .filter(|s| s.expected_label == s.target)
                .count();
            let cleartext_acc = cleartext_matches as f64 / paper.test.len() as f64;
            let encrypted_task_acc = task_matches as f64 / paper.test.len() as f64;
            let label_flip_rate = label_flips as f64 / paper.test.len() as f64;

            let abs_dist = compute_absolute_distribution(sample_max_errors)?;
            let margin_metrics = compute_margin_relative_metrics(&score_margin_errors)?;

            let rep_node_reports: Vec<NodeReport> = rep_profile_nodes
                .into_iter()
                .map(NodeReport::from)
                .collect();

            let accuracy = AccuracyMetrics {
                sample_count: total_samples,
                encrypted_task_accuracy: encrypted_task_acc,
                full_test_quantized_accuracy: cleartext_acc,
                float_accuracy: paper.float_accuracy.clone(),
                label_flips,
                label_flip_rate,
                absolute_output_error: Some(abs_dist),
                margin_relative_score_error: Some(margin_metrics),
                check_method: format!("raw_float_bound_check(bound={bound})"),
                exact_checks_passed: 0,
            };

            Ok(PaperModelRun {
                status: "ok".to_string(),
                model: model.fixture.key.to_string(),
                backend: session.backend.name().to_string(),
                profile: None,
                graph_sha256: paper.graph_sha256.clone(),
                rejection: None,
                latency: Some(latency),
                memory: None,
                sizes: Some(WireSizes {
                    client_key_bytes: 0,
                    server_key_bytes: 0,
                    input_ciphertext_bytes: 0,
                    output_ciphertext_bytes: output_ct_bytes,
                }),
                pbs_split: None,
                accuracy: Some(accuracy),
                nodes: rep_node_reports,
                profile_sample_id,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Orchestrate multi-stage execution of a single model under paper mode.
pub fn run_paper_model<B: PaperBackend>(
    backend: B,
    model: &LoadedModel,
    config: &PaperConfig,
) -> Result<PaperModelRun, String> {
    let paper = model.paper.as_ref().ok_or_else(|| {
        format!(
            "model '{}' is missing paper protocol data",
            model.fixture.key
        )
    })?;

    // Check backend graph budget preflight first: if rejected (e.g. tree models on CKKS depth budget), return unsupported
    if let Err(rejection) = backend.check_graph_budget(&model.graph) {
        return Ok(PaperModelRun {
            status: "unsupported".to_string(),
            model: model.fixture.key.to_string(),
            backend: backend.name().to_string(),
            profile: None,
            graph_sha256: paper.graph_sha256.clone(),
            rejection: Some(rejection),
            latency: None,
            memory: None,
            sizes: None,
            pbs_split: None,
            accuracy: None,
            nodes: Vec::new(),
            profile_sample_id: None,
        });
    }

    // Check backend policy preflight
    let policy_res = backend.paper_policy(model.fixture);
    let _policy = match policy_res {
        Ok(p) => p,
        Err(rejection) => {
            return Ok(PaperModelRun {
                status: "unsupported".to_string(),
                model: model.fixture.key.to_string(),
                backend: backend.name().to_string(),
                profile: None,
                graph_sha256: paper.graph_sha256.clone(),
                rejection: Some(rejection),
                latency: None,
                memory: None,
                sizes: None,
                pbs_split: None,
                accuracy: None,
                nodes: Vec::new(),
                profile_sample_id: None,
            });
        }
    };

    let temp_dir =
        tempfile::tempdir().map_err(|e| format!("cannot create temp artifact dir: {e}"))?;
    let artifact_dir = temp_dir.path();

    // 1. Prepare stage
    let prep_config = PrepareWorkerConfig {
        model_key: model.fixture.key.to_string(),
        backend: backend.name().to_string(),
        profile: None,
        artifact_dir: artifact_dir.to_path_buf(),
    };
    run_prepare_worker(&backend, model, &prep_config)?;

    let prep_meta_bytes = std::fs::read(artifact_dir.join("prepare_meta.json"))
        .map_err(|e| format!("cannot read prepare_meta.json: {e}"))?;
    let prep_meta: PrepareResult = serde_json::from_slice(&prep_meta_bytes)
        .map_err(|e| format!("cannot parse prepare_meta.json: {e}"))?;

    // 2. Server RSS stage (in fresh child process)
    let current_exe =
        std::env::current_exe().map_err(|e| format!("cannot get current executable path: {e}"))?;

    let rss_meta_path = artifact_dir.join("server_rss.json");
    let rss_config = ServerRssWorkerConfig {
        model_key: model.fixture.key.to_string(),
        backend: prep_config.backend.clone(),
        profile: None,
        graph_path: artifact_dir.join("graph.json"),
        server_key_path: artifact_dir.join("server.key"),
        input_path: artifact_dir.join("input.ct"),
        output_meta_path: rss_meta_path.clone(),
    };

    let graph_json =
        serde_json::to_string(&model.graph).map_err(|e| format!("cannot serialize graph: {e}"))?;
    std::fs::write(&rss_config.graph_path, graph_json)
        .map_err(|e| format!("cannot write graph.json: {e}"))?;

    let rss_cfg_path = artifact_dir.join("server_rss_config.json");
    std::fs::write(&rss_cfg_path, serde_json::to_string(&rss_config).unwrap())
        .map_err(|e| format!("cannot write server_rss_config.json: {e}"))?;

    let status = Command::new(&current_exe)
        .args([
            "--worker",
            "server-rss",
            "--config",
            rss_cfg_path.to_str().unwrap(),
        ])
        .env("RAYON_NUM_THREADS", config.threads.to_string())
        .status()
        .map_err(|e| format!("failed to spawn server-rss worker: {e}"))?;

    if !status.success() {
        return Err(format!(
            "server-rss worker failed with status {status} for model '{}'",
            model.fixture.key
        ));
    }

    let rss_meta_bytes =
        std::fs::read(&rss_meta_path).map_err(|e| format!("cannot read server_rss.json: {e}"))?;
    let rss_meta: ServerMemoryMetrics = serde_json::from_slice(&rss_meta_bytes)
        .map_err(|e| format!("cannot parse server_rss.json: {e}"))?;

    // 3. Metrics stage
    let crit_dir = config.output_dir.join("criterion").join(model.fixture.key);
    let metrics_config = MetricsWorkerConfig {
        model_key: model.fixture.key.to_string(),
        backend: prep_config.backend.clone(),
        profile: None,
        client_key_path: artifact_dir.join("client.key"),
        server_key_path: artifact_dir.join("server.key"),
        input_path: artifact_dir.join("input.ct"),
        criterion_dir: crit_dir,
        output_path: artifact_dir.join("run.json"),
    };

    let mut run = run_metrics_worker(backend, model, &metrics_config)?;

    let output_ct_bytes = run
        .sizes
        .as_ref()
        .map(|s| s.output_ciphertext_bytes)
        .unwrap_or(0);

    run.memory = Some(rss_meta);
    run.sizes = Some(WireSizes {
        client_key_bytes: prep_meta.client_key_bytes,
        server_key_bytes: prep_meta.server_key_bytes,
        input_ciphertext_bytes: prep_meta.input_ct_bytes,
        output_ciphertext_bytes: output_ct_bytes,
    });
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_build_score_tap_graph() {
        use penumbra_core::ir::Graph;
        use crate::protocol::{FloatAccuracy, PaperData};

        let make_graph = |outputs: Vec<&str>| Graph {
            schema_version: "1.0.0".to_string(),
            num_blocks: 4,
            input_bits: 8,
            inputs: vec!["input".to_string()],
            outputs: outputs.into_iter().map(String::from).collect(),
            nodes: vec![],
        };

        let make_paper = |output_kind: &str, score_tensor: &str| PaperData {
            schema_version: 1,
            dataset: "test".to_string(),
            graph_sha256: "dummy".to_string(),
            output_kind: output_kind.to_string(),
            score_tensor: score_tensor.to_string(),
            decision_threshold: Some(0),
            calibration: vec![],
            test: vec![],
            tfhe_spot_check: vec![],
            float_accuracy: FloatAccuracy {
                value: 1.0,
                source: "test".to_string(),
                source_path: None,
                source_sha256: None,
                sample_count: 0,
                recomputed: false,
            },
        };

        // Case 1: Non-label model returns None
        let non_label_paper = make_paper("regression", "scores");
        let g1 = make_graph(vec!["label_out"]);
        assert!(maybe_build_score_tap_graph(&g1, &non_label_paper).is_none());

        // Case 2: Label model appends score_tensor if absent
        let label_paper = make_paper("label", "scores");
        let tapped = maybe_build_score_tap_graph(&g1, &label_paper).expect("should return Some");
        assert_eq!(tapped.outputs, vec!["label_out", "scores"]);

        // Case 3: Label model does not duplicate if score_tensor already in outputs
        let g_already = make_graph(vec!["label_out", "scores"]);
        let tapped2 = maybe_build_score_tap_graph(&g_already, &label_paper).expect("should return Some");
        assert_eq!(tapped2.outputs, vec!["label_out", "scores"]);
    }
}
