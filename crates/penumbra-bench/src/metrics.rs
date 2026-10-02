//! Pure statistical reductions and metric computations for paper benchmarks.

use penumbra_core::profile::NodeProfile;

use crate::protocol::{AbsoluteDistribution, MarginRelativeMetrics, PbsSplit, Sample};

/// Nearest-rank quantile on sorted, finite values.
/// Index formula: ceil(p * n) - 1.
pub fn nearest_rank_quantile(sorted_values: &[f64], p: f64) -> Option<f64> {
    if sorted_values.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let n = sorted_values.len();
    let rank = ((p * n as f64).ceil() as usize).max(1);
    let idx = (rank - 1).min(n - 1);
    Some(sorted_values[idx])
}

/// Compute absolute error distribution over per-sample max component errors.
pub fn compute_absolute_distribution(
    mut max_errors: Vec<f64>,
) -> Result<AbsoluteDistribution, String> {
    if max_errors.is_empty() {
        return Err("cannot compute absolute distribution on empty errors".to_string());
    }
    for &e in &max_errors {
        if !e.is_finite() || e < 0.0 {
            return Err(format!("invalid non-finite or negative error value: {e}"));
        }
    }
    max_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max = *max_errors.last().unwrap();
    let median = nearest_rank_quantile(&max_errors, 0.50).unwrap();
    let p95 = nearest_rank_quantile(&max_errors, 0.95).unwrap();
    let p99 = nearest_rank_quantile(&max_errors, 0.99).unwrap();

    Ok(AbsoluteDistribution {
        median,
        p95,
        p99,
        max,
    })
}

/// Compute margin-relative score error metrics.
/// Each entry is (max_score_error, top_two_reference_margin).
pub fn compute_margin_relative_metrics(
    samples: &[(f64, f64)],
) -> Result<MarginRelativeMetrics, String> {
    if samples.is_empty() {
        return Err("cannot compute margin relative metrics on empty samples".to_string());
    }

    let mut finite_relatives: Vec<f64> = Vec::new();
    let mut zero_margin_samples = 0;
    let mut zero_margin_with_nonzero_error = 0;

    for &(max_err, margin) in samples {
        if !max_err.is_finite() || max_err < 0.0 {
            return Err(format!("invalid max score error: {max_err}"));
        }
        if !margin.is_finite() || margin < 0.0 {
            return Err(format!("invalid margin: {margin}"));
        }

        if margin == 0.0 {
            zero_margin_samples += 1;
            if max_err > 0.0 {
                zero_margin_with_nonzero_error += 1;
            }
        } else {
            let rel = max_err / margin;
            if !rel.is_finite() {
                return Err(format!("non-finite relative error from {max_err}/{margin}"));
            }
            finite_relatives.push(rel);
        }
    }

    finite_relatives.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let (median, p95, max) = if finite_relatives.is_empty() {
        (None, None, None)
    } else {
        (
            nearest_rank_quantile(&finite_relatives, 0.50),
            nearest_rank_quantile(&finite_relatives, 0.95),
            finite_relatives.last().copied(),
        )
    };

    Ok(MarginRelativeMetrics {
        median,
        p95,
        max,
        finite_count: finite_relatives.len(),
        zero_margin_samples,
        zero_margin_with_nonzero_error,
    })
}

/// D16 PBS breakdown from node profiles and measured totals.
pub fn compute_pbs_split(
    total_measured_pbs: u64,
    node_profiles: &[NodeProfile],
) -> Result<PbsSplit, String> {
    let mut lookup_pbs: u64 = 0;

    for np in node_profiles {
        match np.op_type {
            "Activation" | "Requant" => {
                let bootstraps = np
                    .counters
                    .iter()
                    .find(|(k, _)| *k == "bootstraps")
                    .map(|(_, v)| *v)
                    .unwrap_or(0);
                lookup_pbs = lookup_pbs.checked_add(bootstraps).ok_or_else(|| {
                    "overflow accumulating lookup PBS from bootstraps".to_string()
                })?;
            }
            "Compare" | "Argmax" => {
                let cmp_ops = np
                    .counters
                    .iter()
                    .find(|(k, _)| *k == "cmp_pbs_ops")
                    .map(|(_, v)| *v)
                    .unwrap_or(0);
                lookup_pbs = lookup_pbs.checked_add(cmp_ops).ok_or_else(|| {
                    "overflow accumulating lookup PBS from cmp_pbs_ops".to_string()
                })?;
            }
            _ => {
                // All other op types contribute zero to logical lookups
            }
        }
    }

    let carry_pbs = total_measured_pbs
        .checked_sub(lookup_pbs)
        .ok_or_else(|| {
            format!(
                "measured PBS underflow: total_measured_pbs ({total_measured_pbs}) < lookup_pbs ({lookup_pbs})"
            )
        })?;

    Ok(PbsSplit {
        total_pbs: total_measured_pbs,
        lookup_pbs,
        carry_pbs,
    })
}

/// Helper for top-two margin calculation from reference integer scores.
pub fn reference_top_two_margin(scores: &[i64]) -> (i64, f64) {
    if scores.len() < 2 {
        return (scores.first().copied().unwrap_or(0), 0.0);
    }
    // Find first maximum (first-tie breaking)
    let mut max_val = scores[0];
    let mut max_idx = 0;
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > max_val {
            max_val = s;
            max_idx = i;
        }
    }

    // Find second maximum (any other element)
    let mut second_val = i64::MIN;
    for (i, &s) in scores.iter().enumerate() {
        if i != max_idx && s > second_val {
            second_val = s;
        }
    }

    let margin = (max_val - second_val) as f64;
    (max_idx as i64, margin)
}

/// Helper to find the index of the first maximum element in `scores`.
///
/// Returns `Err` if `scores` is empty or contains non-finite values (NaN / infinity).
/// On ties, strict greater-than comparison guarantees the first maximum index is selected,
/// matching NumPy / integer reference behavior.
pub fn first_argmax(scores: &[f64]) -> Result<usize, String> {
    if scores.is_empty() {
        return Err("cannot find argmax of empty slice".to_string());
    }
    for (i, &s) in scores.iter().enumerate() {
        if !s.is_finite() {
            return Err(format!("non-finite score at index {i}: {s}"));
        }
    }
    let mut max_idx = 0;
    let mut max_val = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > max_val {
            max_val = s;
            max_idx = i;
        }
    }
    Ok(max_idx)
}

/// Helper for max absolute error between raw floats and reference integer outputs.
///
/// Validates that `raw` contains at least `reference.len()` components and that every
/// logical component is finite before computing reductions. Only the declared logical prefix
/// `raw[..reference.len()]` is examined.
pub fn max_abs_error(raw: &[f64], reference: &[i64]) -> Result<f64, String> {
    if reference.is_empty() {
        return Err("reference cannot be empty".to_string());
    }
    if raw.len() < reference.len() {
        return Err(format!(
            "raw output length {} < reference length {}",
            raw.len(),
            reference.len()
        ));
    }
    let mut max_err = 0.0f64;
    for (i, (&got, &want)) in raw[..reference.len()].iter().zip(reference).enumerate() {
        if !got.is_finite() {
            return Err(format!("non-finite raw output at index {i}: {got}"));
        }
        let err = (got - want as f64).abs();
        if err > max_err {
            max_err = err;
        }
    }
    Ok(max_err)
}

/// Helper for binary score error and decision margin.
///
/// Returns `(abs(raw_score - reference_score), abs(reference_score - threshold))`.
/// Validates finiteness of inputs and results. When reference_score == threshold,
/// margin is 0.0 (explicit zero margin).
pub fn binary_score_error_and_margin(
    raw_score: f64,
    reference_score: i64,
    threshold: i64,
) -> Result<(f64, f64), String> {
    if !raw_score.is_finite() {
        return Err(format!("non-finite raw score: {raw_score}"));
    }
    let error = (raw_score - reference_score as f64).abs();
    let margin = (reference_score - threshold).abs() as f64;
    if !error.is_finite() || !margin.is_finite() {
        return Err(format!(
            "non-finite error or margin from raw={raw_score}, ref={reference_score}, threshold={threshold}"
        ));
    }
    Ok((error, margin))
}

/// Evaluation summary for a single test sample during worker execution.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleEvaluation {
    pub max_err: f64,
    pub pred_label: i64,
    pub label_flip: bool,
    pub task_match: bool,
    pub score_margin: (f64, f64),
}

/// Worker evaluation for a multiclass model sample.
///
/// Slices `raw_floats` strictly to `expected_output.len()`, isolating padded CKKS slots.
/// Validates finite components, verifies the error bound on the logical logits, and
/// determines `pred_label` using first-tie-breaking argmax.
pub fn evaluate_sample_multiclass(
    raw_floats: &[f64],
    expected_output: &[i64],
    expected_label: i64,
    target: i64,
    bound: f64,
) -> Result<SampleEvaluation, String> {
    if raw_floats.len() < expected_output.len() {
        return Err(format!(
            "decoded vector length {} < expected output length {}",
            raw_floats.len(),
            expected_output.len()
        ));
    }
    let logical_raw = &raw_floats[..expected_output.len()];
    let max_err = max_abs_error(logical_raw, expected_output)?;
    if max_err > bound {
        return Err(format!(
            "CKKS error bound violation: error {max_err} > bound {bound}"
        ));
    }
    let pred_idx = first_argmax(logical_raw)?;
    let pred_label = pred_idx as i64;
    let label_flip = pred_label != expected_label;
    let task_match = pred_label == target;
    let (_, margin) = reference_top_two_margin(expected_output);
    Ok(SampleEvaluation {
        max_err,
        pred_label,
        label_flip,
        task_match,
        score_margin: (max_err, margin),
    })
}

/// Worker evaluation for a binary label model sample.
///
/// Decodes and validates the primary label output against `bound` and rounds to retain
/// the actual decrypted decision label for `label_flip` and `task_match`.
/// The continuous score tap is evaluated separately to produce `score_margin` relative
/// to `abs(reference_score - threshold)`.
pub fn evaluate_sample_binary_label(
    sample: &Sample,
    raw_label_output: &[f64],
    raw_score_output: &[f64],
    decision_threshold: i64,
    bound: f64,
) -> Result<SampleEvaluation, String> {
    if raw_label_output.len() < sample.expected_output.len() {
        return Err(format!(
            "decoded label vector length {} < expected label length {}",
            raw_label_output.len(),
            sample.expected_output.len()
        ));
    }
    let logical_label = &raw_label_output[..sample.expected_output.len()];
    let label_err = max_abs_error(logical_label, &sample.expected_output)?;
    if label_err > bound {
        return Err(format!(
            "CKKS error bound violation on label output: error {label_err} > bound {bound}"
        ));
    }
    let dec_float = logical_label
        .first()
        .copied()
        .ok_or_else(|| "empty label output".to_string())?;
    if !dec_float.is_finite() {
        return Err(format!("non-finite label output: {dec_float}"));
    }
    // Retain the actual decrypted binary label for label-flip and task accuracy,
    // rather than deriving a label from the score tap.
    let pred_label = dec_float.round() as i64;
    let label_flip = pred_label != sample.expected_label;
    let task_match = pred_label == sample.target;

    let raw_score = raw_score_output
        .first()
        .copied()
        .ok_or_else(|| "missing score output from score tap".to_string())?;
    let ref_score = sample
        .expected_scores
        .as_ref()
        .and_then(|scores| scores.first().copied())
        .ok_or_else(|| "missing reference score in sample".to_string())?;
    let score_margin = binary_score_error_and_margin(raw_score, ref_score, decision_threshold)?;

    Ok(SampleEvaluation {
        max_err: label_err,
        pred_label,
        label_flip,
        task_match,
        score_margin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_rank_quantiles() {
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let median = nearest_rank_quantile(&values, 0.50).unwrap();
        let p95 = nearest_rank_quantile(&values, 0.95).unwrap();
        let p99 = nearest_rank_quantile(&values, 0.99).unwrap();

        assert_eq!(median, 1.0);
        assert_eq!(p95, 3.0);
        assert_eq!(p99, 3.0);
    }

    #[test]
    fn test_margin_relative_metrics_normal_and_tie() {
        // Normal case: score ref [10, 8], raw [9.5, 8.25]
        let (label, margin) = reference_top_two_margin(&[10, 8]);
        assert_eq!(label, 0);
        assert_eq!(margin, 2.0);
        let err0 = (10.0 - 9.5f64).abs(); // 0.5
        let err1 = (8.0 - 8.25f64).abs(); // 0.25
        let emax = err0.max(err1); // 0.5
        let metrics = compute_margin_relative_metrics(&[(emax, margin)]).unwrap();
        assert_eq!(metrics.median, Some(0.25));
        assert_eq!(metrics.max, Some(0.25));
        assert_eq!(metrics.finite_count, 1);
        assert_eq!(metrics.zero_margin_samples, 0);

        // Tie case: reference [4, 4]
        let (tie_label, tie_margin) = reference_top_two_margin(&[4, 4]);
        assert_eq!(tie_label, 0); // first class on tie
        assert_eq!(tie_margin, 0.0);
        let tie_metrics = compute_margin_relative_metrics(&[(0.5, tie_margin)]).unwrap();
        assert_eq!(tie_metrics.median, None);
        assert_eq!(tie_metrics.max, None);
        assert_eq!(tie_metrics.finite_count, 0);
        assert_eq!(tie_metrics.zero_margin_samples, 1);
        assert_eq!(tie_metrics.zero_margin_with_nonzero_error, 1);
    }

    #[test]
    fn test_absolute_distribution_empty_or_invalid() {
        assert!(compute_absolute_distribution(vec![]).is_err());
        assert!(compute_absolute_distribution(vec![1.0, f64::NAN]).is_err());
        assert!(compute_absolute_distribution(vec![1.0, -0.5]).is_err());
    }

    #[test]
    fn test_pbs_split_requant() {
        let profile = NodeProfile {
            name: "req".to_string(),
            op_type: "Requant",
            build: std::time::Duration::from_millis(1),
            eval: std::time::Duration::from_millis(1),
            input_lens: vec![2],
            output_len: 2,
            counters: vec![("bootstraps", 2), ("cmp_pbs_ops", 6)],
            measured: vec![("pbs", 9)],
        };

        // Measured total 9, lookups 2 -> carry 7
        let split = compute_pbs_split(9, std::slice::from_ref(&profile)).unwrap();
        assert_eq!(split.total_pbs, 9);
        assert_eq!(split.lookup_pbs, 2);
        assert_eq!(split.carry_pbs, 7);

        // Underflow: measured 1 < lookups 2 -> error
        assert!(compute_pbs_split(1, &[profile]).is_err());
    }

    #[test]
    fn first_tie_and_nonfinite_predictions() {
        assert_eq!(first_argmax(&[7.0, 7.0, 1.0]).unwrap(), 0);
        assert!(first_argmax(&[]).is_err());
        assert!(first_argmax(&[1.0, f64::NAN]).is_err());
        assert!(first_argmax(&[f64::INFINITY, 1.0]).is_err());
    }

    #[test]
    fn binary_margin_uses_score_not_decision_label() {
        let (error, margin) = binary_score_error_and_margin(11.0, 10, 0).unwrap();
        assert_eq!((error, margin), (1.0, 10.0));
        let summary = compute_margin_relative_metrics(&[(error, margin)]).unwrap();
        assert_eq!(summary.median, Some(0.1));
        assert_eq!(summary.zero_margin_samples, 0);
    }

    #[test]
    fn binary_margin_zero_and_nonzero_threshold_cases() {
        // Nonzero threshold: ref 15, threshold 10 -> margin 5, score err |12.0 - 15| = 3.0
        let (err_nonzero, margin_nonzero) = binary_score_error_and_margin(12.0, 15, 10).unwrap();
        assert_eq!((err_nonzero, margin_nonzero), (3.0, 5.0));

        // Zero margin with nonzero error: ref == threshold (both 0), raw 2.0 -> (2.0, 0.0)
        let (err_zero, margin_zero) = binary_score_error_and_margin(2.0, 0, 0).unwrap();
        assert_eq!((err_zero, margin_zero), (2.0, 0.0));
        let summary = compute_margin_relative_metrics(&[(err_zero, margin_zero)]).unwrap();
        assert_eq!(summary.zero_margin_samples, 1);
        assert_eq!(summary.zero_margin_with_nonzero_error, 1);
        assert_eq!(summary.median, None);

        // Zero margin with zero error: raw == ref == threshold
        let (err_exact, margin_exact) = binary_score_error_and_margin(0.0, 0, 0).unwrap();
        assert_eq!((err_exact, margin_exact), (0.0, 0.0));
        let summary_exact = compute_margin_relative_metrics(&[(err_exact, margin_exact)]).unwrap();
        assert_eq!(summary_exact.zero_margin_samples, 1);
        assert_eq!(summary_exact.zero_margin_with_nonzero_error, 0);

        // Non-finite score check
        assert!(binary_score_error_and_margin(f64::NAN, 0, 0).is_err());
        assert!(binary_score_error_and_margin(f64::INFINITY, 0, 0).is_err());
    }

    #[test]
    fn output_error_rejects_missing_or_nonfinite_components() {
        assert_eq!(max_abs_error(&[1.25, -1.0], &[1, -2]).unwrap(), 1.0);
        assert!(max_abs_error(&[1.0], &[1, -2]).is_err());
        assert!(max_abs_error(&[], &[1]).is_err());
        assert!(max_abs_error(&[1.0], &[]).is_err());
        assert!(max_abs_error(&[f64::NAN, -1.0], &[1, -2]).is_err());
        assert!(max_abs_error(&[1.0, f64::INFINITY], &[1, -2]).is_err());
    }

    #[test]
    fn test_worker_multiclass_evaluation_and_malformed_buffer() {
        // Short buffer rejected
        assert!(evaluate_sample_multiclass(&[1.0], &[1, 2], 0, 0, 1.0).is_err());
        // Non-finite in logical prefix rejected
        assert!(evaluate_sample_multiclass(&[f64::NAN, 1.0], &[1, 2], 0, 0, 1.0).is_err());
        // Bound violation rejected
        assert!(evaluate_sample_multiclass(&[5.0, 1.0], &[1, 1], 0, 0, 0.5).is_err());
    }

    #[test]
    fn test_worker_binary_evaluation_decision_disagreement_and_malformed_inputs() {
        let sample = Sample {
            id: "0".to_string(),
            inputs: vec![],
            expected_output: vec![1],
            expected_label: 1,
            target: 1,
            expected_scores: Some(vec![10]),
        };

        // Primary label output is 0.9 -> rounds to decision 1.
        // Score tap output is -50.0 (if thresholded at 0, would imply label 0).
        // Worker must retain actual decrypted decision label (1), not derived from score tap!
        let eval = evaluate_sample_binary_label(
            &sample,
            &[0.9],
            &[-50.0],
            0,
            0.2, // bound on label output: |0.9 - 1.0| = 0.1 <= 0.2
        )
        .unwrap();
        assert_eq!(
            eval.pred_label, 1,
            "must retain original decrypted decision"
        );
        assert!(!eval.label_flip, "label flip must be false");
        assert!(eval.task_match, "task match must be true");
        assert_eq!(eval.score_margin, (60.0, 10.0)); // score err |-50 - 10| = 60, margin |10 - 0| = 10

        // Malformed score tap or empty expected_scores rejected
        assert!(evaluate_sample_binary_label(&sample, &[1.0], &[], 0, 0.5).is_err());

        let mut sample_no_scores = sample.clone();
        sample_no_scores.expected_scores = None;
        assert!(evaluate_sample_binary_label(&sample_no_scores, &[1.0], &[10.0], 0, 0.5).is_err());

        let mut sample_empty_scores = sample.clone();
        sample_empty_scores.expected_scores = Some(vec![]);
        assert!(
            evaluate_sample_binary_label(&sample_empty_scores, &[1.0], &[10.0], 0, 0.5).is_err()
        );

        // Short or non-finite label output rejected
        assert!(evaluate_sample_binary_label(&sample, &[], &[10.0], 0, 0.5).is_err());
        assert!(evaluate_sample_binary_label(&sample, &[f64::NAN], &[10.0], 0, 0.5).is_err());
    }

    #[cfg(feature = "ckks")]
    #[test]
    fn test_behavioral_regression_tree_budget_rejection() {
        use crate::ckks_backend;
        use crate::models::{find, load};
        use crate::paper::run_paper_model;
        use crate::protocol::PaperConfig;
        use penumbra_core::backend::Backend;

        for model_key in ["phase8_trees", "phase8_xgb"] {
            let fixture = find(model_key).expect("fixture exists");
            let loaded = load(fixture).expect("loaded model");
            let backend = ckks_backend();

            let expected_err = backend
                .check_graph_budget(&loaded.graph)
                .expect_err("backend budget check must fail for tree model");

            let temp_dir = tempfile::tempdir().unwrap();
            let config = PaperConfig {
                output_dir: temp_dir.path().to_path_buf(),
                threads: 1,
                protocol_version: 1,
            };

            let run = run_paper_model(backend, &loaded, &config)
                .expect("run_paper_model should succeed with unsupported status");

            assert_eq!(run.status, "unsupported");
            assert_eq!(
                run.rejection.as_deref(),
                Some(expected_err.as_str()),
                "model {model_key} rejection must match check_graph_budget error verbatim"
            );
            assert!(run.latency.is_none());
            assert!(run.memory.is_none());
            assert!(run.accuracy.is_none());
            assert!(run.sizes.is_none());
            assert!(
                run.nodes.is_empty(),
                "unsupported run must have empty nodes"
            );
            assert!(
                run.profile_sample_id.is_none(),
                "unsupported run must have no profile_sample_id"
            );
        }
    }
}
