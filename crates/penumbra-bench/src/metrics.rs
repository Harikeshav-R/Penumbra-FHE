//! Pure statistical reductions and metric computations for paper benchmarks.

use penumbra_core::profile::NodeProfile;

use crate::protocol::{AbsoluteDistribution, MarginRelativeMetrics, PbsSplit};

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
}
