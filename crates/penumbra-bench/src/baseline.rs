//! Committed regression baseline and validation logic.
//!
//! Only machine-independent fields appear in a baseline: timing is reported in the benchmark
//! suite but never gated, because CI runners are too noisy for a wall-clock gate. What is gated
//! is the deterministic cost model — `runtime ~= number of bootstraps` (`PROJECT.md` §5) —
//! plus wire sizes and label correctness, all of which are fixed by the crypto params and the
//! graph, not by the CPU.
//!
//! Note on CKKS: CKKS deterministic metrics (cost proxies: depth levels, rescales, rotations,
//! and polynomial evaluations; wire sizes; and label correctness) are gated via
//! `baselines/ckks-baseline.json`. CKKS's `max_abs_err` is a floating-point value that depends on
//! the HAL backend (`FFT64Neon` on AArch64 vs `FFT64Avx` on x86-64, `docs/BENCHMARKS.md:185`),
//! so exact floating-point error is not exact-gated in the baseline file; continuous float
//! accuracy is gated on the nightly job by its seven golden tests against declared bounds
//! (`crates/penumbra-ckks/src/bounds.rs`).

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::report::ModelRun;

/// A single machine-independent regression baseline record for one (model, backend) pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BaselineEntry {
    pub model: String,
    pub backend: String,
    pub profile: Option<String>,
    pub num_blocks: usize,
    pub client_key_bytes: usize,
    pub server_key_bytes: usize,
    pub input_ct_bytes: usize,
    pub output_ct_bytes: usize,
    pub cost_proxy: BTreeMap<String, u64>,
    pub measured_totals: BTreeMap<String, u64>,
    pub labels_matched: usize,
    pub labels_checked: usize,
}

/// A collection of baseline entries forming a regression gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub schema: u32,
    pub note: String,
    pub entries: Vec<BaselineEntry>,
}

/// Build a `Baseline` from a slice of executed model runs.
pub fn baseline_from_runs(runs: &[ModelRun]) -> Baseline {
    let entries = runs
        .iter()
        .map(|run| {
            let labels_matched = run
                .samples
                .iter()
                .filter(|s| s.label_matches == Some(true))
                .count();
            let labels_checked = run
                .samples
                .iter()
                .filter(|s| s.label_matches.is_some())
                .count();
            BaselineEntry {
                model: run.model.clone(),
                backend: run.backend.clone(),
                profile: run.profile.clone(),
                num_blocks: run.num_blocks,
                client_key_bytes: run.client_key_bytes,
                server_key_bytes: run.server_key_bytes,
                input_ct_bytes: run.input_ct_bytes,
                output_ct_bytes: run.output_ct_bytes,
                cost_proxy: run.cost_proxy.clone(),
                measured_totals: run.measured_totals.clone(),
                labels_matched,
                labels_checked,
            }
        })
        .collect();

    Baseline {
        schema: 1,
        note: "Phase-10 committed regression baseline (tfhe-classic profile)".to_string(),
        entries,
    }
}

/// Compare measured `runs` against a committed `Baseline`.
///
/// Returns `Ok(warnings)` on success (where `warnings` names unexercised baseline entries),
/// or `Err(violations)` listing every detected regression or contract break.
pub fn check_against(baseline: &Baseline, runs: &[ModelRun]) -> Result<Vec<String>, Vec<String>> {
    let mut violations = Vec::new();
    let mut exercised = HashSet::new();

    for run in runs {
        let tag = format!("{}/{}", run.model, run.backend);
        exercised.insert((run.model.clone(), run.backend.clone()));

        let entry = match baseline
            .entries
            .iter()
            .find(|e| e.model == run.model && e.backend == run.backend)
        {
            Some(e) => e,
            None => {
                violations.push(format!(
                    "{tag}: missing from baseline; regenerate the baseline with --write-baseline"
                ));
                continue;
            }
        };

        if entry.profile != run.profile {
            violations.push(format!(
                "{tag}: profile mismatch: baseline {:?}, measured {:?}",
                entry.profile, run.profile
            ));
        }

        if entry.num_blocks != run.num_blocks {
            violations.push(format!(
                "{tag}: num_blocks regressed: baseline {}, measured {}",
                entry.num_blocks, run.num_blocks
            ));
        }
        if entry.client_key_bytes != run.client_key_bytes {
            violations.push(format!(
                "{tag}: client_key_bytes regressed: baseline {}, measured {}",
                entry.client_key_bytes, run.client_key_bytes
            ));
        }
        if entry.server_key_bytes != run.server_key_bytes {
            violations.push(format!(
                "{tag}: server_key_bytes regressed: baseline {}, measured {}",
                entry.server_key_bytes, run.server_key_bytes
            ));
        }
        if entry.input_ct_bytes != run.input_ct_bytes {
            violations.push(format!(
                "{tag}: input_ct_bytes regressed: baseline {}, measured {}",
                entry.input_ct_bytes, run.input_ct_bytes
            ));
        }
        if entry.output_ct_bytes != run.output_ct_bytes {
            violations.push(format!(
                "{tag}: output_ct_bytes regressed: baseline {}, measured {}",
                entry.output_ct_bytes, run.output_ct_bytes
            ));
        }

        check_map_exact(
            &tag,
            "cost_proxy",
            &entry.cost_proxy,
            &run.cost_proxy,
            &mut violations,
        );
        check_map_exact(
            &tag,
            "measured_totals",
            &entry.measured_totals,
            &run.measured_totals,
            &mut violations,
        );

        let matched = run
            .samples
            .iter()
            .filter(|s| s.label_matches == Some(true))
            .count();
        let checked = run
            .samples
            .iter()
            .filter(|s| s.label_matches.is_some())
            .count();
        if checked != entry.labels_checked {
            violations.push(format!(
                "{tag}: labels_checked regressed: baseline {}, measured {}",
                entry.labels_checked, checked
            ));
        } else if matched < entry.labels_matched {
            violations.push(format!(
                "{tag}: labels_matched regressed: baseline {}, measured {}",
                entry.labels_matched, matched
            ));
        }
    }

    let mut warnings = Vec::new();
    for entry in &baseline.entries {
        if !exercised.contains(&(entry.model.clone(), entry.backend.clone())) {
            warnings.push(format!(
                "{}/{}: baseline entry not exercised by this run",
                entry.model, entry.backend
            ));
        }
    }

    if violations.is_empty() {
        Ok(warnings)
    } else {
        Err(violations)
    }
}

fn check_map_exact(
    tag: &str,
    field: &str,
    expected: &BTreeMap<String, u64>,
    actual: &BTreeMap<String, u64>,
    violations: &mut Vec<String>,
) {
    for (k, &v_exp) in expected {
        match actual.get(k) {
            Some(&v_act) => {
                if v_exp != v_act {
                    violations.push(format!(
                        "{tag}: {field}.{k} regressed: baseline {v_exp}, measured {v_act}"
                    ));
                }
            }
            None => {
                violations.push(format!(
                    "{tag}: {field}.{k} missing from measured run (baseline {v_exp})"
                ));
            }
        }
    }
    for k in actual.keys() {
        if !expected.contains_key(k) {
            let v = actual[k];
            violations.push(format!(
                "{tag}: {field}.{k} extra key in measured run (value {v}, not in baseline)"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::SampleReport;

    fn dummy_run(model: &str, backend: &str, num_blocks: usize) -> ModelRun {
        let mut cost_proxy = BTreeMap::new();
        cost_proxy.insert("bootstraps".to_string(), 10);
        let mut measured_totals = BTreeMap::new();
        measured_totals.insert("pbs".to_string(), 20);

        ModelRun {
            backend: backend.to_string(),
            profile: Some("classic".to_string()),
            model: model.to_string(),
            fixture: "dummy.json".to_string(),
            num_blocks,
            keygen_secs: 1.0,
            client_key_bytes: 100,
            server_key_bytes: 200,
            input_ct_bytes: 300,
            output_ct_bytes: 400,
            samples: vec![SampleReport {
                index: 0,
                decrypted: vec![1, 2],
                encrypt_secs: 0.1,
                eval_secs: 0.2,
                decrypt_secs: 0.05,
                max_abs_err: None,
                mean_abs_err: None,
                label_matches: Some(true),
                nodes: Vec::new(),
            }],
            op_type_secs: BTreeMap::new(),
            op_type_build_secs: BTreeMap::new(),
            cost_proxy,
            measured_totals,
            accuracy: None,
            bit_plan: None,
            ir_bytes: 500,
            ir_load_secs: 0.001,
        }
    }

    #[test]
    fn test_baseline_exact_match_passes() {
        let run = dummy_run("m1", "tfhe", 6);
        let baseline = baseline_from_runs(std::slice::from_ref(&run));
        let res = check_against(&baseline, &[run]);
        assert!(res.is_ok());
        assert!(res.unwrap().is_empty());
    }

    #[test]
    fn test_baseline_missing_entry_fails() {
        let run1 = dummy_run("m1", "tfhe", 6);
        let run2 = dummy_run("m2", "tfhe", 6);
        let baseline = baseline_from_runs(&[run1]);
        let res = check_against(&baseline, &[run2]);
        assert!(res.is_err());
        let violations = res.unwrap_err();
        assert!(violations
            .iter()
            .any(|v| v.contains("missing from baseline")));
    }

    #[test]
    fn test_baseline_profile_mismatch_fails() {
        let run = dummy_run("m1", "tfhe", 6);
        let baseline = baseline_from_runs(std::slice::from_ref(&run));
        let mut modified = run;
        modified.profile = Some("gaussian".to_string());
        let res = check_against(&baseline, &[modified]);
        assert!(res.is_err());
        assert!(res
            .unwrap_err()
            .iter()
            .any(|v| v.contains("profile mismatch")));
    }

    #[test]
    fn test_baseline_num_blocks_mismatch_fails() {
        let run = dummy_run("m1", "tfhe", 6);
        let baseline = baseline_from_runs(std::slice::from_ref(&run));
        let mut modified = run;
        modified.num_blocks = 7;
        let res = check_against(&baseline, &[modified]);
        assert!(res.is_err());
        assert!(res
            .unwrap_err()
            .iter()
            .any(|v| v.contains("num_blocks regressed")));
    }

    #[test]
    fn test_baseline_unexercised_entry_warns() {
        let run1 = dummy_run("m1", "tfhe", 6);
        let run2 = dummy_run("m2", "tfhe", 6);
        let baseline = baseline_from_runs(&[run1.clone(), run2]);
        let res = check_against(&baseline, &[run1]);
        assert!(res.is_ok());
        let warnings = res.unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("not exercised"));
    }
}
