//! Behavioral regressions for paper metrics and real graph budget rejections (Tasks 2 and 3).

use penumbra_bench::metrics::{
    evaluate_sample_binary_label, evaluate_sample_multiclass, SampleEvaluation,
};
use penumbra_bench::protocol::Sample;

#[test]
fn test_worker_multiclass_logical_prefix_and_padding_isolation() {
    // 3 logical classes with a tie at indices 0 and 1: [7.0, 7.0, 1.0].
    // Padded slots (3..256) contain large values and non-finite values that must NEVER
    // be observed by argmax, bound checking, or error metrics.
    let mut raw_floats = vec![7.0, 7.0, 1.0];
    raw_floats.resize(256, 999.0);
    raw_floats[255] = f64::NAN;

    let expected_output = vec![7, 7, 1];
    let expected_label = 0; // first-tie breaking requires class 0
    let target = 0;
    let bound = 0.5;

    let eval: SampleEvaluation =
        evaluate_sample_multiclass(&raw_floats, &expected_output, expected_label, target, bound)
            .expect("multiclass evaluation should succeed on logical prefix");

    // Argmax must be the first maximum (0), never the last maximum (1) and never a padded slot (>2)
    assert_eq!(eval.pred_label, 0, "first tie must select index 0");
    assert!(!eval.label_flip, "label flip should be false for class 0");
    assert!(eval.task_match, "task match should be true");
    assert_eq!(eval.max_err, 0.0, "error within logical prefix is 0.0");
    // Margin between top two logical classes 7 and 7 is 0.0
    assert_eq!(eval.score_margin, (0.0, 0.0));
}

#[test]
fn test_worker_binary_output_and_score_separation_preserves_decision() {
    // Binary model with label decision output and continuous score tap.
    // Label output: raw 1.1 -> rounds to decision label 1.
    // Score tap: raw 11.0 vs reference score 10 with threshold 0.
    let sample = Sample {
        id: "sample_bin".to_string(),
        inputs: vec![],
        expected_output: vec![1],
        expected_label: 1,
        target: 1,
        expected_scores: Some(vec![10]),
    };
    let raw_label_output = vec![1.1];
    let raw_score_output = vec![11.0];
    let decision_threshold = 0;
    let bound = 0.25; // bound on label output

    let eval: SampleEvaluation = evaluate_sample_binary_label(
        &sample,
        &raw_label_output,
        &raw_score_output,
        decision_threshold,
        bound,
    )
    .expect("binary label evaluation should succeed");

    // Decision label must be preserved from primary output rounding
    assert_eq!(eval.pred_label, 1, "decision label must be 1");
    assert!(!eval.label_flip, "must not flip label");
    assert!(eval.task_match, "must match target");
    assert!(
        (eval.max_err - 0.1).abs() < 1e-6,
        "label error is |1.1 - 1.0| = 0.1"
    );

    // Score margin must use continuous score (11.0 vs 10, threshold 0) -> (1.0, 10.0),
    // NOT the fictitious zero margin from binary decision label (0.0)
    assert_eq!(
        eval.score_margin,
        (1.0, 10.0),
        "score margin must be (score_err, |ref_score - threshold|) = (1.0, 10.0)"
    );
}

#[test]
fn test_worker_binary_retains_actual_decision_on_score_disagreement() {
    // Primary label output is 0.8 -> rounds to decision label 1.
    // Score tap output is -50.0 -> if thresholded at 0, would imply label 0.
    // Worker MUST retain actual decrypted decision label 1, and NOT derive label 0 from score tap!
    let sample = Sample {
        id: "sample_disagree".to_string(),
        inputs: vec![],
        expected_output: vec![1],
        expected_label: 1,
        target: 1,
        expected_scores: Some(vec![10]),
    };
    let eval = evaluate_sample_binary_label(
        &sample,
        &[0.8],
        &[-50.0],
        0,
        0.5, // bound
    )
    .expect("evaluation must succeed");

    assert_eq!(eval.pred_label, 1, "must retain actual decrypted label");
    assert!(!eval.label_flip, "label flip must be false");
    assert!(eval.task_match, "task match must be true");
    assert_eq!(eval.score_margin, (60.0, 10.0));
}

#[cfg(feature = "ckks")]
#[test]
fn test_behavioral_regression_tree_budget_rejection() {
    use penumbra_bench::ckks_backend;
    use penumbra_bench::models::{find, load};
    use penumbra_bench::paper::run_paper_model;
    use penumbra_bench::protocol::PaperConfig;
    use penumbra_core::backend::Backend;

    for model_key in ["phase8_trees", "phase8_xgb"] {
        let fixture = find(model_key).expect("fixture exists");
        let loaded = load(fixture).expect("loaded model");
        let backend = ckks_backend();

        // The real graph budget check error from Backend::check_graph_budget
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
        // Must retain exact check_graph_budget error, NOT fabricated paper_policy error
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
