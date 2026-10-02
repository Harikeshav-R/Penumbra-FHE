# Code Review Refactorings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the four code quality findings (duplicated `NodeProfile` conversion, duplicated `tap_graph` construction, provenance data clumps, and duplicated test logic) identified during the code review of `docs/paper-results`, without changing runtime behavior, schema, or benchmark results.

**Architecture:** Add idiomatic standard trait implementations (`From<NodeProfile>` and `From<&NodeProfile>`) in `report.rs` to collapse repetitive struct conversions; extract graph-tapping logic into a focused helper in `paper.rs`; consolidate system environment provenance capture on `PaperReportMeta::capture` in `protocol.rs`; and retain whole-pipeline regression tests in integration suites (`tests/paper_metrics.rs`) while pruning duplicate copies from unit test modules.

**Tech Stack:** Rust (stable ≥ 1.83, nightly for optional CKKS features), `penumbra-core`, `penumbra-bench`, `rayon`, `serde`, Cargo.

**Spec:** Standards findings from `code-review` run on `docs/paper-results` comparing `main...HEAD`.

## Global Constraints

- No changes to Layer 1, 2, or 3 crates (`crates/penumbra-core`, `crates/penumbra-tfhe`, `crates/penumbra-ckks`, `python/penumbra`).
- All changes are strictly within `crates/penumbra-bench`.
- No changes to serialized JSON artifact schema or committed result files under `docs/results/`.
- All tests must pass: `cargo test -p penumbra-bench --lib` and `cargo +nightly test -p penumbra-bench --features ckks --test paper_metrics`.
- Formatter and linter must remain clean: `cargo fmt --all -- --check` and `cargo clippy -p penumbra-bench --all-targets -- -D warnings`.
- Commits follow Conventional Commits format (`refactor(bench): ...`).

## Review Focus

1. `From<&NodeProfile>` must clone fields correctly while `From<NodeProfile>` takes ownership without redundant allocations.
2. Score tap graph construction must preserve graph properties: non-label models return `None`, label models return `Some` with the score tensor appended only if not already present.
3. Provenance capture must maintain exact error messaging on dirty working trees or commit mismatches across `"paper"`, `"calibrate"`, and `"probe"` modes.
4. Scale probe result JSON serialization must produce identical keys (`machine_model`, `os_product_version`, `kernel_version`, etc.) when reading from `PaperReportMeta`.
5. CKKS budget rejection test in `tests/paper_metrics.rs` must continue to run when CKKS is enabled and remain untouched when removing the redundant copy from `metrics.rs`.

---

### Task 1: Unify `NodeProfile` to `NodeReport` conversion via `From` implementations

**Files:**
- Modify: `crates/penumbra-bench/src/report.rs:24-30`
- Modify: `crates/penumbra-bench/src/paper.rs:386-406,559-579`
- Modify: `crates/penumbra-bench/src/bin/mnist_scale_probe.rs:116-137`
- Test: `crates/penumbra-bench/src/report.rs` (in `mod tests`)

**Interfaces:**
- Consumes: `penumbra_core::profile::NodeProfile`
- Produces: `impl From<NodeProfile> for NodeReport`, `impl From<&NodeProfile> for NodeReport`

- [ ] **Step 1: Write the failing test in `crates/penumbra-bench/src/report.rs`**

Add a test case in `tests` module at the bottom of `crates/penumbra-bench/src/report.rs`:

```rust
#[test]
fn test_node_report_from_node_profile() {
    use penumbra_core::profile::NodeProfile;
    use std::time::Duration;

    let mut profile = NodeProfile {
        name: "test_node".to_string(),
        op_type: "Linear",
        build: Duration::from_millis(15),
        eval: Duration::from_millis(42),
        input_lens: vec![10, 20],
        output_len: 5,
        counters: vec![("mults", 100)].into_iter().collect(),
        measured: vec![("pbs", 4)].into_iter().collect(),
    };

    // Test borrowed conversion
    let rep_ref = NodeReport::from(&profile);
    assert_eq!(rep_ref.name, "test_node");
    assert_eq!(rep_ref.op_type, "Linear");
    assert!((rep_ref.build_secs - 0.015).abs() < 1e-9);
    assert!((rep_ref.eval_secs - 0.042).abs() < 1e-9);
    assert_eq!(rep_ref.input_lens, vec![10, 20]);
    assert_eq!(rep_ref.output_len, 5);
    assert_eq!(rep_ref.counters.get("mults"), Some(&100));
    assert_eq!(rep_ref.measured.get("pbs"), Some(&4));

    // Test owned conversion
    profile.name = "test_owned".to_string();
    let rep_owned = NodeReport::from(profile);
    assert_eq!(rep_owned.name, "test_owned");
    assert_eq!(rep_owned.counters.get("mults"), Some(&100));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p penumbra-bench --lib test_node_report_from_node_profile`
Expected: FAIL with `the trait From<&NodeProfile> is not implemented for NodeReport`

- [ ] **Step 3: Implement `From<NodeProfile>` and `From<&NodeProfile>` on `NodeReport`**

In `crates/penumbra-bench/src/report.rs`, add:

```rust
use penumbra_core::profile::NodeProfile;

impl From<NodeProfile> for NodeReport {
    fn from(n: NodeProfile) -> Self {
        Self {
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
        }
    }
}

impl From<&NodeProfile> for NodeReport {
    fn from(n: &NodeProfile) -> Self {
        Self {
            name: n.name.clone(),
            op_type: n.op_type.to_string(),
            build_secs: n.build.as_secs_f64(),
            eval_secs: n.eval.as_secs_f64(),
            input_lens: n.input_lens.clone(),
            output_len: n.output_len,
            counters: n
                .counters
                .iter()
                .map(|&(k, v)| (k.to_string(), v))
                .collect(),
            measured: n
                .measured
                .iter()
                .map(|&(k, v)| (k.to_string(), v))
                .collect(),
        }
    }
}
```

- [ ] **Step 4: Update call sites in `paper.rs` and `mnist_scale_probe.rs`**

In `crates/penumbra-bench/src/paper.rs:386-406`, replace the manual map with:
```rust
let rep_node_reports: Vec<NodeReport> = rep_profile_nodes
    .into_iter()
    .map(NodeReport::from)
    .collect();
```

In `crates/penumbra-bench/src/paper.rs:559-579`, replace the manual map with:
```rust
let rep_node_reports: Vec<NodeReport> = rep_profile_nodes
    .into_iter()
    .map(NodeReport::from)
    .collect();
```

In `crates/penumbra-bench/src/bin/mnist_scale_probe.rs:116-137`, replace manual mapping with:
```rust
let nodes = p.nodes.iter().map(NodeReport::from).collect();
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `cargo test -p penumbra-bench --lib`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/penumbra-bench/src/report.rs crates/penumbra-bench/src/paper.rs crates/penumbra-bench/src/bin/mnist_scale_probe.rs
git commit -m "refactor(bench): implement From for NodeReport to deduplicate profile conversion"
```

---

### Task 2: Deduplicate `tap_graph` construction in `paper.rs`

**Files:**
- Modify: `crates/penumbra-bench/src/paper.rs:288-296,455-463`
- Test: `crates/penumbra-bench/src/paper.rs` (inline test)

**Interfaces:**
- Consumes: `penumbra_core::ir::Graph`, `crate::protocol::PaperData`
- Produces: `fn maybe_build_score_tap_graph(graph: &Graph, paper: &PaperData) -> Option<Graph>`

- [ ] **Step 1: Write the failing test in `crates/penumbra-bench/src/paper.rs`**

Add a test in the `#[cfg(test)] mod tests` in `crates/penumbra-bench/src/paper.rs`:

```rust
#[test]
fn test_maybe_build_score_tap_graph() {
    use penumbra_core::ir::Graph;
    use crate::protocol::PaperData;

    let mut graph = Graph::new(4);
    graph.outputs = vec!["label_out".to_string()];

    // Case 1: Non-label model returns None
    let non_label_paper = PaperData {
        output_kind: "regression".to_string(),
        score_tensor: "scores".to_string(),
        decision_threshold: 0,
        tfhe_spot_check: vec![],
        calibration: vec![],
        test: vec![],
    };
    assert!(maybe_build_score_tap_graph(&graph, &non_label_paper).is_none());

    // Case 2: Label model appends score_tensor if absent
    let label_paper = PaperData {
        output_kind: "label".to_string(),
        score_tensor: "scores".to_string(),
        decision_threshold: 0,
        tfhe_spot_check: vec![],
        calibration: vec![],
        test: vec![],
    };
    let tapped = maybe_build_score_tap_graph(&graph, &label_paper).expect("should return Some");
    assert_eq!(tapped.outputs, vec!["label_out", "scores"]);

    // Case 3: Label model does not duplicate if score_tensor already in outputs
    let mut already_tapped_graph = Graph::new(4);
    already_tapped_graph.outputs = vec!["label_out".to_string(), "scores".to_string()];
    let tapped2 = maybe_build_score_tap_graph(&already_tapped_graph, &label_paper).expect("should return Some");
    assert_eq!(tapped2.outputs, vec!["label_out", "scores"]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p penumbra-bench --lib test_maybe_build_score_tap_graph`
Expected: FAIL with `cannot find function maybe_build_score_tap_graph`

- [ ] **Step 3: Implement `maybe_build_score_tap_graph`**

In `crates/penumbra-bench/src/paper.rs`:

```rust
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
```

- [ ] **Step 4: Update call sites in `paper.rs`**

In `crates/penumbra-bench/src/paper.rs` at line ~288:
```rust
let tap_graph = maybe_build_score_tap_graph(&model.graph, paper);
```

In `crates/penumbra-bench/src/paper.rs` at line ~455:
```rust
let tap_graph = maybe_build_score_tap_graph(&model.graph, paper);
let eval_g = tap_graph.as_ref().unwrap_or(&model.graph);
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `cargo test -p penumbra-bench --lib`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/penumbra-bench/src/paper.rs
git commit -m "refactor(bench): extract maybe_build_score_tap_graph helper"
```

---

### Task 3: Unify system provenance metadata in `protocol.rs` and eliminate probe data clump

**Files:**
- Modify: `crates/penumbra-bench/src/protocol.rs:382-403`
- Modify: `crates/penumbra-bench/src/bin/report.rs:350-452,548,602`
- Modify: `crates/penumbra-bench/src/bin/mnist_scale_probe.rs:150-256,388-410`
- Test: `crates/penumbra-bench/src/protocol.rs` (in `mod tests`)

**Interfaces:**
- Consumes: `std::process::Command`, `std::env`, `rayon::current_num_threads`
- Produces: `PaperReportMeta::capture(mode: &str, requested_threads: usize) -> Result<PaperReportMeta, String>`

- [ ] **Step 1: Write the failing test in `crates/penumbra-bench/src/protocol.rs`**

Add a test in `crates/penumbra-bench/src/protocol.rs` (create `#[cfg(test)] mod tests` if absent):

```rust
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p penumbra-bench --lib test_paper_report_meta_capture_structure`
Expected: FAIL with `no function or associated item named capture found for struct PaperReportMeta`

- [ ] **Step 3: Implement `PaperReportMeta::capture` in `crates/penumbra-bench/src/protocol.rs`**

In `crates/penumbra-bench/src/protocol.rs`:

```rust
use std::process::Command;

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
```

- [ ] **Step 4: Update `report.rs` to call `PaperReportMeta::capture`**

In `crates/penumbra-bench/src/bin/report.rs`:
- Remove `fn capture_paper_meta(...)` (lines 350-452).
- Update line 548:
  ```rust
  let meta = PaperReportMeta::capture("calibrate", threads)?;
  ```
- Update line 602:
  ```rust
  let meta = PaperReportMeta::capture("paper", threads)?;
  ```

- [ ] **Step 5: Refactor `mnist_scale_probe.rs` to use `PaperReportMeta::capture`**

In `crates/penumbra-bench/src/bin/mnist_scale_probe.rs`:
- Delete `struct ProbeProvenance` and `fn check_provenance()` (lines 204-256).
- Delete `fn run_command_strict()` (lines 185-202).
- In `main()` / probe run:
  ```rust
  let meta = PaperReportMeta::capture("probe", args.threads)?;
  ```
- In constructing `MnistScaleProbeResult`:
  ```rust
  let result = MnistScaleProbeResult {
      schema_version: 1,
      probe: "mnist28_scale_probe".to_string(),
      backend: "tfhe".to_string(),
      machine_model: meta.machine_model,
      os_product_version: meta.os_product_version,
      kernel_version: meta.kernel_version,
      build_commit: meta.build_commit,
      runtime_commit: meta.runtime_commit,
      dirty: meta.dirty,
      rustc_version: meta.rustc_version,
      requested_threads: meta.requested_threads,
      actual_threads: meta.actual_threads,
      // remaining fields unchanged...
  };
  ```

- [ ] **Step 6: Run tests and verify they pass**

Run: `cargo test -p penumbra-bench --lib`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/penumbra-bench/src/protocol.rs crates/penumbra-bench/src/bin/report.rs crates/penumbra-bench/src/bin/mnist_scale_probe.rs
git commit -m "refactor(bench): consolidate system provenance capture on PaperReportMeta"
```

---

### Task 4: Remove redundant CKKS budget rejection test from `metrics.rs`

**Files:**
- Modify: `crates/penumbra-bench/src/metrics.rs:552-599`
- Verify: `crates/penumbra-bench/tests/paper_metrics.rs:108-157`

**Interfaces:**
- Consumes: Integration test suite in `tests/paper_metrics.rs`
- Produces: Clean, non-redundant unit test module in `metrics.rs`

- [ ] **Step 1: Verify the integration test exists and is identical**

Inspect `crates/penumbra-bench/tests/paper_metrics.rs:108-157` and confirm it already contains `test_behavioral_regression_tree_budget_rejection`.
Run: `cargo +nightly test -p penumbra-bench --features ckks --test paper_metrics`
Expected: 4 passed; includes `test_behavioral_regression_tree_budget_rejection`.

- [ ] **Step 2: Remove the duplicate test from `metrics.rs`**

Delete lines 552-599 in `crates/penumbra-bench/src/metrics.rs` (`test_behavioral_regression_tree_budget_rejection` and its imports inside the test function).

- [ ] **Step 3: Run the test suite and verify clean execution**

Run: `cargo test -p penumbra-bench --lib`
Run: `cargo +nightly test -p penumbra-bench --features ckks --test paper_metrics`
Expected: Both commands pass completely.

- [ ] **Step 4: Run clippy and fmt checks across the workspace**

Run: `cargo fmt --all -- --check`
Run: `cargo clippy -p penumbra-bench --all-targets -- -D warnings`
Expected: Zero warnings, zero errors.

- [ ] **Step 5: Commit**

```bash
git add crates/penumbra-bench/src/metrics.rs
git commit -m "refactor(bench): remove duplicate tree budget rejection test from unit tests"
```

---

## Self-Review

1. **Spec coverage:** All four findings from the code review standards axis are covered:
   - Finding 1 (Duplicated `NodeReport` conversion) -> Task 1
   - Finding 2 (Duplicated `tap_graph` construction) -> Task 2
   - Finding 3 (Data clumps & duplicated provenance) -> Task 3
   - Finding 4 (Duplicated test in `metrics.rs` and `paper_metrics.rs`) -> Task 4
2. **Placeholder scan:** No TODOs, TBDs, or vague "handle appropriately" statements. All code blocks contain complete Rust implementations.
3. **Type consistency:** `NodeReport::from` handles both `NodeProfile` and `&NodeProfile`. `maybe_build_score_tap_graph(&Graph, &PaperData) -> Option<Graph>` matches in definition and callers. `PaperReportMeta::capture(&str, usize) -> Result<Self, String>` matches in definition and callers.
4. **Review Focus:**
   - Both owned and borrowed `NodeProfile` conversions tested in Task 1.
   - Non-label vs. label with/without existing score tensor tested in Task 2.
   - Clean tree requirements and struct fields tested in Task 3.
   - Preserved integration test verified before and after pruning in Task 4.
