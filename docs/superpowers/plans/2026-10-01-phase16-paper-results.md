# Phase 16 Full Comparison Runs & External Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` or `subagent-driven-development` to implement this plan task-by-task. Checklist steps track execution, not work already performed.

**Goal:** Complete every Phase 16 task and exit criterion with committed, frozen-revision evidence, without opening a PR or pushing.

**Architecture:** Correct measurement defects only in `penumbra-bench`, then freeze executable code and existing model fixtures before measuring. Keep the raw 28×28 MNIST experiment outside the controlled model registry under the owner's approved D18 amendment. Concrete-ML runs in a separate local repository and environment; it is calibration context, never a Penumbra golden comparator.

**Tech stack:** Existing Rust workspace, nightly for CKKS, tfhe 1.8.1, poulpy-ckks 0.8.3 / FFT64Neon, Criterion 0.7, Python through uv, existing ONNX/PyTorch/quantization stack. External calibration candidates: Python 3.11, concrete-ml 1.9.0, concrete-python 2.10.0, concrete-ml-extensions 0.1.9; the actual resolved transitive environment must be locked and installation exercised.

**Spec:** `ROADMAP.md:902-969`; `docs/PAPER.md` D3/D6/D7/D10/D11/D12/D15/D16/D17/D18 and §6; `AGENTS.md`. Owner already approved: retain the CKKS boundary; measure a raw 28×28 TFHE probe and exclude it from the controlled suite on capacity grounds even if its time is ≤600 seconds. Owner also approved creation of `/Users/hari/Developer/Penumbra-FHE-Paper` for calibration.

## Global constraints

- All Phase 16 Penumbra work uses `docs/paper-results`; no PR, remote creation, push, package publication, tag or Zenodo upload.
- Plan approval precedes implementation. This file is session-local planning output; copy it into `docs/superpowers/plans/2026-10-01-phase16-paper-results.md` during approved execution.
- No IR changes, schema-version changes to the IR, core branching, new primitives, CKKS packing changes, bootstrapping, or crypto/security retuning.
- Existing CKKS bounds and calibration fixtures are fixed before test evaluation; do not rerun calibration to enlarge a failing bound.
- TFHE encrypted outputs equal `evaluate_graph_int` bit-for-bit. CKKS compares raw output against the same integer reference within its existing declared bound; report measured errors and label flips.
- Canonical cross-backend latency is Criterion median with 95% CI. Per-op elapsed times are diagnostic breakdowns, not alternate headline timings.
- Pin 11 Rayon threads on the observed Apple M3 Pro (Mac15,6, 11 cores, 18 GB); disclose single-threaded FFT64Neon CKKS. Capture actual hardware/product OS/kernel/compiler from the machine at execution, not from prompt metadata.
- Freeze only after executable/test/fixture additions. Stage output under ignored `target/` until all frozen runs finish; do not trip the clean-tree guard by writing new untracked result files mid-run.
- Historical JSON is immutable. Missing historical evidence is marked unavailable, never reconstructed as a measurement.
- All Python commands use uv. Default CI must remain hermetic: committed fixture tests do not download MNIST or require optional ML packages; tests needing live ML use `pytest.importorskip`.

## Approval-sensitive calibration detail

`ROADMAP.md:938` says `p_error` “at its minimum”; D15 says “near zero.” [Pinned Concrete-ML source](https://github.com/zama-ai/concrete-ml/blob/v1.9.0/src/concrete/ml/common/utils.py) sets `2**-40` as its default, not a universal minimum. [Pinned optimizer source](https://github.com/zama-ai/concrete/blob/v2.10.0/compilers/concrete-optimizer/concrete-optimizer/src/noise_estimator/error.rs) passes the requested probability to `puruspe::inverfc`; this inspected function does not establish a supported global floor. Do not claim the default is a minimum or use zero without verified compiler support. Preferred explicit amendment, requiring approval with this plan: use `rounding_threshold_bits=None`, `p_error=2**-40`, and `global_p_error=None`, disclose configured and compiler-reported probabilities, and describe the row as unrounded, low-failure-probability external calibration. A compilation failure is not an installation failure and does not automatically authorize the published-number fallback.

## Review focus

1. Tied multiclass logits must select the first maximum, matching NumPy/integer reference, not Rust iterator last-maximum behavior.
2. Binary-logreg score error must use `score_tensor` and `expected_scores`; a scalar decision label must not generate a fictitious all-zero top-two margin.
3. NaN/infinity or malformed decoded vectors must fail before argmax/error folding; padded CKKS slots must never enter classification or error statistics.
4. CKKS tree rejection must come from real graph budget preflight before keygen, not a model-name policy string.
5. All final rows must share frozen build/runtime revision, correct graph hashes, thread count and sample protocol; historical timing paths must remain distinguishable.

## File ownership and task boundaries

- Harness correction owner: `crates/penumbra-bench/src/{metrics,paper,paper_backend,protocol}.rs`, `src/bin/report.rs` only for report metadata/version emission, tests colocated with metrics and new `tests/paper_metrics.rs` as needed.
- Scale-probe owner: new `examples/mnist/mnist28_export.py`, `examples/mnist/phase16_mnist28_fixture.json`, `tests/test_mnist28_fixture.py`, new `crates/penumbra-bench/src/bin/mnist_scale_probe.rs`, explicit Cargo binary registration. `.gitignore` only if the new downloaded-data/export paths are not already covered. No edit to `MODELS`, IR, op registry, quantizer, or crypto crates.
- Calibration owner: separate `/Users/hari/Developer/Penumbra-FHE-Paper/{pyproject.toml,uv.lock,scripts/concrete_calibration.py,tests/test_concrete_calibration.py}` and calibration result/log artifacts. No Concrete-ML dependency in Penumbra.
- Integration owner: freeze/build/run protocol, artifact audit, result promotion, `docs/{PAPER,BENCHMARKS,COMPARISON,MODEL-ZOO}.md`, `examples/mnist/README.md`, `CHANGELOG.md`, `ROADMAP.md`, and final commits.
- Harness and scale implementation are independent slices; delegate together after approval. Calibration preparation is independent but encrypted measurements must not overlap with Penumbra runs. Agents skip aggregate build/lint/format/test commands mid-flight; integration owner runs verification at the test-first and integration barriers. One owner integrates docs and shared files.

---

## Task 1 — Establish the branch and record approved protocol amendments

**Files:** plan path above; `docs/PAPER.md:215-228`; `ROADMAP.md:927-942`.

- [ ] Inspect current branch/worktree once at execution start; preserve unexpected user changes. Create `docs/paper-results` from the inspected checkout, rather than silently changing its base.

```bash
git switch -c docs/paper-results
```

- [ ] Persist the plan and record the already-approved D18 amendment in both PAPER and ROADMAP: real input has 784 elements; fixed CKKS linear-transform capacity is 256; keep the TFHE probe separate regardless of the timing gate. Preserve the ≤600-second timing observation as a scale result.
- [ ] Resolve the probability wording as approved, with pinned-source citation. Do not rewrite decisions not implicated by these amendments.
- [ ] Create the separate local paper repository only if the path remains absent; inspect rather than overwrite an existing directory. Initialize git and its uv project without a remote. Keep data, environments, models and generated ONNX/key/ciphertext files ignored. Commit locally using Conventional Commits.

**Check:** protocol amendments match the owner's answers; no changes to IR, backend implementations or controlled model membership.

## Task 2 — Correct consumer-visible accuracy metrics before freezing

**Files:** `metrics.rs:19-99,154-178`; `paper.rs:230-538`; tests in `metrics.rs` and `tests/paper_metrics.rs`.

**Consumes:** `PaperData::{output_kind,score_tensor,decision_threshold}`, `Sample::{expected_output,expected_scores}`, `eval_server`, existing `Comparator` and bounds.

**Produces:** deterministic metric helpers used by the actual metrics worker, not a parallel test-only implementation.

- [ ] First write behavioral regression cases for tied maxima, padded slots, finite checking, and the binary score/label distinction. Proposed helper contracts:

```rust
fn first_argmax(scores: &[f64]) -> Result<usize, String>;
fn max_abs_error(raw: &[f64], reference: &[i64]) -> Result<f64, String>;
fn binary_score_error_and_margin(
    raw_score: f64,
    reference_score: i64,
    threshold: i64,
) -> Result<(f64, f64), String>;
```

`first_argmax([7.0, 7.0, 1.0]) == 0`; empty/NaN/infinite inputs fail. `max_abs_error([1.25, -1.0], [1, -2]) == 1.0`; short vectors fail; only the declared logical prefix is supplied by the caller. Binary reference score 10, threshold 0, decoded score 11 yields `(1.0, 10.0)` and relative error 0.1, not a zero-margin scalar-label statistic. Threshold score 0 with nonzero error increments existing zero-margin counters rather than dividing by zero.

```rust
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
fn output_error_rejects_missing_or_nonfinite_components() {
    assert_eq!(max_abs_error(&[1.25, -1.0], &[1, -2]).unwrap(), 1.0);
    assert!(max_abs_error(&[1.0], &[1, -2]).is_err());
    assert!(max_abs_error(&[f64::NAN, -1.0], &[1, -2]).is_err());
}
```
- [ ] Run those tests against current behavior to establish the defect before correction. Test the worker-facing calculation/transition as well as helpers; avoid tests that merely duplicate arithmetic in a mock.
- [ ] Implement strict-greater first-maximum selection. Validate every logical decoded component before reductions; iterate only `expected_output.len()` logits, never padding.
- [ ] In the CKKS accuracy path only, add the declared score tensor as a temporary graph output for label models, as the TFHE score-tap path already does. Read `expected_scores` and `decision_threshold` from the fixture; do not hardcode `logit`/0 in new logic. Preserve the original graph for Criterion and RSS; the score tap is a diagnostic observation, not a different timed model.
- [ ] Keep the existing absolute bound on the original label/logit output. Report binary score error relative to `abs(reference_score - threshold)` with that binary-specific meaning stated in docs; do not reuse the label-output bound as a newly invented score-output bound. Retain the actual decrypted binary label for label-flip/task accuracy, rather than deriving a more favorable label from the tap.
- [ ] Run focused tests, then a real release paper-mode logreg smoke at a clean committed revision. Verify exact TFHE scores and decisions, CKKS bounded decision outputs, and non-fabricated score margins. The smoke artifact is not final paper evidence and lives under ignored `target/`.

**Check:** existing multiclass error/margin semantics and all fixed CKKS bounds remain unchanged; zero margins remain explicit; no new inference semantics.

## Task 3 — Preserve real budget rejections and include existing cost profiles

**Files:** `paper.rs:559-578` and accuracy profile capture; `paper_backend.rs:109-138`; `protocol.rs:356-398`; reuse `report.rs::NodeReport` and conversion pattern at `report.rs:200-224`; metadata emitter in `bin/report.rs`.

- [ ] Write a real-graph regression using `phase8_trees` and `phase8_xgb`: invoke configured backend `check_graph_budget` and assert the paper unsupported result retains that exact returned error. Do not pin incidental message wording. An unsupported row must not have latency/RSS/accuracy claims.
- [ ] Run actual graph budget preflight before paper policy and before key generation. Capture its error verbatim in `PaperModelRun.rejection`. Remove the fabricated named-tree rejection from the paper policy; trees have no CKKS accuracy bound because graph preflight rejects them first. Do not change the backend depth checker.
- [ ] Carry the existing representative profile into paper rows for both schemes: reuse `NodeReport` fields `name`, `op_type`, `build_secs`, `eval_secs`, `input_lens`, `output_len`, `counters`, `measured`. Capture profiles from original-graph evaluations, not score-tap-only work. Add `nodes` and `profile_sample_id` to `PaperModelRun`, empty/absent for unsupported rows. Identify the actual first seeded TFHE spot-check sample or first CKKS test sample used for the profile; do not imply that its data-dependent counters came from the Criterion representative if those IDs differ. TFHE retains its measured lookup/carry split; CKKS exposes rotations, rescale and polynomial/depth counters actually present in profiles.
- [ ] This is a benchmark-report format extension only, not an IR schema change. Emit paper-report `schema_version=2`; keep fixture paper-input schema and protocol sampling version at 1. Migrate every report constructor and consumer in this change; old result JSON stays immutable. Document report version/provenance and label node timings diagnostic.
- [ ] Verify both actual tree preflights without encrypted evaluation; run the integrated real logreg smoke to observe cost profile values. Add permanent tests only for profile aggregation/count invariants or rejection behavior, not copied-field wiring.

**Check:** no generic “CKKS cannot” claim; rejection is “leveled CKKS at these parameters,” with model/node/depth evidence from the backend. No Layer-1/2 changes.

## Task 4 — Build the raw 28×28 MNIST scale probe via Layer 3

**Files:** new generator/fixture/Python fixture test/bench binary; `crates/penumbra-bench/Cargo.toml`; model/bench/example documentation.

**Architecture chosen for the small probe:** `Conv2d(1,4,kernel_size=3,stride=4,bias=False) → ReLU → Flatten → Linear(196,10)`. Input remains 1×1×28×28 (784 encrypted pixels); output is 10 logits. The 4×7×7 activation avoids a large dense head, without preprocessing/downsampling outside the encrypted graph.

```python
class Mnist28CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, stride=4, bias=False)
        self.fc = nn.Linear(196, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(self.conv(x)).flatten(1))
```

- [ ] Fix experiment constants before timing: seed 0; official MNIST 60,000 training / 10,000 test split; pixels normalized by 255; Adam learning rate 1e-3, batch size 128, five epochs; quantization input_bits=3, weight_bits=(5,6), act_bits=2, max_mult_bits=1, per_channel=True, calibration='mse'. Select 128 calibration examples from training with seed 1507. No test-set tuning or repeated architecture search to get below the time gate.
- [ ] Use official MNIST gzip IDX resources from `https://ossci-datasets.s3.amazonaws.com/mnist/`, verifying published checksums and IDX image/label shapes/counts. Store downloads in ignored `data/`; reuse a valid local download. Network/corruption failures name URL/file and propagate; do not substitute sklearn digits.
- [ ] Write the fixture-contract regression first: exactly 784 integer inputs per sample; graph Conv2d in_h/in_w=28 and stride=4; 10 reference logits; `evaluate_graph_int` reproduces stored logits and first-maximum labels. Default tests inspect the committed fixture without importing torch or using the network. Add a guarded synthetic-data lowering test for ONNX export/quantization, independent of downloading/training.
- [ ] Train once and export ONNX using the existing legacy exporter convention, opset 13. Lower with `fhe.load_onnx`; quantize with the existing `Model.quantize`; compute frozen cleartext reference with `evaluate_graph_int`. Commit JSON containing graph, scale/bit plan, two fixed held-out samples, expected logits/labels, dataset checksums, training protocol and float/quantized held-out accuracy. Do not commit generated ONNX/model/data artifacts. No insertion into `MODELS`.
- [ ] Implement a bench binary `penumbra-mnist-scale-probe` accepting `--fixture`, `--threads`, `--out`. Use the existing bench `Session::new`, `encrypt`, `eval`, `decrypt` and `GraphProfile` APIs. Do not write another FHE execution loop. Assert full decrypted logits equal the fixture integer reference; reject mismatch before writing success evidence. Time only the server `Session::eval` call, excluding keygen/encrypt/decrypt; label it a one-sample feasibility observation, not Criterion headline latency.

```rust
let encrypted = session.encrypt(&sample.inputs);
let started = std::time::Instant::now();
let (output, profile) = session.eval(&fixture.graph, &encrypted)?;
let elapsed_secs = started.elapsed().as_secs_f64();
let got = session.decrypt(&output);
if got != sample.expected_logits {
    return Err("MNIST28 exactness violation".into());
}
```

- [ ] Record typed JSON with frozen build/runtime SHA, graph/fixture SHA, machine/compiler/threads, sample ID, elapsed_secs, ≤600-second boolean, exact expected/actual logits, profile counters, input length 784, CKKS capacity 256, and controlled-suite exclusion reason. Reuse provenance guard logic; fail dirty/mismatched build at measurement time.
- [ ] Observe the existing CKKS `encrypt_raw` error for this 784-value input in a standalone smoke, plus the linear-map capacity source. Record the actual error, not a new fake backend preflight. This is capacity evidence only, not a CKKS golden run. Keep any needed keys/material temporary.
- [ ] Run the TFHE probe after freezing. Its actual encrypted exactness check is the new graph's golden gate. Report the result even if >600 seconds; never kill at 600 seconds and report the timeout as a measured latency. If execution fails, diagnose and do not fabricate a time.

**Check:** raw 28×28 input, normal ONNX/quantization path, no crypto/core edits, registry remains 14 models, explicit D18 exclusion rather than silently claiming parity.

## Task 5 — Prepare and run external Concrete-ML calibration separately

**Files:** separate local paper repository listed above; eventual Penumbra results JSON references that repository commit and script hash.

- [ ] Use uv to resolve the pinned Python 3.11 environment natively, commit `uv.lock`, and test actual import of Concrete-ML/compiler/extensions. Record command, architecture, package versions, stdout/stderr and exit status. Wheel availability alone is not an installation success or failure.
- [ ] Select one representative calibration row: the existing `phase6_onnx` 8×8 CNN architecture (Conv 1→12, 3×3 stride 2, ReLU, Linear 108→10). Import its committed ONNX weights; do not retrain a different model under the same row name. Verify model file hash and architecture. Reconstruct calibration/training and held-out splits from the same seed/data conventions; verify sample IDs against the committed Penumbra protocol. Concrete's quantizer remains its own, explicitly disclosed.
- [ ] Adapt only the external ONNX copy to opset 14 with `onnx.version_converter.convert_version`: [Concrete-ML compile.py](https://github.com/zama-ai/concrete-ml/blob/v1.9.0/src/concrete/ml/torch/compile.py) requires its export opset, and [convert.py](https://github.com/zama-ai/concrete-ml/blob/v1.9.0/src/concrete/ml/onnx/convert.py) defines it as 14. Preserve initializer bytes and architecture; verify original and converted float logits agree on the full held-out split using ONNX Runtime. Record original/converted model hashes. Never mutate the committed Penumbra ONNX or relabel an unverified conversion.
- [ ] Test deterministic sample selection, timing boundaries and failure-result handling with hermetic input; use a real compiler/encryption smoke to prove live APIs. Compile the converted imported ONNX through pinned `compile_onnx_model`, with `n_bits=6`, `rounding_threshold_bits=None`, approved `p_error=2**-40`, `global_p_error=None`, CPU backend and 128-bit security. Six-bit Concrete quantization is fixed before runs and explicitly differs from Penumbra's per-layer bit plan; do not optimize it against observed latency. No high-p_error speed row disguised as exact mode.
- [ ] Evaluate float and Concrete quantized-cleartext accuracy on the full matching held-out set. Check actual encrypted output against Concrete's own quantized reference on the predeclared 30 held-out spot samples; report failures/probability limitations explicitly. Do not compare Concrete integer outputs with Penumbra's incompatible quantizer.
- [ ] Measure 10 repeated encrypted server `fhe_circuit.run` evaluations of the first held-out sample after one untimed warmup. Encrypt once outside the timing loop, keygen outside timing, decrypt outside timing. These repeats reuse one key/ciphertext, not independent cryptographic trials. Record raw durations plus median and descriptive uncertainty under a clearly named external timing method; do not label it Criterion or combine it with controlled scheme ratios. Record exact thread/runtime environment and compiler-reported probabilities/statistics.
- [ ] Emit calibration JSON with status, hardware/product OS/compiler/runtime, repository commit/script hash/lock hash, model/data hashes, quantization/rounding/probability/security configuration, accuracy methods/counts and raw timing data. Commit the calibration implementation and evidence in the local paper repo; copy the source-linked calibration result to Penumbra only after all frozen Penumbra measurements finish.
- [ ] If native install/import genuinely fails, preserve complete failure evidence and use the permitted published-context fallback. The [official pinned Concrete-ML notebook](https://github.com/zama-ai/concrete-ml/blob/v1.9.0/use_case_examples/white_paper_experiment/WhitePaperExperiments.ipynb) reports NN-20 0.995 s and NN-50 3.03 s on hpc7a with 192 cores, rounding to 6 bits and p_error=0.1. Cite the pinned notebook, model, machine, approximate configuration and accuracy definition; these are not same-machine/unrounded calibration. Do not infer comparable speedup. Compilation/inference failures after successful install are a separate unresolved prerequisite, not this fallback.

**Check:** no Concrete dependencies or scripts added to the Penumbra package; no claim of deterministic cryptographic exactness from merely disabling rounding.

## Task 6 — Integrate, verify, commit and freeze executable state

- [ ] Integrate both implementation slices and update feature/bug documentation plus CHANGELOG. Read all affected callers/tests and migrate report consumers. Keep existing comments and unrelated user work.
- [ ] Run focused regression tests and the actual small logreg smoke at a committed clean revision. Diagnose every failure before measurement; do not suppress warnings/assertions.
- [ ] Run relevant Python suite in both hermetic and ML-enabled environments through uv, and format/lint all touched Python. Run workspace Rust fmt/clippy/tests, enabling CKKS.

```bash
cargo fmt --all -- --check
cargo +nightly clippy --workspace --all-targets --features penumbra-bench/ckks -- -D warnings
uv run ruff check .
uv run black --check .
uv run pytest
uv run --extra ml pytest tests/test_mnist28_fixture.py tests/test_paper_protocol.py
```


- [ ] Commit all executable changes, test inputs and experiment methods. Record the resulting full SHA as F in a measurement manifest under ignored `target/phase16/`; do not create a tag. Build the release binaries at F. Record fixture/model hashes, full rustc -Vv, resolved packages, hardware, OS and actual threads. Run existing clean-tree/build-SHA guards.

```bash
cargo +nightly build -p penumbra-bench --features ckks --release --bins
```
- [ ] Execute existing ignored TFHE golden targets covering real models on this revision: `golden_digits`, `golden_qat`, `golden_onnx`, `golden_sklearn`, `golden_faces`, and second-framework `golden_trees` (plus any other ignored encrypted golden discovered in the test inventory). Existing non-ignored workspace tests cover default goldens. Run all CKKS golden tests with fixed bounds and measured error output. Do not run expensive verification concurrently with latency collection.

```bash
cargo +nightly test -p penumbra-fhe-runtime --release --test golden_onnx -- --ignored --nocapture
cargo +nightly test -p penumbra-ckks --features ckks --release -- --nocapture
```
- [ ] Execute the non-ignored workspace golden suite at F as well, recording the frozen revision with its output:

```bash
cargo +nightly test --workspace --features penumbra-bench/ckks --release
```

**Freeze rule:** after F, only results/docs commits. Any executable/model/protocol correction creates a new F and reruns the full final protocol; no stale mix of commits.

## Task 7 — Execute the full pinned protocol and the MNIST observation

**Output staging:** `target/phase16/` only while measurements are in progress.

- [ ] Confirm power/idle conditions using actual machine observations. Finish builds/tests/calibration preparation first; no agents, training or external FHE evaluations during controlled timing.
- [ ] Run all 14 current models, both backends, 11 threads, release paper mode:

```bash
RAYON_NUM_THREADS=11 ./target/release/penumbra-bench-report \
  --mode paper --models all --backends tfhe,ckks --threads 11 \
  --format json --out target/phase16/phase16-paper-final.json
```

Do not pass `--samples` or profile overrides in paper mode. Existing policy selects 30 TFHE spot checks and full CKKS test sets. Expect 28 rows: 14 TFHE success rows, 12 CKKS success rows and 2 real CKKS tree-rejection rows; unexpected additional rejection is a failure, not a narrower completed suite.
- [ ] Run the separate MNIST28 probe once at the same F and thread setting:

```bash
RAYON_NUM_THREADS=11 ./target/release/penumbra-mnist-scale-probe \
  --fixture examples/mnist/phase16_mnist28_fixture.json --threads 11 \
  --out target/phase16/phase16-mnist28-probe.json
```

- [ ] Preserve raw Criterion artifacts in the separate local paper repo (or a reproducible compressed evidence artifact there), with checksums/source linkage; never commit key/ciphertext material. Retain logs of golden outputs and exact commands in the evidence manifest.
- [ ] Audit JSON against fixtures before promotion: exact model/backend pair coverage; graph hashes match committed inputs; `runtime_commit == build_commit == F`; dirty=false; requested/actual Rayon threads=11; at least 10 Criterion samples and ordered finite 95% CI; positive wire sizes/RSS; TFHE exact_checks_passed=30 and PBS total=lookup+carry; CKKS sample_count equals its full test split and max error ≤ existing bound; margin denominators/counts meaningful; profile sample IDs match their actual fixture samples; both tree errors agree with actual preflight evidence. Record faces' mixed calibration/test spot-check policy, not an invented 30 held-out faces claim.
- [ ] If any bound fails, stop final measurement acceptance and diagnose scale/indexing/polynomial behavior without enlarging the bound. Raise any required crypto/core change to the owner before implementation.

**Check:** this is the shared harness exercised end-to-end, not merely unit-test success.

## Task 8 — Derive traceable before/after tables and update final write-ups

**Files:** new results below; `docs/BENCHMARKS.md`; `docs/COMPARISON.md`; `ROADMAP.md`.

- [ ] Promote successful final files only after all measurements finish:
  - `docs/results/phase16-paper-final.json`
  - `docs/results/phase16-mnist28-probe.json`
  - `docs/results/phase16-concrete-calibration.json` (measured row or honest installation-failure/published-context artifact)
  - `docs/results/phase16-tfhe-before-after.json`
  - `docs/results/phase16-run-manifest.json` (F, methods, hashes, commands, verification/evidence links).
- [ ] Derive D17 tables from immutable evidence. Preserve Phase 12/13 historical sources and caveats. Use the Phase-14 **pre-fix** baseline as the supplemental same-fixture radix reference, explicitly identifying it as pre-fix, not post-fix. Phase-10 results predate the fix; do not mislabel chronology.
- [ ] For each historical model with a matching baseline, derive pre/post diagnostic mean server time and measured PBS from the same report methodology, identifying source file, run SHA, fixture/bit-plan differences, sample counts and method. Mark absent baselines unavailable. A changed fixture is a confounded historical comparison, not the isolated gain of per-tensor radix.
- [ ] For logreg, use the retained Phase-13 Criterion median/CI as the canonical pre-fix timing comparison. For other models without pre-fix Criterion evidence, label timing ratios diagnostic; do not divide historical diagnostic timing by final Criterion latency and call it an implementation speedup. Report final cross-scheme M× only from the two final Criterion rows. To obtain post-fix matched diagnostic observations, run the existing diagnostics mode at F under 11 threads with two samples per baseline model; preserve its raw new JSON before deriving ratios. This is breakdown/cost evidence, not another headline timing path.

```bash
RAYON_NUM_THREADS=11 ./target/release/penumbra-bench-report \
  --mode diagnostics --models all --backends tfhe --threads 11 --samples 2 \
  --format json --out target/phase16/phase16-tfhe-diagnostics.json
```

- [ ] Commit the raw diagnostic file as `docs/results/phase16-tfhe-diagnostics.json`; derived JSON contains source hashes/paths and formulas `N = pre_tfhe / post_tfhe` (matched diagnostic or explicitly Criterion), `M = final_tfhe_criterion / final_ckks_criterion`. Keep method distinction in table headers and prose.
- [ ] BENCHMARKS: final Criterion median/CI, server peak RSS, key/ciphertext sizes, lookup/carry PBS, CKKS cost counters, float/quantized/encrypted accuracy, label flips and error/margin distributions; named source JSON and frozen revision for every number. Security setup cites existing Phase-15 lattice estimates, not retuned parameters.
- [ ] COMPARISON: replace provisional logreg Results/Verdict with evidence from final canonical CIs; overlapping/small differences are not findings. Explain measured implementation gain against final cross-scheme gap without mixing methods. Trees always use “leveled CKKS at these parameters.” Threats cover one M3 Pro, ≤256/8192 utilized slots, CKKS maturity/level budget, unmeasured batching, quantization/polynomial error, small Criterion sample statistics, scale probe and parity exclusion, and external calibration's separate quantizer/method.
- [ ] Add the 28×28 graph as a separately labelled probe in MODEL-ZOO/example docs, never as a backend-parity suite member. Record both measured TFHE time and fixed CKKS input capacity even if TFHE passes the ≤600-second gate.
- [ ] Record external calibration as a separate context table; if installation fails, the table says no native measured row and cites published approximate rows with hardware/configuration caveats.
- [ ] Update reproduction commands: full final protocol uses paper mode, not the old seven-model diagnostic loop or golden-test timing as canonical evidence. Preserve historical sections with labels rather than silently replacing their provenance.
- [ ] Run strict documentation build and artifact audit before checking off ROADMAP tasks/exit criteria:

```bash
uv run --group docs mkdocs build --strict
```

Check each Phase-16 checkbox against an actual committed artifact/method. Do not mark completion if installation succeeded but calibration remains unimplemented, a golden failed, or any frozen row is missing.
- [ ] Commit results/docs locally in logical Conventional Commits. Final report names F, result paths, executed verification, measured MNIST time/capacity finding and calibration status; no PR or push.

## Acceptance map

| Roadmap requirement | Task/evidence |
|---|---|
| Frozen library commit | 6; final JSON and run manifest |
| Full protocol, both backends, all committed models | 7; 28 final rows, 30 TFHE checks/model, CKKS full splits |
| Criterion / RSS / sizes / costs | 3 + 7; final report v2 with existing profiles |
| Trees exact TFHE / real CKKS rejection | 3 + 7; tree goldens and rejection rows |
| Raw 28×28 gate without backend edits | 4 + 7; separate probe under approved D18 amendment |
| External calibration or legitimate installation fallback | 5 + 8; separately sourced calibration artifact |
| Before/after N× versus final cross-scheme M× | 8; raw diagnostics plus source-linked derived table |
| Final BENCHMARKS/COMPARISON + strict docs | 8 |
| Golden gates on frozen revision | 6 + 7; existing gates plus probe exactness |
| No PR/push; uv; raise conflicts | Global constraints and explicit stop gates |

## Current evidence versus planned work

Discovery and planning only. No Phase-16 branch, code, installation, build, test or encrypted measurement is claimed by this plan. Existing source inspection established the harness defects/capacity constraint; outcome values remain unmeasured. Rust LSP initialization previously failed because the installed stable toolchain lacks rust-analyzer; graph/source inspection was used instead. Recheck availability before symbol changes during implementation and use LSP references if available.
