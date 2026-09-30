# Phase 14 Code Review Remediations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all standards and specification findings from the two-axis code review on branch `perf/tfhe-per-tensor-radix` (PR #31), fixing CI regression failure, removing silent error suppression in bit-width derivation, refactoring Fowler code smells, and clarifying server-side input trimming.

**Architecture:** 
1. **Benchmark Integrity:** Restore the deleted `phase8_trees` baseline entry in `crates/penumbra-bench/baselines/tfhe-classic.json` with its post-fix per-tensor radix measurements (`1699` PBS), unblocking CI.
2. **Fail Loudly & Early:** Replace `std::panic::catch_unwind(...).ok()` in `crates/penumbra-core/src/eval.rs` with explicit checked bit-width derivation (`op_spec_output_bits_multi_checked` in `crates/penumbra-core/src/bitwidth.rs`), propagating actionable error messages naming the node and op type.
3. **Clean Code & Smell Elimination:** Eliminate Fowler baseline smells in `crates/penumbra-tfhe/src/ops/mac.rs` (Mysterious Names) and across `linear.rs`/`conv2d.rs` (Duplicated Code) by introducing a helper `NodeWidths::linear_op_blocks`.
4. **Boundary Clarity:** Document the zero-PBS server-side lazy input trimming mechanism in `docs/NOTES-tfhe.md` and `docs/BACKENDS.md`, confirming why trimming at op entrypoints satisfies the D14 Amendment without adding scheme-specific seams to Layer 2.

**Tech Stack:** Rust (`cargo`), `tfhe-rs`, `penumbra-core`, `penumbra-tfhe`, `penumbra-bench`, MkDocs Material.

**Spec:** Two-axis code review findings against `ROADMAP.md` Phase 14 (lines 745–823), `docs/PAPER.md` Decision D14 & D14 Amendment (lines 239–262, 326–329), and repo standards in `AGENTS.md` (§1.4, §6).

---

## Global Constraints

- **Golden invariant is sacred:** TFHE must remain bit-for-bit exact on every committed model; no golden test may be altered or weakened (`AGENTS.md` §1.1).
- **Core narrow-waist discipline:** No IR schema changes, no new scheme-specific branches in `penumbra-core`, and no CKKS modifications (`AGENTS.md` §1.2, `ROADMAP.md:807`).
- **Fail loudly, early:** No silent swallowing of errors or panics; invalid inputs or bit-width derivation failures must produce actionable errors naming the offending node (`AGENTS.md` §1.4).
- **Clean formatting and linter passes:** `cargo fmt` and `cargo clippy -D warnings` must remain completely clean (`AGENTS.md` §6).
- **Commit discipline:** Use Conventional Commits (`fix(...)`, `refactor(...)`, `chore(...)`, `docs(...)`) with no AI/agent attribution (`AGENTS.md` §8).

## Review Focus

1. **Missing baseline entry failure:** Benchmark regression runner fails if any model in the run is missing from the baseline JSON. Verified by running `penumbra-bench-report` against `phase8_trees`.
2. **Invalid operator input arity:** If a node declares fewer or more inputs than its `OpSpec` expects (e.g. `Add` with 1 input), `op_spec_output_bits_multi_checked` must return an `Err` describing the arity mismatch, rather than panicking or swallowing.
3. **Radix calculation preservation:** Renaming variables in `evaluate_weighted_mac` must preserve bit manipulation and shifted ciphertext arithmetic bit-for-bit.
4. **Decoupled linear width extraction:** `linear_op_blocks` helper must return identical `(in_bits, out_bits, acc_blocks)` under both `Uniform` and `PerTensor` variants.
5. **CI pipeline pass:** Full local CI commands (`cargo test --workspace --release`, benchmark regression check, and MkDocs) must pass cleanly before merge.

---

### Task 1: Restore `phase8_trees` in TFHE Classic Regression Baseline

**Files:**
- Modify: `crates/penumbra-bench/baselines/tfhe-classic.json:60-65`
- Test: Benchmark regression check via `cargo run -p penumbra-bench --release --bin penumbra-bench-report`

**Interfaces:**
- Consumes: Post-fix benchmark measurements from `docs/results/phase14-tfhe-per-tensor.json:1328-1534`.
- Produces: Restored `phase8_trees` entry in `tfhe-classic.json` with `pbs: 1699`.

- [ ] **Step 1: Verify current failure**

Run:
```bash
cargo run -p penumbra-bench --release --bin penumbra-bench-report -- --models phase8_trees --backends tfhe --samples 1 --baseline crates/penumbra-bench/baselines/tfhe-classic.json
```
Expected: FAIL with `REGRESSION: phase8_trees/tfhe: missing from baseline; regenerate the baseline with --write-baseline`.

- [ ] **Step 2: Add `phase8_trees` entry to `crates/penumbra-bench/baselines/tfhe-classic.json`**

Insert the following JSON entry into the models array in `crates/penumbra-bench/baselines/tfhe-classic.json`:
```json
    {
      "model": "phase8_trees",
      "backend": "tfhe",
      "profile": "classic",
      "num_blocks": 7,
      "client_key_bytes": 23940,
      "server_key_bytes": 120419227,
      "input_ct_bytes": 3457700,
      "output_ct_bytes": 230532,
      "cost_proxy": {
        "cmp_pbs_ops": 67,
        "ct_add": 114,
        "scalar_add": 38,
        "scalar_mul": 76
      },
      "measured_totals": {
        "pbs": 1699
      },
      "labels_matched": 1,
      "labels_checked": 1
    }
```

- [ ] **Step 3: Run regression check to verify it passes**

Run:
```bash
cargo run -p penumbra-bench --release --bin penumbra-bench-report -- --models phase2_logreg,phase4_cnn,phase6_sklearn,phase8_trees --backends tfhe --samples 1 --baseline crates/penumbra-bench/baselines/tfhe-classic.json
```
Expected: PASS with `0 baseline violation(s)`.

- [ ] **Step 4: Commit**

```bash
git add crates/penumbra-bench/baselines/tfhe-classic.json
git commit -m "chore(bench): restore phase8_trees in TFHE classic baseline"
```

---

### Task 2: Robust Bit-Width Error Handling in `penumbra-core`

**Files:**
- Modify: `crates/penumbra-core/src/bitwidth.rs:40-140`
- Modify: `crates/penumbra-core/src/eval.rs:135-155`
- Test: `crates/penumbra-core/tests/eval_node_errors.rs`

**Interfaces:**
- Consumes: `spec: &OpSpec`, `input_bits: &[usize]`.
- Produces: `op_spec_output_bits_multi_checked(spec: &OpSpec, input_bits: &[usize]) -> Result<Vec<usize>, String>`.

- [ ] **Step 1: Write failing test for invalid node input count**

In `crates/penumbra-core/tests/eval_node_errors.rs`, append:
```rust
#[test]
fn node_with_invalid_input_count_fails_loudly_before_eval() {
    use std::collections::HashMap;
    use penumbra_core::backend::{Backend, CtVec, EvalCtx};
    use penumbra_core::eval::evaluate_graph;
    use penumbra_core::ir::{Graph, Node, OpSpec, SCHEMA_VERSION};
    use penumbra_core::ops::Op;

    struct DummyBackend;
    impl Backend for DummyBackend {
        type Ciphertext = i64;
        type ServerKey = ();
        type ClientKey = ();
        fn name(&self) -> &'static str { "dummy" }
        fn build_op(&self, _spec: &OpSpec) -> Result<Box<dyn Op<Self>>, String> {
            Ok(Box::new(DummyOp))
        }
    }
    struct DummyOp;
    impl Op<DummyBackend> for DummyOp {
        fn eval(&self, _ctx: &EvalCtx<DummyBackend>, _inputs: &CtVec<DummyBackend>) -> CtVec<DummyBackend> {
            vec![0]
        }
    }

    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 8,
        input_bits: 4,
        inputs: vec!["a".to_string()],
        outputs: vec!["out".to_string()],
        nodes: vec![Node {
            name: "malformed_add".to_string(),
            inputs: vec!["a".to_string()], // Add requires 2 inputs
            outputs: vec!["out".to_string()],
            op: OpSpec::Add {},
        }],
    };

    let backend = DummyBackend;
    let ctx = EvalCtx::new(&(), 4);
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vec![1]);

    let err = evaluate_graph(&backend, &ctx, &graph, inputs)
        .expect_err("node with invalid input count must fail loudly");
    assert!(
        err.contains("node 'malformed_add' (Add): Add is a two-input op: expected 2 inputs, got 1"),
        "unexpected error message: {err}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cargo test -p penumbra-core --test eval_node_errors node_with_invalid_input_count_fails_loudly_before_eval
```
Expected: FAIL (currently panics or fails with different message due to `catch_unwind`).

- [ ] **Step 3: Implement checked bit-width derivation in `crates/penumbra-core/src/bitwidth.rs`**

In `crates/penumbra-core/src/bitwidth.rs`:
```rust
/// Checked version of [`op_spec_output_bits_n`] returning an actionable error message on input count mismatch.
pub fn op_spec_output_bits_n_checked(spec: &OpSpec, input_bits: &[usize]) -> Result<usize, String> {
    match spec {
        OpSpec::Linear { weights, bias, weight_bits } => {
            if input_bits.len() != 1 {
                return Err(format!("Linear is a single-input op: expected 1 input, got {}", input_bits.len()));
            }
            let in_b = input_bits[0];
            let n = weights.first().map_or(0, Vec::len);
            let sum_growth = ceil_log2(n);
            let sum_bits = in_b + weight_bits + sum_growth;
            let bias_bits = bias.iter().map(|&b| magnitude_bits(b)).max().unwrap_or(0);
            Ok(sum_bits.max(bias_bits) + 2)
        }
        OpSpec::Conv2d { bias, weight_bits, in_channels, kernel_h, kernel_w, .. } => {
            if input_bits.len() != 1 {
                return Err(format!("Conv2d is a single-input op: expected 1 input, got {}", input_bits.len()));
            }
            let in_b = input_bits[0];
            let fan_in = in_channels * kernel_h * kernel_w;
            let sum_growth = ceil_log2(fan_in);
            let sum_bits = in_b + weight_bits + sum_growth;
            let bias_bits = bias.iter().map(|&b| magnitude_bits(b)).max().unwrap_or(0);
            Ok(sum_bits.max(bias_bits) + 2)
        }
        OpSpec::Activation { output_bits, .. } => {
            if input_bits.len() != 1 {
                return Err(format!("Activation is a single-input op: expected 1 input, got {}", input_bits.len()));
            }
            Ok(*output_bits)
        }
        OpSpec::Requant { out_bits, .. } => {
            if input_bits.len() != 1 {
                return Err(format!("Requant is a single-input op: expected 1 input, got {}", input_bits.len()));
            }
            Ok(*out_bits)
        }
        OpSpec::Pool { mode, pool_h, pool_w, .. } => {
            if input_bits.len() != 1 {
                return Err(format!("Pool is a single-input op: expected 1 input, got {}", input_bits.len()));
            }
            let in_b = input_bits[0];
            let k = pool_h * pool_w;
            let parsed_mode = match mode.as_str() {
                "avg" => PoolMode::Avg,
                "max" => PoolMode::Max,
                _ => PoolMode::Avg,
            };
            match parsed_mode {
                PoolMode::Avg => Ok(in_b + ceil_log2(k)),
                PoolMode::Max => Ok(in_b),
            }
        }
        OpSpec::Add {} => {
            if input_bits.len() != 2 {
                return Err(format!("Add is a two-input op: expected 2 inputs, got {}", input_bits.len()));
            }
            Ok(input_bits[0].max(input_bits[1]) + 1)
        }
        OpSpec::Argmax { .. } => {
            if input_bits.len() != 1 {
                return Err(format!("Argmax is a single-input op: expected 1 input, got {}", input_bits.len()));
            }
            Ok(1)
        }
        OpSpec::Compare { .. } => {
            if input_bits.len() != 1 {
                return Err(format!("Compare is a single-input op: expected 1 input, got {}", input_bits.len()));
            }
            Ok(1)
        }
        OpSpec::Concat { sizes } => {
            if input_bits.len() != sizes.len() {
                return Err(format!("Concat takes one input per declared segment: expected {}, got {}", sizes.len(), input_bits.len()));
            }
            Ok(input_bits.iter().copied().max().unwrap_or(0))
        }
        OpSpec::Split { .. } => {
            if input_bits.len() != 1 {
                return Err(format!("Split is a single-input op: expected 1 input, got {}", input_bits.len()));
            }
            Ok(input_bits[0])
        }
    }
}

/// Checked version of [`op_spec_output_bits_multi`].
pub fn op_spec_output_bits_multi_checked(spec: &OpSpec, input_bits: &[usize]) -> Result<Vec<usize>, String> {
    match spec {
        OpSpec::Split { sizes } => {
            if input_bits.len() != 1 {
                return Err(format!("Split is a single-input op: expected 1 input, got {}", input_bits.len()));
            }
            Ok(vec![input_bits[0]; sizes.len()])
        }
        _ => op_spec_output_bits_n_checked(spec, input_bits).map(|b| vec![b]),
    }
}
```

In `crates/penumbra-core/src/lib.rs`, re-export `op_spec_output_bits_multi_checked`.

In `crates/penumbra-core/src/eval.rs:140-152`, replace `catch_unwind` with:
```rust
        let output_bits = crate::bitwidth::op_spec_output_bits_multi_checked(&node.op, &input_bits)
            .map_err(|e| format!("node '{}' ({}): {e}", node.name, node.op.op_type()))?;

        let t_build = profile.as_ref().map(|_| Instant::now());
        let op = backend
            .build_op_with_bits(&node.op, &input_bits, &output_bits)
            .map_err(|e| format!("node '{}': {e}", node.name))?;
        let build = t_build.map(|t| t.elapsed()).unwrap_or_default();
```

- [ ] **Step 4: Run test suite to verify it passes**

Run:
```bash
cargo test -p penumbra-core --release
```
Expected: PASS (all 23 library tests and integration test suites pass).

- [ ] **Step 5: Commit**

```bash
git add crates/penumbra-core/src/bitwidth.rs crates/penumbra-core/src/eval.rs crates/penumbra-core/src/lib.rs crates/penumbra-core/tests/eval_node_errors.rs
git commit -m "fix(core): fail loudly on bit-width derivation error without catch_unwind"
```

---

### Task 3: Refactor Mysterious Names in Weighted MAC

**Files:**
- Modify: `crates/penumbra-tfhe/src/ops/mac.rs:55-135`
- Test: `crates/penumbra-tfhe/src/ops/mac.rs:133-164` (inline unit tests)

**Interfaces:**
- Consumes: `WidthCache`, `groups: &BTreeMap<i64, Vec<usize>>`, `bias: i64`, `in_bits: usize`, `acc_blocks: usize`.
- Produces: `evaluate_weighted_mac` with self-explanatory, domain-revealing variable names.

- [ ] **Step 1: Verify current tests pass**

Run:
```bash
cargo test -p penumbra-tfhe --lib ops::mac::tests --release
```
Expected: PASS.

- [ ] **Step 2: Refactor single-letter names in `evaluate_weighted_mac`**

In `crates/penumbra-tfhe/src/ops/mac.rs:55-131`, refactor variable names:
```rust
pub(crate) fn evaluate_weighted_mac(
    sk: &ServerKey,
    cache: &WidthCache,
    groups: &BTreeMap<i64, Vec<usize>>,
    bias: i64,
    in_bits: usize,
    acc_blocks: usize,
) -> SignedRadixCiphertext {
    let mut pos_terms: Vec<SignedRadixCiphertext> = Vec::new();
    let mut neg_terms: Vec<SignedRadixCiphertext> = Vec::new();

    for (&weight, idxs) in groups {
        let k = idxs.len();
        let sum_bits = in_bits + penumbra_core::bitwidth::ceil_log2(k);
        let group_blocks = value_blocks(sum_bits, acc_blocks);
        let group_sum_narrow = if idxs.len() == 1 {
            cache.get(group_blocks, idxs[0]).clone()
        } else {
            sk.sum_ciphertexts_parallelized(idxs.iter().map(|&i| cache.get(group_blocks, i)))
                .expect("non-empty cts group")
        };
        let group_sum_acc = resize(sk, &group_sum_narrow, acc_blocks).into_owned();

        let weight_mag = weight.unsigned_abs();
        let mut pre: Vec<SignedRadixCiphertext> = Vec::with_capacity(MESSAGE_BITS);
        pre.push(group_sum_acc);
        for j in 1..MESSAGE_BITS {
            let needed = (0..64).any(|bit| ((weight_mag >> bit) & 1) != 0 && (bit % MESSAGE_BITS == j));
            if needed {
                let shifted = sk.unchecked_scalar_left_shift_parallelized(&pre[0], j as u64);
                pre.push(shifted);
            } else {
                pre.push(sk.create_trivial_zero_radix(acc_blocks));
            }
        }

        for bit in 0..64 {
            if ((weight_mag >> bit) & 1) != 0 {
                let block_idx = bit / MESSAGE_BITS;
                if block_idx < acc_blocks {
                    let shifted = sk.blockshift(&pre[bit % MESSAGE_BITS], block_idx);
                    if weight > 0 {
                        pos_terms.push(shifted);
                    } else {
                        neg_terms.push(shifted);
                    }
                }
            }
        }
    }

    if bias > 0 {
        pos_terms.push(
            sk.create_trivial_radix::<u64, SignedRadixCiphertext>(bias.unsigned_abs(), acc_blocks),
        );
    } else if bias < 0 {
        neg_terms.push(
            sk.create_trivial_radix::<u64, SignedRadixCiphertext>(bias.unsigned_abs(), acc_blocks),
        );
    }

    let pos_sum = if pos_terms.is_empty() {
        sk.create_trivial_zero_radix(acc_blocks)
    } else {
        sk.unchecked_sum_ciphertexts_vec_parallelized(pos_terms)
            .unwrap_or_else(|| sk.create_trivial_zero_radix(acc_blocks))
    };

    if neg_terms.is_empty() {
        pos_sum
    } else {
        let neg_sum = sk
            .unchecked_sum_ciphertexts_vec_parallelized(neg_terms)
            .unwrap_or_else(|| sk.create_trivial_zero_radix(acc_blocks));
        sk.sub_parallelized(&pos_sum, &neg_sum)
    }
}
```

- [ ] **Step 3: Run unit tests to verify behavior is unchanged**

Run:
```bash
cargo test -p penumbra-tfhe --lib ops::mac::tests --release
```
Expected: PASS (both `test_mac_cache_widths` and `test_evaluate_weighted_mac_correctness` pass).

- [ ] **Step 4: Commit**

```bash
git add crates/penumbra-tfhe/src/ops/mac.rs
git commit -m "refactor(tfhe): clarify variable names in evaluate_weighted_mac"
```

---

### Task 4: Deduplicate Linear and Conv2d Node Width Extraction

**Files:**
- Modify: `crates/penumbra-tfhe/src/width.rs:78-100`
- Modify: `crates/penumbra-tfhe/src/ops/linear.rs:43-47`
- Modify: `crates/penumbra-tfhe/src/ops/conv2d.rs:187-191`
- Test: `crates/penumbra-tfhe/src/width.rs` and `crates/penumbra-tfhe/tests/per_tensor_width.rs`

**Interfaces:**
- Consumes: `&NodeWidths`, `num_blocks: usize`.
- Produces: `NodeWidths::linear_op_blocks(&self, num_blocks: usize) -> (usize, usize, usize)`.

- [ ] **Step 1: Add unit test in `crates/penumbra-tfhe/src/width.rs`**

In `crates/penumbra-tfhe/src/width.rs:100-137`, add:
```rust
    #[test]
    fn test_linear_op_blocks() {
        let w = NodeWidths::PerTensor {
            inputs: vec![4],
            outputs: vec![12],
        };
        let (ib, ob, acc) = w.linear_op_blocks(8);
        assert_eq!(ib, 4);
        assert_eq!(ob, 12);
        assert_eq!(acc, signed_blocks(12, 8)); // 12 / 2 = 6 blocks
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cargo test -p penumbra-tfhe --lib width::tests::test_linear_op_blocks --release
```
Expected: FAIL (`linear_op_blocks` method not found).

- [ ] **Step 3: Implement `linear_op_blocks` and adopt in `linear.rs` and `conv2d.rs`**

In `crates/penumbra-tfhe/src/width.rs`:
```rust
impl NodeWidths {
    ...
    /// Extracts `(input_bits, output_bits, acc_blocks)` for single-input, single-output linear operations.
    pub(crate) fn linear_op_blocks(&self, num_blocks: usize) -> (usize, usize, usize) {
        let ib = self.input_bits(0, num_blocks);
        let ob = self.output_bits(0, num_blocks);
        let acc = signed_blocks(ob, num_blocks);
        (ib, ob, acc)
    }
}
```

In `crates/penumbra-tfhe/src/ops/linear.rs:43-46`, replace with:
```rust
        let (ib, _ob, acc) = widths.linear_op_blocks(ctx.num_blocks);
```

In `crates/penumbra-tfhe/src/ops/conv2d.rs:187-190`, replace with:
```rust
        let (ib, _ob, acc) = widths.linear_op_blocks(ctx.num_blocks);
```

- [ ] **Step 4: Run tests to verify it passes**

Run:
```bash
cargo test -p penumbra-tfhe --lib width::tests --release && cargo test -p penumbra-tfhe --test per_tensor_width --release
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/penumbra-tfhe/src/width.rs crates/penumbra-tfhe/src/ops/linear.rs crates/penumbra-tfhe/src/ops/conv2d.rs
git commit -m "refactor(tfhe): deduplicate linear operator width extraction"
```

---

### Task 5: Document Server-Side Input Trimming Architecture

**Files:**
- Modify: `docs/NOTES-tfhe.md`
- Modify: `docs/BACKENDS.md`
- Test: `uv run mkdocs build --strict`

**Interfaces:**
- Consumes: Phase 14 D14 Amendment rationale (`docs/PAPER.md:255-262`), `width.rs` `resize_tensor` zero-PBS block trimming.
- Produces: Updated notes in `docs/NOTES-tfhe.md` and `docs/BACKENDS.md` clarifying that server-side trimming executes at op entrypoints (e.g. `WidthCache::build` / `resize_tensor`) for $0$ PBS, satisfying the requirement without adding scheme-specific seams to `penumbra-core`.

- [ ] **Step 1: Update `docs/NOTES-tfhe.md`**

In `docs/NOTES-tfhe.md` under `## Per-tensor radix width (Phase 14)`, add a dedicated paragraph:
```markdown
### Server-side Input Trimming

Under the D14 Amendment, client-side encryption remains uniform: clients encrypt input tensors at the model-level radix ceiling `num_blocks` without needing graph-wide per-tensor width knowledge. On the server side, input ciphertexts are trimmed to each consuming operation's required width (e.g. `WidthCache::build` in `Linear`/`Conv2d` or `resize_tensor` in `Requant`/`Add`) via `sk.cast_to_signed(ct, target_blocks)`. Because dropping higher radix blocks costs 0 PBS and allocates no crypto noise, trimming is computationally free. Performing trimming lazily at the operator boundary preserves the backend-neutral Layer-2 eval loop contract (`penumbra-core`) with zero scheme-specific entrypoint branching.
```

- [ ] **Step 2: Update `docs/BACKENDS.md`**

In `docs/BACKENDS.md` under `### Sizing & Bit-Width Budgets`, add:
```markdown
- **Input Trimming on Arrival:** TFHE clients encrypt inputs using `num_blocks` (the model-level ceiling). Sizing to the input tensor's derived bit width occurs at the server operator boundary via block trimming (`sk.cast_to_signed`). Dropping higher blocks costs $0$ PBS and requires no client-side multi-width coordination.
```

- [ ] **Step 3: Verify MkDocs build**

Run:
```bash
uv run mkdocs build --strict
```
Expected: PASS with 0 warnings.

- [ ] **Step 4: Commit**

```bash
git add docs/NOTES-tfhe.md docs/BACKENDS.md
git commit -m "docs(tfhe): document server-side input trimming architecture"
```

---

### Task 6: Full Verification & Golden Test Assertion

**Files:**
- Test: Whole workspace test suite, clippy, fmt, python tests.

- [ ] **Step 1: Run formatting and clippy across workspace**

Run:
```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings
```
Expected: Clean pass with 0 warnings.

- [ ] **Step 2: Run Python tests and linters**

Run:
```bash
uv run ruff check . && uv run black --check . && uv run pytest
```
Expected: Clean pass.

- [ ] **Step 3: Run full Rust workspace tests in release mode**

Run:
```bash
cargo test --workspace --release
```
Expected: All tests pass, including golden tests and per-tensor width tests.

- [ ] **Step 4: Run full benchmark regression check across all models**

Run:
```bash
cargo run -p penumbra-bench --release --bin penumbra-bench-report -- --models phase2_logreg,phase4_cnn,phase6_sklearn,phase8_trees --backends tfhe --samples 1 --baseline crates/penumbra-bench/baselines/tfhe-classic.json
```
Expected: PASS with 0 violations.
