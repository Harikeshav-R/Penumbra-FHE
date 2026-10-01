# Phase 15 Code Review Remediations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all specification traceability and code standards findings from the two-axis code review on branch `feat/bench-paper-metrics`: fix incorrect decision citations (`D10` -> `D7`) for the CKKS calibration protocol, eliminate the unused direct `tfhe` dependency in `penumbra-bench`, and deduplicate model-specific CKKS backend construction across the benchmark harness.

**Architecture:**
1. **Spec Traceability Alignment:** Update `crates/penumbra-ckks/src/bounds.rs`, `docs/BACKENDS.md`, and `docs/COMPARISON.md` so that the CKKS calibration protocol consistently references decision **D7** ("CKKS error bounds and metrics") instead of **D10** ("Peak server memory").
2. **Layering & Dependency Hygiene:** Remove the redundant direct `tfhe` crate dependency from `crates/penumbra-bench/Cargo.toml`, relying on the workspace's Layer-1 `penumbra-tfhe` crate and honoring `AGENTS.md` §7 and `CONTRIBUTING.md`.
3. **Smell Elimination & Backend Construction Centralization:** Introduce `ckks_backend_for_model(model_key)` in `crates/penumbra-bench/src/lib.rs` and `ModelFixture::ckks_backend(&self)` in `crates/penumbra-bench/src/models.rs`, replacing the five duplicate `if fixture.key == "phase8_branch"` match blocks across `report.rs`, `paper.rs`, `inference.rs`, and `backend_parity.rs`.

**Tech Stack:** Rust (`cargo`), `penumbra-core`, `penumbra-tfhe`, `penumbra-ckks`, `penumbra-bench`, Python 3.12 (`uv`, `pytest`), MkDocs Material (`mkdocs`).

**Spec:** Code review findings against `ROADMAP.md` Phase 15 (lines 825–899), `docs/PAPER.md` decisions D3, D6, D7, D10, D16, D19 (lines 184–232), and repository standards in `AGENTS.md` (§1.1, §1.2, §6, §7).

---

## Global Constraints

- **Golden invariant is sacred:** TFHE must remain bit-for-bit exact; CKKS must stay strictly within declared error bounds (`AGENTS.md` §1.1).
- **Core narrow-waist discipline:** No IR schema changes, no new scheme-specific branches in `penumbra-core`, and no backend leaks (`AGENTS.md` §1.2).
- **Fail loudly, early:** Load-time and compile-time validation remain strict (`AGENTS.md` §1.4).
- **Clean formatting and linter passes:** `cargo fmt` and `cargo clippy --workspace --all-targets` must be completely clean (`AGENTS.md` §6).
- **Conventional Commits:** Follow `AGENTS.md` §8 (`fix(...)`, `refactor(...)`, `docs(...)`) with no AI attribution.

## Review Focus

1. **Decision citation accuracy:** Verify that no other documentation references `D10` when describing the calibration protocol.
2. **Feature flag isolation:** Ensure `ckks_backend_for_model` is properly gated behind `#[cfg(feature = "ckks")]` so `penumbra-bench` compiles cleanly without default features.
3. **Polynomial degree preservation:** Confirm `phase8_branch` still receives `max_poly_degree = 3` while all other models receive standard default parameters.
4. **Compile-time dependency verification:** Verify `penumbra-bench` compiles and links without errors when `tfhe` is removed from its direct `Cargo.toml` dependencies.
5. **Full CI verification:** Ensure `cargo test --workspace --release`, `mkdocs build --strict`, and pytest pass without regression.

---

### Task 1: Fix Decision Citation for CKKS Calibration Protocol (D10 -> D7)

**Files:**
- Modify: `crates/penumbra-ckks/src/bounds.rs:5-7`
- Modify: `docs/BACKENDS.md:135`
- Modify: `docs/COMPARISON.md:63`
- Test: Documentation search and `mkdocs build --strict`

**Interfaces:**
- Consumes: `docs/PAPER.md:196-204` (Decision D7 definition).
- Produces: Corrected D7 citations across doc comments and documentation files.

- [ ] **Step 1: Verify incorrect references before edit**

Run:
```bash
grep -n "D10.*calibration\|calibration.*D10" crates/penumbra-ckks/src/bounds.rs docs/BACKENDS.md docs/COMPARISON.md
```
Expected: 3 matches referencing `D10`.

- [ ] **Step 2: Update `crates/penumbra-ckks/src/bounds.rs`**

In `crates/penumbra-ckks/src/bounds.rs`, update lines 5–7:
```rust
//! Calibration protocol (Phase 15, D7): derived as exactly 2.0 * p99 error over
//! the calibration split (never the test split), committed before test evaluation.
//! Source artifact: `docs/results/phase15-ckks-calibration.json`.
```

- [ ] **Step 3: Update `docs/BACKENDS.md`**

In `docs/BACKENDS.md`, line 135, change `(D10)` to `(D7)`:
```markdown
3. **Calibration chronology (D7):** All twelve supported model error bounds in `crates/penumbra-ckks/src/bounds.rs` are derived as exactly $2.0 \times \text{p99}$ over the training calibration split and committed to git *before* evaluating test rows.
```

- [ ] **Step 4: Update `docs/COMPARISON.md`**

In `docs/COMPARISON.md`, line 63, change `(D10)` to `(D7)`:
```markdown
4. **Calibration chronology (D7):** CKKS error bounds are derived over the training calibration split ($2.0 \times \text{p99}$),
   committed to `crates/penumbra-ckks/src/bounds.rs` and `docs/results/phase15-ckks-calibration.json` *before* evaluating test rows.
```

- [ ] **Step 5: Verify no remaining improper D10 citations and check docs build**

Run:
```bash
grep -n "D10.*calibration\|calibration.*D10" crates/penumbra-ckks/src/bounds.rs docs/BACKENDS.md docs/COMPARISON.md
uv run mkdocs build --strict
```
Expected: 0 grep matches and `INFO - Documentation built`.

- [ ] **Step 6: Commit**

```bash
git add crates/penumbra-ckks/src/bounds.rs docs/BACKENDS.md docs/COMPARISON.md
git commit -m "docs(ckks): correct calibration protocol decision citation from D10 to D7"
```

---

### Task 2: Remove Unused Direct `tfhe` Dependency in `penumbra-bench`

**Files:**
- Modify: `crates/penumbra-bench/Cargo.toml:22-25`
- Test: `cargo check -p penumbra-bench` and `cargo test -p penumbra-bench`

**Interfaces:**
- Consumes: `penumbra-tfhe = { workspace = true }`.
- Produces: Cleaner `Cargo.toml` decoupled from direct `tfhe` crate dependency.

- [ ] **Step 1: Check existing `crates/penumbra-bench/Cargo.toml` dependencies**

Read `crates/penumbra-bench/Cargo.toml:20-25`:
```toml
libc = "0.2"
tempfile = "3"
criterion = { version = "0.7", default-features = false, features = ["cargo_bench_support"] }
tfhe = { version = "1.6", features = ["shortint", "integer", "pbs-stats"] }
poulpy-core = { version = "=0.8.3", optional = true }
```

- [ ] **Step 2: Remove the `tfhe` line from `crates/penumbra-bench/Cargo.toml`**

Remove line 23 (`tfhe = { version = "1.6", features = ["shortint", "integer", "pbs-stats"] }`).
The section becomes:
```toml
libc = "0.2"
tempfile = "3"
criterion = { version = "0.7", default-features = false, features = ["cargo_bench_support"] }
poulpy-core = { version = "=0.8.3", optional = true }
```

- [ ] **Step 3: Verify compilation and tests pass**

Run:
```bash
cargo check -p penumbra-bench
cargo test -p penumbra-bench --test backend_parity
```
Expected: PASS with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add crates/penumbra-bench/Cargo.toml
git commit -m "chore(bench): remove unused direct tfhe crate dependency"
```

---

### Task 3: Centralize Model-Specific CKKS Backend Construction

**Files:**
- Modify: `crates/penumbra-bench/src/lib.rs:30-40`
- Modify: `crates/penumbra-bench/src/models.rs:90-120`
- Modify: `crates/penumbra-bench/src/bin/report.rs:470-530,650-665`
- Modify: `crates/penumbra-bench/src/paper.rs:68-76`
- Modify: `crates/penumbra-bench/benches/inference.rs:48-60`
- Modify: `crates/penumbra-bench/tests/backend_parity.rs:40-55`
- Test: `cargo test -p penumbra-bench` and `cargo test -p penumbra-bench --features ckks`

**Interfaces:**
- Consumes: `model_key: &str` or `fixture: &ModelFixture`.
- Produces: `penumbra_bench::ckks_backend_for_model(model_key: &str) -> penumbra_ckks::CkksBackend` and `fixture.ckks_backend() -> penumbra_ckks::CkksBackend`.

- [ ] **Step 1: Add `ckks_backend_for_model` in `crates/penumbra-bench/src/lib.rs`**

In `crates/penumbra-bench/src/lib.rs`, immediately after `ckks_backend()`, add:
```rust
/// Return a configured instance of the CKKS backend tailored for the given model fixture key.
///
/// For `phase8_branch`, overrides `max_poly_degree` to 3 to satisfy depth requirements
/// while maintaining standard default parameters for all other models.
#[cfg(feature = "ckks")]
pub fn ckks_backend_for_model(model_key: &str) -> penumbra_ckks::CkksBackend {
    if model_key == "phase8_branch" {
        let params = penumbra_ckks::params::DEFAULT_PARAMS
            .with_max_poly_degree(3)
            .expect("phase8_branch requires valid degree-3 polynomial parameters");
        penumbra_ckks::CkksBackend::new(params)
    } else {
        ckks_backend()
    }
}
```

- [ ] **Step 2: Add `ModelFixture::ckks_backend` in `crates/penumbra-bench/src/models.rs`**

In `crates/penumbra-bench/src/models.rs`, inside `impl ModelFixture`:
```rust
impl ModelFixture {
    /// Return a configured instance of the CKKS backend tailored for this model.
    #[cfg(feature = "ckks")]
    pub fn ckks_backend(&self) -> penumbra_ckks::CkksBackend {
        crate::ckks_backend_for_model(self.key)
    }
}
```

- [ ] **Step 3: Refactor duplicate sites in `crates/penumbra-bench/src/bin/report.rs`**

Replace:
1. Lines 470–477:
```rust
                "ckks" => {
                    let be = model.fixture.ckks_backend();
                    run_prepare_worker(&be, &model, &config)?;
                }
```
2. Lines 495–502:
```rust
                "ckks" => {
                    let be = model.fixture.ckks_backend();
                    run_server_rss_worker(be, &model, &config)?;
                }
```
3. Lines 520–527:
```rust
                "ckks" => {
                    let be = model.fixture.ckks_backend();
                    run_metrics_worker(be, &model, &config)?
                }
```
4. Lines 654–661:
```rust
                        "ckks" => {
                            let be = loaded.fixture.ckks_backend();
                            let run = run_paper_model(be, &loaded, &paper_config)?;
                            runs.push(run);
                        }
```

- [ ] **Step 4: Refactor duplicate site in `crates/penumbra-bench/src/paper.rs`**

In `crates/penumbra-bench/src/paper.rs:68-76`, replace:
```rust
    let backend = if model.fixture.key == "phase8_branch" {
        let p = penumbra_ckks::params::DEFAULT_PARAMS
            .with_max_poly_degree(3)
            .map_err(|e| format!("invalid max_poly_degree: {e}"))?;
        penumbra_ckks::CkksBackend::new(p)
    } else {
        penumbra_bench::ckks_backend()
    };
```
with:
```rust
    let backend = model.fixture.ckks_backend();
```

- [ ] **Step 5: Refactor duplicate site in `crates/penumbra-bench/benches/inference.rs`**

In `crates/penumbra-bench/benches/inference.rs:49-59`, replace:
```rust
                    let backend = if fixture.key == "phase8_branch" {
                        let p = penumbra_ckks::params::DEFAULT_PARAMS
                            .with_max_poly_degree(3)
                            .unwrap();
                        penumbra_ckks::CkksBackend::new(p)
                    } else {
                        penumbra_bench::ckks_backend()
                    };
```
with:
```rust
                    let backend = fixture.ckks_backend();
```

- [ ] **Step 6: Refactor duplicate site in `crates/penumbra-bench/tests/backend_parity.rs`**

In `crates/penumbra-bench/tests/backend_parity.rs:43-52`, replace:
```rust
                let branch_backend;
                let b = if fixture.key == "phase8_branch" {
                    let p = penumbra_ckks::params::DEFAULT_PARAMS
                        .with_max_poly_degree(3)
                        .unwrap();
                    branch_backend = penumbra_ckks::CkksBackend::new(p);
                    &branch_backend
                } else {
                    &ckks
                };
```
with:
```rust
                let branch_backend;
                let b = if fixture.key == "phase8_branch" {
                    branch_backend = fixture.ckks_backend();
                    &branch_backend
                } else {
                    &ckks
                };
```

- [ ] **Step 7: Verify compilation, tests, and formatting**

Run:
```bash
cargo check -p penumbra-bench
cargo test -p penumbra-bench --test backend_parity
cargo fmt --check
cargo clippy -p penumbra-bench --all-targets -- -D warnings
```
Expected: PASS with 0 warnings/errors.

- [ ] **Step 8: Commit**

```bash
git add crates/penumbra-bench/src/lib.rs crates/penumbra-bench/src/models.rs crates/penumbra-bench/src/bin/report.rs crates/penumbra-bench/src/paper.rs crates/penumbra-bench/benches/inference.rs crates/penumbra-bench/tests/backend_parity.rs
git commit -m "refactor(bench): centralize model-specific CKKS backend construction"
```

---

### Task 4: Full Workspace Verification

**Files:**
- None (verification and quality gate task)

**Interfaces:**
- Consumes: All updated files.
- Produces: Green CI checks across Rust workspace, Python tests, and docs.

- [ ] **Step 1: Run Rust workspace formatting and clippy**

Run:
```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```
Expected: Clean output with 0 warnings.

- [ ] **Step 2: Run Rust workspace unit and integration tests**

Run:
```bash
cargo test --workspace --release
```
Expected: All tests pass.

- [ ] **Step 3: Run Python test suite**

Run:
```bash
uv run pytest
```
Expected: All Python tests pass.

- [ ] **Step 4: Run documentation build**

Run:
```bash
uv run mkdocs build --strict
```
Expected: Build succeeds with 0 warnings.
