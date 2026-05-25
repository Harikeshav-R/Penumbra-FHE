# AGENTS.md

> **Read this entire document before making any change to this repository.** If you are an AI coding agent (Claude Code, Cursor, Aider, GitHub Copilot Workspace, etc.) you are bound by everything in this file. Following these rules is not optional. Compliance is checked in code review.

This document is exhaustive on purpose. The project implements cryptography. The cost of a wrong shortcut here is not "a bug a user might hit." It is "a silent failure of the privacy guarantee that is the entire point of the project." We assume agents are capable, careful, and willing to ask questions; we are still going to write the rules down.

---

## Part 0 — Identity and scope

| | |
|---|---|
| **Project** | Penumbra — a homomorphic inference engine for privacy-preserving ML |
| **Owner** | [@Harikeshav-R](https://github.com/Harikeshav-R) |
| **Languages** | Rust (core, ~70%), Python (ingestion + API, ~30%) |
| **Build tools** | `cargo`, `maturin` |
| **Python version** | 3.12+ |
| **Rust channel** | stable |
| **License** | Apache-2.0 |
| **Status** | Pre-alpha |

---

## Part 1 — Required reading

Before your first edit to this repository, you **must** read these files in this order. Do not skim. The information in them is not duplicated in this document.

1. **`README.md`** — context for what this project is.
2. **`PHILOSOPHY.md`** — the values that resolve disputes.
3. **`ARCHITECTURE.md`** — component boundaries, type signatures, invariants. The canonical reference for "what goes where."
4. **`ROADMAP.md`** — milestone-by-milestone deliverables and definitions of done.
5. **`SECURITY.md`** — the threat model and disclosure policy.
6. **`CONTRIBUTING.md`** — dev setup, commit conventions, PR process.
7. **This file (`AGENTS.md`).**

After your initial read, you do not need to re-read these on every change. You do need to re-read the specific section that governs the area you are editing. If your task touches the analyzer, re-read `ARCHITECTURE.md` §3.2 before you begin.

---

## Part 2 — The hard rules

Violations of any of these are blocking. Do not merge a PR that violates one of these, even if a human reviewer approved it.

### 2.1 Crypto safety

1. **Do not modify polynomial coefficients in `crates/penumbra-compiler/src/poly/coefficients.rs`** without re-deriving them from source. Each coefficient block includes a derivation reference; updates must update the reference, the maximum error metadata, and the corresponding test fixtures **in the same commit**.
2. **Do not bypass `penumbra-analyzer`.** If you are adding a new op that performs encrypted multiplication, you must update the depth-cost table in the same PR and verify with a benchmark that the table value is correct.
3. **Do not introduce `unwrap()` or `panic!()` in any code path that touches a `Ciphertext`, `PublicKey`, or `PrivateKey`.** Use `Result` types. CI enforces this with `clippy::unwrap_used` and `clippy::expect_used` set to deny for the relevant crates.
4. **Do not read from `thread_rng()` or `OsRng` directly in library code.** Cryptographic randomness is drawn only inside `penumbra_runtime::keygen`, and only through the `SecurityParams::rng_seed` parameter. Determinism is mandatory.
5. **Do not log, print, or expose ciphertext contents, key material, or noise estimates** in user-facing output. Debug logging that exposes these is gated behind a `debug-internals` cargo feature that is *not* enabled by default and *cannot* be enabled on a published wheel.
6. **Do not add a new dependency on a cryptographic library other than TFHE-rs** without an issue first. The architecture is single-backend by design; adding a second backend is a v0.4 conversation.
7. **Do not silently downgrade security parameters.** If a user passes parameters that would compromise security, fail loudly with a `PenumbraSecurityError`. Do not "round down to the nearest safe value."

### 2.2 Architectural boundaries

8. **`penumbra-ir` does not depend on FHE libraries.** It is pure data. If you add a `tfhe` dependency to `penumbra-ir/Cargo.toml`, the PR is rejected. Period.
9. **`penumbra-runtime` does not depend on `penumbra-ir`.** The runtime executes lowered operations; it does not know what graph they came from. If you find yourself wanting this dependency, the thing you want belongs in `penumbra-compiler`.
10. **Only `penumbra-py` depends on `pyo3`.** No other crate imports Python types or Python-specific machinery.
11. **The IR is the contract.** Cross-component communication goes through `penumbra_ir::Graph` and its serialized forms. There are no back-channel passes of TFHE-rs types between non-adjacent components.
12. **Public API additions to `python/penumbra_fhe/__init__.py` require a docstring with `:param:` / `:returns:` annotations, a corresponding `.pyi` stub, and a test.** Three artifacts. Same commit. No exceptions.

### 2.3 Determinism

13. **Same input → same output, byte-for-byte.** If you introduce a source of non-determinism (hashmap iteration order, parallel reduction, time-based seeding), it is a bug. Fix it before merging.
14. **Test fixtures must be checked in, not generated at test time** (with rare, documented exceptions). If you find yourself writing `let weights = random_matrix(...)` in a test that asserts a specific output, you are doing it wrong.

### 2.4 Documentation

15. **Every new public function must have rustdoc (Rust) or a docstring (Python).** Docs include: purpose, parameters, return value, errors, at least one example. CI fails on missing docs (`#![warn(missing_docs)]`).
16. **A user-visible behavior change requires a `CHANGELOG.md` entry in the same PR.** Format: see `CHANGELOG.md`.
17. **A new public API requires a tutorial or example update.** Either `docs/tutorials/` or `examples/`.

### 2.5 Process

18. **Branch names follow the convention.** See Part 6.
19. **Commit messages follow Conventional Commits.** See Part 6.
20. **PRs require all CI green and at least one human reviewer's approval before merge.** Agents do not self-approve.

---

## Part 3 — Component scope rules

Before making a change, identify which component you are touching. The component governs what you may and may not do.

### 3.1 `penumbra-ir`

| | |
|---|---|
| Lives in | `crates/penumbra-ir/` |
| Imports allowed | `serde`, `serde_json`, `thiserror`, `bitflags`, the Rust standard library |
| Imports forbidden | Anything related to FHE; anything related to Python; the other workspace crates |
| Tests live in | `crates/penumbra-ir/tests/`, `crates/penumbra-ir/src/**/tests.rs` |

**Common tasks:**
- Adding a new op variant: update `Op` enum, add shape inference, add a test, update `docs/architecture/op_set.rst`.
- Changing serialization: must remain backward-compatible within a minor version (provide a migration if not).

**Forbidden in this crate:**
- Anything that performs an actual computation. The IR describes; it does not execute.
- Importing `tfhe`, `numpy`, `onnx`, or `pyo3`.

### 3.2 `penumbra-analyzer`

| | |
|---|---|
| Lives in | `crates/penumbra-analyzer/` |
| Imports allowed | `penumbra-ir`, `thiserror`, the Rust standard library |
| Imports forbidden | `tfhe`, `pyo3`, `penumbra-runtime`, `penumbra-compiler` |
| Tests live in | `crates/penumbra-analyzer/tests/` |

**Common tasks:**
- Adding a new bootstrapping placement policy: extend the `PlacementPolicy` enum.
- Updating depth costs: requires a corresponding benchmark verification in `benchmarks/depth_costs/`. Do not update the table without empirical evidence.

**Forbidden in this crate:**
- Actually inserting `Bootstrap` nodes in the IR returned by a placement that has not been validated. Use `PlacementPolicy::Manual { positions }` to test placements first.

### 3.3 `penumbra-compiler`

| | |
|---|---|
| Lives in | `crates/penumbra-compiler/` |
| Imports allowed | `penumbra-ir`, `penumbra-runtime`, `thiserror`, the Rust standard library |
| Imports forbidden | `pyo3`, anything Python |
| Tests live in | `crates/penumbra-compiler/tests/` |

**Common tasks:**
- Adding a lowering rule for a new IR op: implement the rule, add a unit test on an isolated single-op graph, update the lowering table in `docs/architecture/lowering.rst`.
- Adjusting polynomial activation defaults: see crypto-safety rule 1.

**Forbidden in this crate:**
- Performing FHE operations directly. Lowering produces a description; execution belongs to the runtime.
- Hardcoding polynomial coefficients in lowering rules. Always reference `penumbra_compiler::poly::coefficients`.

### 3.4 `penumbra-runtime`

| | |
|---|---|
| Lives in | `crates/penumbra-runtime/` |
| Imports allowed | `tfhe`, `thiserror`, the Rust standard library |
| Imports forbidden | `penumbra-ir`, `pyo3`, anything not directly involved in FHE primitives |
| Tests live in | `crates/penumbra-runtime/tests/` |

**Common tasks:**
- Adding a new encrypted operation: see Part 4 (Definition of Done — Encrypted Primitive).

**Forbidden in this crate:**
- Importing the IR. The runtime executes lowered ops; it does not know about graph structure.
- Implicit RNG. All randomness flows through explicit parameters.

### 3.5 `penumbra-py`

| | |
|---|---|
| Lives in | `crates/penumbra-py/` |
| Imports allowed | `pyo3`, all workspace crates, `thiserror` |
| Tests live in | `python/tests/` (integration tests in Python) |

**Common tasks:**
- Exposing a new Rust function to Python: add a `#[pyfunction]`, register it in the module, add a Python smoke test, add a `.pyi` type stub.

**Forbidden in this crate:**
- Business logic. This crate is translation only. If you find yourself implementing an algorithm here, move it to one of the other crates.
- Catching Python exceptions and translating them silently. Errors flow Rust → Python; Python errors during a Python → Rust call should propagate as `PyErr`, not be swallowed.

### 3.6 `python/penumbra_fhe/`

| | |
|---|---|
| Lives in | `python/penumbra_fhe/` |
| Imports allowed | `onnx`, `numpy`, the Python standard library, the Rust extension (`penumbra_fhe._bindings`) |
| Tests live in | `tests/python/` |

**Common tasks:**
- ONNX op handling: extend `ingestion.py`, add a fixture in `tests/python/fixtures/`, add a golden test.
- API polish: improve docstrings, type stubs, error messages.

**Forbidden in this crate:**
- Calling TFHE-rs directly. All FHE operations go through the Rust extension.
- Heavy computation. The Python layer is ingestion + API ergonomics. Numerical work happens in Rust.

---

## Part 4 — Definition of done

A change is "done" only when **all** of the following are true. Do not declare work complete based on a passing test alone.

### 4.1 General DoD (every change)

- [ ] All CI checks pass on Ubuntu, macOS, and Windows x86_64.
- [ ] `cargo fmt --all -- --check` passes.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes.
- [ ] `cargo test --workspace` passes.
- [ ] `ruff check .` passes.
- [ ] `pyrefly check python/` passes.
- [ ] `pytest tests/` passes.
- [ ] At least one human reviewer has approved.

### 4.2 New feature DoD

In addition to general DoD:

- [ ] User-facing changes documented in `CHANGELOG.md` under `## [Unreleased]`.
- [ ] If new public API: docstring, type annotation, `.pyi` stub if Python, rustdoc example if Rust.
- [ ] If new public API: a tutorial entry or an example.
- [ ] Test coverage for the new code path. Minimum: one happy-path test and one error-path test.
- [ ] Performance impact assessed if relevant.

### 4.3 Encrypted primitive DoD (highest bar)

In addition to new feature DoD:

- [ ] Correctness test: `decrypt(op(encrypt(x), encrypt(y))) == op(x, y)` for at least 100 randomized inputs using `proptest`.
- [ ] Depth-cost entry added to `crates/penumbra-analyzer/src/depth_costs.rs` with empirical justification.
- [ ] A benchmark in `benchmarks/runtime/` measuring the operation's latency.
- [ ] Reviewer count: **two** (not one). The second reviewer should ideally be from outside the immediate working context.
- [ ] No `unwrap()`, `panic!`, `expect()` in the implementation path.
- [ ] Explicit RNG handling. No implicit `thread_rng()`.

### 4.4 Bug fix DoD

In addition to general DoD:

- [ ] A test that **fails on the buggy code and passes on the fixed code** is included in the PR.
- [ ] `CHANGELOG.md` entry under `### Fixed`.
- [ ] If the bug had security implications: a `SECURITY.md` cross-reference (or a security advisory if disclosure has been coordinated).

### 4.5 Documentation-only DoD

- [ ] Docs build (`make -C docs html`) succeeds.
- [ ] No broken links (`make -C docs linkcheck`).
- [ ] Examples in docstrings still run (doctest passes).

---

## Part 5 — Forbidden operations

In addition to the hard rules, the following operations are **forbidden** unless you have an explicit instruction to perform them from a human maintainer with write access to the repo:

1. **`git push --force` to `main`.** Even with permission, prefer `--force-with-lease` and only on your own feature branches.
2. **Squashing commits in a PR before review.** Reviewers benefit from your incremental commits. Squash after approval, not before.
3. **Deleting branches that have open PRs.**
4. **Bumping the version in `Cargo.toml` or `pyproject.toml`.** Versions are managed via the release workflow (`.github/workflows/release.yml`).
5. **Publishing to PyPI or crates.io.** Releases go through the workflow.
6. **Disabling CI checks** (`# clippy::allow`, `# noqa`, `# type: ignore`) without a comment explaining why the rule does not apply and a tracking issue if the suppression is temporary.
7. **Reformatting unrelated code.** Style-only changes go in their own commits, separate from logic changes.
8. **Adding TODO comments without a corresponding GitHub issue** (the format is `// TODO(#issue): description`).
9. **Editing `LICENSE`, `NOTICE`, or `CODE_OF_CONDUCT.md`.** Changes to these require explicit maintainer action.
10. **Adding files outside the documented directory structure** (see `ROADMAP.md` § Repository file manifest). New top-level directories require an architecture discussion.

---

## Part 6 — Commit messages, branches, PRs

### 6.1 Conventional Commits

All commit messages follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

**Allowed types:**

| Type | Meaning |
|---|---|
| `feat` | New feature (user-visible) |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting / whitespace; no logic change |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or modifying tests |
| `build` | Build system or dependency changes |
| `ci` | CI configuration |
| `chore` | Miscellaneous; use sparingly |
| `roadmap` | Updates to `ROADMAP.md` |
| `arch` | Updates to `ARCHITECTURE.md` |
| `crypto` | Changes to cryptographic primitives or polynomial coefficients (requires two reviewers) |

**Allowed scopes:** `ir`, `analyzer`, `compiler`, `runtime`, `py`, `python`, `docs`, `bench`, `examples`, `ci`, or omitted.

**Examples:**

```
feat(analyzer): greedy bootstrapping placement policy

Adds Greedy { threshold } variant to PlacementPolicy. Inserts a
Bootstrap node when remaining budget falls below the threshold.

Closes #42.
```

```
crypto(compiler): replace degree-3 ReLU coefficients with minimax-derived

Old coefficients were Chebyshev-truncated; new ones are minimax
over [-4, 4]. Max error improves from 0.21 to 0.15.

Derivation: docs/architecture/polynomial_derivation.rst#relu-deg3-v2
```

```
fix(runtime): bootstrap was not refreshing noise budget under feature `nightly-avx512`

Closes #117.
```

### 6.2 Branch naming

Pattern: `<type>/<short-kebab-description>` matching commit types.

Examples:
- `feat/greedy-bootstrap-placement`
- `fix/avx512-bootstrap-refresh`
- `docs/clarify-quantization-scale`
- `crypto/degree-3-relu-minimax`

### 6.3 PR description template

PRs use the template in `.github/PULL_REQUEST_TEMPLATE.md`. Don't bypass it. Fill in every section.

### 6.4 PR size

Aim for PRs under **400 lines of diff** (excluding generated files, fixture data, and docs). Larger PRs are not categorically rejected but require justification in the PR description and benefit from being broken into logical commits.

---

## Part 7 — Tooling and workflow

### 7.1 Required local tooling

Before your first commit, verify these are available:

```bash
rustup show                              # rust toolchain
cargo --version                          # >=1.78
maturin --version                        # >=1.7
python --version                         # >=3.12
ruff --version
pyrefly --version
pre-commit --version
```

`pre-commit install` is mandatory. CI enforces what pre-commit catches; running it locally saves you a round trip.

### 7.2 The standard build/test loop

```bash
# Rust changes
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# Python changes
ruff check . && ruff format --check .
pyrefly check python/
maturin develop --release        # rebuild the Rust extension
pytest tests/

# Docs changes
make -C docs html
make -C docs linkcheck
```

Do this before every commit. Do not rely on CI to catch what you could have caught locally.

### 7.3 Benchmarks

Run benchmarks before any change to a hot path:

```bash
cargo bench --workspace
```

Compare against the baseline in `benchmarks/results/`. A regression of >20% on any tracked benchmark requires explicit reviewer approval to merge.

### 7.4 The development feedback loop

When working on Rust + Python together:

```bash
maturin develop --release        # 1. build the Rust extension into the active venv
pytest -xvs tests/python/test_thing.py::test_specific_case   # 2. run the one test you care about
# 3. edit Rust or Python
# 4. goto 1
```

Don't run the full test suite for every iteration. Use `pytest -x` (stop on first failure) and `cargo test -p <crate>` (test one crate) during inner loops.

---

## Part 8 — When to ask before editing

Agents should **stop and ask the maintainer** before:

1. **Modifying anything in `crates/penumbra-compiler/src/poly/coefficients.rs`.**
2. **Changing depth-cost values in `crates/penumbra-analyzer/src/depth_costs.rs`** without empirical justification ready.
3. **Adding a new dependency to any `Cargo.toml`** beyond what's needed for an in-progress task already discussed.
4. **Removing a test, even if it appears to be flaky.** Flaky tests get investigated, not deleted.
5. **Renaming public API.** Even if it would be more consistent. Stability beats consistency in a pre-1.0 project that wants to attract users.
6. **Refactoring across crate boundaries.** A change that touches three crates simultaneously almost certainly belongs in a design discussion.
7. **Changing the license, the copyright notice, or `NOTICE`.**
8. **Disabling a CI check, even temporarily.**

If unsure, ask. The cost of asking is one round-trip; the cost of a wrong autonomous decision in this codebase can be significant.

---

## Part 9 — Code style hard rules

### 9.1 Rust

- **Formatting:** `rustfmt` with the project's `rustfmt.toml`. Non-negotiable.
- **Linting:** `clippy -- -D warnings`. The `pedantic` group is enabled selectively (see `Cargo.toml`); the rules listed there are not allowed to be `#[allow]`'d at a function level without a comment explaining why.
- **Imports:** grouped (std, external crates, workspace crates, super::, self::); within each group, alphabetical.
- **Error handling:** `Result<T, E>` with `thiserror`-derived error types. Never `Box<dyn Error>`. Never `anyhow` in library code; `anyhow` is allowed in `examples/` and `benchmarks/`.
- **Comments:** explain *why*, not *what*. The code says what it does; the comment says why it does it that way.
- **Documentation:** every public item has rustdoc. CI enforces `missing_docs`.
- **Tests:** in-module `#[cfg(test)] mod tests` for unit tests; `tests/` directory for integration tests.

### 9.2 Python

- **Formatting:** `ruff format`.
- **Linting:** `ruff check` with the rules in `pyproject.toml`.
- **Type hints:** mandatory on every public function. Verified with `pyrefly`.
- **Imports:** sorted by `ruff`.
- **Docstrings:** Sphinx style (`:param:`, `:returns:`, `:raises:`). Every public function and class.
- **Error handling:** Penumbra-specific exception types from `penumbra_fhe.errors`. Never catch `Exception` broadly; catch specific types.
- **Tests:** `pytest`. One assertion per test where reasonable. Fixtures in `conftest.py`.

### 9.3 Both

- **Line length:** 100 characters (relaxed from 80 because Rust types are verbose; relaxed for Python for consistency).
- **No trailing whitespace.** Pre-commit catches this.
- **Files end with exactly one newline.**

---

## Part 10 — Common pitfalls (and how to avoid them)

A non-exhaustive list of mistakes that have either happened or that we anticipate. Read this before your first PR.

### 10.1 "It works on my machine"

Probably you ran `cargo build` but not `maturin develop --release`. The Python tests use the maturin-built extension, not the cargo-built one. They are not the same.

### 10.2 "The property test passes"

100 cases is the minimum, not the goal. For crypto-adjacent code, run `proptest` with at least 10,000 cases (locally; CI runs the 100 case version). Set `PROPTEST_CASES=10000` for high-confidence local validation.

### 10.3 "I added a TODO"

A bare TODO is a future-bug-and-also-an-irritation. If the work is needed, file an issue and reference it: `// TODO(#42): handle the case where ...`. If the work isn't needed, delete the TODO.

### 10.4 "I'll add tests later"

Tests are part of the change. They land in the same commit. "Later" means "never" in code that other people will rely on for privacy.

### 10.5 "I refactored while I was in there"

Stylistic refactors in the middle of a logic change make review harder and bisection worse. Refactor in a separate commit, ideally a separate PR.

### 10.6 "I bumped the dependency version because the patch release was out"

Dependency bumps go through Dependabot. Manual bumps require a justification in the PR description ("CVE-2026-XXXX fixed in 1.2.3"). For TFHE-rs specifically, see crypto-safety rule 6.

### 10.7 "I caught the unwrap with an if-let so it's fine"

It's not fine if the alternative path produces a default value silently. An error condition that the user can't see is worse than a panic — at least a panic produces a stack trace. Use `Result` and propagate.

### 10.8 "I added a function, the test exists, I documented the parameters"

Did you also: add a Python `.pyi` stub if it's a `pyfunction`? Add a tutorial mention? Update the `CHANGELOG.md`? Run the doctest? The bar is **not** "code + test." The bar is **complete**.

### 10.9 "The CI is wrong"

Probably it's not. If you genuinely believe CI is wrong (transient infrastructure issue), say so in a comment and re-run. Do not bypass.

### 10.10 "I asked the user and they said go ahead"

User permission to bypass a rule in this file requires an explicit acknowledgment in the PR description ("This bypasses §X because Y, approved by @username in <link>"). Implicit permission is not sufficient.

---

## Part 11 — Agent-specific guidance

### 11.1 Tool use posture

- **Read before you write.** When opening a file, read enough surrounding context to understand the local conventions, not just the lines you're about to edit.
- **Run tests after editing.** Do not declare a change complete based on type-checking or compilation alone. Run the relevant test(s).
- **Search the codebase before introducing a new abstraction.** Penumbra is small; there's a good chance the abstraction you want already exists in a different shape.
- **Prefer small, verified steps over large, untested ones.** A PR with five commits, each tested, is better than one giant commit with all the tests at the end.

### 11.2 Asking for clarification

You are encouraged to ask. Specifically, ask when:
- The acceptance criteria for a task are not clear from the issue.
- A change would affect public API.
- A change would touch the polynomial coefficients or depth-cost table.
- You see a tension between two of the rules in this document.
- You're unsure whether to add a dependency.

You are *not* expected to ask when:
- The task is a clear bug fix with an obvious test case.
- The change is documentation-only and you have read the relevant section.
- You're operating within a single component and the change is mechanical.

### 11.3 What to do when you encounter something undocumented

1. Check if it's a bug in this document. If a hard rule conflicts with a documented behavior in `ARCHITECTURE.md`, flag the conflict in the PR.
2. Check `docs/` and rustdoc. The detailed documentation lives there; this file is summary.
3. Ask. Open a discussion or comment on the issue.
4. **Do not** invent a convention and proceed silently. The cost of one round-trip with a maintainer is much lower than the cost of a divergent style or architecture.

### 11.4 Handling failure

If a tool call fails, a test breaks unexpectedly, or you cannot make progress:
- **Stop.** Do not flail. Do not try variations.
- **Diagnose.** Read the error. Read the test. Read the relevant code.
- **Surface.** If you can't resolve it in 2–3 attempts, surface it to the human user with a clear summary of what you tried.

A blocked agent that asks for help is more useful than an unblocked agent that has made things worse.

### 11.5 Working with cryptographic code

Cryptographic code receives extra scrutiny. When editing it:

- **Read the surrounding code twice before changing anything.**
- **Do not "clean up" or "modernize" idioms** in cryptographic functions. If something looks weird, it may be weird for a reason. Ask before changing.
- **Verify with property tests, not just unit tests.** A unit test that passes once is not enough.
- **Mark your PR with the `crypto` scope** in the commit message. This triggers the two-reviewer requirement.
- **Do not optimize for performance without measuring.** "Looks faster" is not a justification.

---

## Part 12 — Quick reference card

If you remember nothing else from this document:

```
                BEFORE  YOU  EDIT
┌────────────────────────────────────────────────────┐
│  1. Have you read PHILOSOPHY, ARCHITECTURE,        │
│     ROADMAP, SECURITY for this area?               │
│  2. Which component is this? What are its imports? │
│  3. Is this a crypto change? Two reviewers + tests │
│     before merge.                                  │
│  4. Have you set up pre-commit hooks?              │
└────────────────────────────────────────────────────┘

                BEFORE  YOU  COMMIT
┌────────────────────────────────────────────────────┐
│  1. cargo fmt + clippy + test passes               │
│  2. ruff + pyrefly + pytest passes                 │
│  3. Conventional commit message                    │
│  4. CHANGELOG entry if user-visible                │
│  5. Docstring/rustdoc for new public API           │
└────────────────────────────────────────────────────┘

                BEFORE  YOU  MERGE
┌────────────────────────────────────────────────────┐
│  1. CI green on all three platforms                │
│  2. At least one human reviewer approved           │
│     (two if crypto)                                │
│  3. No unaddressed review comments                 │
│  4. Branch is up-to-date with main                 │
└────────────────────────────────────────────────────┘
```

---

> **Last words.** If this document feels heavy, that's intentional. The cost of getting one of these rules wrong is high enough that being explicit is worth the friction. If you find a rule that seems to make no sense, open an issue — we'll either explain it or change it. We will not, however, accept silent violation.

> **For human maintainers:** when this document goes out of date, *update it in the same PR* that obsoletes a rule. Stale agent directives are worse than no directives.
