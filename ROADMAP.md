# Penumbra Roadmap

> **Living document.** Edit as scope shifts. Every change requires a corresponding commit message starting with `roadmap:`.

This roadmap expands the three-month milestone plan into week-by-week deliverables, each with a concrete **definition of done** (DoD) and explicit dependencies on prior work. Treat it as a contract: a milestone is not "done" until its DoD is met.

## Reading this document

| Symbol | Meaning |
|---|---|
| **DoD** | Definition of done — the milestone is complete when these conditions hold |
| **Depends on** | Hard prerequisite milestones |
| **Risk** | Identified failure mode with mitigation |
| **Stretch** | Optional; pursue only if base milestone lands early |
| **Out** | Explicitly out of scope for this milestone |

---

## Phase 0 — Repository Bootstrap (Pre-Week 1, ~3 days)

The repository scaffold and CI must exist before any technical work begins. This phase is short but blocking.

### 0.1 — Repository hygiene

**DoD:**
- [x] All documents in [the repo manifest](#repository-file-manifest) committed to `main`.
- [x] `LICENSE`, `NOTICE`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CONTRIBUTING.md`, `AGENTS.md` present.
- [x] Branch protection enabled on `main` (require PR, require CI green, require signed commits).
- [x] Issue/PR templates populated.
- [x] Repository description, topics, and homepage URL set on GitHub.

### 0.2 — Toolchain

**DoD:**
- [x] `rust-toolchain.toml` pins stable channel with `rustfmt`, `clippy`, `rust-src`.
- [x] `pyproject.toml` declares `requires-python = ">=3.12"`.
- [x] `maturin develop` succeeds locally on macOS aarch64 and Linux x86_64.
- [x] `cargo test --workspace` succeeds (with empty crates).
- [x] `pytest` succeeds (with one smoke test).

### 0.3 — CI green

**DoD:**
- [x] `.github/workflows/ci.yml` passes on Ubuntu latest, macOS latest, Windows latest (x86_64).
- [x] `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test --workspace`, `ruff check`, `pyrefly check`, `pytest` all run in CI.
- [x] `.github/workflows/docs.yml` builds Sphinx + rustdoc and deploys to GitHub Pages.

### 0.4 — Pre-commit hooks

**DoD:**
- [x] `.pre-commit-config.yaml` runs `rustfmt`, `clippy`, `ruff`, `pyrefly`, conventional-commit lint, and end-of-file fixers.
- [x] Hook installed in dev setup (documented in `CONTRIBUTING.md`).

---

## Phase 1 — Month 1: Foundation

**Phase goal:** End of Month 1 — an encrypted linear layer that produces correct outputs, verified against plaintext, with the IR walker in place. We can encrypt, multiply, add, decrypt, and prove correctness.

### Week 1 — TFHE-rs + PyO3 plumbing

This is the highest-risk week. TFHE-rs and PyO3 have a steep setup curve, especially around feature flags.

**Deliverables:**
- A minimal `penumbra-runtime` crate that depends on `tfhe-rs` with `boolean`, `shortint`, `integer` features.
- A minimal `penumbra-py` crate that uses PyO3 to expose a `version()` function.
- A maturin build pipeline that produces a wheel installable via `pip install ./dist/*.whl`.
- A Python integration test that imports `penumbra_fhe`, calls `version()`, and asserts the result.

**DoD:**
- [x] `cargo build --workspace --release` succeeds in <10 minutes on the reference machine.
- [x] `maturin build --release` produces a wheel for cpython-3.12.
- [x] CI green on all three platforms.
- [x] An issue tagged `documentation/setup` exists capturing every TFHE-rs gotcha encountered.

**Depends on:** Phase 0 complete.

**Risk — TFHE-rs feature flag incompatibility:** TFHE-rs has subtly interacting feature flags (`nightly-avx512`, `pbs-stats`, etc.). Mitigation: pin to a known-working flag combination; document it in `crates/penumbra-runtime/README.md`; do not change without benchmark.

**Risk — PyO3 ABI mismatch on macOS aarch64:** Apple Silicon wheels occasionally break with new PyO3 releases. Mitigation: pin `pyo3` exactly in `Cargo.toml`; only upgrade with intent.

### Week 2 — Encrypted scalar operations

**Deliverables:**
- `penumbra-runtime` exposes `Ciphertext` (newtype over TFHE-rs's `FheUint32` to start).
- Free functions for encrypted add, encrypted sub, encrypted scalar multiply.
- A property-based test suite (`proptest`) that verifies:
  - `decrypt(encrypt(x) + encrypt(y)) == x + y` for `x, y` in valid range.
  - `decrypt(encrypt(x) * k) == x * k` for scalar `k`.
  - Bootstrapping refreshes ciphertext (noise budget restored).
- Python bindings: `penumbra_fhe.encrypt(int) -> Ciphertext`, `penumbra_fhe.decrypt(Ciphertext) -> int`, `Ciphertext.__add__`, `Ciphertext.__mul__`.

**DoD:**
- [ ] Property tests pass with 1000 cases each, no failures.
- [ ] Documented round-trip example in `docs/tutorials/scalars.rst`.
- [ ] No clippy warnings, no `unwrap()` in non-test code (use `?` or explicit panic with message).

**Depends on:** Week 1.

**Out:** Vectorized operations. Matrix multiplication. ONNX. Polynomial activations.

### Week 3 — Encrypted dense layer

**Deliverables:**
- `penumbra-runtime` exposes `EncryptedTensor` (a `Vec<Ciphertext>` with shape metadata).
- A `dense_layer(input: &EncryptedTensor, weights: &Matrix<i64>, bias: &Vec<i64>) -> EncryptedTensor` function.
- Quantization helpers: float weights → fixed-point i64 with documented scale factor.
- Correctness test: for a randomly generated `(W, b, x)`, compare encrypted `Wx + b` to plaintext `Wx + b` within tolerance.

**DoD:**
- [ ] For 100 random (16×16) weight matrices and 100 random input vectors, encrypted result matches plaintext within ±2 LSB of the quantization scale.
- [ ] Latency profile committed to `benchmarks/week3.json` (will be wrong, that's fine — establishes the format).
- [ ] Module-level rustdoc on `dense_layer` explains the noise budget impact and quantization assumptions.

**Depends on:** Week 2.

**Risk — Quantization error compounding:** Fixed-point quantization introduces error per multiply. Mitigation: pick scale conservatively (`2^16` to start); document the chosen scale; add a stretch goal to characterize error vs scale.

### Week 4 — ONNX walker and IR sketch

**Deliverables:**
- `python/penumbra_fhe/ingestion.py` parses an ONNX file, walks the graph, prints `(op_type, input_shape, output_shape)` per node.
- `crates/penumbra-ir/src/lib.rs` defines `Op`, `Tensor`, `Graph` types covering: `MatMul`, `Add`, `Gemm`, `Relu`, `Conv` (placeholder), `MaxPool` (placeholder).
- A Python → Rust IR bridge: Python parses ONNX, builds a JSON intermediate, Rust deserializes into `Graph`.
- One golden-file test: `tests/python/test_ingestion.py` parses a fixture MLP and asserts the IR matches a checked-in snapshot.

**DoD:**
- [ ] All ONNX op types appearing in the MNIST MLP fixture are classified (linear/non-linear/structural).
- [ ] IR survives JSON round-trip without data loss.
- [ ] `pyrefly check` is clean on `ingestion.py`.

**Depends on:** Week 3.

**Month 1 retrospective (end of Week 4):**
- [ ] Open a tracking issue: "Month 1 retrospective".
- [ ] Document: what surprised us, what took longer than expected, what scope to cut from Month 2.
- [ ] Tag the commit as `v0.0.1-foundation`. No PyPI release yet.

---

## Phase 2 — Month 2: Graph Lowering & Activation Approximation

**Phase goal:** End of Month 2 — end-to-end encrypted MLP inference on MNIST, correct within 1% of plaintext accuracy.

### Week 5 — Depth budget analyzer

**Deliverables:**
- `crates/penumbra-analyzer` implements:
  - Per-op depth cost (with a published table in `docs/architecture/depth_costs.rst`).
  - Whole-graph depth analysis: traverse the IR, accumulate cost, flag overruns.
  - Greedy bootstrapping placement: insert `Bootstrap` nodes when remaining budget falls below threshold.
- Visualization: a `print_depth_profile(graph)` function that emits an ASCII diagram.

**DoD:**
- [ ] For a 3-layer MLP with degree-3 polynomial activations, analyzer correctly predicts depth overflow without bootstrapping and zero overflow with bootstrapping.
- [ ] Unit tests for greedy placement on synthetic graphs (handcrafted).
- [ ] `print_depth_profile` output checked into a tutorial as a visual reference.

**Depends on:** Week 4.

**Risk — Greedy is wrong:** Greedy placement may insert too many bootstraps. Mitigation: characterize greedy vs optimal (handcrafted DP for small graphs); accept ≤2x overhead for v0.1; file an issue for DP-based placement as Month 3 stretch.

### Week 6 — Polynomial activations

**Deliverables:**
- `crates/penumbra-compiler` exposes a `PolyActivation` enum with variants `Degree3`, `Degree5`, `Degree7`.
- Three families of approximations:
  - **ReLU minimax** — derived numerically over a target input range.
  - **Sigmoid Chebyshev** — for the bounded range `[-8, 8]`.
  - **Tanh Chebyshev** — for the bounded range `[-4, 4]`.
- Coefficients checked into source as `const` arrays. **Coefficients must not be derived at runtime** (reproducibility, audit trail).
- Plaintext correctness tests: max error over the target range vs the true function.

**DoD:**
- [ ] Degree-3 ReLU has max error <0.15 over `[-4, 4]`.
- [ ] Degree-5 ReLU has max error <0.05 over `[-4, 4]`.
- [ ] Degree-7 ReLU has max error <0.02 over `[-4, 4]`.
- [ ] Approximation coefficients have a reference (paper, derivation script) checked into `docs/architecture/polynomial_derivation.rst`.

**Depends on:** Week 3 (encrypted multiplication).

**Risk — Out-of-range inputs:** If activations receive inputs outside the approximation range, error explodes. Mitigation: emit a warning during compilation when network statistics suggest range overrun; document range assumptions in user-facing API.

### Week 7 — FHE lowering pass

**Deliverables:**
- `crates/penumbra-compiler` exposes `lower(graph: Graph) -> CompiledGraph`.
- Each IR op has a corresponding lowering rule producing TFHE-rs operations.
- `CompiledGraph::run(inputs: &EncryptedTensor, pk: &PublicKey) -> EncryptedTensor` executes the lowered graph.
- BatchNorm folding: scale and shift folded into the preceding `Gemm`/`Conv` op during lowering.

**DoD:**
- [ ] Lowering is total over the MLP op set defined in Week 4.
- [ ] Lowering is deterministic (same input → same `CompiledGraph` bytes).
- [ ] Unit tests for each lowering rule on isolated single-op graphs.

**Depends on:** Weeks 5, 6.

**Out:** Conv2d lowering (Week 9). Attention. Quantization-aware training.

### Week 8 — End-to-end MLP on MNIST

**Deliverables:**
- A reference 3-layer MLP trained on MNIST checked into `examples/mnist_mlp.py`.
- A `examples/mnist_inference.py` that loads the model, exports to ONNX, compiles, encrypts a test image, runs inference, decrypts, asserts correctness.
- A user-facing API in `python/penumbra_fhe/__init__.py` that exposes `compile()`, `keygen()`, `encrypt()`, `decrypt()`.

**DoD:**
- [ ] Encrypted MNIST accuracy is within **1 percentage point** of plaintext accuracy (target: plaintext ~97%, encrypted ≥96%).
- [ ] One encrypted inference completes in <300 seconds on the reference machine.
- [ ] The full example runs from a clean checkout in <5 minutes excluding the inference time.
- [ ] Tag commit as `v0.1.0-mlp`.

**Depends on:** Week 7.

**Month 2 retrospective (end of Week 8):**
- [ ] Open a tracking issue: "Month 2 retrospective".
- [ ] Publish blog draft (private repo, not yet public).
- [ ] Decide: shallow CNN in Month 3, or polish MLP further? Default: shallow CNN.

---

## Phase 3 — Month 3: Polish, Benchmarks, Launch

**Phase goal:** End of Month 3 — installable library with honest benchmarks and a write-up ready to publish.

### Week 9 — Shallow CNN support

**Deliverables:**
- `Conv2d` lowering rule in `penumbra-compiler` (small kernels, stride 1, padding 'same' or 'valid').
- `MaxPool2d` replaced with `AvgPool2d` during ingestion (warning emitted) — max requires comparison, which FHE cannot do.
- A CNN example: small ConvNet (2 conv layers + 2 dense) on MNIST, end-to-end.

**DoD:**
- [ ] CNN MNIST accuracy within 1% of plaintext.
- [ ] Conv2d unit tests covering: 1×1, 3×3, 5×5 kernels; stride 1; padding 'valid' and 'same'.
- [ ] Documented in `docs/tutorials/cnn.rst`.

**Depends on:** Week 8.

**Stretch:** Strided convolution. `AvgPool` as a first-class op (currently emerges from conv folding).

### Week 10 — Benchmarks

**Deliverables:**
- A benchmarks harness in `benchmarks/` using `criterion.rs` for Rust microbenchmarks and `pytest-benchmark` for end-to-end Python timings.
- Per-op latency table: encrypted add, encrypted multiply, scalar multiply, dense layer (various sizes), polynomial activation (degree 3/5/7), bootstrapping.
- End-to-end inference latency on:
  - 3-layer MLP, MNIST
  - 2-conv + 2-dense ConvNet, MNIST
- **Depth budget breakdown:** for each model, time-per-layer + percentage spent in bootstrapping.

**DoD:**
- [ ] All benchmarks reproducible from a clean checkout with one command.
- [ ] Results checked into `benchmarks/results/`, with reference machine documented.
- [ ] A `benchmarks/README.md` explaining what each number means, how to interpret it, and how to reproduce.

**Reference machine for v0.1 benchmarks:**
- MacBook Pro 14" (2023)
- Apple M3 Pro (11-core CPU, 14-core GPU; GPU unused)
- 18 GB RAM
- macOS (record version at benchmark time)
- All benchmarks run with the machine plugged in, lid open, in low-power mode disabled, with no other workloads.

**Depends on:** Week 9.

### Week 11 — Documentation

**Deliverables:**
- Sphinx site built from `docs/`, deployed to GitHub Pages.
- API reference auto-generated via `sphinx-autodoc` for Python and `rustdoc` for Rust crates (linked from Sphinx).
- Tutorials:
  - Scalars: encrypt, decrypt, arithmetic.
  - MLP: train, export, compile, infer.
  - CNN: same flow.
  - Benchmarks: how to read and reproduce.
  - Architecture: a guided tour of the four components.
- Migration notes: anticipated breaking changes between v0.1 and v0.2.

**DoD:**
- [ ] Every public Python function and class has a docstring with type hints.
- [ ] Every public Rust function has a rustdoc comment with at least one example.
- [ ] `mkdocs build` (or Sphinx equivalent) emits zero warnings.
- [ ] Linkcheck passes (no broken internal links).

**Depends on:** Week 10.

### Week 12 — Launch

**Deliverables:**
- v0.1.0 release on GitHub with detailed release notes.
- `pip install penumbra-fhe==0.1.0` works (PyPI publish).
- `cargo add penumbra-fhe` works (crates.io publish; the runtime crate also published as needed).
- Blog post or arXiv draft on the depth budget problem and our bootstrapping placement heuristic, ready to publish.
- Announcement post drafted (HackerNews, r/MachineLearning, r/cryptography, r/rust).

**DoD:**
- [ ] v0.1.0 tag exists on `main`, signed.
- [ ] All published artifacts (PyPI, crates.io, GitHub Release) match the tagged commit.
- [ ] Release notes credit every contributor and dependency.
- [ ] At least one external reviewer (someone outside the project) has read the blog draft and given feedback.

**Depends on:** Week 11.

---

## Post-v0.1 (Vision, not commitment)

The following are **not** v0.1 commitments. They exist here to capture direction without overcommitting.

| Item | Why it matters | Tentative |
|---|---|---|
| Bootstrapping placement via DP | Greedy is wasteful; DP can prove optimality bounds | v0.2 |
| Quantization-aware training helpers | Reduces accuracy loss vs naive quantization | v0.2 |
| Encrypted batch inference | Amortizes bootstrapping cost | v0.2 |
| Streaming inference (don't load whole graph) | Memory savings for large models | v0.3 |
| GPU offload for plaintext key generation | Doesn't require GPU FHE | v0.3 |
| TenSEAL / Concrete-ML compatibility shim | Lets users try alternative backends | v0.4 |
| Formal verification of polynomial coefficients | Audit trail for high-stakes deployments | v1.0 |

---

## Risk register

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | TFHE-rs API breaks between minor versions | M | H | Pin exact version in `Cargo.toml`; review release notes before bumping | core |
| R2 | PyO3 + maturin breaks on Apple Silicon | L | H | Pin PyO3 version; test wheel building in CI on macOS aarch64 | core |
| R3 | Polynomial activation accuracy insufficient for >4 layers | H | M | Document depth-vs-accuracy curve; scope explicitly limits to 4 layers in v0.1 | research |
| R4 | Bootstrapping placement is NP-hard in general | M | M | Greedy for v0.1; DP for v0.2; document the gap | research |
| R5 | Quantization error compounds beyond tolerance | M | H | Use generous fixed-point scale; document range assumptions; offer 32-bit fallback | core |
| R6 | TFHE-rs noise budget tables drift across versions | L | H | Verify depth-cost table per minor version of TFHE-rs in CI | core |
| R7 | Solo dev burnout over 3 months | M | H | Strict scope discipline; explicit week-level DoD; weekly retros | self |
| R8 | Security advisory in TFHE-rs after v0.1 ships | L | H | Subscribe to Zama security mailing list; have a hotfix release process documented in `SECURITY.md` | self |

---

## Repository file manifest

> Initial commit should contain everything listed here. Files marked **stub** are committed as placeholders to establish structure; their content will fill in over the corresponding milestone.

```
penumbra-fhe/
├── README.md                              # landing page (done in Phase 0)
├── ROADMAP.md                             # this file
├── ARCHITECTURE.md                        # technical design (done in Phase 0)
├── PHILOSOPHY.md                          # design principles (done in Phase 0)
├── AGENTS.md                              # AI coding agent directives
├── CONTRIBUTING.md                        # contributor guide
├── CODE_OF_CONDUCT.md                     # Contributor Covenant 2.1
├── SECURITY.md                            # disclosure policy
├── CHANGELOG.md                           # Keep a Changelog format
├── LICENSE                                # MIT
├── .gitignore
├── .gitattributes
├── .editorconfig
├── Cargo.toml                             # workspace root
├── pyproject.toml                         # maturin config
├── rust-toolchain.toml                    # pinned stable
├── rustfmt.toml
├── clippy.toml
├── .pre-commit-config.yaml
├── .github/
│   ├── CODEOWNERS
│   ├── dependabot.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   ├── feature_request.yml
│   │   ├── crypto_concern.yml
│   │   └── config.yml
│   └── workflows/
│       ├── ci.yml
│       ├── docs.yml
│       └── release.yml
├── crates/
│   ├── penumbra-ir/         (stub)
│   ├── penumbra-analyzer/   (stub)
│   ├── penumbra-compiler/   (stub)
│   ├── penumbra-runtime/    (stub)
│   └── penumbra-py/         (stub)
├── python/
│   └── penumbra_fhe/        (stub)
├── docs/                    (Sphinx scaffold)
├── tests/                   (test harness)
├── benchmarks/              (criterion + pytest-benchmark)
└── examples/                (filled in over Weeks 8, 9)
```
