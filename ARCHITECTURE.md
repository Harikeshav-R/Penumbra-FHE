# Penumbra Architecture

> **Audience:** contributors and AI agents working on the codebase. If you are an end user, see [`README.md`](README.md) and the tutorials in `docs/tutorials/`.

This document specifies the architecture of Penumbra: component boundaries, type signatures, data flow, error model, concurrency model, performance budgets, and testing strategy. It is the **single source of truth** for "what goes where." If you are about to introduce a cross-component dependency or a new top-level abstraction, edit this document in the same PR.

## Table of contents

1. [Design principles](#1-design-principles)
2. [System overview](#2-system-overview)
3. [Component contracts](#3-component-contracts)
4. [Data flow](#4-data-flow)
5. [Error model](#5-error-model)
6. [Concurrency model](#6-concurrency-model)
7. [Security model](#7-security-model)
8. [Performance budget](#8-performance-budget)
9. [Testing strategy](#9-testing-strategy)
10. [Cross-cutting decisions](#10-cross-cutting-decisions)

---

## 1. Design principles

These are the rules that resolve design disputes. When in doubt, the **earlier** principle wins.

1. **Correctness over performance.** A correct slow inference is a feature; an incorrect fast inference is a bug masquerading as a feature. Optimization that risks correctness is rejected.
2. **Mathematics is not negotiable.** Where FHE constraints (depth budget, lack of comparison, polynomial activation error) limit us, we accept the limit and document it. We do not paper over math with heuristics that lie.
3. **Components are separable.** Each crate must be usable on its own. `penumbra-ir` must not import `tfhe-rs`. `penumbra-analyzer` must not know about Python. This rule is enforced in `Cargo.toml` dependencies.
4. **The IR is the contract.** Every cross-component conversation happens through the IR. No back-channels. No "just this once" type imports.
5. **Determinism by default.** Same inputs → same outputs, byte-for-byte. Randomness is injected via an explicit RNG parameter, never read from `thread_rng()` implicitly.
6. **Honesty by default.** Benchmarks are reported with reference hardware. Accuracy degradation is documented. We do not say "fast" when we mean "less slow than the alternatives."
7. **Crypto code is special.** It is reviewed twice. It is not refactored for style. It is changed only with intent and a passing test.

---

## 2. System overview

Penumbra is a **compiler** plus a **runtime**, packaged as a Python library with a Rust core.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        USER-FACING PYTHON API                        │
│                       (python/penumbra_fhe/)                         │
│                                                                      │
│  penumbra_fhe.compile(onnx_path) -> CompiledModel                    │
│  penumbra_fhe.keygen() -> (PublicKey, PrivateKey)                    │
│  penumbra_fhe.encrypt(x, pk) -> Ciphertext                           │
│  penumbra_fhe.decrypt(ct, sk) -> ndarray                             │
└──────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ PyO3
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        RUST CORE (workspace)                         │
│                                                                      │
│  ┌──────────────┐                                                    │
│  │ penumbra-py  │  PyO3 bindings — translates Python ↔ Rust types    │
│  └──────┬───────┘                                                    │
│         │ uses                                                       │
│         ▼                                                            │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐    │
│  │ penumbra-    │    │ penumbra-        │    │ penumbra-        │    │
│  │ runtime      │◄───┤ compiler         │◄───┤ analyzer         │    │
│  │              │    │                  │    │                  │    │
│  │ TFHE-rs ops, │    │ Lowering pass:   │    │ Depth cost,      │    │
│  │ encrypt/     │    │ IR → executable  │    │ bootstrapping    │    │
│  │ decrypt,     │    │ graph; poly      │    │ placement        │    │
│  │ execution    │    │ approximations   │    │                  │    │
│  └──────────────┘    └──────────────────┘    └────────┬─────────┘    │
│                                                       │              │
│                              ┌────────────────────────┘              │
│                              ▼                                       │
│                         ┌──────────────┐                             │
│                         │ penumbra-ir  │  Op types, Graph, Tensor    │
│                         │              │  metadata, shapes           │
│                         └──────────────┘                             │
└──────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ JSON
                                  │
┌──────────────────────────────────────────────────────────────────────┐
│                       PYTHON INGESTION LAYER                         │
│                                                                      │
│  ONNX file ──► parse ──► classify ops ──► emit IR JSON ──► (to Rust) │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 Why this split?

The split is driven by **dependency direction** and **language strength**:

- **Python for ingestion** because the ONNX Python bindings (`onnx`) are mature; doing this in Rust would require reimplementing protobuf parsing for ONNX with no payoff.
- **Rust for everything else** because correctness in cryptography demands a strong type system, and TFHE-rs is Rust.
- **PyO3 only at the edge** so the rest of the Rust code stays free of Python types.

The IR is the membrane: Python emits IR JSON, Rust consumes it. Neither side has to know about the other's quirks.

---

## 3. Component contracts

Each component is specified by **what it imports**, **what it exports**, and **what invariants it guarantees**.

### 3.1 `penumbra-ir`

**Purpose:** Define the typed intermediate representation. Nothing else.

**Imports:**
- `serde` for JSON serialization
- `thiserror` for error types
- **Nothing from FHE libraries. Nothing from Python. Nothing from the analyzer or compiler.**

**Exports:**

```rust
pub struct Graph {
    pub nodes:   Vec<Node>,
    pub inputs:  Vec<TensorRef>,
    pub outputs: Vec<TensorRef>,
}

pub struct Node {
    pub id:        NodeId,
    pub op:        Op,
    pub inputs:    Vec<TensorRef>,
    pub outputs:   Vec<TensorRef>,
    pub metadata:  NodeMetadata,
}

pub enum Op {
    // Linear (FHE-native)
    MatMul { transpose_a: bool, transpose_b: bool },
    Gemm   { alpha: f32, beta: f32, transpose_a: bool, transpose_b: bool },
    Add,
    Sub,
    Mul,                     // element-wise; scalar multiply via broadcast

    // Convolution (FHE via decomposition into matmul)
    Conv2d   { kernel: (usize, usize), stride: (usize, usize), padding: Padding },
    AvgPool2d { kernel: (usize, usize), stride: (usize, usize) },

    // Non-linear (requires polynomial approximation)
    Relu     { degree: PolynomialDegree, range: ApproxRange },
    Sigmoid  { degree: PolynomialDegree, range: ApproxRange },
    Tanh     { degree: PolynomialDegree, range: ApproxRange },

    // Structural (no FHE work)
    Reshape  { shape: Vec<i64> },
    Flatten  { axis: i64 },

    // FHE-internal (inserted by analyzer)
    Bootstrap,
}

pub struct Tensor {
    pub shape:  Vec<usize>,
    pub dtype:  Dtype,
    pub data:   Option<TensorData>,  // None for activation tensors, Some for weights/biases
}

pub enum Dtype { F32, I32, I64, QFixed { scale_bits: u32 } }
```

**Invariants:**
- `Graph` is a DAG (no cycles). Validated on construction; constructor returns `Result<Graph, IrError>`.
- Every `TensorRef` in `Node::inputs` resolves to a `TensorRef` in some earlier `Node::outputs` or to a `Graph::input`.
- `Bootstrap` nodes are produced **only** by the analyzer. Ingestion may not emit them.
- Shape inference is closed: every `Node::outputs[i].shape` is computable from its inputs and op kind.

**Serialization:** `serde_json` with deterministic field ordering. The same `Graph` serializes to the same bytes.

### 3.2 `penumbra-analyzer`

**Purpose:** Compute depth cost per op, place bootstrapping operations.

**Imports:**
- `penumbra-ir`
- `thiserror`

**Exports:**

```rust
pub struct DepthBudget {
    pub max_depth: u32,            // total multiplicative depth before bootstrap
    pub current:   u32,            // current accumulated depth
}

pub fn analyze(graph: &Graph) -> Result<DepthProfile, AnalyzerError>;

pub fn place_bootstraps(
    graph:   &Graph,
    budget:  DepthBudget,
    policy:  PlacementPolicy,
) -> Result<Graph, AnalyzerError>;

pub enum PlacementPolicy {
    Greedy { threshold: u32 },     // insert when remaining budget < threshold
    Manual { positions: Vec<NodeId> },
}

pub struct DepthProfile {
    pub per_node:    HashMap<NodeId, u32>,
    pub cumulative:  HashMap<NodeId, u32>,
    pub overflows:   Vec<NodeId>,  // empty if profile is within budget
}
```

**Invariants:**
- `analyze` is pure: same `Graph` → same `DepthProfile`.
- `place_bootstraps` returns a graph that is equivalent to the input (decrypts to the same value) but has bootstrap nodes inserted such that no path exceeds `budget.max_depth`.
- A graph that already contains bootstraps may be re-analyzed; existing bootstraps are respected, not re-placed.

**Depth cost table** lives in `penumbra-analyzer::depth_costs::TABLE`. Costs are sourced from TFHE-rs documentation **and** empirically verified by a `cargo bench` benchmark in CI. The table is versioned alongside the pinned TFHE-rs version.

### 3.3 `penumbra-compiler`

**Purpose:** Lower IR ops to executable TFHE-rs operations. Apply polynomial approximations.

**Imports:**
- `penumbra-ir`
- `penumbra-runtime` (for the operations being lowered to)
- `thiserror`

**Exports:**

```rust
pub struct CompiledGraph {
    // Opaque type. Holds the lowered, executable representation.
    // Internally: a Vec of operations against penumbra-runtime primitives.
}

pub fn compile(
    graph:    &Graph,
    options:  CompileOptions,
) -> Result<CompiledGraph, CompilerError>;

pub struct CompileOptions {
    pub activation_degree:  PolynomialDegree,
    pub quantization:       Quantization,
    pub determinism:        bool,             // true = no parallel non-determinism
}

pub struct Quantization {
    pub scale_bits:  u32,                     // fixed-point scale; 2^scale_bits
    pub clamp_to:    Option<(i64, i64)>,      // pre-encryption clamp
}
```

**Invariants:**
- Compilation is total over the documented op set (see Section 3.1). Unknown ops produce a `CompilerError::UnsupportedOp`, never a panic.
- Compilation is deterministic: same `Graph` + same `CompileOptions` → same `CompiledGraph` bytes.
- Polynomial coefficients are loaded from `const` arrays. They are not regenerated at compile time. (See [`PHILOSOPHY.md`](PHILOSOPHY.md) §4 for why.)

**Polynomial approximation source:** coefficients in `crates/penumbra-compiler/src/poly/coefficients.rs`. Each coefficient block has a comment containing: the function approximated, the range, the degree, the maximum error over the range, and a citation to the derivation. **Do not** modify coefficients without re-running the derivation and updating all four pieces of metadata.

### 3.4 `penumbra-runtime`

**Purpose:** Wrap TFHE-rs. Provide encrypted tensor operations. Execute compiled graphs.

**Imports:**
- `tfhe` (TFHE-rs)
- `thiserror`
- **Not** `penumbra-ir` (runtime operates on lowered ops, not IR).

**Exports:**

```rust
pub struct PublicKey   (/* opaque */);
pub struct PrivateKey  (/* opaque */);
pub struct Ciphertext  (/* opaque */);
pub struct EncryptedTensor {
    pub shape: Vec<usize>,
    pub data:  Vec<Ciphertext>,
}

pub fn keygen(params: SecurityParams) -> Result<(PublicKey, PrivateKey), RuntimeError>;

pub fn encrypt(value: &[i64], pk: &PublicKey) -> Result<EncryptedTensor, RuntimeError>;
pub fn decrypt(ct: &EncryptedTensor, sk: &PrivateKey) -> Result<Vec<i64>, RuntimeError>;

// Primitive operations (lowering target)
impl EncryptedTensor {
    pub fn add(&self, other: &Self, pk: &PublicKey) -> Result<Self, RuntimeError>;
    pub fn scalar_mul(&self, k: i64, pk: &PublicKey) -> Result<Self, RuntimeError>;
    pub fn dense(&self, w: &[i64], b: &[i64], pk: &PublicKey) -> Result<Self, RuntimeError>;
    pub fn bootstrap(&mut self, pk: &PublicKey) -> Result<(), RuntimeError>;
}

pub struct SecurityParams {
    pub bits: u32,                // 128 by default; 192 / 256 also supported
}
```

**Invariants:**
- All operations are constant-time with respect to plaintext values (inherited from TFHE-rs).
- `keygen` is the only function that draws cryptographic randomness; takes an optional explicit RNG seed for testing.
- An `EncryptedTensor`'s shape is stored in cleartext (this is intentional and documented; the **values** are encrypted, the **structure** is not).

### 3.5 `penumbra-py`

**Purpose:** PyO3 bindings. The only crate that knows about Python.

**Imports:**
- `pyo3`
- `penumbra-ir`, `penumbra-analyzer`, `penumbra-compiler`, `penumbra-runtime`

**Exports:** Python classes and functions mirroring the Rust API, with Pythonic naming.

**Invariants:**
- Every Python-callable function has a docstring with `:param:` and `:returns:` annotations.
- Errors are translated from Rust `Result::Err` to Python exceptions of named types (`PenumbraError`, `PenumbraCompilerError`, etc.) defined in `__init__.py`.
- No business logic in this crate. It is a translation layer.

### 3.6 `python/penumbra_fhe/`

**Purpose:** ONNX ingestion, user-facing API, polish.

**Layout:**

```
python/penumbra_fhe/
├── __init__.py           # public API surface
├── ingestion.py          # ONNX → IR JSON
├── _bindings.pyi         # type stubs for the Rust extension
├── errors.py             # Python exception classes
└── py.typed              # PEP 561 marker
```

**Invariants:**
- Every public function has a type annotation. Verified by `pyrefly` in CI.
- The module's `__all__` lists every public symbol; nothing else is part of the API.
- ONNX parsing failures produce `PenumbraIngestionError`, not raw `onnx` exceptions.

---

## 4. Data flow

### 4.1 Compile-time flow

```
┌─────────────────┐
│ model.onnx      │
└────────┬────────┘
         │
         ▼  (Python: ingestion.py)
┌─────────────────┐
│ onnx.ModelProto │
└────────┬────────┘
         │
         ▼  classify ops, infer shapes
┌─────────────────┐
│ Graph (Python)  │
└────────┬────────┘
         │
         ▼  serialize to JSON
┌─────────────────┐
│ IR JSON         │
└────────┬────────┘
         │
         ▼  (Rust: penumbra-py)
┌─────────────────┐
│ Graph (Rust)    │
└────────┬────────┘
         │
         ▼  (analyzer)
┌─────────────────┐
│ Graph + bootstraps │
└────────┬────────┘
         │
         ▼  (compiler)
┌─────────────────┐
│ CompiledGraph   │
└─────────────────┘
```

### 4.2 Inference flow

```
                   PUBLIC KEY ─────────┐
                                       │
┌─────────────┐                        │
│ plaintext   │                        ▼
│ input  x    │ ──► encrypt ──► ┌─────────────┐
└─────────────┘    (client)     │ Enc(x)      │
                                └──────┬──────┘
                                       │
                                       ▼
                                ┌─────────────┐
                                │CompiledGraph│
                                │    .run     │  ← server side
                                │             │  ← public key only
                                └──────┬──────┘
                                       │
                                       ▼
                                ┌─────────────┐
                                │ Enc(y)      │
                                └──────┬──────┘
                                       │
PRIVATE KEY ─────────────────────┐     │
                                 ▼     ▼
                                decrypt
                                       │
                                       ▼
                                ┌─────────────┐
                                │ plaintext   │
                                │ output  y   │
                                └─────────────┘
```

**Key property:** the private key is **never** present on the server. The server holds only `pk` and the compiled graph.

---

## 5. Error model

### 5.1 Rust

- All public functions return `Result<T, ComponentError>` where `ComponentError` is the crate's error type, defined with `thiserror`.
- **`unwrap()` is forbidden** in non-test code. CI enforces this with a clippy lint (`clippy::unwrap_used`).
- **`panic!` is forbidden** in non-test code except for documented invariant violations, marked with a `// PANIC:` comment explaining why this cannot be a `Result`.
- Errors carry context. Wrap with `.map_err(|e| MyError::Context { source: e, ... })`. Never lose causal chains.

### 5.2 Python

- Rust errors are translated to typed Python exceptions in `python/penumbra_fhe/errors.py`:
  - `PenumbraError` (root)
  - `PenumbraIngestionError`
  - `PenumbraCompilerError`
  - `PenumbraRuntimeError`
  - `PenumbraDepthBudgetError`
- Original Rust message preserved in `str(exc)`.
- A traceback chain is preserved across the FFI boundary where possible.

### 5.3 What is not an error

- Slow inference is not an error. It is documented behavior.
- Approximation error within the documented tolerance is not an error. Approximation error **above** the documented tolerance is a `PenumbraCompilerError::AccuracyDegradation`.

---

## 6. Concurrency model

**v0.1:** single-threaded execution. TFHE-rs internal parallelism is allowed and welcomed, but Penumbra's own code does not spawn threads.

**Why:** debugging concurrency in cryptographic code is painful. We pay the latency cost in v0.1 to keep correctness easy to reason about.

**Future (v0.2+):**
- Operator-level parallelism: independent branches of the graph executed in parallel.
- Batch-level parallelism: multiple inputs processed concurrently.
- Both will be opt-in via `CompileOptions::concurrency`.

**Determinism:** with `CompileOptions::determinism = true`, all parallelism is disabled regardless of policy.

---

## 7. Security model

### 7.1 Threat model

**Adversary:** a malicious server that:
- Holds the public key.
- Holds the compiled graph.
- Receives ciphertexts.
- Executes the compiled graph honestly (semi-honest model).
- Returns ciphertexts to the client.

**Goals:**
- The server learns nothing about the client's plaintext input beyond its **structure** (tensor shape, dtype).
- The server learns nothing about the client's plaintext output.

**Out of threat model for v0.1:**
- A malicious server that returns wrong results (no integrity / no verifiable computation).
- A malicious server that tries to learn input through timing side channels at the FHE library level (we inherit TFHE-rs's defenses, no more).
- A malicious server that learns input through tensor shape / network architecture (this is by design — shape is cleartext).
- A malicious client that tries to extract weights via crafted inputs (model extraction attacks; this is a separate research area).

### 7.2 What you cannot conclude from this project

- **This is not a cryptographic audit.** No external review has been performed. Use at your own risk.
- **This is not a constant-time guarantee.** We inherit TFHE-rs's properties, no more, no less.
- **This is not a post-quantum security analysis.** TFHE is believed to be quantum-resistant (lattice-based), but we make no claim beyond what TFHE-rs makes.

See [`SECURITY.md`](SECURITY.md) for the full disclosure policy.

---

## 8. Performance budget

These are **targets**, not guarantees. They are revised at each phase retrospective.

### 8.1 Latency targets (reference machine: MacBook Pro M3 Pro 11-core)

| Operation | v0.1 target | Stretch |
|---|---|---|
| Encrypt a 784-element vector (MNIST input) | <500 ms | <100 ms |
| Single encrypted add | <10 ms | <5 ms |
| Single encrypted multiply | <100 ms | <50 ms |
| Bootstrapping (one ciphertext) | <10 s | <5 s |
| Dense layer (128 → 128, degree-3 ReLU) | <30 s | <15 s |
| 3-layer MNIST MLP, single inference | <120 s | <60 s |
| 2-conv + 2-dense ConvNet MNIST, single inference | <300 s | <180 s |

### 8.2 Accuracy targets

| Model | v0.1 target |
|---|---|
| 3-layer MLP on MNIST | within 1% of plaintext accuracy |
| 2-conv + 2-dense ConvNet on MNIST | within 1% of plaintext accuracy |
| Tabular logistic regression (UCI Adult, Iris, etc.) | within 2% of plaintext accuracy |

### 8.3 Resource targets

| Resource | v0.1 target |
|---|---|
| Peak RAM during MLP inference | <4 GB |
| Peak RAM during ConvNet inference | <8 GB |
| Wheel size (cpython-3.12 macos arm64) | <50 MB |
| Cold-start time (`python -c "import penumbra_fhe"`) | <2 s |

---

## 9. Testing strategy

### 9.1 By component

| Component | Test types |
|---|---|
| `penumbra-ir` | Unit tests for graph construction; property tests (`proptest`) for serialization round-trips |
| `penumbra-analyzer` | Unit tests on handcrafted graphs; property tests for "bootstraps placed → no path overflows" |
| `penumbra-compiler` | Unit tests per lowering rule; golden-file tests for compiled output |
| `penumbra-runtime` | Property tests for `decrypt(op(encrypt(x), encrypt(y))) == op(x, y)`; correctness tests vs reference plaintext |
| `penumbra-py` | Smoke tests for every public function; doctest in docstrings |
| `python/penumbra_fhe` | Unit tests for ingestion on fixture ONNX files; type checking with `pyrefly` |

### 9.2 Integration tests

Located in `tests/integration/`. Each test:
- Loads a fixture model (small, checked in).
- Runs end-to-end (compile → keygen → encrypt → infer → decrypt).
- Asserts both **correctness** (within tolerance) and **latency** (within budget × 2 — generous because CI machines are slow).

### 9.3 Benchmark gates

Performance-sensitive PRs include a benchmark diff in the PR description. A regression of >20% on any tracked benchmark requires explicit reviewer approval to merge.

---

## 10. Cross-cutting decisions

### 10.1 Why not Concrete-ML / TenSEAL / SEAL?

- **Concrete-ML** (Zama): excellent project, but its Python-side compiler is opinionated about model structure and harder to extend in directions we want (e.g., a separable analyzer crate, an IR we control). We use Zama's lower-level **TFHE-rs** instead so the rest of the architecture is ours.
- **TenSEAL** (OpenMined): BFV/CKKS based; CKKS has different tradeoffs (approximate arithmetic, no bootstrapping in most modes). TFHE's bootstrapping-friendly nature is the right fit for deeper networks.
- **Microsoft SEAL**: C++; loses us the Rust correctness benefits, and the Rust bindings are not first-class.

This decision is revisited at v0.4 when we may consider adding a CKKS backend behind a feature flag.

### 10.2 Why fixed-point quantization, not native floats?

TFHE-rs supports integer ciphertexts (`FheUint`, `FheInt`). True floating-point under FHE exists but is dramatically slower. Fixed-point quantization (model weights → i64 with scale factor) is the standard tradeoff. We document the scale factor and let the user override it.

### 10.3 Why ONNX, not PyTorch directly?

- **Static graph.** ONNX is a frozen computational graph; PyTorch is dynamic. Static graph dramatically simplifies the IR.
- **Framework-agnostic.** Users from TensorFlow, JAX, or other frameworks can also export to ONNX.
- **No PyTorch runtime dependency.** Penumbra doesn't ship PyTorch in its install.

The cost is that some PyTorch-native ops (custom autograd functions) don't export cleanly. We accept this cost.

### 10.4 Why workspaces, not a single crate?

- **Compile times.** Changes to the analyzer don't trigger rebuilds of the runtime.
- **Dependency hygiene.** `penumbra-ir` has no FHE dependencies; it can be used by external tools (e.g., a debugger or visualizer) without pulling in TFHE-rs.
- **Future modularity.** A v0.4 CKKS backend can be a parallel `penumbra-runtime-ckks` crate without restructuring everything.

### 10.5 Versioning

- Semver: `MAJOR.MINOR.PATCH`.
- v0.x: minor versions may break the API. We document migration in `CHANGELOG.md`.
- v1.0 is committed only when the API is judged stable enough that breaking changes are rare.

### 10.6 What "production-ready" means and when we'll claim it

We will not claim "production-ready" until:
1. The project has had at least one external cryptographic review (audit).
2. The polynomial coefficient derivation is documented and reproducible.
3. The depth-cost table is empirically verified for at least two consecutive TFHE-rs versions.
4. There is a documented procedure for emergency disclosure (in `SECURITY.md`) **and** that procedure has been tested with a dry run.

None of these are v0.1 commitments.
