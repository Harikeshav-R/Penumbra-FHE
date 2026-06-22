# Penumbra-FHE — Build Roadmap

> A start-to-finish, task-level roadmap for building **Penumbra-FHE**: a library that loads
> ONNX models and runs encrypted inference on them via `tfhe-rs`.
>
> Read [`PROJECT.md`](./PROJECT.md) first — this roadmap assumes its architecture (the
> three-layer "narrow waist," the ~8-op vocabulary, the IR bridge, and the quantization +
> exactness invariant).

## How to use this document

- Work **top to bottom**. Each phase ends in a **working, demoable artifact** — you are
  never blocked on the whole system.
- Every task has a checkbox. Every phase has **Exit Criteria** — do not advance until they
  all pass.
- The **golden invariant** appears throughout: *FHE output must equal quantized-cleartext
  output, bit-for-bit.* Wire it in early (Phase 2) and never let it regress.
- Effort estimates assume one developer new to FHE but comfortable with Rust + Python. They
  are relative, not promises.

### Phase overview

| Phase | Goal | Headline artifact | Rough effort |
|---|---|---|---|
| 0 | Foundations & environment | Repo builds, CI green, hello-FHE runs | 1 week |
| 1 | TFHE primitives spike | Encrypt → LUT → decrypt in Rust | 1 week |
| 2 | Narrow waist, 3 ops | Encrypted logistic regression / 1-layer MNIST | 2 weeks |
| 3 | IR + serialization bridge | Python emits IR, Rust runs it | 1.5 weeks |
| 4 | Multi-layer CNN + bit-width | Encrypted small CNN on MNIST | 2.5 weeks |
| 5 | Quantization service | PTQ/QAT module, calibration | 2 weeks |
| 6 | ONNX front door | `load_onnx()` → validated IR | 2.5 weeks |
| 7 | Second use case (faces) | Face classifier, zero backend edits | 1.5 weeks |
| 8 | Op coverage expansion | Trees, more activations, pooling | 2 weeks |
| 9 | Ergonomics & PyO3 | One-call Python API, real bindings | 2 weeks |
| 10 | Performance & params | Profiling, PBS reduction, param tuning | 2 weeks |
| 11 | Hardening & release | Docs, tests, packaging, v1.0 | 2 weeks |

---

## Phase 0 — Foundations & Environment

**Goal:** A clean, reproducible repo where both the Rust runtime and Python front end build
and test in CI, with a trivial `tfhe-rs` program proving the toolchain works.

### Tasks

- [ ] Create the repo skeleton from `PROJECT.md` §13 (`python/`, `runtime/`, `examples/`,
      `tests/`, `docs/`).
- [ ] Initialize git; add `.gitignore` (Rust `target/`, Python `__pycache__/`, venv, ONNX
      artifacts, `*.fhe` files).
- [ ] **Rust:** `cargo init --lib runtime/`; add `tfhe`, `serde`, `serde_json` to
      `Cargo.toml`. Confirm `cargo build` works.
- [ ] **Python:** set up `pyproject.toml` managed with **`uv`** (project standard — not
      poetry); add `onnx`, `numpy`, `torch`, `brevitas`, `pytest`. Create
      `python/penumbra/__init__.py`.
- [ ] Add the **Apache 2.0** `LICENSE` file.
- [ ] Write a minimal `README.md` (one-paragraph pitch + "see PROJECT.md / ROADMAP.md").
- [ ] **CI:** GitHub Actions (or equivalent) with two jobs — `cargo test` and `pytest`.
      Make both green on an empty placeholder test.
- [ ] Add `rustfmt` + `clippy` (Rust) and `ruff` + `black` (Python) to CI; enforce on PRs.
- [ ] Document the dev setup in `docs/DEVELOPMENT.md` (toolchain versions, how to build/test).

### Exit Criteria

- `cargo test` and `pytest` both pass in CI from a clean checkout.
- Linters run clean.
- A new contributor can follow `docs/DEVELOPMENT.md` to a working build.

### Pitfalls

- `tfhe-rs` needs a recent stable Rust and benefits from `--release` for any timing work
  (debug builds are *extremely* slow for FHE). Note this in docs now.

---

## Phase 1 — TFHE Primitives Spike

**Goal:** Internalize the `tfhe-rs` API and the two operations everything is built from:
**plaintext-weight arithmetic** (cheap) and **programmable bootstrapping / LUT** (expensive).
Throwaway exploration — but keep it as a reference example.

### Tasks

- [ ] Read the `tfhe-rs` docs for the `shortint` and `integer` high-level APIs. Note the
      types you'll use for small quantized integers.
- [ ] Spike A — **keygen + encrypt + decrypt**: generate client/server keys, encrypt a small
      integer, decrypt it back. Confirm round-trip.
- [ ] Spike B — **plaintext-weight arithmetic**: multiply an encrypted integer by a *cleartext*
      scalar and add a cleartext bias. Decrypt; confirm correctness. (This is the `Linear`
      core.)
- [ ] Spike C — **programmable bootstrapping / LUT**: build a lookup table (e.g. ReLU on a
      small integer range) and apply it to a ciphertext. Decrypt; confirm it matches the
      table. (This is the `Activation`/`Requant` core.)
- [ ] Spike D — **bit-width experiment**: measure how PBS latency changes as you increase the
      precision (e.g. 2-bit vs 4-bit vs 6-bit vs 8-bit message). Record numbers in
      `docs/NOTES-tfhe.md`. This directly informs the bit-width budget design.
- [ ] Write down, in `docs/NOTES-tfhe.md`: which concrete `tfhe-rs` types/params you chose,
      the default parameter profile, and the cost ratio between a LUT op and an add/mul.

### Exit Criteria

- You can confidently write, from memory, the four primitives: keygen, encrypt/decrypt,
  plaintext-weight mul+add, and LUT-via-PBS.
- You have empirical latency numbers for PBS at several bit-widths.

### Pitfalls

- Choosing too-large message precision early makes everything slow. Confirm small integers
  (≤ 6–8 bits) are your working range.
- `tfhe-rs` parameter sets bundle security + precision + noise; don't hand-roll — start from
  a provided default profile.

---

## Phase 2 — Narrow Waist with 3 Ops (Encrypted Logistic Regression)

**Goal:** Prove the entire end-to-end pipeline with the minimal op set — `Linear`,
`Activation(LUT)`, `Argmax`. This is the spine of the whole project. **Establish the golden
exactness test here.**

### Tasks

- [ ] **Runtime — op trait:** define a Rust `Op` interface: takes encrypted inputs + server
      key, returns encrypted outputs. All ops implement it.
- [ ] **Runtime — `Linear`:** matvec of encrypted inputs against plaintext weights + bias.
      Cheap (no PBS).
- [ ] **Runtime — `Activation(LUT)`:** apply a provided lookup table via PBS. Start with
      sigmoid/ReLU on a small range.
- [ ] **Runtime — `Argmax`:** return the index of the max over a small encrypted vector
      (LUT/compare-based). For a first cut, a 2-class threshold is fine.
- [ ] **Runtime — eval loop:** a hardcoded sequence `Linear → Activation → Argmax` walking a
      `Vec<Op>`.
- [ ] **Python — toy model:** train logistic regression (or a 1-layer net) on a 2-class
      subset of MNIST (e.g. 0 vs 1) with scikit-learn / PyTorch.
- [ ] **Python — manual quantization:** by hand, quantize weights to small integers, compute
      scales, and build the activation LUT. (Automated in Phase 5 — manual is fine now.)
- [ ] **Python — hand-write IR:** emit the model as a hardcoded structure (JSON or even
      in-code) the Rust side reads. (Real IR is Phase 3.)
- [ ] **Cleartext reference:** implement the *quantized-integer* forward pass in plain Python
      (no FHE). This is the oracle.
- [ ] **GOLDEN TEST:** assert FHE output == quantized-cleartext output, **bit-for-bit**, over
      a batch of test inputs. Wire into CI.
- [ ] Measure and record end-to-end encrypted-inference latency for one sample.

### Exit Criteria

- Encrypted logistic regression classifies MNIST 0-vs-1 with the same labels as the
  quantized-cleartext model.
- The golden exactness test passes in CI.
- You can articulate the data flow: train → quantize → encrypt → eval → decrypt.

### Pitfalls

- If FHE ≠ cleartext, it is **never crypto noise** (TFHE is exact) — it's a quantization or
  indexing bug. Debug the cleartext path first.
- Keep precision tiny; correctness before speed.

---

## Phase 3 — The IR & Serialization Bridge

**Goal:** Replace the hardcoded model with a real, serializable **Intermediate Representation**
defined consistently on both sides. The IR is the product's backbone.

### Tasks

- [ ] **Design the IR schema** (see `PROJECT.md` §7): a graph of op nodes, each with op type,
      input/output edges, attributes, quantized params (int weights, bias), scales/zero-points,
      and (for nonlinear ops) the precomputed LUT.
- [ ] Start with **JSON** as the wire format (human-inspectable, easy to debug). Document the
      schema in `docs/IR-SPEC.md`.
- [ ] **Python — `ir.py`:** data classes for nodes + graph; `to_json()` / `from_json()`.
- [ ] **Rust — `ir.rs`:** mirror structs with `serde` deserialization. **Add a schema-version
      field** so format changes are detectable.
- [ ] **Cross-language conformance test:** Python emits an IR file; Rust loads it; assert the
      op graph matches expectations. Run in CI (Python writes a fixture, Rust reads it).
- [ ] **Refactor the eval loop** to consume the deserialized IR graph instead of a hardcoded
      `Vec` — walk nodes in topological order.
- [ ] Re-run the Phase 2 logistic-regression example **through the IR** end to end; golden
      test must still pass.
- [ ] Add a small `penumbra inspect model.fhe` debug command (Rust or Python) that prints the
      op graph + bit-widths for human inspection.

### Exit Criteria

- The Phase 2 model runs entirely from a serialized IR file — no hardcoded model anywhere.
- Python-emitted IR and Rust-consumed IR agree (conformance test green).
- Golden exactness test still passes.

### Pitfalls

- Keep the two IR definitions in lockstep. The schema-version field + conformance test is
  what prevents silent drift.
- Don't over-engineer the format yet (no binary format, no compression) — JSON until Phase 10.

---

## Phase 4 — Multi-Layer CNN + Bit-Width Budget

**Goal:** Add the ops needed for a real small CNN and implement the **automatic bit-width
management** that keeps multi-layer models feasible. This is where the hard engineering lives.

### Tasks

- [ ] **Runtime — `Conv2d`:** MACs of encrypted input against plaintext kernel weights.
      Reuse the `Linear` plaintext-weight pattern.
- [ ] **Runtime — `Pool`:** average pool (adds) and max pool (LUT/compare).
- [ ] **Runtime — `Requant`:** rescale a wide accumulator back to a small integer via LUT.
- [ ] **Runtime — `Add`:** ciphertext addition (for residuals).
- [ ] **Bit-width tracker (Python):** for each op, compute output bit-width from inputs. A
      `Linear`/`Conv` over N terms grows the accumulator by ~`log2(N)` bits (see `PROJECT.md`
      §9).
- [ ] **Automatic `Requant` insertion:** the compiler step inserts `Requant` nodes wherever
      accumulator bit-width exceeds the next op's LUT budget. Make this automatic, not manual.
- [ ] **Budget enforcement:** emit a clear **error/warning** when required precision exceeds
      what the PBS/LUT can handle, naming the offending layer.
- [ ] **Python — small CNN:** define a tiny conv net (e.g. 1–2 conv + pool + 1–2 dense) for
      10-class MNIST; quantize (still manual or semi-manual).
- [ ] Run full 10-class MNIST encrypted inference end to end through the IR.
- [ ] **Golden test extended:** FHE == quantized-cleartext for the CNN over a test batch.
- [ ] Record accuracy (vs float model) and latency. Note the accuracy lost to quantization.

### Exit Criteria

- Encrypted small CNN classifies 10-class MNIST; labels match quantized-cleartext exactly.
- `Requant` is inserted automatically; over-budget models fail loudly with a useful message.
- Accuracy and latency are recorded in `docs/BENCHMARKS.md`.

### Pitfalls

- **Accumulator overflow is the #1 bug here.** A conv summing many products needs many bits;
  if you don't requantize, results wrap or the LUT range is wrong. The bit-width tracker must
  be correct.
- Max pool needs comparisons (LUT) — more expensive than average pool. Prefer average pool in
  early models.

---

## Phase 5 — Quantization Service

**Goal:** Turn quantization from a manual chore into a **library service**. Users supply a
trained model + calibration data; the library produces int weights, scales, and LUTs.

### Tasks

- [ ] **Post-Training Quantization (PTQ):** given a trained float model + calibration data,
      compute per-tensor (then optionally per-channel) scales and zero-points; quantize
      weights to N bits.
- [ ] **Calibration:** run calibration data through the model to observe activation ranges;
      choose activation scales. Support a configurable `n_bits`.
- [ ] **LUT generation:** auto-generate the lookup table for each activation/requant from its
      function + the chosen input/output scales.
- [ ] **Quantization-Aware Training (QAT) via Brevitas:** integrate Brevitas so users can
      train with simulated quantization and export to the same int-graph form. Provide a
      documented example.
- [ ] **Accuracy harness:** a utility that reports float-accuracy vs quantized-accuracy vs
      FHE-accuracy on a test set, so users can see the quantization gap.
- [ ] **Refactor Phases 2 & 4 examples** to use the quantization service instead of manual
      quantization. Golden test still passes.
- [ ] Document quantization in `docs/QUANTIZATION.md`: PTQ vs QAT, choosing `n_bits`, the
      accuracy/speed tradeoff, and the bit-width budget link.

### Exit Criteria

- A user can call `model.quantize(calibration_data, n_bits=...)` and get a working int graph
  with no manual scale math.
- QAT path works end to end on at least one example.
- Quantized-cleartext still matches FHE exactly (the service didn't break the invariant).

### Pitfalls

- Per-tensor vs per-channel scales materially affect accuracy; start per-tensor, offer
  per-channel for weights.
- The LUT must be generated in the *quantized integer domain* consistent with the scales —
  an off-by-scale here silently wrecks accuracy.

---

## Phase 6 — The ONNX Front Door

**Goal:** The headline feature. `load_onnx()` parses an ONNX model, **validates** it against
the supported-op registry, quantizes it, and lowers it to IR. This is the inflection point
where Penumbra-FHE becomes a *library*, not a demo.

### Tasks

- [ ] **Op registry:** a declarative table mapping supported ONNX ops → internal ops. Start
      with: `Gemm`/`MatMul` → `Linear`, `Conv` → `Conv2d`, `Relu`/`Sigmoid` → `Activation`,
      `MaxPool`/`AveragePool` → `Pool`, `Add` → `Add`, plus shape ops (`Reshape`, `Flatten`).
- [ ] **Decide which ONNX ops are FHE-viable** before building the registry: an op is viable
      only if it reduces to your TFHE primitives (plaintext-weight arithmetic, adds, or a
      single-input LUT) within the bit-width budget. Document the rationale per op.
- [ ] **Parser:** load ONNX with the `onnx` package; extract the graph, initializers
      (weights), and node attributes.
- [ ] **Validator:** walk every node; if any op isn't in the registry, **fail at load time**
      with `"operator X (node 'name') not supported"`. List all unsupported ops at once, not
      one at a time.
- [ ] **Lowering:** translate the validated ONNX graph into the internal IR graph, attaching
      quantized params + LUTs from the quantization service.
- [ ] **Shape handling:** resolve tensor shapes (needed for bit-width tracking and conv/pool).
      Handle `Reshape`/`Flatten`/`Transpose` as no-ops or layout changes where possible.
- [ ] **Round-trip test:** export a PyTorch model → ONNX → `load_onnx()` → IR → encrypted
      inference; assert labels match the original quantized model.
- [ ] **Multi-framework test:** repeat with a scikit-learn model and (if feasible) a Keras
      model exported to ONNX, proving "train anywhere."
- [ ] Document supported ops + constraints in `docs/SUPPORTED-OPS.md`. Be explicit about the
      bounded meaning of "any model" (`PROJECT.md` §10).

### Exit Criteria

- `load_onnx("model.onnx")` works for at least two models from two different frameworks.
- Unsupported ops fail loudly at load time with actionable messages (verified by a test that
  feeds an unsupported model).
- The documented supported-op list matches what the validator actually accepts (test this).

### Pitfalls

- ONNX has 150+ ops; **resist supporting more than you need.** Each op is real backend work.
- ONNX opset versions differ — pin a supported opset range and validate it.
- Shape inference is fiddly; lean on `onnx.shape_inference` before lowering.

---

## Phase 7 — Second Use Case (Faces): The Abstraction Validation

**Goal:** Add a completely different use case **without touching the Rust backend (Layers
1–2)**. This is the proof that the narrow waist holds.

### Tasks

- [ ] Pick a **closed-set face classification** task (is this one of N enrolled people?) — a
      fixed-output small CNN, very FHE-friendly. Avoid open-set embedding+distance for now
      (`PROJECT.md` §11).
- [ ] Train a small face classifier (or fine-tune a tiny CNN) on a small face dataset; export
      to ONNX.
- [ ] Run it through the **existing** `load_onnx → quantize → IR → encrypted inference`
      pipeline.
- [ ] **THE VALIDATION:** confirm this required **zero edits to `runtime/src/ops/` or
      `eval.rs`**. If it did require backend edits, the abstraction leaked — fix the
      abstraction, not the use case.
- [ ] Add the example under `examples/faces/` with a README.
- [ ] Record accuracy + latency in `docs/BENCHMARKS.md`.
- [ ] (Optional stretch) Prototype open-set: add an encrypted distance + threshold compare,
      noting any new ops needed.

### Exit Criteria

- A face classifier runs encrypted inference through the unchanged backend.
- Git diff for this phase touches **no Layer-1 crypto code** (only Python adapters/examples
  and possibly new declarative registry entries).

### Pitfalls

- If you find yourself editing `Conv2d` or adding a special case to `eval.rs`, stop — the
  right fix is a more general op or a missing registry entry, not a use-case hack.

---

## Phase 8 — Op Coverage Expansion

**Goal:** Broaden the supported model families now that the architecture is proven. Add ops
that unlock new model classes, each via the same add-an-op discipline.

### Tasks

- [ ] **Tree ensembles (decision trees / XGBoost):** trees are often *easier* in FHE than NNs
      — comparisons are LUTs. Add a tree-to-IR adapter and the compare/select ops needed.
- [ ] **More activations:** tanh, GELU, leaky ReLU, hardswish — all are single-input LUTs, so
      mostly LUT-generation work in the quantization service.
- [ ] **Concat / split / multi-input graphs:** support branching graphs (not just linear
      chains) in the IR walker.
- [ ] **Batch norm folding:** fold BN into preceding conv/linear at quantization time (common
      ONNX pattern) so it costs nothing at runtime.
- [ ] **Additional pooling / global average pool.**
- [ ] For each new op: registry entry + Rust impl + bit-width rule + golden test.
- [ ] Expand `docs/SUPPORTED-OPS.md` and add a model-zoo of validated examples.

### Exit Criteria

- At least one tree-based model and one additional NN architecture run encrypted end to end.
- Every new op has a passing golden test and a documented bit-width rule.

### Pitfalls

- Multi-input/branching graphs break the naive linear eval loop — make sure the walker does
  true topological ordering with intermediate-result storage.

---

## Phase 9 — Ergonomics & PyO3 Bindings

**Goal:** Make the library pleasant to use. Replace the file/subprocess bridge with real
in-process bindings and a clean one-call API.

### Tasks

- [ ] **PyO3 bindings:** expose the Rust runtime (keygen, encrypt, evaluate, decrypt) to
      Python directly, eliminating the subprocess/file round-trip.
- [ ] **One-call API:** implement `model.predict_encrypted(x)` that internally does
      keygen (or reuse) → encrypt → evaluate → decrypt and returns the prediction
      (`PROJECT.md` §12).
- [ ] **Key management API:** `keygen()`, save/load keys, reuse keys across calls; document
      which key goes where (client vs server).
- [ ] **Client/server split example:** a runnable demo with an actual process boundary —
      client encrypts and sends, server evaluates and returns, client decrypts. Proves the
      privacy story (server only ever touches ciphertext).
- [ ] **Error messages:** audit all failure modes (unsupported op, over-budget bit-width,
      shape mismatch, key mismatch) for clear, actionable text.
- [ ] **Crypto-param profile API:** ship a secure default; expose a single override knob
      (`PROJECT.md` §12). Do not surface raw `tfhe-rs` params to users.
- [ ] Build wheels so `pip install penumbra-fhe` works (PyO3 + maturin).

### Exit Criteria

- `pip install` + `load_onnx` + `predict_encrypted` works in a fresh environment with no
  manual Rust steps.
- A genuine client/server demo runs with a process boundary.
- All known error paths produce actionable messages (tested).

### Pitfalls

- PyO3 + heavy `tfhe-rs` types: be deliberate about what crosses the boundary (pass IR +
  ciphertext handles, not giant copies).
- Key (de)serialization for server keys can be large — measure and document.

---

## Phase 10 — Performance & Parameter Tuning

**Goal:** Make it as fast as the design allows. Runtime ≈ number of bootstraps, so the work
is reducing and parallelizing PBS, plus tuning crypto params.

### Tasks

- [ ] **Profile:** instrument the eval loop to count PBS ops and time per op type. Identify
      the dominant cost (almost always activations/requant).
- [ ] **Reduce bootstraps:** fuse adjacent requant/activation where possible; skip
      unnecessary requant; choose op orderings that minimize PBS.
- [ ] **Parallelism:** evaluate independent ciphertexts/channels in parallel (rayon). PBS over
      a layer's outputs is embarrassingly parallel.
- [ ] **Parameter tuning:** experiment with `tfhe-rs` `shortint` parameter sets to trade
      noise/security/speed; keep security fixed, optimize speed for your bit-widths. Record a
      tuned default profile.
- [ ] **Bit-width minimization:** revisit models to use the smallest viable precision per
      layer (you own this since there's no optimizer — `PROJECT.md` §3).
- [ ] **Binary IR format (optional):** replace JSON with a compact binary format if
      load/serialization shows up in profiles.
- [ ] **Benchmark suite:** standardized latency/accuracy numbers for MNIST, the CNN, faces,
      and a tree model. Track regressions in CI.
- [ ] Document tuning guidance in `docs/PERFORMANCE.md` (what knobs exist, their tradeoffs).

### Exit Criteria

- Measurable, documented latency improvement over the Phase 4/7 baselines.
- A benchmark suite runs in CI and flags regressions.
- A tuned default parameter profile is committed and justified.

### Pitfalls

- Never trade away security for speed silently — security level is a hard constraint; only
  optimize within it.
- Premature micro-optimization before profiling wastes time; measure first.

---

## Phase 11 — Hardening & v1.0 Release

**Goal:** Turn a working system into a releasable library: docs, tests, examples, packaging,
and a clear statement of scope.

### Tasks

- [ ] **Test coverage:** unit tests per op, integration tests per example, the golden
      exactness test across all models, the unsupported-op failure test, cross-language IR
      conformance. Target high coverage on Layers 1–2.
- [ ] **Property/fuzz tests:** random small models → assert FHE == quantized-cleartext.
- [ ] **Documentation site:** getting-started, tutorial (MNIST end to end), supported ops,
      quantization guide, performance guide, API reference, architecture (link `PROJECT.md`).
- [ ] **Examples polished:** `examples/mnist/`, `examples/faces/`, a tabular example, a tree
      example — each with a README and one-command run.
- [ ] **Scope statement:** prominently document the bounded meaning of "any ONNX model"
      (`PROJECT.md` §17) and latency expectations, so users aren't surprised.
- [ ] **Security note:** state the threat model (server sees only ciphertext), the parameter
      security level, and that this is research/prototype-grade, not audited production crypto.
- [ ] **Packaging:** publish wheels (PyPI) and the Rust crate (crates.io if desired);
      versioning + changelog.
- [ ] **CONTRIBUTING.md:** how to add an op (the canonical extension path: registry entry +
      Rust impl + bit-width rule + golden test).
- [ ] Tag **v1.0**.

### Exit Criteria

- A new user goes from `pip install` to encrypted MNIST inference using only the docs.
- All tests green in CI; coverage targets met.
- Scope, security, and performance expectations are documented honestly.
- v1.0 is tagged and packages are published.

---

## Cross-Cutting Practices (apply in every phase)

- **The golden invariant is sacred:** every model, every phase — FHE output == quantized-
  cleartext output, bit-for-bit. It's your truth oracle; never let it regress in CI.
- **New use case ⇒ new graph, never new crypto.** If a use case forces backend edits, the
  abstraction leaked (`PROJECT.md` §4). Fix the abstraction.
- **Fail loudly, early.** Unsupported ops and over-budget bit-widths are caught at
  compile/load time with actionable messages — never silently at runtime.
- **Bit-width budget is everything** (`PROJECT.md` §9). Track it centrally; it governs both
  accuracy and speed.
- **Runtime ≈ number of bootstraps.** Keep PBS count in mind for every op you add.
- **Build in release mode for any timing.** Debug FHE is misleadingly slow.
- **Keep the two IR definitions in lockstep** (Python ↔ Rust) via the conformance test +
  schema version.

## Dependency Graph (what unblocks what)

```
P0 ──▶ P1 ──▶ P2 ──▶ P3 ──▶ P4 ──▶ P5 ──▶ P6 ──▶ P7 ──▶ P8 ──▶ P9 ──▶ P10 ──▶ P11
                │             │      │      │
                └─ golden test established and carried forward ─┘
                                     │
        (P5 quantization + P4 bit-width feed P6 ONNX lowering;
         P6 must exist before P7 faces; P8 broadens after P7 validates the waist)
```

## Definition of Done (the whole project)

- [ ] Load an ONNX model from any supported framework, composed of supported ops.
- [ ] Quantize it via the library (PTQ or QAT) with a measurable, documented accuracy gap.
- [ ] Run encrypted inference where the server only ever touches ciphertext.
- [ ] FHE results match quantized-cleartext results exactly.
- [ ] Adding a new use case requires no crypto-backend changes.
- [ ] Installable via `pip`, documented, tested, benchmarked, and honestly scoped.
