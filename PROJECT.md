# Penumbra-FHE

> A library for running **encrypted inference** on machine-learning models. Export any
> supported model to ONNX, load it into Penumbra-FHE, and run inference directly on
> encrypted data using Fully Homomorphic Encryption (FHE) — without ever writing
> cryptography code.

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Background: FHE and the Crypto Landscape](#2-background-fhe-and-the-crypto-landscape)
3. [Why These Technology Choices](#3-why-these-technology-choices)
4. [The Core Architecture: The Narrow Waist](#4-the-core-architecture-the-narrow-waist)
5. [How ML Operations Map onto TFHE](#5-how-ml-operations-map-onto-tfhe)
6. [The Operator Set (Narrow Waist Vocabulary)](#6-the-operator-set-narrow-waist-vocabulary)
7. [The Intermediate Representation (IR)](#7-the-intermediate-representation-ir)
8. [Quantization: The Hardest Part](#8-quantization-the-hardest-part)
9. [Bit-Width Budget Management](#9-bit-width-budget-management)
10. [The ONNX Front Door](#10-the-onnx-front-door)
11. [Client/Server Deployment Model](#11-clientserver-deployment-model)
12. [Public API Design](#12-public-api-design)
13. [Repository Layout](#13-repository-layout)
14. [Build Order & Milestones](#14-build-order--milestones)
15. [Technology Stack & Dependencies](#15-technology-stack--dependencies)
16. [Scope, Limits & Honest Caveats](#16-scope-limits--honest-caveats)
17. [Glossary](#17-glossary)

---

## 1. What This Project Is

**Penumbra-FHE** is an FHE machine-learning inference library. Its promise:

> Load any ONNX model **composed of supported operators**, that **quantizes acceptably**
> and is **small enough to be practical**, and run it under encryption — without writing
> any cryptography code.

The end-to-end flow you are building toward:

```
   any_model.onnx  ──▶  Penumbra-FHE  ──▶  encrypted inference
   (PyTorch / sklearn /                     (tfhe-rs underneath;
    Keras / XGBoost export)                  user never touches crypto)
```

```python
import penumbra as fhe

m = fhe.load_onnx("model.onnx")
m.quantize(calibration_data)        # float graph → int graph + lookup tables
m.compile()                          # map ONNX ops → internal op registry
pred = m.predict_encrypted(x)        # client encrypts → server evaluates → client decrypts
```

It is built **directly on the `tfhe-rs` cryptographic library** — implementing a fixed,
small set of ML operations against TFHE primitives rather than going through a general-purpose
FHE compiler.

### What it is NOT

- It is **not** a compiler. We do not trace arbitrary programs or build MLIR circuits.
- It is **not** a general FHE framework. It does one thing: ML inference.
- It does **not** support arbitrary ONNX graphs — only a documented operator subset.
- It is **not** for large LLMs. Targets small/medium models (image classifiers, tabular
  models, small CNNs, tree ensembles, face classification).

---

## 2. Background: FHE and the Crypto Landscape

**Fully Homomorphic Encryption (FHE)** lets you compute directly on encrypted data without
decrypting it. The server processes ciphertext and returns ciphertext; only the client
holds the key to decrypt the result. The server learns nothing about the input or output.

### The scheme landscape (this matters more than library choice)

| Scheme | Arithmetic | Strength | Weakness |
|---|---|---|---|
| **TFHE / CGGI** (tfhe-rs) | Exact, small integers | Arbitrary functions via **lookup tables** (programmable bootstrapping); exact results | No SIMD batching → large linear algebra is costly |
| **CKKS** (OpenFHE) | Approximate reals, **SIMD-batched** | Excellent at big matmuls/convolutions | Nonlinearities need polynomial approximation; depth budgeting; approximate |
| **BGV/BFV** (OpenFHE) | Exact integers, batched | Good integer SIMD | Nonlinearities hard; leveled depth |

**Key takeaway:** TFHE is best for **exact, nonlinearity-heavy, discrete** models
(quantized integer NNs, decision trees, comparisons, small classifiers). CKKS is best for
**linear-algebra-dominated, approximation-tolerant** models at scale.

For small classifiers with ReLU/argmax (MNIST, faces, tabular), **TFHE is the correct
choice** — exact, arbitrary activations as lookup tables, no batching needed.

---

## 3. Why These Technology Choices

### Why a fixed op set over `tfhe-rs`, and not a general FHE compiler

A general-purpose FHE compiler exists to take **arbitrary programs** and solve a hard
optimization problem: pick crypto parameters, place bootstraps, lower to executable code.
That is a large, complex piece of compiler + cryptography engineering.

You don't need it, because ML models use a **tiny, fixed vocabulary** of operations
(~8 op types). Instead of a compiler that dynamically figures out any program, you
**hand-implement each ML op once** against `tfhe-rs`. The work a compiler would do at
runtime, you do by hand, ahead of time, for a fixed menu of layers.

```
What a general FHE compiler automates:              You do MANUALLY for ~8 ML ops:
  trace program → circuit                            you already know the ops (it's a NN)
  pick crypto parameters (an optimizer)       ──▶    use tfhe-rs default param profile
  schedule/place bootstraps                          activations/requant = bootstrap (obvious)
  lower to executable                                write the Rust eval loop once
```

`tfhe-rs`'s high-level `integer` API already gives you bootstrapped integer arithmetic and
lookup tables directly. You **consume** those primitives rather than generating them.

### What you give up (and why it's acceptable)

| A compiler gives | Without it (direct tfhe-rs) | Matters for you? |
|---|---|---|
| Optimal crypto params per circuit | Take tfhe-rs default profile | No — defaults are secure & work |
| Auto bootstrap placement | You decide (activation = bootstrap) | No — obvious for NNs |
| Compile *any* program | Only your fixed ML ops | No — you only need ML ops |
| Max performance squeeze | Somewhat slower | Only at scale; fine for these models |

The hard part (crypto-param + bit-width decisions) **shrinks and moves** to your library:
you make those choices by hand, but centrally and once.

---

## 4. The Core Architecture: The Narrow Waist

The entire design hinges on finding a **narrow waist** — a small, fixed set of operations
that every model compiles down to — so the crypto layer **never changes** as use cases
multiply.

```
┌─ Layer 3: MODEL ADAPTERS (grows per use case — NO crypto here) ────┐
│  MNIST CNN │ face classifier │ tabular MLP │ XGBoost │ ...          │
│         each just produces a graph of standard ops                  │
└──────────────────────────────┬──────────────────────────────────────┘
                                │  ◀── stable IR (the narrow waist)
┌─ Layer 2: IR + OP REGISTRY (fixed — the heart of the library) ─────┐
│  a graph of ~8 op types: Linear, Conv, ReLU/LUT, Requant, ...       │
└──────────────────────────────┬──────────────────────────────────────┘
                                │  ◀── stable op-eval interface
┌─ Layer 1: TFHE BACKEND (written ONCE — never changes per use) ─────┐
│  each op implemented against tfhe-rs primitives                     │
└──────────────────────────────────────────────────────────────────────┘
```

### The discipline that keeps it general

> **A new use case only ever adds a Layer-3 adapter (or just a new ONNX file).
> It never touches Layers 1–2.**

This is the litmus test: **if adding face recognition forces you to edit the crypto
backend, your abstraction leaked.** Adding a use case must mean adding a graph, never
adding crypto.

---

## 5. How ML Operations Map onto TFHE

Two cost regimes dominate everything:

- **Linear ops** (matmul, conv) where weights are **plaintext** and only the data is
  encrypted → cheap: scalar-multiply ciphertext by each plaintext weight, then add.
  **No bootstrap.**
- **Nonlinear ops** (activations, requantization) → **programmable bootstrapping (PBS)**:
  apply a lookup table to a ciphertext. **Expensive — this dominates runtime.**

| ML operation | TFHE realization | Cost |
|---|---|---|
| Linear / Conv (encrypted input, **plaintext** weights) | scalar-mul + adds | cheap |
| Activation (ReLU, sigmoid, …) | programmable bootstrap = apply lookup table | expensive |
| Requantization (rescale wide accumulator → small int) | also a LUT/PBS | expensive |
| Compare / Argmax | LUT | expensive |
| Add / residual | ciphertext addition | cheap |

**Runtime ≈ number of bootstraps.** Minimizing PBS operations is the central performance
lever. The deployment model is: **client encrypts the input; weights stay plaintext on the
server** → linear layers are cheap, and bootstraps occur only at activations/requant.

Rust server-side eval loop (sketch):

```rust
use tfhe::prelude::*;
set_server_key(server_key);

let mut acc = input_ciphertexts;                 // encrypted by client
for layer in &model.layers {
    match layer {
        Layer::Linear { weights, bias } =>
            acc = matvec_plaintext_weights(&acc, weights, bias),   // cheap
        Layer::Activation(lut) | Layer::Requant(lut) =>
            acc = acc.iter().map(|c| apply_lut(c, lut)).collect(),  // bootstrap
    }
}
// client decrypts `acc`
```

---

## 6. The Operator Set (Narrow Waist Vocabulary)

~8 operators cover an enormous range of models. Implement each **once**, correctly, with
bit-width management:

| Op | Covers | TFHE realization |
|---|---|---|
| `Linear` (matmul + bias) | dense layers, logistic/linear regression | ciphertext × plaintext weights → cheap |
| `Conv2d` | CNNs (MNIST, faces) | MACs against plaintext weights |
| `Activation(LUT)` | ReLU, sigmoid, GELU, any 1-input function | programmable bootstrap |
| `Requant` | rescale wide accumulator → small int | LUT |
| `Pool` / `Sum` | avg/max pool, reductions | adds (+ LUT for max) |
| `Compare` / `Argmax` | classification head, trees, thresholds | LUT |
| `Add` / `Concat` | residuals, skip connections | adds |

MNIST, face classification, tabular MLPs, small CNNs, and tree ensembles all compile to
combinations of these. **The library's value is implementing these correctly with
automatic bit-width management** — everything above is just graphs.

---

## 7. The Intermediate Representation (IR)

**The IR is your real product.** A clean, serializable op graph is what makes everything
compose. Get it right and every model is just data; get it wrong and every use case becomes
a special case.

### Design guidance

- Model it as a **directed graph of op nodes**, each with: op type, inputs, attributes
  (kernel size, stride, etc.), quantized parameters (int weights, bias), scales/zero-points,
  and (for nonlinear ops) the precomputed lookup table.
- Make it **serializable** (JSON to start; a compact binary format later). The Python side
  emits it; the Rust runtime consumes it.
- Consider it a **tiny subset of ONNX** — this is deliberate, because the ONNX loader's job
  becomes "lower ONNX graph → this IR."

### The two stable interfaces

1. **Python → IR file** (the export boundary). Adapters and the ONNX loader produce IR.
2. **IR file → Rust runtime** (the eval boundary). The runtime reads *any* IR and walks the
   op graph. It never changes per use case.

```python
# Layer 3: a use case is just a graph definition + quantized weights
model = fhe.Model([
    fhe.Conv2d(weights=w1, stride=1, padding=0),
    fhe.Activation(fhe.ReLU, bits=6),
    fhe.Linear(weights=w2, bias=b2),
    fhe.Argmax(),
])
model.quantize(calibration_data)   # PTQ/QAT → int weights, scales, LUTs
model.export("model.fhe")          # serialize IR for the Rust runtime
```

```rust
// Layer 1+2: the runtime reads ANY exported model. Never edited per use case.
let model = Model::load("model.fhe");
let ct    = client.encrypt(&input);
let out   = model.evaluate(&server_key, &ct);   // walks the op graph
let pred  = client.decrypt(&out);
```

---

## 8. Quantization: The Hardest Part

TFHE computes on **small integers**. You must convert float models to low-bit integer
models (weights, activations, accumulators). **This is ~80% of the engineering effort and
where ML accuracy lives or dies.**

### Make quantization a library service, not the user's problem

Provide a quantization module that turns any float graph into the int graph + scales +
lookup tables the backend needs. This is what makes the library usable by non-crypto people.

- **Post-Training Quantization (PTQ):** quantize a trained float model using calibration
  data to choose scales. Easiest path; start here.
- **Quantization-Aware Training (QAT):** train with quantization simulated in the loop;
  recovers most accuracy lost to low-bit integers. Use **Brevitas** rather than writing
  your own. Needed for harder models.

### Verification invariant

> FHE output must match the **quantized-cleartext** output **bit-for-bit**.

TFHE is **exact** — any discrepancy is a quantization or implementation bug, **never crypto
noise**. This gives you a powerful, deterministic test oracle: run the quantized model in
plain integers, run it under FHE, assert equality.

---

## 9. Bit-Width Budget Management

This is the manual remnant of the parameter/precision tuning a general FHE compiler's
optimizer would automate — centralized into the library instead.

- TFHE LUT/PBS cost grows sharply with precision. An 8-bit table has 256 entries. Keep
  activation bit-widths **small (≤ 6–8 bits)**.
- A `Linear`/`Conv` summing N products of b-bit values produces an accumulator needing
  ~`b + log2(N)` bits. You **must requantize back down** before the next layer, or cost
  explodes.
- **Enforce this centrally:** each op declares how it grows bit-width; the library inserts
  `Requant` automatically and **warns/errors** when precision exceeds what the LUT/PBS can
  handle.

This is the #1 lever for both accuracy and speed, and the place projects most often die.

---

## 10. The ONNX Front Door

ONNX is the universal export format — PyTorch, sklearn, Keras, and XGBoost all emit it.
"Train anywhere, run encrypted here" is the goal.

### What the loader does

1. **Parse** the ONNX graph.
2. **Validate** every node against the supported-op registry → **fail loudly at load time**
   with a clear "operator X not supported" message (never fail mysteriously at runtime).
3. **Quantize** to int weights + scales + lookup tables (using calibration data).
4. **Lower** the ONNX graph to the internal IR (the narrow waist).
5. Hand the IR to the tfhe-rs runtime for encrypted eval.

### The honest meaning of "any ONNX model"

ONNX has **150+ operators**; you will support a **subset** (the ~8–15 that matter:
Gemm/MatMul, Conv, Relu, Sigmoid, MaxPool/AveragePool, Add, Reshape, etc.). "Any" therefore
means:

> Any ONNX model **composed of supported operators**, that **quantizes acceptably**, and is
> **small enough to be practical**.

Two real constraints beyond op coverage:
- **Quantization must succeed** — the model must tolerate low-bit integers without
  unacceptable accuracy loss.
- **Size must be feasible** — op set matching ≠ runs in reasonable time. A full transformer
  may match the op set but be unusably slow.

This still covers a huge range from a single entry point: MNIST, face classification,
tabular MLPs, small CNNs, tree ensembles.

---

## 11. Client/Server Deployment Model

The privacy promise: **the server never sees the plaintext input or output.**

```
┌── CLIENT ──┐                       ┌──────── SERVER ────────┐
│ input      │   encrypt input        │ runs ENTIRE model       │
│   │        │ ──────────FHE─────────▶│ under FHE on ciphertext │
│ encrypt    │                        │  Conv/Linear: ×plaintext│
│            │                        │   weights (cheap)       │
│ decrypt ◀──┼─────────FHE────────────│  ReLU/Requant: LUT(PBS) │
│ result     │   encrypted output      │ never sees the input    │
└────────────┘                        └────────────────────────┘
```

### Roles & key material

- **Client:** generates keys, encrypts input, decrypts result. Holds the secret key.
- **Server:** holds the **server/evaluation key** (public, enables bootstrapping) and the
  **plaintext model weights**. Runs the encrypted forward pass. Learns nothing.

For small classifiers, **full-model-on-server under FHE is genuinely runnable** (one
encrypted forward pass, seconds-ish — unlike LLMs which need this per token). This gives the
cleanest privacy claim: the server runs the entire model on ciphertext and never sees the
input.

### Note on closed-set vs open-set (relevant for face recognition)

- **Classification** (digit 0–9; is this one of N enrolled faces?) → fixed-output small net,
  very FHE-friendly. **Start here.**
- **Embedding + distance matching** (open-set "who is this?") → adds an encrypted distance
  computation + comparison. Doable in TFHE (comparisons are LUTs) but a step harder.

---

## 12. Public API Design

### Python (front end: load, quantize, export, drive inference)

```python
import penumbra as fhe

# Load any supported ONNX model
model = fhe.load_onnx("model.onnx")

# Quantize using calibration data (becomes int weights + scales + LUTs)
model.quantize(calibration_data, n_bits=6)

# Lower to IR + validate ops (fails loudly on unsupported ops)
model.compile()

# Serialize IR for the runtime
model.export("model.fhe")

# Convenience: full client-side round trip (encrypt → eval → decrypt)
pred = model.predict_encrypted(x)
```

### Rust (runtime: keys, encrypt, evaluate, decrypt)

```rust
let (client_key, server_key) = penumbra::keygen(&PARAMS);
let model = penumbra::Model::load("model.fhe");

let ct   = penumbra::encrypt(&client_key, &input);
let out  = model.evaluate(&server_key, &ct);     // walks the op graph
let pred = penumbra::decrypt(&client_key, &out);
```

### Design principles

- **Crypto params:** ship a secure default profile; expose **one** override knob. Never make
  users choose parameters.
- **Quantization is a service:** the library owns it; users supply calibration data, not
  scales.
- **Fail loudly, early:** unsupported ops and infeasible bit-widths are caught at
  compile/load time with actionable messages.

---

## 13. Repository Layout

Proposed structure (Python front end + Rust runtime, bridged by the IR file format):

```
penumbra-fhe/
├── PROJECT.md                      # this document
├── README.md
├── LICENSE                         # Apache 2.0
│
├── python/                         # Layer 3 + quantization + ONNX loader + IR emitter
│   └── penumbra/
│       ├── __init__.py
│       ├── onnx_loader.py          # parse → validate → lower ONNX to IR
│       ├── op_registry.py          # supported op definitions + ONNX op mapping
│       ├── ir.py                   # IR graph data structures + (de)serialization
│       ├── quantization/           # PTQ / QAT (wraps Brevitas), calibration
│       ├── adapters/               # optional convenience builders (sklearn, torch)
│       └── client.py               # PyO3 bindings or subprocess bridge to Rust runtime
│
├── runtime/                        # Layer 1 + 2: tfhe-rs backend (Rust crate)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── keys.rs                 # keygen, param profiles
│       ├── ir.rs                   # IR deserialization (mirrors python/penumbra/ir.py)
│       ├── ops/                    # one module per op: linear, conv, activation, ...
│       ├── eval.rs                 # graph walker / eval loop
│       └── encrypt.rs              # encrypt / decrypt helpers
│
├── examples/
│   ├── mnist/                      # train → quantize → export → encrypted inference
│   └── faces/                      # second use case (validates abstraction: no crypto edits)
│
└── tests/
    ├── test_quantized_vs_fhe.py    # exactness invariant (FHE == quantized cleartext)
    └── ...
```

### The IR bridge

The IR data structures must be defined **consistently on both sides**
(`python/penumbra/ir.py` ↔ `runtime/src/ir.rs`). Start with JSON for the file format
(human-inspectable, easy to debug); move to a compact binary format later if needed.

---

## 14. Build Order & Milestones

Sequenced so you are **never blocked on the whole thing**. Each milestone is end-to-end.

### M0 — Spike (prove the crypto plumbing)
Stand up `tfhe-rs`, encrypt a value, apply a lookup table, decrypt. Confirm you understand
the `integer`/`shortint` API and programmable bootstrapping.

### M1 — Narrow waist with 3 ops → logistic regression / 1-layer MNIST
Implement `Linear`, `Activation(LUT)`, `Argmax` in the Rust runtime + minimal IR.
Run end-to-end: train → quantize → export IR → encrypt → evaluate → decrypt.
**Proves the waist.** Assert FHE output == quantized-cleartext output.

### M2 — Add `Conv2d`, `Pool`, `Requant` → small CNN on MNIST
Now you hit the real engineering problem: **accumulator bit-width**. Implement automatic
`Requant` insertion and bit-width tracking. **Proves multi-layer + bit-width management.**

### M3 — Quantization module + ONNX import
Wrap Brevitas/PTQ; build the ONNX loader (parse → validate → lower to IR). Users can now
**bring their own models**. **This is the inflection point where it becomes a library, not
a demo.**

### M4 — Second use case (faces) with ZERO backend changes
Add a face classifier purely as a new ONNX model / Layer-3 graph. **If it requires no
Layer-1 edits, the abstraction holds.** This is your validation milestone.

### M5 — Op coverage + ergonomics
Trees/XGBoost, more activations, clean Python API, PyO3 bindings, error messages, docs,
serialization format hardening, parameter-profile tuning.

---

## 15. Technology Stack & Dependencies

### Rust runtime (Layer 1 + 2)
- **`tfhe-rs`** — the FHE primitives. Use the high-level `integer` / `shortint` API
  for bootstrapped integer arithmetic and programmable bootstrapping (lookup tables).
- **`serde` / `serde_json`** — IR (de)serialization.
- **`PyO3`** (optional, recommended for M5) — expose the Rust runtime to Python directly,
  avoiding a subprocess/file bridge.

### Python front end (Layer 3 + quantization + ONNX)
- **`onnx`** — parse and inspect ONNX graphs.
- **`Brevitas`** — quantization-aware training; reuse instead of writing your own quantizer.
- **`numpy`** — numerical work, calibration.
- **PyTorch / scikit-learn / XGBoost** — for producing/exporting models in examples.
- Optional: `skorch`, `onnxruntime` (cleartext reference inference for the exactness test).
- **Packaging/env:** **`uv`** for dependency + environment management (project standard — not poetry).

### Bridge options (Python ↔ Rust)
1. **IR file + subprocess** — simplest; Python writes `model.fhe`, Rust runtime reads it.
   Start here.
2. **PyO3 bindings** — call the Rust runtime from Python directly. Better ergonomics;
   adopt in M5.

### Crypto parameters
Start with `tfhe-rs` **default secure parameter profile**. Expose a single override knob.
Hand-tuning `shortint` parameters (a noise/security/speed tradeoff) is a later optimization
you now own, since there is no compiler optimizer choosing them for you.

### License
Penumbra-FHE is licensed under **Apache 2.0**.

---

## 16. Scope, Limits & Honest Caveats

- **This is a real engineering project.** The surface is bounded because you target
  *inference* on *small models* over `tfhe-rs`'s high-level API, skipping any
  compiler/optimizer layer.
- **"Any model" is bounded** — only supported ops, only models that quantize acceptably,
  only sizes that run in reasonable time. Be precise about this boundary; it's what separates
  a working library from one that quietly breaks on the second model.
- **Bit-width budget is everything** — the dominant constraint on both accuracy and speed
  (Section 9). Centralize it.
- **Runtime ≈ number of bootstraps** — minimize activations/requant; linear ops with
  plaintext weights are cheap.
- **Latency** — even small models take seconds per inference. This is research/prototype
  territory, not real-time serving. Set expectations accordingly.
- **Exactness is your friend** — TFHE is exact, so FHE output == quantized-cleartext output.
  Any gap is a bug, not noise. Use this as your test oracle everywhere.
- **You own crypto-parameter selection** — there's no optimizer choosing params for you.
  Defaults work to start; tuning is a later, optional optimization.
- **Not for LLMs** — large transformers are matmul-dominated (TFHE's weakness) and need a
  forward pass per token. If you ever want LLM privacy, that's a *hybrid* design (cleartext
  backbone + one encrypted slice), a separate project from this library.

---

## 17. Glossary

- **FHE (Fully Homomorphic Encryption):** compute on ciphertext without decrypting.
- **TFHE / CGGI:** the FHE scheme `tfhe-rs` implements — exact small-integer arithmetic with
  arbitrary functions via lookup tables.
- **PBS (Programmable Bootstrapping):** the TFHE operation that simultaneously reduces noise
  and applies a **lookup table (LUT)** to a ciphertext. How activations/requant are realized.
  Expensive; runtime is dominated by PBS count.
- **LUT (Lookup Table):** a table mapping input integers to output integers, applied to a
  ciphertext via PBS. Realizes any single-input function (ReLU, sigmoid, requant, compare).
- **Quantization:** converting float weights/activations to low-bit integers. PTQ
  (post-training) or QAT (quantization-aware training).
- **Calibration data:** representative inputs used to choose quantization scales.
- **Bit-width budget:** the number of bits an integer value occupies; accumulators grow it,
  requantization shrinks it. The central performance/accuracy constraint.
- **Narrow waist:** the small fixed op set that all models compile to, keeping the crypto
  backend stable across use cases.
- **IR (Intermediate Representation):** the serializable op-graph that the Python front end
  emits and the Rust runtime consumes.
- **Client/Server keys:** client holds the secret key (encrypt/decrypt); server holds the
  public evaluation/server key (enables bootstrapping) and plaintext weights.
- **`tfhe-rs`:** the Rust TFHE library — the cryptographic foundation Penumbra-FHE is
  built on.
