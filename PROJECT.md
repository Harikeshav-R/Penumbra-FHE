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
5. [How ML Operations Map onto FHE](#5-how-ml-operations-map-onto-fhe)
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
18. [The Backend Comparison Study](#18-the-backend-comparison-study)

---

## 1. What This Project Is

**Penumbra-FHE** is an FHE machine-learning inference library. Its promise:

> Load any ONNX model **composed of supported operators**, that **quantizes acceptably**
> and is **small enough to be practical**, and run it under encryption — without writing
> any cryptography code.

The end-to-end flow you are building toward:

```
   any_model.onnx  ──▶  Penumbra-FHE  ──▶  encrypted inference
   (PyTorch / sklearn /                     (a pluggable FHE backend underneath;
    Keras / XGBoost export)                  user never touches crypto)
```

```python
import penumbra as fhe

m = fhe.load_onnx("model.onnx")
m.quantize(calibration_data)        # float graph → int graph + lookup tables
m.compile()                          # map ONNX ops → internal op registry
pred = m.predict_encrypted(x)        # client encrypts → server evaluates → client decrypts
```

It is built **directly on FHE libraries** — implementing a fixed, small set of ML operations
against their primitives rather than going through a general-purpose FHE compiler. The
reference backend is [`tfhe-rs`](https://github.com/zama-ai/tfhe-rs) (the TFHE scheme); a
second backend over [`poulpy-ckks`](https://github.com/phantomzone-org/poulpy) (the CKKS
scheme) exists to make a controlled comparison between the two schemes possible (§18,
[`docs/BACKENDS.md`](./docs/BACKENDS.md)).

### What it is NOT

- It is **not** a compiler. We do not trace arbitrary programs or build MLIR circuits.
- It is **not** a general FHE framework. It does one thing: ML inference. Supporting two
  schemes does not change that — the second backend serves a specific comparison (§18), and
  backend feature work stops where that comparison's needs stop.
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
| **CKKS** (poulpy-ckks, OpenFHE) | Approximate reals, **SIMD-batched** | Excellent at big matmuls/convolutions | Nonlinearities need polynomial approximation; depth budgeting; approximate |
| **BGV/BFV** (OpenFHE) | Exact integers, batched | Good integer SIMD | Nonlinearities hard; leveled depth |

**Key takeaway:** TFHE is best for **exact, nonlinearity-heavy, discrete** models
(quantized integer NNs, decision trees, comparisons, small classifiers). CKKS is best for
**linear-algebra-dominated, approximation-tolerant** models at scale.

From those first principles, for small classifiers with ReLU/argmax (MNIST, faces, tabular),
**TFHE should be the correct choice** — exact, arbitrary activations as lookup tables, no
batching needed. That is why Penumbra was built on TFHE, and it remains the reference
backend.

> **This is a hypothesis, not a settled result.** It is well-motivated but has never been
> measured on this codebase, on these models, under one harness. The CKKS backend (§18)
> exists to test it: same IR, same models, same measurement code, only the scheme varies.
> The comparison may confirm the reasoning above — a useful result — or locate the crossover
> where SIMD batching overtakes exactness on Penumbra's own workloads. See
> [`docs/COMPARISON.md`](./docs/COMPARISON.md).

BGV/BFV is **not** in scope. It shares CKKS's batching and its difficulty with
nonlinearities, so it would mostly re-measure the same axis at additional cost.

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

The argument is stated against `tfhe-rs` because that is the reference backend, but it is
scheme-independent: it says a fixed ~8-op vocabulary is small enough to hand-implement
against *any* FHE library's primitives. That is precisely why a second backend is a bounded
piece of work rather than a second project — and it is the reason the `Backend` trait sits
where it does (§4).

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
                                │  ◀── waist 1: the stable IR
┌─ Layer 2: IR + OP REGISTRY + EVAL (fixed — the heart of the lib) ──┐
│  a graph of ~8 op types: Linear, Conv, ReLU/LUT, Requant, ...       │
│  backend-neutral: no crypto, no scheme branch                       │
└──────────────────────────────┬──────────────────────────────────────┘
                                │  ◀── waist 2: the `Backend` trait
┌─ Layer 1: FHE BACKENDS (one per scheme — never change per use) ────┐
│   penumbra-tfhe (tfhe-rs)      │      penumbra-ckks (poulpy-ckks)   │
│   exact ints, LUT via PBS      │      approx reals, polynomials     │
└──────────────────────────────────────────────────────────────────────┘
```

There are **two** waists, and they constrain growth along the two axes the project actually
grows in. The IR stops the op vocabulary from growing as *use cases* multiply; the `Backend`
trait stops the op implementations from growing as *schemes* multiply.

### The discipline that keeps it general

> **A new use case only ever adds a Layer-3 adapter (or just a new ONNX file).
> It never touches Layers 1–2.**
>
> **A new backend only ever adds a Layer-1 crate. It never touches Layers 2–3.**

These are the litmus tests, and they are duals:

- **If adding face recognition forces you to edit a crypto backend, the op vocabulary
  leaked.** Adding a use case must mean adding a graph, never adding crypto.
- **If adding CKKS forces you to change the IR, the op vocabulary, or the eval loop, the
  backend abstraction leaked.** Adding a scheme must mean adding a crate, never changing the
  waist above it.

The second rule buys something the first does not: **backend parity**. Because both backends
consume the same IR and run under the same harness, differences in their output are
attributable to the schemes rather than to two separately-built pipelines. That is a
correctness property of the comparison, not a convenience — see
[`docs/BACKENDS.md`](./docs/BACKENDS.md) and §18.

---

## 5. How ML Operations Map onto FHE

The same split applies under both schemes, but what makes each half expensive differs:

- **Linear ops** (matmul, conv) where weights are **plaintext** and only the data is
  encrypted → cheap under both. Under TFHE: scalar-multiply ciphertext by each plaintext
  weight, then add, no bootstrap. Under CKKS: the same, *and* batched across slots.
- **Nonlinear ops** (activations, requantization) → the expensive half, for opposite reasons.
  Under TFHE they are **programmable bootstraps (PBS)**: apply a lookup table to a
  ciphertext — exact, arbitrary, and the runtime bottleneck. Under CKKS there is **no LUT and
  no PBS at all**; they become **polynomial approximations**, which are cheaper in time but
  spend multiplicative depth and introduce error.

| ML operation | TFHE realization | CKKS realization |
|---|---|---|
| Linear / Conv (encrypted input, **plaintext** weights) | scalar-mul + adds — cheap | plaintext-mul + adds, SIMD-batched — cheap, spends one level |
| Activation (ReLU, sigmoid, …) | programmable bootstrap = apply lookup table — expensive | fitted polynomial — spends depth, approximate |
| Requantization (rescale wide accumulator → small int) | also a LUT/PBS — expensive | ReLU polynomial + a native rescale |
| Compare / Argmax | LUT — expensive | polynomial step function; poor at low degree |
| Add / residual | ciphertext addition — cheap | ciphertext addition — cheap |

The two cost models are therefore different, and neither generalizes to the other:

> **TFHE: runtime ≈ number of bootstraps.**
> **CKKS: runtime ≈ multiplicative depth × (rotations + rescales), amortized over slots.**

Minimizing PBS count is the central TFHE performance lever; minimizing depth and packing more
values per ciphertext is the central CKKS one. Where the rest of these docs say "runtime ≈
number of bootstraps" without qualification, read it as scoped to the TFHE backend.

The deployment model is shared: **client encrypts the input; weights stay plaintext on the
server** → linear layers are cheap under either scheme, and the expensive work is concentrated
at activations/requant.

Server-side eval loop (sketch) — note that it is generic over the backend, and that the
scheme appears nowhere in it:

```rust
let mut acc = input_ciphertexts;                 // encrypted by client
for layer in &model.layers {
    match layer {
        Layer::Linear { weights, bias } =>
            acc = backend.matvec_plaintext_weights(&acc, weights, bias),   // cheap
        Layer::Activation(f) | Layer::Requant(f) =>
            acc = backend.apply_univariate(&acc, f),   // PBS under TFHE, polynomial under CKKS
    }
}
// client decrypts `acc`
```

`apply_univariate` is the one place the schemes genuinely diverge, and it is exactly the
operation that dominates cost. See [`docs/BACKENDS.md`](./docs/BACKENDS.md) for the full
primitive contract.

---

## 6. The Operator Set (Narrow Waist Vocabulary)

~8 operators cover an enormous range of models. Implement each **once**, correctly, with
bit-width management:

| Op | Covers | TFHE realization | CKKS realization |
|---|---|---|---|
| `Linear` (matmul + bias) | dense layers, logistic/linear regression | ciphertext × plaintext weights → cheap | same, batched across slots |
| `Conv2d` | CNNs (MNIST, faces) | MACs against plaintext weights | MACs, batched; rotations for the window |
| `Activation(LUT)` | ReLU, sigmoid, GELU, any 1-input function | programmable bootstrap | fitted polynomial |
| `Requant` | rescale wide accumulator → small int | LUT | ReLU polynomial + native rescale |
| `Pool` / `Sum` | avg/max pool, reductions | adds (+ LUT for max) | adds/rotations (+ polynomial for max) |
| `Compare` / `Argmax` | classification head, trees, thresholds | LUT | polynomial step |
| `Add` / `Concat` | residuals, skip connections | adds | adds |

MNIST, face classification, tabular MLPs, small CNNs, and tree ensembles all compile to
combinations of these. **The library's value is implementing these correctly with
automatic bit-width management** — everything above is just graphs.

The vocabulary is deliberately **the same for every backend**. A scheme that cannot realize
an op rejects it loudly at load time; it never gets a private op, and the op set never forks
per backend (`AGENTS.md` §1.2, §1.4).

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

### The IR is backend-neutral

Every backend deserializes the **same** graph — byte for byte, no scheme tag, no
discriminator. This is not incidental: it is what makes a scheme comparison valid, because it
guarantees the two backends are being handed identical work (§18).

Some payload fields are historically TFHE-shaped — `num_blocks` (radix width),
`Requant.clamp_lut`, `Activation.lut`. A second backend **reinterprets** them (fitting a
polynomial to the same tabulated function, ignoring an advisory radix width); it does not get
fields of its own. Any pressure to add scheme-tagged fields is an architectural fork, not a
routine schema bump — see [`docs/IR-SPEC.md`](./docs/IR-SPEC.md) and `AGENTS.md` §5.

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

> Encrypted output must match the **quantized-cleartext** output: **bit-for-bit under TFHE;
> within the model's declared error bound under CKKS**.

The **reference never changes** — the quantized model run in plain integers
(`python/penumbra/reference.py`). Only the comparator is per-backend, which is exactly what
keeps the two backends comparable.

TFHE is **exact**, so under it any discrepancy is a quantization or implementation bug,
**never crypto noise**. That gives a powerful, deterministic test oracle: run the quantized
model in plain integers, run it under FHE, assert equality.

CKKS is **approximate by construction**, so its gate is a committed per-model error bound,
with the measured error always reported. Exceeding the bound is still a bug first — scale,
level, or polynomial degree — and noise second. CKKS is never compared against a friendlier
reference (such as the float model) to flatter its numbers.

Full statement and rationale: `AGENTS.md` §1.1 and
[`docs/BACKENDS.md`](./docs/BACKENDS.md#correctness-one-invariant-two-comparators).

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

### The budget is per-backend

Bit-width growth is a property of the **quantized graph**, so the tracker itself is
scheme-neutral and both backends reuse it. What differs is the capacity it is checked
against — each scheme has its own hard resource, and each fails loudly with the offending
layer named:

| | TFHE | CKKS |
|---|---|---|
| The resource | radix capacity, `num_blocks × MESSAGE_BITS` | multiplicative depth / level budget, and scale precision |
| Consumed by | accumulator growth (`b + log2(N)`) | every plaintext multiply and every polynomial degree |
| Restored by | `Requant` (a PBS) narrows back to one block | rescale, or (expensively) bootstrapping |
| Overflow looks like | silently wrong ciphertext | precision collapse, then noise |

Treat "bit-width budget" throughout these docs as TFHE's instance of a general resource
budget; `docs/BACKENDS.md` states both.

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

# Same model, same IR, different scheme — the only thing the user changes
pred = model.predict_encrypted(x, backend="ckks")
```

### Rust (runtime: keys, encrypt, evaluate, decrypt)

```rust
let backend = penumbra::backend("tfhe")?;        // or "ckks"
let (client_key, server_key) = backend.keygen(&params);
let model = penumbra::Model::load("model.fhe");  // the same file either way

let ct   = backend.encrypt(&client_key, &input);
let out  = model.evaluate(&backend, &server_key, &ct);   // walks the op graph
let pred = backend.decrypt(&client_key, &out);
```

### Design principles

- **Crypto params:** ship a secure default profile **per backend**; expose **one** override
  knob per backend. Never make users choose parameters.
- **Backend selection is not parameter exposure.** Users name a backend (`"tfhe"`, `"ckks"`);
  they never see `tfhe-rs` or `poulpy` types. Selection does not count against the one-knob
  budget.
- **Quantization is a service:** the library owns it; users supply calibration data, not
  scales.
- **Fail loudly, early:** unsupported ops and infeasible bit-widths are caught at
  compile/load time with actionable messages — including ops a *particular backend* cannot
  realize, and key or ciphertext material handed to the wrong backend.

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
├── crates/                         # the Rust workspace
│   ├── penumbra-core/              # Layer 2 — BACKEND-NEUTRAL: no crypto, no scheme branch
│   │   └── src/
│   │       ├── ir.rs               # IR deserialization (mirrors python/penumbra/ir.py)
│   │       ├── eval.rs             # graph walker / eval loop
│   │       ├── bitwidth.rs         # growth rules + budget propagation
│   │       └── backend.rs          # the `Backend` trait — waist 2
│   ├── penumbra-tfhe/              # Layer 1 — the tfhe-rs backend (the reference)
│   │   └── src/
│   │       ├── keys.rs             # keygen, param profiles
│   │       ├── ops/                # one module per op: linear, conv, activation, ...
│   │       └── encrypt.rs          # encrypt / decrypt helpers
│   ├── penumbra-ckks/              # Layer 1 — the poulpy-ckks backend (Phase 12)
│   └── penumbra-bench/             # the shared comparison harness — both backends
│
├── examples/
│   ├── mnist/                      # train → quantize → export → encrypted inference
│   └── faces/                      # second use case (validates abstraction: no crypto edits)
│
└── tests/
    ├── test_quantized_vs_fhe.py    # exactness invariant (FHE == quantized cleartext)
    └── ...
```

Until the Phase-12.1 workspace refactor lands, all Rust code lives in a single `runtime/`
crate; `ir.rs`/`eval.rs` map to `penumbra-core` and everything else to `penumbra-tfhe`.

### The IR bridge

The IR data structures must be defined **consistently on both sides**
(`python/penumbra/ir.py` ↔ `penumbra-core`'s `ir.rs`). Start with JSON for the file format
(human-inspectable, easy to debug); move to a compact binary format later if needed. There is
exactly **one** Rust-side definition, shared by every backend — a per-backend IR would defeat
the whole point (§7).

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

### M6 — Second *backend* (CKKS) with ZERO waist changes
Spike `poulpy-ckks` standalone, extract the `Backend` trait from the existing TFHE code,
implement CKKS against it, and run both under one benchmarking harness. **If it requires no
IR change, no op-vocabulary change, and no scheme branch in the eval loop, the backend
abstraction holds** — the mirror image of M4. Ends in the two-backend comparison (§18).
Tasks: `ROADMAP.md` Phase 12.

---

## 15. Technology Stack & Dependencies

### Rust runtime (Layer 1 + 2)
- **`tfhe-rs`** — the FHE primitives for the reference backend. Use the high-level
  `integer` / `shortint` API for bootstrapped integer arithmetic and programmable
  bootstrapping (lookup tables).
- **`poulpy-ckks`** (M6) — the CKKS backend. Pure Rust: no C++ toolchain, no CMake, no native
  linking, so it drops into the Cargo workspace the same way `tfhe-rs` does. Chosen over
  OpenFHE/SEAL bindings for exactly that reason. Authored by Jean-Philippe Bossuat (also the
  author of Lattigo), incubated by PhantomZone, and being integrated as a backend by Google's
  HEIR project.
  - Sits on **`poulpy-hal`** (a trait-based hardware abstraction layer) with pluggable CPU
    backends: `poulpy-cpu-ref` (reference, correctness-oriented), `poulpy-cpu-arm` (NEON,
    AArch64), `poulpy-cpu-avx` / `-avx512` (x86-64). The HAL's extension points are also the
    documented seam for a custom SIMD backend later — **out of scope**, but don't design the
    integration in a way that forecloses it.
  - ⚠️ Its own docs state the CKKS crate's public API is **subject to change**. Pin the exact
    version; log breakage rather than working around it (`docs/NOTES-ckks.md`).
  - ⚠️ Upstream pins a **nightly** toolchain and depends on `libm`'s `unstable-float`.
    Whether the workspace needs nightly is a blocking spike question (`ROADMAP.md` Phase 12.0).
- **`serde` / `serde_json` / `bincode`** — IR, key, and ciphertext (de)serialization.
- **`criterion`** (M6) — the shared benchmark harness both backends are measured by.
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
Start with each library's **default secure parameter profile** — `tfhe-rs`'s default, and
`poulpy-ckks`'s `presets`. Expose a single override knob per backend. Hand-tuning parameters
(a noise/security/speed tradeoff) is a later optimization you now own, since there is no
compiler optimizer choosing them for you. The two backends' **security levels must match**,
or any comparison between them is unfair by construction.

### License
Penumbra-FHE is licensed under **Apache 2.0**.

---

## 16. Scope, Limits & Honest Caveats

- **This is a real engineering project.** The surface is bounded because you target
  *inference* on *small models* over FHE libraries' high-level APIs, skipping any
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
- **Exactness is your friend, under TFHE** — TFHE is exact, so FHE output ==
  quantized-cleartext output. Any gap is a bug, not noise. Use this as your test oracle
  everywhere the TFHE backend runs.
- **CKKS is approximate by construction** — there is no bit-exactness to lean on. Its gate is
  a declared per-model error bound against the *same* reference, with the measured error
  always reported (§8). Do not read "FHE is exact" as a project-wide claim.
- **"Runtime ≈ number of bootstraps" is a TFHE statement**, not a project-wide one. CKKS's
  cost is depth, rotations, and rescales, amortized across slots (§5).
- **The scheme comparison is scoped** — it covers Penumbra's existing supported operations
  and committed models, on one pinned machine, and it compares a mature library against a
  young one. Those asymmetries are real and must be stated alongside any result
  (`docs/COMPARISON.md`).
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
  requantization shrinks it. TFHE's instance of a resource budget, and the central
  performance/accuracy constraint there.
- **Narrow waist:** the small fixed op set that all models compile to, keeping the crypto
  backends stable across use cases. Penumbra has two waists — the IR, and the `Backend` trait
  beneath it (§4).
- **IR (Intermediate Representation):** the serializable op-graph that the Python front end
  emits and the Rust runtime consumes. Backend-neutral: every scheme reads the same file.
- **Backend:** one Layer-1 crate implementing the `Backend` trait against exactly one FHE
  library — `penumbra-tfhe`, `penumbra-ckks`. Not to be confused with `poulpy`'s *HAL*
  backends, which are CPU implementations (`poulpy-cpu-arm`, …) one level further down.
- **Backend parity:** the rule that every backend consumes the same IR, runs the same models,
  and is measured by the same harness — what makes a scheme comparison valid (§18).
- **Client/Server keys:** client holds the secret key (encrypt/decrypt); server holds the
  public evaluation/server key (enables bootstrapping) and plaintext weights. Key material is
  **not** portable between backends.
- **`tfhe-rs`:** the Rust TFHE library — the cryptographic foundation of the reference backend.
- **CKKS:** an FHE scheme over **approximate reals** with **SIMD slot batching**. Excellent at
  large linear algebra; has no lookup table, so nonlinearities become polynomial
  approximations.
- **`poulpy` / `poulpy-ckks`:** the pure-Rust FHE library providing Penumbra's CKKS backend.
- **Slot / SIMD packing:** a CKKS ciphertext holds many values ("slots") that are operated on
  simultaneously. The source of CKKS's throughput advantage — and of the biggest open design
  question in the backend (`docs/BACKENDS.md`).
- **Rescale:** the CKKS operation that divides out accumulated scale after a multiplication,
  consuming one level. CKKS's analogue of narrowing an accumulator.
- **Level / multiplicative depth:** how many multiplications a CKKS ciphertext can still
  undergo before it must be bootstrapped. CKKS's hard budget, the counterpart to TFHE's radix
  capacity.
- **Polynomial approximation:** replacing a nonlinearity (ReLU, clamp, comparison) with a
  fitted polynomial, because CKKS cannot apply an arbitrary function. Trades accuracy against
  depth against latency.
- **HAL (Hardware Abstraction Layer):** `poulpy-hal`, the trait layer letting `poulpy-ckks`
  run over different CPU implementations. The documented extension point for a custom SIMD
  backend later — out of scope today.

---

## 18. The Backend Comparison Study

### Why a second backend at all

§2 concludes, from first principles, that TFHE is the right scheme for the models Penumbra
targets. The reasoning is sound and it is why the project exists in its current form — but it
has never been **measured**. A second backend turns that argument into a result.

The deliverable is the **comparison**, not a general-purpose FHE toolkit: latency, accuracy
degradation, and ciphertext/computation overhead for both schemes, produced by one shared
harness, in a form that can go directly into a short paper or preprint. Correctness and
reproducibility of that comparison matter more here than breadth of feature coverage.

### What makes the comparison valid

A second backend is only useful as evidence if everything *except* the scheme is held fixed:

- the same ONNX models and committed fixtures,
- the same IR file, byte for byte — no schema change and no scheme tag (§7),
- the same quantization output: identical integer weights, scales, and LUTs,
- the same accuracy reference, `python/penumbra/reference.py` (§8),
- the same eval loop — neither backend gets a private fast path,
- the same measurement code (`penumbra-bench`), on one pinned machine,
- matched security levels (§15).

Two independently built pipelines would produce differences that no analysis could separate
from genuine scheme differences. This is why **backend parity** (§4) is a correctness
requirement rather than a stylistic preference.

### What it deliberately does not attempt

- **Not feature completeness.** Backend work stops at what the comparison needs on Penumbra's
  existing supported operations (§6). Chasing coverage beyond that is out of scope.
- **Not a custom SIMD backend.** `poulpy-hal`'s extension points make one possible later; it
  is explicitly future work. The integration must simply not foreclose it (§15).
- **Not a claim beyond this workload class.** The result speaks to small quantized
  classifiers with ReLU/argmax heads, on one machine, at one set of parameters.
- **Not a scheme survey.** BGV/BFV are excluded (§2); they would re-measure the same axis.

### Known threats to the result

Stated here because they bound what the numbers mean, and in full in
[`docs/COMPARISON.md`](./docs/COMPARISON.md):

1. **SIMD packing.** If the CKKS backend keeps one value per ciphertext, it discards the only
   thing CKKS is better at, and its latency numbers are close to meaningless. The packing
   decision is the single most consequential open fork (`docs/BACKENDS.md`).
2. **The graph is quantized for TFHE.** Penumbra caps activations at one 2-bit block because
   a PBS is only feasible over a narrow value. Feeding CKKS that same graph is what makes the
   comparison apples-to-apples — *and* handicaps CKKS on accuracy. Both halves must be said.
3. **Asymmetric maturity and effort.** `tfhe-rs` is a mature, heavily optimized production
   library; `poulpy-ckks` is at 0.8.x with an API it describes as subject to change, and its
   Penumbra backend is new. A latency difference partly measures engineering investment.

### Where the work is tracked

`ROADMAP.md` **Phase 12**, staged spike-first: prove one real operation against
`poulpy-ckks` standalone before any refactor; then extract the `Backend` trait mechanically;
then implement CKKS against it; then the shared harness; then the numbers. The spike is
blocking on purpose — if something core is missing or broken upstream, that must surface
before a trait boundary is built around it.
