<h1 align="center">Penumbra-FHE</h1>

<p align="center">
  <strong>Run encrypted inference on machine-learning models — without writing crypto code.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <img alt="Status: pre-alpha" src="https://img.shields.io/badge/status-pre--alpha-orange.svg">
</p>

---

Export any supported model to **ONNX**, load it into Penumbra-FHE, and run inference
directly on **encrypted data** using Fully Homomorphic Encryption (FHE). The server
computes on ciphertext and never sees your input or output.

```python
import penumbra as fhe

model = fhe.load_onnx("model.onnx")           # ONNX front door: parse + validate + lower to a Model
model.quantize(calibration_data, n_bits=6)   # float graph → int graph + lookup tables
pred = model.predict_encrypted(x)             # client encrypts → server evaluates → client decrypts
```

The **ONNX front door, quantization service, and the encrypted round trip work today**
(Phases 5–6, plus the first Phase-9 slice). `load_onnx` parses an ONNX model, **validates every
op at load time** (failing loudly with all problems at once if a model uses an unsupported op —
validation *is* the compile step), and lowers it to an `fhe.Model`. `predict_encrypted` then runs
the real encrypted forward pass (keygen → encrypt → evaluate → decrypt) via the Rust runtime and
returns the client-side prediction — it needs a Rust toolchain (`cargo`) and takes seconds-to-
minutes per sample. The current bridge shells out to the runtime; in-process PyO3 bindings and
wheels are the remaining Phase-9 work. You can also assemble a model by hand from the op
vocabulary — the same `Model` `load_onnx` produces:

```python
import penumbra as fhe

model = fhe.Model([
    fhe.Conv2d(weight=w1, in_h=8, in_w=8, in_channels=1, stride=2),
    fhe.Activation(lambda v: max(v, 0.0)),   # ReLU, fused into the conv's requantization
    fhe.Linear(weight=w2, bias=b2),
])
model.quantize(calibration_data, n_bits=4)   # PTQ (or QAT) → int weights, scales, lookup tables
model.export("model.fhe")                     # serialize for the Rust runtime
```

It implements a small, fixed set of ML operations against FHE primitives — no general-purpose
FHE compiler involved — over **pluggable backends**. The reference backend is
[`tfhe-rs`](https://github.com/zama-ai/tfhe-rs) (the TFHE scheme: exact, lookup-table based);
a second backend over [`poulpy-ckks`](https://github.com/phantomzone-org/poulpy) (the CKKS
scheme: approximate, SIMD-batched) is in progress, so the two schemes can be compared on
identical model-loading, inference, and measurement code.

## How it works

Penumbra-FHE has a **three-layer "narrow waist"** architecture with **two** waists: a small,
fixed set of ~8 operations that every model compiles down to, and a `Backend` trait beneath
it. The op vocabulary never changes as use cases multiply, and the op implementations never
change as schemes multiply.

- **Python front end** — load ONNX, quantize, lower to a serializable Intermediate
  Representation (IR).
- **Rust core** — read the IR and walk the op graph. Backend-neutral: no crypto here.
- **Rust backends** — `penumbra-tfhe` (`tfhe-rs`) and `penumbra-ckks` (`poulpy-ckks`)
  evaluate the ops under encryption. Both read the *same* IR file.

> **The golden invariant:** encrypted output matches the quantized-cleartext output — for
> TFHE **bit-for-bit** (TFHE is exact, so any discrepancy is a bug, never crypto noise); for
> CKKS, within a declared per-model error bound, since CKKS is approximate by construction.
> Same reference, one comparator per backend — see [`docs/BACKENDS.md`](docs/BACKENDS.md).

## Project status

**Pre-alpha — under active construction.** This is research/prototype-grade software, not
audited production cryptography. It targets *small* models (image classifiers, tabular
models, small CNNs, tree ensembles); inference takes seconds, not milliseconds. "Any ONNX
model" means: composed of supported ops, quantizes acceptably, and small enough to be
practical.

Current focus is **Phase 12**: a second (CKKS) backend and a controlled comparison of the two
schemes — latency, accuracy degradation, and overhead, under one shared harness. The TFHE
backend is the reference implementation and its exactness gate is unchanged.

## Documentation

- [`PROJECT.md`](PROJECT.md) — architecture, rationale, and the full design.
- [`ROADMAP.md`](ROADMAP.md) — the task-level build plan (phases P0–P12).
- [`docs/BACKENDS.md`](docs/BACKENDS.md) — the backend boundary: the `Backend` contract,
  per-scheme cost and budget models, and how to add a backend.
- [`docs/COMPARISON.md`](docs/COMPARISON.md) — the TFHE vs CKKS study: hypothesis, method,
  threats to validity, and the measured results.
- [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md) — the quantization service: PTQ/QAT, `n_bits`,
  per-channel scales, the bit-width budget, and the accuracy/speed tradeoff.
- [`docs/SUPPORTED-OPS.md`](docs/SUPPORTED-OPS.md) — the operators the runtime implements.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — accuracy and latency for the example models.
- [`docs/NOTES-tfhe.md`](docs/NOTES-tfhe.md) · [`docs/NOTES-ckks.md`](docs/NOTES-ckks.md) —
  per-scheme parameter profiles, primitives, and measured costs.
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — toolchain, build, and test instructions.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to contribute (and the canonical "add an op" path).
- [`AGENTS.md`](AGENTS.md) — guidelines for AI agents working in this repo.

## Quick start (development)

```bash
# Rust runtime (build in --release; debug FHE is very slow)
cargo test --workspace --release

# Python front end (managed with uv)
uv sync --all-extras && uv run pytest

The CKKS backend is not wired up yet; when it lands it gains its own test target and may
require a separate toolchain — see [`docs/NOTES-ckks.md`](docs/NOTES-ckks.md).

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for full setup.

## License

[Apache 2.0](LICENSE).
