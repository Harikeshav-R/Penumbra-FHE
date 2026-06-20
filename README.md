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

model = fhe.load_onnx("model.onnx")
model.quantize(calibration_data, n_bits=6)   # float graph → int graph + lookup tables
model.compile()                               # map ONNX ops → internal op registry
pred = model.predict_encrypted(x)             # client encrypts → server evaluates → client decrypts
```

Built directly on [`tfhe-rs`](https://github.com/zama-ai/tfhe-rs) (the TFHE scheme), it
implements a small, fixed set of ML operations against TFHE primitives — no general-purpose
FHE compiler involved.

## How it works

Penumbra-FHE has a **three-layer "narrow waist"** architecture: a small, fixed set of ~8
operations that every model compiles down to, so the cryptography layer never changes as
use cases multiply.

- **Python front end** — load ONNX, quantize, lower to a serializable Intermediate
  Representation (IR).
- **Rust runtime** (`tfhe-rs`) — read the IR and evaluate the op graph under encryption.

> **The golden invariant:** TFHE is *exact*, so FHE output equals the quantized-cleartext
> output **bit-for-bit**. Any discrepancy is a bug, never crypto noise.

## Project status

**Pre-alpha — under active construction.** This is research/prototype-grade software, not
audited production cryptography. It targets *small* models (image classifiers, tabular
models, small CNNs, tree ensembles); inference takes seconds, not milliseconds. "Any ONNX
model" means: composed of supported ops, quantizes acceptably, and small enough to be
practical.

## Documentation

- [`PROJECT.md`](PROJECT.md) — architecture, rationale, and the full design.
- [`ROADMAP.md`](ROADMAP.md) — the task-level build plan (phases P0–P11).
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — toolchain, build, and test instructions.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to contribute (and the canonical "add an op" path).
- [`AGENTS.md`](AGENTS.md) — guidelines for AI agents working in this repo.

## Quick start (development)

```bash
# Rust runtime (build in --release; debug FHE is very slow)
cd runtime && cargo test --release

# Python front end (managed with uv)
cd python && uv sync --all-extras && uv run pytest
```

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for full setup.

## License

[Apache 2.0](LICENSE).
