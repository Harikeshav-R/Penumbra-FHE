<div align="center">

# Penumbra

**A homomorphic inference engine for privacy-preserving ML.**

[![CI](https://github.com/Harikeshav-R/penumbra-fhe/actions/workflows/ci.yml/badge.svg)](https://github.com/Harikeshav-R/penumbra-fhe/actions/workflows/ci.yml)
[![Docs](https://github.com/Harikeshav-R/penumbra-fhe/actions/workflows/docs.yml/badge.svg)](https://harikeshav-r.github.io/penumbra-fhe/)
[![PyPI](https://img.shields.io/pypi/v/penumbra-fhe.svg)](https://pypi.org/project/penumbra-fhe/)
[![Crates.io](https://img.shields.io/crates/v/penumbra-fhe.svg)](https://crates.io/crates/penumbra-fhe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust: stable](https://img.shields.io/badge/rust-stable-orange.svg)](rust-toolchain.toml)
[![Python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

*"In astronomy, the penumbra is the region of partial shadow — light is present but obscured."*

</div>

---

## What is this?

Penumbra lets you run neural network inference on **encrypted inputs**, on a server that **never sees the plaintext**, and return an **encrypted result that only you can decrypt**. The math is Fully Homomorphic Encryption (FHE); the contribution is making it usable.

```python
import penumbra_fhe as penumbra

# One-time setup
model = penumbra.compile("my_model.onnx")
public_key, private_key = penumbra.keygen()

# --- Client side ---
enc_input = penumbra.encrypt(x, public_key)

# --- Server side --- (sees only ciphertext)
enc_output = model.run(enc_input, public_key)

# --- Client side ---
result = penumbra.decrypt(enc_output, private_key)
```

The server is *mathematically incapable* of learning anything about your input. Not "promised not to look." Not "encrypted at rest." Genuinely cannot.

## Why does it matter?

Every cloud ML inference API today — medical diagnosis, fraud scoring, personal finance classification — receives your raw plaintext data. You trust the operator. FHE removes that trust requirement entirely. The reason it hasn't been deployed is engineering, not math: existing FHE libraries require expert-level cryptography knowledge to wire into a neural network. Penumbra closes that gap with an ONNX-in, encrypted-inference-out workflow.

## Status

> **Pre-alpha.** This project is in active development. APIs will change. Do not use in production. Do not use for anything that actually requires privacy yet.

See [`ROADMAP.md`](ROADMAP.md) for the milestone schedule.

## Features

- **ONNX-in workflow.** Export your PyTorch model with `torch.onnx.export(...)`. Penumbra handles the rest.
- **Depth-budget analyzer.** Automatic bootstrapping placement — the hard part of FHE engineering, done for you.
- **Polynomial activation library.** Drop-in polynomial approximations for ReLU, sigmoid, tanh at degrees 3, 5, 7.
- **Honest benchmarks.** Per-layer latency profiling, depth-cost breakdown, accuracy degradation reporting. We tell you exactly what FHE costs.
- **Pip-installable.** `pip install penumbra-fhe`. The crypto runs in Rust; you write Python.
- **MIT licensed.** Use it commercially. Modify it. Ship it.

## Supported model classes

| Architecture | Status |
|---|---|
| MLPs (2–4 layers) | Planned for v0.2 |
| Shallow CNNs | Planned for v0.3 |
| Tabular classifiers | Planned for v0.3 |
| Transformers / attention | **Out of scope** |
| GPU-accelerated FHE | Out of scope (TFHE-rs limitation) |
| Encrypted training | Out of scope (inference only) |

See [`ROADMAP.md`](ROADMAP.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the rationale.

## Installation

> Will be available once v0.1 ships. See [`ROADMAP.md`](ROADMAP.md).

**Python (end users):**
```bash
pip install penumbra-fhe
```

**Rust (library users):**
```bash
cargo add penumbra-fhe
```

**From source:**
```bash
git clone https://github.com/Harikeshav-R/penumbra-fhe.git
cd penumbra-fhe
maturin develop --release
```

System requirements:
- Rust toolchain (stable)
- Python 3.12+
- A CPU with AVX2 (recommended; aarch64 with NEON is also supported)
- ~8 GB RAM for MNIST-scale models

## Quick start (MLP on MNIST)

> Available once Month 2 milestones land. Tracking issue: TBD.

```python
import torch
import penumbra_fhe as penumbra

# 1. Train a small MLP in PyTorch (or use a pretrained one)
model = MyMLP()
# ... training ...

# 2. Export to ONNX
torch.onnx.export(model, dummy_input, "mlp.onnx")

# 3. Compile under FHE
encrypted_model = penumbra.compile(
    "mlp.onnx",
    activation_degree=3,   # polynomial approximation degree
    security_level=128,    # bits of cryptographic security
)

# 4. Generate keys and encrypt
pk, sk = penumbra.keygen()
ciphertext_in = penumbra.encrypt(x_test, pk)

# 5. Run encrypted inference (this is what a server would do)
ciphertext_out = encrypted_model.run(ciphertext_in, pk)

# 6. Decrypt and verify
prediction = penumbra.decrypt(ciphertext_out, sk)
```

A single MNIST inference will likely take **30–120 seconds** on a modern CPU. This is not a bug — it is the current frontier of FHE. See the [benchmark documentation](docs/tutorials/benchmarks.rst) for a detailed breakdown.

## How does FHE work? (30-second version)

Normal arithmetic:

```
2 + 3 = 5
```

Homomorphic arithmetic:

```
Enc(2) + Enc(3) = Enc(5)        # server never sees 2, 3, or 5
Enc(2) * Enc(3) = Enc(6)        # same — multiplication too
```

A neural network is mostly multiplications and additions, so most of it works directly. The hard parts are:

1. **Multiplicative depth.** Every encrypted multiplication adds noise. After ~10–15 levels of multiplication, the noise overwhelms the signal. A "bootstrapping" operation refreshes the ciphertext but costs ~1–10 seconds. Penumbra automatically places bootstrapping operations to balance correctness and latency.

2. **No conditionals.** You can't compute `if x > 0` without revealing `x`. This kills ReLU. Penumbra replaces non-linear activations with low-degree polynomial approximations (e.g., `ReLU(x) ≈ 0.5x + 0.125x³`), trading a small amount of accuracy for FHE-compatibility.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the deep version.

## Architecture

Four components, each separable, each in its own Cargo crate (plus a Python ingestion layer):

```
ONNX model
    │
    ▼
┌──────────────────────────┐
│ 1. Ingestion (Python)    │  →  parses ONNX, builds typed IR
└──────────────────────────┘
              │
              ▼
┌──────────────────────────┐
│ 2. Analyzer (Rust)       │  →  depth-cost per op, bootstrapping placement
└──────────────────────────┘
              │
              ▼
┌──────────────────────────┐
│ 3. Compiler (Rust)       │  →  lower to TFHE-rs primitives
└──────────────────────────┘
              │
              ▼
┌──────────────────────────┐
│ 4. Runtime (Rust + PyO3) │  →  encrypt / run / decrypt
└──────────────────────────┘
```

Workspace layout:

```
crates/
  penumbra-ir/         # IR types, op definitions
  penumbra-analyzer/   # depth analysis + bootstrapping placement
  penumbra-compiler/   # lowering pass + polynomial approximations
  penumbra-runtime/    # encrypted execution + TFHE-rs wrapper
  penumbra-py/         # PyO3 bindings (the maturin-built crate)
python/
  penumbra_fhe/        # Python package (ONNX ingestion + user API)
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for component contracts, type signatures, and data flow.

## Documentation

- **[`ROADMAP.md`](ROADMAP.md)** — week-by-week milestones, definition of done, risk register
- **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — system design, component contracts, performance budgets
- **[`PHILOSOPHY.md`](PHILOSOPHY.md)** — design principles, non-goals, what this project is and isn't
- **[`AGENTS.md`](AGENTS.md)** — directives for AI coding agents (Claude Code, Cursor, etc.)
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — contributor guide, dev setup, PR process
- **[`SECURITY.md`](SECURITY.md)** — disclosure policy, threat model, what this protects against

Full API documentation: [harikeshav-r.github.io/penumbra-fhe](https://harikeshav-r.github.io/penumbra-fhe/) *(once docs ship)*.

## Built on

| Dependency | Role |
|---|---|
| [**TFHE-rs**](https://github.com/zama-ai/tfhe-rs) (Zama) | Core FHE library. The TFHE scheme, SIMD-accelerated, actively maintained. |
| [**PyO3**](https://github.com/PyO3/pyo3) | Rust ↔ Python FFI. |
| [**maturin**](https://github.com/PyO3/maturin) | Build/publish tooling for PyO3 projects. |
| [**ONNX**](https://onnx.ai/) | Model interchange format. PyTorch and TensorFlow both export to it. |

## Contributing

Contributions are welcome. **Start by reading [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md).** This project has hard rules around cryptographic correctness that all contributors (human and AI) must follow.

Good places to start:
- Issues tagged [`good first issue`](https://github.com/Harikeshav-R/penumbra-fhe/labels/good%20first%20issue)
- Issues tagged [`help wanted`](https://github.com/Harikeshav-R/penumbra-fhe/labels/help%20wanted)
- Anything in the [`benchmarks/`](benchmarks/) directory

## Security

This project implements cryptography. If you find a security issue, **do not open a public issue.** See [`SECURITY.md`](SECURITY.md) for the disclosure process.

## License

Licensed under the **MIT License**. See [`LICENSE`](LICENSE).

Penumbra builds on [TFHE-rs](https://github.com/zama-ai/tfhe-rs), licensed BSD-3-Clause-Clear. We are deeply indebted to the Zama team for making practical FHE possible.

## Citation

If you use Penumbra in academic work:

```bibtex
@software{penumbra_fhe,
  author = {Harikeshav R},
  title  = {Penumbra: A Homomorphic Inference Engine for Privacy-Preserving ML},
  year   = {2026},
  url    = {https://github.com/Harikeshav-R/penumbra-fhe},
}
```
