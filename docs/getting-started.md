# Getting Started

This guide walks through prerequisites, installation, and running your first encrypted inference with Penumbra-FHE.

## Prerequisites

- **Python:** 3.10, 3.11, or 3.12 required.
- **Rust:** stable ≥ 1.83 (install via [rustup](https://rustup.rs)) and **uv** (install via [Astral](https://docs.astral.sh/uv/)) are needed only for source builds.

## Installation

```bash
pip install penumbra-fhe
```

Or with uv:

```bash
uv add penumbra-fhe
```

Prebuilt abi3 wheels exist for Linux x86_64 (manylinux2014), macOS arm64, and Windows x86_64. Elsewhere pip builds the sdist, which needs Rust stable ≥ 1.83 and takes several minutes (it compiles tfhe-rs). Wheels include only the TFHE backend (see [Optional: CKKS Backend](#optional-ckks-backend)).

### From source (contributors)

Clone the repository and sync dependencies:

```bash
git clone https://github.com/Harikeshav-R/Penumbra-FHE.git
cd Penumbra-FHE
uv sync
```

`uv sync` creates a virtual environment at `.venv/`, downloads all Python dependencies, and compiles the native Rust PyO3 extension (`penumbra._penumbra`). The first build compiles `tfhe-rs` and takes several minutes.

## First Encrypted Inference

Here is a minimal, complete example constructing a small linear model, quantizing it with calibration data, and evaluating on encrypted input:

```python
import numpy as np
import penumbra as fhe
from penumbra.client import KeySet
from penumbra.quantization.spec import QuantSpec
from penumbra.reference import evaluate_graph_int

# 1. Create a model: Linear (8 features -> 3 outputs)
rng = np.random.default_rng(42)
weights = rng.normal(size=(3, 8))
bias = rng.normal(size=3)
model = fhe.Model([fhe.Linear(weight=weights, bias=bias)], input_bits=4)

# 2. Calibrate and quantize to integers
calibration_data = rng.uniform(0.0, 16.0, size=(64, 8))
model.quantize(calibration_data, n_bits=4)

# 3. Encrypted inference on a sample
x = rng.uniform(0.0, 16.0, size=8)
label, logits = model.predict_encrypted(x, return_logits=True)

# 4. Assert bit-for-bit agreement with the quantized-cleartext oracle
xq = QuantSpec(scale=model.input_scale, bits=model.input_bits, signed=False).quantize(x).tolist()
oracle = evaluate_graph_int(model.graph, {"x": xq})[model.graph.outputs[0]]
assert logits == oracle, f"Golden invariant failed: {logits} != {oracle}"

print(f"Predicted class: {label}")
print(f"Encrypted logits: {logits} (matches cleartext oracle bit-for-bit!)")
```

### Key Generation and Reuse

In the snippet above, `predict_encrypted` generates an ephemeral key pair on the fly. In production client/server setups or multi-sample batches, generate the key pair once and pass it explicitly:

```python
# Generate keys for the model's radix capacity
keys = KeySet.generate(model.graph.num_blocks)

# Re-use keys across multiple encrypted inferences
label, logits = model.predict_encrypted(x, keys=keys, return_logits=True)
```

## Client / Server Demo

To see a simulated untrusted server evaluating ciphertext without access to the secret key, run the client/server demo (requires a clone of the repository; see [From source](#from-source-contributors)):

```bash
uv run python examples/client_server/demo.py
```
## Run the examples

Each example directory includes a self-contained runner that replays the committed test fixtures
in-process under FHE (keygen → encrypt → evaluate → decrypt) and validates results against the
quantized-cleartext oracle (running these scripts requires a clone of the repository; see [From source](#from-source-contributors)):

```bash
# MNIST digit classification (CNN, logistic regression, MLP):
uv run python examples/mnist/run.py

# Olivetti face recognition CNN:
uv run python examples/faces/run.py

# Wisconsin Breast Cancer tabular MLP:
uv run python examples/tabular/run.py

# Wisconsin Breast Cancer tree ensembles (Random Forest, XGBoost):
uv run python examples/trees/run.py
```

See the example directories for detailed walkthroughs, architecture diagrams, and reproduction commands:
- [MNIST Examples](https://github.com/Harikeshav-R/Penumbra-FHE/tree/main/examples/mnist)
- [Faces Example](https://github.com/Harikeshav-R/Penumbra-FHE/tree/main/examples/faces)
- [Tabular MLP Example](https://github.com/Harikeshav-R/Penumbra-FHE/tree/main/examples/tabular)
- [Tree Ensembles Example](https://github.com/Harikeshav-R/Penumbra-FHE/tree/main/examples/trees)


## Optional: CKKS Backend

Penumbra also includes a CKKS backend based on `poulpy-ckks` for approximate real-number arithmetic (SIMD packed into a single ciphertext).

Building the CKKS backend requires a nightly Rust toolchain:

```bash
RUSTUP_TOOLCHAIN=nightly uv run --with "maturin>=1.9,<2.0" maturin develop --release --features ckks
```

Once built with CKKS feature support, pass `backend="ckks"` to encrypted inference:

```python
label, logits = model.predict_encrypted(x, backend="ckks", return_logits=True)
```

See [FHE Backends](BACKENDS.md) and [TFHE vs CKKS Comparison](COMPARISON.md) for details on trade-offs between exact integer arithmetic and approximate real arithmetic.
