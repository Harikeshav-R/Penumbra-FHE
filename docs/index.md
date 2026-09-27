# Penumbra-FHE

Export any supported model to **ONNX**, load it into Penumbra-FHE, and run inference directly on **encrypted data** using Fully Homomorphic Encryption (FHE). The server computes on ciphertext and never sees your input or output.

```python
import penumbra as fhe

model = fhe.load_onnx("model.onnx")           # ONNX front door: parse + validate + lower to a Model
model.quantize(calibration_data, n_bits=6)   # float graph → int graph + lookup tables
pred = model.predict_encrypted(x)             # client encrypts → server evaluates → client decrypts
```

## The Golden Invariant

> **Encrypted output must match the quantized-cleartext output: for TFHE, bit-for-bit; for CKKS, within the model's declared error bound.**

Every backend is judged against the exact same quantized-cleartext integer forward pass (`evaluate_graph_int`).
- **TFHE is exact:** Any discrepancy between the encrypted evaluation and the quantized-cleartext oracle is a quantization or implementation bug — never cryptographic noise.
- **CKKS is approximate:** Evaluates real-valued polynomial approximations within calibrated, committed per-model error bounds. Exceeding a declared bound is treated as a bug (scale, level, or polynomial degree) rather than accepted noise.

!!! warning "Status: Research / Prototype Grade"
    Penumbra-FHE is a research library and prototype, not audited production cryptography software. In-process encrypted inference latency is seconds to minutes per sample; this is not real-time serving. See [SECURITY.md](https://github.com/Harikeshav-R/Penumbra-FHE/blob/main/SECURITY.md) and [PROJECT.md](https://github.com/Harikeshav-R/Penumbra-FHE/blob/main/PROJECT.md) for architecture, scope, and security considerations.

## Where to Go Next

- **[Getting started](getting-started.md):** Prerequisites, source installation, and your first encrypted inference.
- **[Tutorial: MNIST end to end](tutorial-mnist.md):** Train a CNN, export to ONNX, quantize, and verify against cleartext and encrypted oracles.
- **Guides:**
    - [Supported ops](SUPPORTED-OPS.md): Operator support matrix across ONNX, TFHE, and CKKS.
    - [Quantization](QUANTIZATION.md): Calibration, bit-width tracking, and LUT generation.
    - [Performance](PERFORMANCE.md): PBS bottlenecks, depth budgets, and optimization passes.
    - [Backends](BACKENDS.md): Backend boundaries, TFHE vs CKKS, and adding backends.
- **[API reference](api.md):** Full Python API reference for models, layers, IR, and encrypted client execution.
- **[Architecture](architecture.md):** The three layers, two narrow waists, and graph evaluation flow.
