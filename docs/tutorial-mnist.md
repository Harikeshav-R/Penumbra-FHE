# Tutorial: MNIST Digit Classification End-to-End

This tutorial guides you through loading a real convolutional neural network exported to ONNX, quantizing it into integer arithmetic with fixed-point rescales, and running encrypted inference.

The workflow reproduces the committed Phase 6 ONNX digit CNN fixture (`examples/mnist/digit_cnn.onnx`) and verifies predictions against cleartext and encrypted oracles.

## The End-to-End Pipeline

Here is the complete script. Run it from the repository root:

```bash
uv run --with scikit-learn python - << 'EOF'
import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import penumbra as fhe
from penumbra.reference import evaluate_graph_int

# 1. Load the scikit-learn 8x8 digits dataset (values 0..16)
digits = load_digits()
x = digits.images.reshape(len(digits.images), -1).astype(np.float64)   # (1797, 64)
x_tr, x_te, y_tr, y_te = train_test_split(
    x, digits.target, test_size=0.2, random_state=0, stratify=digits.target
)

# 2. Load the trained ONNX model
model = fhe.load_onnx("examples/mnist/digit_cnn.onnx", input_bits=3)

# 3. Post-Training Quantization (PTQ) with MSE calibration
graph = model.quantize(
    x_tr,
    n_bits=[5, 6],
    act_bits=2,
    per_channel=True,
    max_mult_bits=1,
    calibration="mse",
)

# 4. Measure quantized-cleartext accuracy over the test set
q = np.clip(np.round(x_te / model.input_scale), 0, 7).astype(np.int64)
preds = [
    int(np.argmax(evaluate_graph_int(graph, {"x": r.tolist()})[graph.outputs[0]]))
    for r in q
]
accuracy = np.mean(np.array(preds) == y_te)
print(f"Quantized-cleartext accuracy: {accuracy:.4f}")

# 5. Encrypted inference on sample 0
label, logits = model.predict_encrypted(x_te[0], return_logits=True)
oracle_logits = evaluate_graph_int(graph, {"x": q[0].tolist()})[graph.outputs[0]]

assert logits == oracle_logits, "Golden invariant violated!"
print(f"Sample 0 prediction: class {label} (logits: {logits})")
print("Encrypted output matches quantized-cleartext oracle bit-for-bit!")
EOF
```

## Step-by-Step Walkthrough

### 1. ONNX Model Loading and Validation (`load_onnx`)

Penumbra-FHE uses ONNX as its primary model interchange format. When you call `fhe.load_onnx("examples/mnist/digit_cnn.onnx", input_bits=3)`, the loader:
- Traverses the ONNX computational graph;
- Validates every operator against the supported op registry (`python/penumbra/op_registry.py`);
- Fails loudly with an `UnsupportedModelError` listing all unsupported operators and nodes if any exist;
- Folds constant initializers and batch normalization parameters into convolution weights and biases;
- Lowers the validated ONNX graph into a Python `Model` abstraction.

### 2. Quantization Service (`model.quantize`)

FHE schemes operate on integers or polynomials rather than floating-point values. `model.quantize` performs Post-Training Quantization (PTQ) using calibration data:
- **Weights & Biases:** Quantized to integer values (5 bits for convolution kernels, 6 bits for dense layer weights).
- **Per-Channel Scales:** With `per_channel=True`, each convolution kernel carries its own fixed-point scale multiplier.
- **Requant Insertion:** Wide accumulator values are narrowed down to narrow representations using fused `Requant` operations (shift + multiply + clamp LUT).
- **Radix Sizing:** Analyzes peak bit-widths across all nodes and determines the minimal `num_blocks` needed to avoid accumulator overflow without wasting radix capacity.

Verification confirms that this quantization pipeline produces the exact graph committed in `examples/mnist/phase6_onnx_fixture.json["graph"]`, achieving **91.67% accuracy** on the quantized test set.

### 3. Cleartext Integer Oracle (`evaluate_graph_int`)

The function `evaluate_graph_int` in `penumbra.reference` evaluates the integer IR graph using exact integer arithmetic. It represents the ground-truth specification:
- Any divergence between TFHE encrypted evaluation and `evaluate_graph_int` is a bug, never cryptographic noise.
- Every encrypted evaluation asserts exact equality against this oracle.

### 4. Encrypted Inference and Performance

During `model.predict_encrypted(x_te[0])`:
1. Client generates keys for `model.graph.num_blocks` (or reuses provided keys);
2. Client quantizes input `x` using `model.input_scale` and encrypts each integer into ciphertext blocks;
3. Server evaluates the graph homomorphically (convolutions, requantization bootstraps, dense layers);
4. Client decrypts the result and takes the argmax.

#### Latency: TFHE vs CKKS

- **TFHE Backend (`penumbra-tfhe`):** Evaluates exact 2-bit radix blocks. For `phase6_onnx`, evaluating 12 convolution channels over 3×3 spatial positions requires 108 Programmable Bootstrapping (PBS) operations per sample. As reported in `docs/BENCHMARKS.md`, execution takes approximately **218 seconds per sample** on Apple Silicon M-series hardware.
- **CKKS Backend (`penumbra-ckks`):** Evaluates SIMD-packed real-number polynomial approximations. Because convolutions and linear projections collapse into parallel slot multiplications without bootstrapping, inference executes in approximately **1.3 seconds per sample** (a ~160× speedup). To use CKKS, install the nightly extension as described in [Getting Started](getting-started.md) and pass `backend="ckks"`.

## Training Your Own Models

To train and export your own models for Penumbra:
- Inspect `examples/mnist/onnx_export.py` to see the PyTorch training loop, batch normalization folding, and `torch.onnx.export` invocation.
- For scikit-learn models, inspect `examples/mnist/sklearn_export.py` which uses `skl2onnx`.
- For gradient-boosted trees, inspect `examples/trees/xgb_export.py`.
