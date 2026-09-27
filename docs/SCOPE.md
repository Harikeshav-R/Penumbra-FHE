# Scope & Expectations

This document defines the operating scope, design boundaries, and realistic performance expectations for Penumbra-FHE.

---

## What "Any ONNX Model" Means

In [`PROJECT.md`](https://github.com/Harikeshav-R/Penumbra-FHE/blob/main/PROJECT.md), the project sets a narrow-waist design goal:

> Any ONNX model **composed of supported operators**, that **quantizes acceptably**, and is **small enough to be practical**.

"Any ONNX model" does not mean arbitrary ONNX computational graphs. ONNX defines over 150 operators across multiple domains; Penumbra-FHE intentionally implements a targeted subset that covers standard classification architectures (linear models, multi-layer perceptrons, convolutional neural networks, and tree ensembles).

To run on Penumbra-FHE, a model must satisfy the following enforced conditions:

1. **Supported operators:** Every node in the graph must map directly to an entry in the [Supported operators table](SUPPORTED-OPS.md). Unrecognized operators are rejected immediately at load time.
2. **DAG topology:** The graph must form a strict Directed Acyclic Graph with exactly one input tensor and one output tensor ([Notes on ONNX models](SUPPORTED-OPS.md#notes--phase-6)). This single-input/single-output constraint is enforced at model load and by the runtime session bridge (`crates/penumbra-py/src/session.rs`).
3. **ONNX opset range:** The model must be exported with an `ai.onnx` opset version between 11 and 22 (`python/penumbra/op_registry.py`). Custom operator domains outside `ai.onnx` are rejected ([Domain limits](SUPPORTED-OPS.md#notes--phase-6)).
4. **Acceptable quantization:** Post-`Requant` activations are capped at $2$ bits (`MESSAGE_BITS = 2`, the radix capacity limit under TFHE). The model must maintain acceptable utility under Post-Training Quantization (PTQ) or Quantization-Aware Training (QAT) ([Quantization](QUANTIZATION.md)).
5. **Backend resource budgets:**
   - **TFHE:** The graph's maximum bit-width must fit within the radix capacity ($num\_blocks \times 2$ bits) ([The bit-width budget](QUANTIZATION.md#the-bit-width-budget-why-requantization-exists)).
   - **CKKS:** Multiplicative depth and scale precision must fit within the 330-bit level budget ($k = 360, \log\Delta = 30$). Chained sharp comparisons (tree ensembles) and non-polynomial operations like `Pool(max)` exceed this level budget and are rejected at load time ([Backend support](SUPPORTED-OPS.md#backend-support)).

All operator and topology checks occur during `penumbra.load_onnx()`. If a model violates constraints, the loader fails loudly and lists **all** unsupported operators and validation errors at once.

---

## What It Is Not

Penumbra-FHE is purpose-built for privacy-preserving ML inference. To preserve architectural clarity, several non-goals are strictly enforced:

- **Not an optimizing compiler:** Penumbra does not feature a polyhedral loop optimizer, automatic tiling engine, or general-purpose graph rewrite engine. Optimization is centered on op fusion (e.g. ReLU into `Requant`) and bit-width tracking.
- **Not a general FHE framework:** Penumbra does not provide a general homomorphic programming language or arbitrary arithmetic circuit evaluator. It evaluates fixed ML inference graphs.
- **No homomorphic training:** Training under FHE requires continuous bootstrapping and severe depth penalties. All training occurs in cleartext (e.g. PyTorch, scikit-learn, XGBoost) before export to ONNX.
- **No LLMs or Transformers:** Large language models, multi-head self-attention, and auto-regressive decoding loops are outside the design scope.
- **Not real-time serving:** Homomorphic evaluations require milliseconds to seconds per layer. It is not designed for sub-millisecond real-time web services.

---

## Latency Expectations

Homomorphic operations incur substantial computational overhead compared to plaintext floating-point execution. The following table records wall-clock execution time per sample across committed example models:

| Model | Architecture | TFHE `classic` Eval Total | CKKS Eval Total | TFHE Speedup / Ratio |
|---|---|---:|---:|---:|
| `phase2_logreg` | `Linear → Argmax` | 0.52 s | 0.65 s | 0.8× |
| `phase4_cnn` | `Conv2d → Requant → Pool → Linear` | 27.15 s | 0.66 s | 41.1× |
| `phase5_digits` | `Conv2d → Requant → Linear` | 222.33 s | 1.37 s | 162.6× |
| `phase5_qat` | `Conv2d → Requant → Linear` | 178.93 s | 1.17 s | 152.9× |
| `phase6_onnx` | `Conv2d → Requant → Linear` | 217.97 s | 1.35 s | 162.1× |
| `phase6_sklearn` | `Linear` | 40.56 s | 0.51 s | 79.8× |
| `phase7_faces` | `Conv2d → Requant → Linear` | 370.55 s | 2.30 s | 161.1× |
| `phase8_trees` | `Compare → Linear → Compare → Linear` | ~10.40 s | *Rejected at load* | — |

*Measured on Apple M3 Pro, macOS 25.6.0, `rustc 1.100.0-nightly (bba531001 2026-09-20)`, HAL backend `FFT64Neon` (`poulpy-ckks 0.8.3`), commit `9b38c1b`, 2026-09-24, `--samples 2` (see [`docs/BENCHMARKS.md`](BENCHMARKS.md)).*

### Key Generation and Memory Overhead

- **TFHE (`classic` profile):**
  - Keygen latency: ~0.50 s
  - Client secret key: ~0.03 MB
  - Server evaluation key: 114.84 MB
- **CKKS (`poulpy-ckks` default profile):**
  - Keygen latency: ~2.18 s
  - Client secret key: ~0.52 MB
  - Server evaluation key: 1782.50 MB (~1.78 GB, containing Galois automorphism keys for BSGS diagonal rotations)

### Rules of Thumb

- **TFHE cost:** Dominated by the total count of Programmable Bootstrapping (PBS) operations (`Requant`, non-linear `Activation` LUTs, and `Compare`). Linear matrix-vector multiplies without intermediate activations execute orders of magnitude faster than bootstrapped operations.
- **CKKS cost:** Dominated by multiplicative depth (number of levels consumed) and vector rotations needed for BSGS diagonal transforms.
- **Release builds:** Compiling the Rust runtime in `--release` mode is mandatory. Unoptimized debug builds run roughly 50× to 100× slower.
- **Initial build duration:** The initial `uv sync` compiles `tfhe-rs` from source, which can take several minutes on multi-core systems.

For full benchmarks and optimization options, see [Performance](PERFORMANCE.md) and [Benchmarks](BENCHMARKS.md).

---

## Accuracy Expectations

Quantization inevitably introduces rounding and representation error compared to 32-bit floating-point models:

- Every model in the [Model Zoo](MODEL-ZOO.md) reports both float accuracy and quantized-cleartext accuracy.
- While simple models maintain parity, aggressive low-bit integer activations can result in noticeable accuracy gaps:
  - **Olivetti Faces (`phase7_faces`):** Float accuracy $0.950 \rightarrow$ Quantized accuracy $0.900$ (a $0.050$ gap from 2-bit activation capping on 16×16 inputs).
  - **Global Average Pooling CNN (`phase8_gap_cnn`):** Float accuracy $0.833 \rightarrow$ Quantized accuracy $0.561$ (accumulated integer truncation across cascading average-pooling stages).

---

## Backend Comparison & Parity

Penumbra-FHE supports two cryptographic backends evaluated against the exact same quantized-cleartext reference:

1. **TFHE (`penumbra-tfhe`):**
   - Exact bit-for-bit equivalence with integer cleartext evaluation.
   - Built on `tfhe-rs` (Zama).
   - Included in standard wheels and default `uv sync` installations.
2. **CKKS (`penumbra-ckks`):**
   - Approximate real-number arithmetic evaluated over SIMD-packed slots.
   - Built on `poulpy-ckks` (Phantom Zone).
   - Requires a nightly Rust toolchain and must be built from source (`--features ckks`). It is omitted from published wheels.
   - Evaluates within declared per-model error bounds calibrated in `crates/penumbra-ckks/src/bounds.rs`.

Key material, ciphertexts, and parameters are **not portable** between backends. Loading keys generated for one backend into another produces an actionable load-time error.

---

## Maturity & Status

Penumbra-FHE is **research- and prototype-grade software**. It has not been audited by third-party cryptography specialists and should not be used in critical production environments handling sensitive real-world secrets. For the complete threat model and parameter security claims, see [Security](security.md).
