# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The Python package and the four published crates share one version, and the IR `schema_version` is versioned separately (currently `0.10.0`).

## [Unreleased]

### Added
- `Backend::build_op_with_bits` default-implemented trait method on `penumbra_core::backend::Backend`, allowing Layer 2 to pass derived input/output tensor bit widths to backends during topological evaluation.

### Changed
- TFHE per-tensor radix widths (Phase 14): sized each tensor's radix representation to its Layer-2 derived bit width with progressive widening and deferred carry propagation in linear operations. Yields a 2.37x overall speedup across the test suite (up to 3.27x on branching networks) and a 56.6% reduction in PBS operations, while preserving bit-for-bit exactness.
## [1.0.0] - 2026-09-27

### Added — Python front end
- `load_onnx` ONNX front door. It validates every node at load time and supports opset 11–22. Supported ONNX ops: Add, ArgMax, AveragePool, BatchNormalization, Cast, Concat, Conv, Elu, Flatten, Gelu, Gemm, GlobalAveragePool, HardSigmoid, HardSwish, LeakyRelu, LogSoftmax, MatMul, MaxPool, Relu, Reshape, Sigmoid, Softmax, Split, Tanh, Transpose.
- `Model` builder layers (`Conv2d`, `Linear`, `Pool`, `Activation`, `Add`, `Concat`, `Split`).
- Quantization service: PTQ with calibration (incl. `calibration="mse"`), per-channel scales, QAT via Brevitas (`ml` extra), BatchNorm folding, automatic `Requant` insertion, and bit-width minimization.
- Tree-ensemble adapters `penumbra.adapters.from_sklearn` and `penumbra.adapters.from_xgboost`.
- `predict_encrypted` and `run_encrypted` in-process encrypted inference.
- `KeySet` key persistence and reuse.
- `CryptoProfile.tfhe(...)` and `CryptoProfile.ckks(...)`.
- The quantized-cleartext oracle `evaluate_graph_int`.

### Added — Rust runtime
- `penumbra-core`: backend-neutral IR (schema 0.10.0), topological eval loop, bit-width propagation, and the `Backend` trait.
- Internal op vocabulary: Linear, Conv2d, Activation, Argmax, Compare, Requant, Pool (avg/max), Add, Concat, Split.
- `penumbra-tfhe` on tfhe-rs, exact, with profiles classic/gaussian/multibit2/multibit3/multibit4.
- `penumbra-ckks` on poulpy-ckks `=0.8.3`, feature `ckks`, nightly, with `max_poly_degree` as its single knob. `Pool(max)` is rejected loudly.
- `penumbra-fhe-runtime` CLIs `keygen`, `encrypt`, `serve`, `decrypt`, `predict`, `inspect`.
- Scheme-tagged key/ciphertext wire format. Cross-backend material is rejected.
- Shared comparison harness `penumbra-bench` (unpublished).

### Added — Examples & docs
- `examples/{mnist,faces,tabular,trees,client_server}` with one-command runners.
- MkDocs site.
- `SECURITY.md` threat model.
- `docs/SCOPE.md`.
- TFHE-vs-CKKS comparison (`docs/COMPARISON.md`, `docs/BENCHMARKS.md`).

### Packaging
- abi3 wheels for CPython 3.10–3.12 on Linux x86_64 (manylinux2014), macOS arm64, and Windows x86_64, plus an sdist.
- Wheels ship the TFHE backend only.
- Crates `penumbra-core`, `penumbra-tfhe`, `penumbra-ckks`, `penumbra-fhe-runtime`.

[Unreleased]: https://github.com/Harikeshav-R/Penumbra-FHE/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Harikeshav-R/Penumbra-FHE/releases/tag/v1.0.0
