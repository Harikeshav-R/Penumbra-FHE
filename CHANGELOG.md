# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The Python package and the four published crates share one version, and the IR `schema_version` is versioned separately (currently `0.10.0`).

## [Unreleased]

### Added
- Phase 16 full comparison run artifacts (`docs/results/`): final 28-row matrix evaluation (`phase16-paper-final.json`), raw 28×28 scale probe (`phase16-mnist28-probe.json`), post-fix TFHE diagnostics (`phase16-tfhe-diagnostics.json`), D17 before/after within-scheme vs cross-scheme analysis (`phase16-tfhe-before-after.json`), external Concrete-ML calibration (`phase16-concrete-calibration.json`), and execution manifest (`phase16-run-manifest.json`).
- Separate raw 28×28 MNIST scale experiment (Phase 16): a frozen Layer-3 CNN fixture and `penumbra-mnist-scale-probe` command verify TFHE logits against the integer reference and observe the existing CKKS input-capacity rejection. The probe is not a member of the controlled 14-model suite.
- Paper benchmark evaluation protocol (Phase 15): added `examples/paper_protocol.py` establishing canonical evaluation datasets, frozen input quantization, and exact reference outputs across all 14 model fixtures under a root `paper` schema (version 1).
- Registered `phase8_tanh` and `phase8_xgb` in `penumbra-bench` model fixtures.
- Added `Backend::build_op_with_bits` default-implemented trait method on `penumbra_core::backend::Backend`, allowing Layer 2 to pass derived input/output tensor bit widths to backends during topological evaluation.

### Changed
- Unified benchmark harness protocol (`penumbra-bench`): added `--mode paper|calibrate|diagnostics|security-inputs`, multi-stage worker isolation for server peak RSS capture (`getrusage`), canonical Criterion latency parsing with confidence intervals, and D16 logical lookup vs carry PBS breakdown.
- Paper report format version 2: preserve original-graph node cost profiles and their actual sample IDs, and retain CKKS graph-budget rejection messages before key generation. Input fixture and protocol versions remain 1; the IR is unchanged.
- TFHE per-tensor radix widths (Phase 14): sized each tensor's radix representation to its Layer-2 derived bit width with progressive widening and deferred carry propagation in linear operations. Yields a 2.37x overall speedup across the test suite (up to 3.27x on branching networks) and a 56.6% reduction in PBS operations, while preserving bit-for-bit exactness.

### Fixed
- Paper accuracy metrics now select the first tied maximum, reject non-finite logical outputs, ignore CKKS padding, and distinguish binary decision-output error from score-tap error relative to the decision threshold. Original graphs remain the Criterion/RSS measurement surface.
- Reuse the first CKKS multiclass evaluation for both its representative profile and accuracy, rather than evaluating the identical graph twice.

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
