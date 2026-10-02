# Model zoo — validated examples

Every controlled-suite row is a committed fixture whose FHE output is gated against `evaluate_graph_int`. TFHE is bit-for-bit. CKKS is within the named bound, or rejected loudly at load. Accuracies are the fixture's recorded test-split numbers. The separately labelled MNIST28 scale probe is not a controlled-suite member; its executable enforces full-logit TFHE equality on every run.

## Validated models

| Fixture | Generator | Architecture | Dataset | IR ops | Float acc | Quantized acc | TFHE golden | CKKS | Bench key |
|---|---|---|---|---|---|---|---|---|---|
| `examples/faces/phase7_faces_fixture.json` | `examples/faces/olivetti_export.py` | CNN with Conv2d and Linear | Olivetti faces (8 identities) | `Conv2d → Requant → Linear` | 0.950 | 0.900 | `runtime/tests/golden_faces.rs` | `crates/penumbra-ckks/tests/ckks_golden_faces.rs` (`PHASE7_FACES`) | `phase7_faces` |
| `examples/mnist/phase2_fixture.json` | `examples/mnist/train_quantize_export.py` | Logistic regression with Argmax | synthetic 2-class digits | `Linear → Argmax` | 1.000 | 1.000 | `runtime/tests/golden_logreg.rs` | `crates/penumbra-ckks/tests/ckks_golden_logreg.rs` (`PHASE2_LOGREG`) | `phase2_logreg` |
| `examples/mnist/phase4_cnn_fixture.json` | `examples/mnist/cnn_export.py` | CNN with Conv, AveragePool, Linear | synthetic 10-class digits | `Conv2d → Requant → Pool → Linear` | 0.980 | 0.957 | `runtime/tests/golden_cnn.rs` | `crates/penumbra-ckks/tests/ckks_golden_cnn.rs` (`PHASE4_CNN`) | `phase4_cnn` |
| `examples/mnist/phase5_digits_fixture.json` | `examples/mnist/real_digits_export.py` | CNN with Conv and Linear (PTQ) | sklearn load_digits (8x8 digits) | `Conv2d → Requant → Linear` | 0.964 | 0.917 | `runtime/tests/golden_digits.rs` | `crates/penumbra-ckks/tests/ckks_golden_digits.rs` (`PHASE5_DIGITS`) | `phase5_digits` |
| `examples/mnist/phase5_qat_fixture.json` | `examples/mnist/qat_export.py` | Brevitas QAT CNN with Conv and Linear | sklearn load_digits (8x8 digits) | `Conv2d → Requant → Linear` | 0.933 | 0.936 | `runtime/tests/golden_qat.rs` | `crates/penumbra-ckks/tests/ckks_golden_qat.rs` (`PHASE5_QAT`) | `phase5_qat` |
| `examples/mnist/phase6_onnx_fixture.json` | `examples/mnist/onnx_export.py` | PyTorch CNN exported to ONNX | sklearn load_digits (8x8 digits) | `Conv2d → Requant → Linear` | 0.964 | 0.917 | `runtime/tests/golden_onnx.rs` | `crates/penumbra-ckks/tests/ckks_golden_onnx.rs` (`PHASE6_ONNX`) | `phase6_onnx` |
| `examples/mnist/phase6_sklearn_fixture.json` | `examples/mnist/sklearn_export.py` | scikit-learn linear classifier (MLPRegressor) | sklearn load_digits (8x8 digits) | `Linear` | 0.894 | 0.881 | `runtime/tests/golden_sklearn.rs` | `crates/penumbra-ckks/tests/ckks_golden_sklearn.rs` (`PHASE6_SKLEARN`) | `phase6_sklearn` |
| `examples/mnist/phase8_bn_cnn_fixture.json` | `examples/mnist/bn_cnn_export.py` | PyTorch CNN with folded BatchNorm and AveragePool | sklearn load_digits (8x8 digits) | `Conv2d → Requant → Pool → Linear` | 0.958 | 0.928 | `runtime/tests/golden_bn_cnn.rs` | `crates/penumbra-ckks/tests/ckks_golden_bn_cnn.rs` (`PHASE8_BN_CNN`) | `phase8_bn_cnn` |
| `examples/mnist/phase8_branch_fixture.json` | `examples/mnist/branch_mlp_export.py` | Branching PyTorch MLP with Split, Concat, and Add | sklearn load_digits (8x8 digits) | `Linear → Requant → Split → Linear → Requant → Linear → Requant → Concat → Add → Linear` | 0.967 | 0.906 | `runtime/tests/golden_branch_mlp.rs` | `crates/penumbra-ckks/tests/ckks_golden_branch_mlp.rs` (`PHASE8_BRANCH`, `max_poly_degree = 3`) | `phase8_branch` |
| `examples/mnist/phase8_gap_cnn_fixture.json` | `examples/mnist/gap_cnn_export.py` | PyTorch CNN with padded AveragePool and GlobalAveragePool | sklearn load_digits (8x8 digits) | `Conv2d → Requant → Pool → Pool → Linear` | 0.833 | 0.561 | `runtime/tests/golden_gap_cnn.rs` | `crates/penumbra-ckks/tests/ckks_golden_gap_cnn.rs` (`PHASE8_GAP_CNN`) | `phase8_gap_cnn` |
| `examples/mnist/phase8_tanh_fixture.json` | `examples/mnist/tanh_mlp_export.py` | PyTorch Tanh MLP | sklearn load_digits (8x8 digits) | `Linear → Requant → Activation → Linear` | 0.961 | 0.692 | `runtime/tests/golden_tanh_mlp.rs` | `crates/penumbra-ckks/tests/ckks_golden_tanh_mlp.rs` (`PHASE8_TANH`) | `phase8_tanh` |
| `examples/trees/phase8_trees_fixture.json` | `examples/trees/tree_export.py` | scikit-learn RandomForestClassifier lowered to Compare and Linear | Wisconsin Breast Cancer (30 features, 2 classes) | `Compare → Linear → Compare → Linear` | 0.956 | 0.956 | `runtime/tests/golden_trees.rs` | rejected at load (depth budget 360 > 330 bits): `crates/penumbra-ckks/tests/ckks_unsupported_ops.rs` | `phase8_trees` |
| `examples/trees/phase8_xgb_fixture.json` | `examples/trees/xgb_export.py` | XGBoost XGBClassifier lowered to Compare and Linear | Wisconsin Breast Cancer (30 features, 2 classes) | `Compare → Linear → Compare → Linear` | 0.965 | 0.965 | `runtime/tests/golden_trees.rs` | rejected at load (same Compare→Linear→Compare→Linear shape as phase8_trees, 360 > 330 bits) | `phase8_xgb` |
| `examples/tabular/phase11_tabular_mlp_fixture.json` | `examples/tabular/mlp_export.py` | PyTorch MLP (Gemm → ReLU → Gemm) exported to ONNX | Wisconsin Breast Cancer (30 features, 2 classes) | `Linear → Requant → Linear` | 0.965 | 0.956 | `runtime/tests/golden_tabular_mlp.rs` | `crates/penumbra-ckks/tests/ckks_golden_tabular_mlp.rs` (`PHASE11_TABULAR_MLP`) | `phase11_tabular_mlp` |
| `examples/mnist/phase16_mnist28_fixture.json` | `examples/mnist/mnist28_export.py` | **Separate scale probe:** raw 28×28 Conv 1→4, kernel 3, stride 4; Linear 196→10 | official MNIST (60,000 train / 10,000 test) | `Conv2d → Requant → Linear` | 0.905 | 0.871 | exact full-logit gate in `crates/penumbra-bench/src/bin/mnist_scale_probe.rs` | excluded under approved D18: 784 inputs exceed fixed 256-element linear-transform capacity | — |

### Separate raw MNIST28 scale probe

The controlled benchmark registry remains **14 models**. The raw 28×28 probe
encrypts all 784 pixels; stride-4 convolution occurs inside the encrypted graph,
not as external downsampling. D18 excludes it from backend-parity measurements
regardless of whether its one-sample TFHE server evaluation is ≤ 600 seconds.

The frozen fixture records official dataset checksums, training/calibration
indices, two held-out reference vectors, and full 10,000-example cleartext
accuracies. The probe binary checks full decrypted logits before emitting
evidence. Its server-evaluation duration is a feasibility observation, not
Criterion headline latency or a cross-scheme ratio.


## Regenerating

Each example script can be re-run with:

```bash
uv run --extra ml --system-certs python examples/faces/olivetti_export.py
uv run --extra ml --system-certs python examples/mnist/train_quantize_export.py
uv run --extra ml --system-certs python examples/mnist/cnn_export.py
uv run --extra ml --system-certs python examples/mnist/real_digits_export.py
uv run --extra ml --system-certs python examples/mnist/qat_export.py
uv run --extra ml --system-certs python examples/mnist/onnx_export.py
uv run --extra ml --system-certs python examples/mnist/sklearn_export.py
uv run --extra ml --system-certs python examples/mnist/bn_cnn_export.py
uv run --extra ml --system-certs python examples/mnist/branch_mlp_export.py
uv run --extra ml --system-certs python examples/mnist/gap_cnn_export.py
uv run --extra ml --system-certs python examples/mnist/tanh_mlp_export.py
uv run --extra ml --system-certs python examples/trees/tree_export.py
uv run --extra ml --system-certs python examples/trees/xgb_export.py
uv run --extra ml --system-certs python examples/tabular/mlp_export.py
uv run --extra ml python examples/mnist/mnist28_export.py
```

### Paper protocol evaluation data

The evaluation data, calibration splits, and full-test references under each fixture's `paper` object are regenerated with:

```bash
uv run --extra ml --system-certs python examples/paper_protocol.py --models all
```

### Split counts and baseline notes

- **Synthetic 2-class (`phase2_logreg`):** 400 train, 256 test (128 calibration, seed 1507).
- **Synthetic 10-class CNN (`phase4_cnn`):** 600 train, 256 test (128 calibration, seed 1507). Float accuracy (0.980) is a hybrid baseline where the linear head was trained/evaluated on quantized pooled features from the conv/requant stages.
- **Sklearn load_digits 8x8 (`phase5_digits`, `phase5_qat`, `phase6_onnx`, `phase6_sklearn`, `phase8_branch`, `phase8_bn_cnn`, `phase8_gap_cnn`, `phase8_tanh`):** 1437 train, 360 test (128 calibration, seed 1507; stratified 80/20 seed 0). For `phase5_qat`, float accuracy (0.933) is the historical committed fixture baseline (no tracked float checkpoint).
- **Olivetti faces (`phase7_faces`):** 60 train, 20 test (60 calibration, seed 1507; stratified 75/25 seed 0 across first 8 identities).
- **Wisconsin Breast Cancer (`phase8_trees`, `phase8_xgb`, `phase11_tabular_mlp`):** 455 train, 114 test (128 calibration, seed 1507; 80/20 seed 42).

Note that TFHE goldens for the digits, faces, and Phase-8 models are marked `#[ignore]` by default due to runtime duration. Run them with:

```bash
cargo test --workspace --release --test <test_name> -- --ignored
```

## Adding a model

To add a new validated model to the repository:

1. **Generator script:** write an export script in `examples/` that trains or defines the model, exports it (or lowers it), and calls `fmodel.quantize(...)`.
2. **Committed fixture:** emit `<model>_fixture.json` containing the serialized IR graph, quantization scales, bit plan, test inputs, expected logits, and expected labels. If the script generates an `.onnx` file loaded by tests, add a `.gitignore` exception `!<path>.onnx` and commit it.
3. **Fixture guard:** write `tests/test_<model>_fixture.py` asserting that the graph round-trips, fits its bit-width budget, `insert_requants` is idempotent, committed logits match `evaluate_graph_int`, and reported accuracy satisfies expected floors.
4. **TFHE golden test:** add `runtime/tests/golden_<model>.rs` verifying that TFHE evaluation reproduces the expected cleartext logits and labels bit-for-bit.
5. **CKKS golden test and bound:** add `crates/penumbra-ckks/tests/ckks_golden_<model>.rs`, declare a calibration-derived bound in `crates/penumbra-ckks/src/bounds.rs` (or write a load-time rejection test in `crates/penumbra-ckks/tests/ckks_unsupported_ops.rs` if unsupported by depth/scheme).
6. **Benchmark registry:** register the fixture in `crates/penumbra-bench/src/models.rs` `MODELS` so it is covered by backend parity and comparison harnesses.
7. **Model zoo row:** add the model's row to the table above.
