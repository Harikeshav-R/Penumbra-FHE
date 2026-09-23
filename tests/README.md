# Tests

Cross-cutting tests for Penumbra-FHE. Per-op Rust unit tests and the FHE golden tests live in
`runtime/tests/`; this directory holds the Python-side and cross-language tests.

Run them with `uv run pytest` (pytest's `testpaths` points here).

## The golden test (sacred — `AGENTS.md` §1.1)

> Encrypted output must match the quantized-cleartext output: **bit-for-bit under TFHE,
> within the declared error bound under CKKS.**

The reference never changes — `python/penumbra/reference.py`'s `evaluate_graph_int`. Only the
comparator is per-backend, which is what keeps the two backends comparable
([`docs/BACKENDS.md`](../docs/BACKENDS.md)).

TFHE is exact, so any discrepancy there is a quantization or implementation bug, never crypto
noise. This is the project's truth oracle — wired into CI from Phase 2 onward, and it must
never regress. CKKS is approximate, so its gate is a committed per-model error bound with the
measured error always reported; a CKKS backend is **never** compared against a friendlier
reference to flatter its numbers.

## Current tests

### The invariant and its oracle

- `test_quantized_vs_fhe.py` — the golden invariant across models: the committed fixtures'
  `expected_labels` match the quantized-int reference, and the graphs fit the bit-width budget.
- `test_reference.py` — pins the integer oracle itself (Requant fixed-point and per-channel
  formulas, loud failures on bad input). Everything else trusts this file, so it is tested
  directly.

### Cross-language lockstep

- `test_ir_conformance.py` — IR conformance, Python half: the IR round-trips and the committed
  fixture graph is exactly what `ir.py` emits (the drift guard). Rust half:
  `runtime/tests/ir_conformance.rs`. Together they keep `ir.py` ↔ the Rust `ir.rs` in lockstep
  (`AGENTS.md` §5). See [`docs/IR-SPEC.md`](../docs/IR-SPEC.md).
- `test_bitwidth_conformance.py` — the Python bit-width rules reproduce every case in
  `fixtures/bitwidth_cases.json`. Rust half: `runtime/tests/bitwidth_conformance.rs`.
- `test_supported_ops_doc.py` — parses the ONNX mapping table out of
  [`docs/SUPPORTED-OPS.md`](../docs/SUPPORTED-OPS.md) and asserts it equals the registry
  exactly, so the documented op list cannot drift from what the validator accepts.

### The ONNX front door

- `test_onnx_loader.py` — in-memory ONNX models lower to the expected layers (Gemm, MatMul+Add
  bias folding, Conv, pooling, Cast/Transpose folding, terminal Softmax drop).
- `test_onnx_unsupported.py` — the loud-failure gate: unsupported ops, branching, opset range,
  grouped conv, and friends, with **all** problems reported at once (`AGENTS.md` §1.4).
- `test_onnx_fidelity.py` — differential check against **onnxruntime** on the original
  `.onnx`, catching lowering bugs the golden invariant structurally cannot see. Skipped when
  `onnxruntime` is absent (including in CI).

### Quantization service

- `test_quantization_ptq.py` — observers, per-layer quantizers, and `choose_requant_params`.
- `test_quantization_lut.py` · `test_quantization_accuracy.py` — LUT generation against the
  single-block contract; accuracy/SQNR reporting.
- `test_model_quantize.py` · `test_compile.py` — `Model.quantize` end to end, and automatic
  `Requant` insertion (placement, idempotency, loud over-budget errors naming the layer).

### Fixtures and the encrypted round trip

- `test_cnn_fixture.py` · `test_real_digits_fixture.py` · `test_qat_fixture.py` ·
  `test_onnx_fixture.py` · `test_sklearn_fixture.py` · `test_faces_fixture.py` — fast NumPy
  guards on the committed fixtures, so every CI run checks self-consistency even though the
  matching FHE golden tests are `#[ignore]`d (minutes per sample).
- `test_predict_bridge.py` · `test_key_management.py` — the Python→Rust bridge and `KeySet`,
  with fast fakes by default. The real-FHE cases run only under `PENUMBRA_E2E=1`.

## Planned tests

- **CKKS tolerance gate** (Phase 12.2) — the CKKS backend's correctness test: assert the
  decrypted output is within the model's declared error bound of the same integer oracle, and
  report the measured error distribution rather than a pass/fail alone.
- **Backend parity** (Phase 12.3) — assert both backends consume the identical IR file and run
  the same node list, so a measured difference is attributable to the scheme.
- **Property/fuzz** — random small models → assert the invariant at each backend's comparator
  (Phase 11).
