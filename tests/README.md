# Tests

Cross-cutting tests for Penumbra-FHE. Per-op Rust unit tests live in `runtime/`; this
directory holds the Python-side and cross-language tests.

## The golden exactness test (sacred — `AGENTS.md` §1.1)

> FHE output must equal the quantized-cleartext output, **bit-for-bit**.

TFHE is exact, so any discrepancy is a quantization or implementation bug, never crypto
noise. This is the project's truth oracle — it is wired into CI from Phase 2 onward and
must never regress.

## Current tests

- `test_quantized_vs_fhe.py` — the golden exactness invariant across all models (Phase 2+);
  recovers the model parameters from the embedded IR graph (`fx["graph"]`).
- `test_ir_conformance.py` — cross-language IR conformance, Python half: the IR round-trips
  and the committed fixture graph is exactly what `ir.py` emits (the drift guard). The Rust
  half is `runtime/tests/ir_conformance.rs`. Together they keep `ir.py` ↔ `runtime/src/ir.rs`
  in lockstep (`AGENTS.md` §5). See [`docs/IR-SPEC.md`](../docs/IR-SPEC.md).

## Planned tests

- Unsupported-op failure — feed an unsupported ONNX model, assert a loud, actionable
  load-time error (Phase 6).
- Property/fuzz — random small models → assert FHE == quantized-cleartext (Phase 11).
