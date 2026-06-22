# Tests

Cross-cutting tests for Penumbra-FHE. Per-op Rust unit tests live in `runtime/`; this
directory holds the Python-side and cross-language tests.

## The golden exactness test (sacred — `AGENTS.md` §1.1)

> FHE output must equal the quantized-cleartext output, **bit-for-bit**.

TFHE is exact, so any discrepancy is a quantization or implementation bug, never crypto
noise. This is the project's truth oracle — it is wired into CI from Phase 2 onward and
must never regress.

## Planned tests

- `test_quantized_vs_fhe.py` — the golden exactness invariant across all models (Phase 2+).
- Cross-language IR conformance — Python emits IR, Rust loads it, assert agreement
  (Phase 3; keeps `ir.py` ↔ `runtime/src/ir.rs` in lockstep, `AGENTS.md` §5).
- Unsupported-op failure — feed an unsupported ONNX model, assert a loud, actionable
  load-time error (Phase 6).
- Property/fuzz — random small models → assert FHE == quantized-cleartext (Phase 11).
