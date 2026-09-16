# penumbra-fhe (Python front end)

The Python half of [Penumbra-FHE](https://github.com/Harikeshav-R/Penumbra-FHE): run
inference on **encrypted** data, without writing cryptography code.

This package is **Layer 3** of the project — it loads an ONNX model, quantizes it to a low-bit
integer graph, and emits a serializable IR. The encrypted forward pass itself runs in the Rust
runtime, over a pluggable FHE backend.

```python
import penumbra as fhe

model = fhe.load_onnx("model.onnx")          # parse + validate + lower to a Model
model.quantize(calibration_data, n_bits=6)   # float graph -> int graph + lookup tables
pred = model.predict_encrypted(x)            # client encrypts -> server evaluates -> client decrypts
```

`load_onnx` validates every operator at load time and fails loudly, listing **all** problems at
once, if a model uses anything outside the supported subset. `model.export("model.fhe")` writes
the IR for the runtime.

## What you need

- **For loading, quantizing, and exporting IR:** this package alone (`onnx` + `numpy`).
- **For `predict_encrypted`:** a Rust toolchain (`cargo`) and a checkout of the repository —
  the current bridge shells out to the runtime. Expect seconds-to-minutes per sample; this is
  research/prototype-grade software, not real-time serving.
- **For the example generators:** the optional `ml` extra (`torch`, `brevitas`,
  `scikit-learn`).

## Documentation

Everything lives in the [repository](https://github.com/Harikeshav-R/Penumbra-FHE):
`PROJECT.md` for the architecture, `docs/SUPPORTED-OPS.md` for the operator list,
`docs/QUANTIZATION.md` for the quantization service, and `docs/DEVELOPMENT.md` for setup.

Licensed under Apache 2.0.
