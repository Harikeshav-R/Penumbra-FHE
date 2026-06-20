"""Penumbra-FHE — encrypted ML inference on ONNX models via TFHE.

Load any supported ONNX model, quantize it with the library's quantization service,
lower it to the Intermediate Representation (IR), and run inference under Fully
Homomorphic Encryption — without writing any cryptography code.

This is the Python front end (Layer 3 + quantization + ONNX loader + IR emitter). The
TFHE backend (Layers 1 + 2) lives in the ``runtime/`` Rust crate; the two are bridged by
the IR file format (``ir.py`` <-> ``runtime/src/ir.rs``). See ``PROJECT.md`` §4, §13.

The public API is intentionally small (``PROJECT.md`` §12)::

    import penumbra as fhe

    model = fhe.load_onnx("model.onnx")
    model.quantize(calibration_data, n_bits=6)
    model.compile()
    model.export("model.fhe")
    pred = model.predict_encrypted(x)

This package is currently a scaffold (ROADMAP.md Phase 0); the API above is built out in
later phases.
"""

__version__ = "0.0.0"

__all__ = ["__version__"]

# TODO(phase-6): re-export `load_onnx`.
# TODO(phase-3): re-export IR builders (`Model`, `Conv2d`, `Linear`, `Activation`, ...).
