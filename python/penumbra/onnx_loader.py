"""The ONNX front door: parse -> validate -> quantize -> lower to IR.

ONNX is the universal export format (PyTorch, sklearn, Keras, XGBoost all emit it):
"train anywhere, run encrypted here" (``PROJECT.md`` §10).

What ``load_onnx()`` does:
    1. Parse the ONNX graph (``onnx`` package): nodes, initializers, attributes.
    2. Validate every node against ``op_registry`` -> fail loudly at load time, listing
       *all* unsupported ops at once (``AGENTS.md`` §1.4).
    3. Quantize to int weights + scales + LUTs (via the ``quantization`` service).
    4. Lower the validated ONNX graph to the internal IR (``ir.py``).
    5. Hand the IR to the runtime for encrypted eval.

"Any ONNX model" is bounded: only supported ops, only models that quantize acceptably,
only sizes that run in reasonable time (``PROJECT.md`` §10, §16). Be precise about this.

TODO(phase-6): implement `load_onnx(path)` returning a compilable model object.
"""
