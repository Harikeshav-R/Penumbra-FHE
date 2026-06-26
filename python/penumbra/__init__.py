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

# Float layer builders (Layer 3). Imported here so the public API reads `fhe.Conv2d(...)`,
# matching the PROJECT.md §7 sketch. Named distinctly from the int `*Spec` IR payloads.
from penumbra import layers
from penumbra.bitwidth import (
    check_bit_width_budget,
    output_bits,
    propagate_bit_widths,
    radix_capacity_bits,
)
from penumbra.compile import insert_requants
from penumbra.ir import (
    SCHEMA_VERSION,
    ActivationSpec,
    AddSpec,
    ArgmaxSpec,
    Conv2dSpec,
    Graph,
    LinearSpec,
    Node,
    OpSpec,
    PoolSpec,
    RequantSpec,
    build_linear_argmax_graph,
)
from penumbra.layers import Activation, Conv2d, Linear, Pool, QuantConfig
from penumbra.model import Model
from penumbra.reference import evaluate_graph_int

__version__ = "0.0.0"

__all__ = [
    "__version__",
    # IR (the Python ↔ Rust bridge, PROJECT.md §7)
    "SCHEMA_VERSION",
    "Graph",
    "Node",
    "OpSpec",
    "LinearSpec",
    "Conv2dSpec",
    "ActivationSpec",
    "ArgmaxSpec",
    "RequantSpec",
    "PoolSpec",
    "AddSpec",
    "build_linear_argmax_graph",
    # Bit-width tracking + compile pass (Phase 4, PROJECT.md §9)
    "output_bits",
    "propagate_bit_widths",
    "check_bit_width_budget",
    "radix_capacity_bits",
    "insert_requants",
    # Quantization service: float model -> int IR (Phase 5, PROJECT.md §7, §8, §12)
    "Model",
    "layers",
    "Conv2d",
    "Linear",
    "Pool",
    "Activation",
    "QuantConfig",
    "evaluate_graph_int",
]

# TODO(phase-6): re-export `load_onnx`.
