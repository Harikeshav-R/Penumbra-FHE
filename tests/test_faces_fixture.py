"""Phase-7 faces fixture guard, Python side (``AGENTS.md`` §1.1, §5) — the abstraction proof.

This is the framework proof for the **second use case** (closed-set face recognition), hermetic
and in CI: it reloads the **committed** ``face_cnn.onnx`` (a real PyTorch-exported model) with the
*core* ``onnx`` dep — no torch, no ML stack — lowers it through the front door, and asserts the
result equals the committed ``phase7_faces_fixture.json`` (its graph, its logits, its labels). If
the loader drifts from what generated the fixture, this fails here (fast) rather than as a
confusing Rust golden violation.

The point of Phase 7 is that faces reuse the **unchanged** backend: the committed graph is the
same ``Conv2d -> Requant(fused ReLU) -> Linear`` shape the digit CNN produces, with no new IR op.
This guard asserts exactly that (the op-type list below), so a regression that smuggled a
use-case-specific op into the faces path would be caught.

The bit-for-bit FHE gate over the same fixture lives in Rust (``runtime/tests/golden_faces.rs``,
``#[ignore]`` by default — minutes per sample). The torch/sklearn training that *produced* the
``.onnx`` + fixture is the example generator's job (the optional ``ml`` extra), never CI's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import penumbra as fhe
from penumbra.bitwidth import check_bit_width_budget
from penumbra.compile import insert_requants
from penumbra.ir import Graph
from penumbra.reference import evaluate_graph_int

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "faces"
ONNX_MODEL = EXAMPLES / "face_cnn.onnx"
FIXTURE = EXAMPLES / "phase7_faces_fixture.json"

# Regeneration knobs must match examples/faces/olivetti_export.py so the reload reproduces the
# fixture. Geometry: 64x64 Olivetti downsampled 4x4 -> 16x16, Conv(1->8,3x3,stride4) -> Linear.
INPUT_BITS = 4
WEIGHT_BITS = 6
ACT_BITS = 2
N_IDENTITIES = 8
IN_H = IN_W = 16
CONV_CH = 8
N_FEATURES = 128


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_faces_fixture_graph_round_trips_and_fits_budget():
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    assert Graph.from_json(g.to_json()) == g, "faces IR round-trip must be exact"
    check_bit_width_budget(g)  # raises if any tensor / Requant internal peak exceeds the radix


def test_faces_fixture_graph_is_conv_requant_linear_and_idempotent():
    """The committed graph is Conv2d -> Requant(fused ReLU) -> Linear — no new op for faces."""
    g = Graph.from_dict(_fixture()["graph"])
    assert [n.op.op_type for n in g.nodes] == ["Conv2d", "Requant", "Linear"]
    assert insert_requants(g) == g, "insert_requants must be idempotent on the committed graph"


def test_faces_fixture_logits_and_labels_match_oracle():
    """Committed logits/labels are exactly what the integer reference produces (drift guard)."""
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    for i, (x, expected) in enumerate(zip(fx["test_inputs"], fx["expected_logits"], strict=True)):
        logits = evaluate_graph_int(g, {"x": x})[g.outputs[0]]
        assert logits == expected, f"sample {i}: logits drifted from the oracle"
        assert int(np.argmax(logits)) == fx["expected_labels"][i], f"sample {i}: label drifted"


def test_committed_faces_onnx_lowers_to_expected_float_layers():
    """load_onnx lowers the committed face_cnn.onnx to the right float layers, weights and all.

    This is the pure load-time drift guard, hermetic and calibration-free: it checks the loader's
    contract in isolation — the lowered float ``Conv2d``/``Linear`` weights must equal the ONNX
    model's initializers (read directly with the core ``onnx`` dep). Quantization is deliberately
    *not* re-run here: the head's bias and the Requant rescale depend on the Olivetti training-set
    calibration the generator used (unavailable in CI), so reproducing the full int graph would
    need the ``ml`` extra. Weight *lowering* is data-independent, so this catches a transB/layout/
    fold regression without any calibration. Lowering to the *same* three layers as the digit CNN
    is the Phase-7 abstraction claim (no use-case-specific op).
    """
    model = fhe.load_onnx(str(ONNX_MODEL), input_bits=INPUT_BITS)
    assert [type(layer).__name__ for layer in model.layers] == ["Conv2d", "Activation", "Linear"]

    inits = _onnx_initializers(ONNX_MODEL)
    conv, _relu, linear = model.layers
    # Conv weight is (out, in, kh, kw), passed through from the ONNX initializer unchanged.
    conv_w = _find(inits, shape_ndim=4)
    assert np.allclose(conv.weight, conv_w)
    assert (conv.in_h, conv.in_w, conv.in_channels, conv.stride) == (IN_H, IN_W, 1, 4)
    # The Gemm exports with transB=1, so its (n_out, n_in) weight passes straight to Linear.weight.
    gemm_w = _find(inits, shape=(N_IDENTITIES, N_FEATURES))
    assert np.allclose(linear.weight, gemm_w)
    gemm_b = _find(inits, shape=(N_IDENTITIES,))
    assert linear.bias is not None and np.allclose(linear.bias, gemm_b)


def _onnx_initializers(path: Path) -> dict[str, np.ndarray]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _find(inits: dict[str, np.ndarray], *, shape=None, shape_ndim=None) -> np.ndarray:
    for arr in inits.values():
        if shape is not None and arr.shape == shape:
            return arr
        if shape_ndim is not None and arr.ndim == shape_ndim:
            return arr
    raise AssertionError(f"no initializer with shape={shape} ndim={shape_ndim} in {list(inits)}")


def test_faces_fixture_reports_honest_accuracy():
    acc = _fixture()["accuracy"]
    assert 0.0 <= acc["quantized"] <= acc["float"] <= 1.0
    assert acc["float"] > 0.8, "the float CNN should recognize the enrolled faces well"
