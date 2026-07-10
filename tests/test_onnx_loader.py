"""Hermetic tests for the ONNX front door (``penumbra.load_onnx``), Phase 6.

These build tiny ONNX models **in memory** with ``onnx.helper``/``onnx.numpy_helper`` — no ML
framework, no committed file, so they run in the default CI ``python`` job (``onnx`` is a core
dep). They pin the lowering contract: an ONNX graph parses and validates, lowers to the right
:mod:`penumbra.layers` layer list with correct weight orientation/shapes, and the resulting
:class:`~penumbra.model.Model` flows through the unchanged Phase-5 ``quantize`` service to an IR
graph the integer oracle (:func:`penumbra.reference.evaluate_graph_int`) can evaluate — the full
ONNX -> Model -> IR -> oracle round trip.

The framework end-to-end proof (a real PyTorch-exported ``.onnx``) is the committed-fixture test
(``test_onnx_fixture.py``); the loud-failure gate is ``test_onnx_unsupported.py``; the
doc<->registry lockstep is ``test_supported_ops_doc.py``.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import penumbra as fhe
from penumbra.layers import Activation, Conv2d
from penumbra.reference import evaluate_graph_int

OPSET = 13


def _save(nodes, inits, inputs, outputs, tmp_path, name="m.onnx") -> str:
    graph = helper.make_graph(nodes, "g", inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)])
    onnx.checker.check_model(model)
    path = str(tmp_path / name)
    onnx.save(model, path)
    return path


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _f32(arr, name):
    return numpy_helper.from_array(arr.astype(np.float32), name)


def _quantize_input(model: fhe.Model, x: np.ndarray) -> list[int]:
    """Quantize a single float input row to the model's int input domain (unsigned)."""
    hi = (1 << model.input_bits) - 1
    return np.clip(np.round(x / model.input_scale), 0, hi).astype(np.int64).tolist()


# --- lowering shape/structure -------------------------------------------------------------


def test_lowers_gemm_relu_gemm(tmp_path):
    """Gemm(transB=1) -> Relu -> Gemm(transB=1) -> Softmax lowers to Linear, Activation, Linear."""
    rng = np.random.default_rng(0)
    w1 = rng.normal(size=(6, 4))  # (n_out, n_in) with transB=1
    b1 = rng.normal(size=6)
    w2 = rng.normal(size=(3, 6))
    b2 = rng.normal(size=3)
    nodes = [
        helper.make_node("Gemm", ["x", "w1", "b1"], ["h0"], name="fc1", transB=1),
        helper.make_node("Relu", ["h0"], ["h1"], name="relu1"),
        helper.make_node("Gemm", ["h1", "w2", "b2"], ["h2"], name="fc2", transB=1),
        helper.make_node("Softmax", ["h2"], ["y"], name="sm", axis=1),
    ]
    inits = [_f32(w1, "w1"), _f32(b1, "b1"), _f32(w2, "w2"), _f32(b2, "b2")]
    path = _save(nodes, inits, [_vi("x", [1, 4])], [_vi("y", [1, 3])], tmp_path)

    model = fhe.load_onnx(path)
    assert [type(layer).__name__ for layer in model.layers] == ["Linear", "Activation", "Linear"]
    assert model.layers[0].weight.shape == (6, 4)  # (n_out, n_in)
    assert model.layers[0].bias.shape == (6,)
    assert model.layers[2].weight.shape == (3, 6)


def test_gemm_transb0_transposes_weight(tmp_path):
    """A Gemm with transB=0 has weight (n_in, n_out); the loader transposes to (n_out, n_in)."""
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 6))  # (n_in=4, n_out=6), transB=0
    nodes = [helper.make_node("Gemm", ["x", "w"], ["y"], name="fc", transB=0)]
    path = _save(nodes, [_f32(w, "w")], [_vi("x", [1, 4])], [_vi("y", [1, 6])], tmp_path)

    model = fhe.load_onnx(path)
    assert model.layers[0].weight.shape == (6, 4)
    # Fidelity: our Linear.forward must equal x @ w (the ONNX Gemm) for transB=0.
    x = rng.uniform(size=(2, 4))
    assert np.allclose(model.layers[0].forward(x), x @ w)


def test_matmul_plus_const_add_folds_to_linear_bias(tmp_path):
    """MatMul(x@W) + Add(const) is one dense layer: the Add folds into Linear.bias."""
    rng = np.random.default_rng(2)
    w = rng.normal(size=(4, 5))  # MatMul weight is (n_in, n_out)
    bias = rng.normal(size=5)
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"], name="mm"),
        helper.make_node("Add", ["h", "bias"], ["y"], name="addbias"),
    ]
    inits = [_f32(w, "w"), _f32(bias, "bias")]
    path = _save(nodes, inits, [_vi("x", [1, 4])], [_vi("y", [1, 5])], tmp_path)

    model = fhe.load_onnx(path)
    assert [type(layer).__name__ for layer in model.layers] == ["Linear"]
    lin = model.layers[0]
    assert lin.weight.shape == (5, 4)  # transposed from (n_in, n_out)
    assert np.allclose(lin.bias, bias)
    # The folded layer computes x @ W + bias exactly.
    x = rng.uniform(size=(3, 4))
    assert np.allclose(lin.forward(x), x @ w + bias)


def test_lowers_conv_relu_flatten_gemm(tmp_path):
    """Conv -> Relu -> Flatten -> Gemm: Flatten folds away, Conv weight passes through as-is."""
    rng = np.random.default_rng(3)
    wc = rng.normal(size=(3, 1, 3, 3))  # (out, in, kh, kw)
    wg = rng.normal(size=(5, 3 * 3 * 3))  # stride-2 on 8x8 -> 3x3 map -> 27 features
    bg = rng.normal(size=5)
    nodes = [
        helper.make_node("Conv", ["x", "wc"], ["c"], name="conv1", strides=[2, 2], group=1),
        helper.make_node("Relu", ["c"], ["r"], name="relu1"),
        helper.make_node("Flatten", ["r"], ["f"], name="flat", axis=1),
        helper.make_node("Gemm", ["f", "wg", "bg"], ["y"], name="fc", transB=1),
    ]
    inits = [_f32(wc, "wc"), _f32(wg, "wg"), _f32(bg, "bg")]
    path = _save(nodes, inits, [_vi("x", [1, 1, 8, 8])], [_vi("y", [1, 5])], tmp_path)

    model = fhe.load_onnx(path)
    assert [type(layer).__name__ for layer in model.layers] == ["Conv2d", "Activation", "Linear"]
    conv = model.layers[0]
    assert isinstance(conv, Conv2d)
    assert conv.weight.shape == (3, 1, 3, 3)
    assert (conv.in_h, conv.in_w, conv.in_channels, conv.stride, conv.padding) == (8, 8, 1, 2, 0)


def test_terminal_softmax_and_reshape_are_dropped_and_folded(tmp_path):
    """A terminal Softmax is dropped; a Reshape between layers folds to a layout no-op."""
    rng = np.random.default_rng(4)
    w1 = rng.normal(size=(6, 4))
    w2 = rng.normal(size=(3, 6))
    shape = numpy_helper.from_array(np.array([-1, 6], dtype=np.int64), "shp")
    nodes = [
        helper.make_node("Gemm", ["x", "w1"], ["h0"], name="fc1", transB=1),
        helper.make_node("Reshape", ["h0", "shp"], ["h0r"], name="rs"),
        helper.make_node("Gemm", ["h0r", "w2"], ["h2"], name="fc2", transB=1),
        helper.make_node("Softmax", ["h2"], ["y"], name="sm", axis=1),
    ]
    inits = [_f32(w1, "w1"), _f32(w2, "w2"), shape]
    path = _save(nodes, inits, [_vi("x", [1, 4])], [_vi("y", [1, 3])], tmp_path)

    model = fhe.load_onnx(path)
    # Reshape and Softmax emit no layer.
    assert [type(layer).__name__ for layer in model.layers] == ["Linear", "Linear"]


# --- full round trip: ONNX -> Model -> IR -> oracle vs a NumPy float reference -------------


def test_round_trip_quantize_matches_float_argmax(tmp_path):
    """Lower + quantize a Gemm->Relu->Gemm; the oracle's argmax matches a NumPy float reference.

    This is the ONNX -> Model -> IR -> integer-oracle round trip. We don't demand bit-exact
    logits (quantization is lossy by design) — we demand the *label* agrees with the same tiny
    network evaluated in float, which is the classification contract that matters.
    """
    rng = np.random.default_rng(5)
    w1 = rng.normal(size=(8, 6)) * 0.5
    b1 = rng.normal(size=8) * 0.1
    w2 = rng.normal(size=(4, 8)) * 0.5
    b2 = rng.normal(size=4) * 0.1
    nodes = [
        helper.make_node("Gemm", ["x", "w1", "b1"], ["h0"], name="fc1", transB=1),
        helper.make_node("Relu", ["h0"], ["h1"], name="relu1"),
        helper.make_node("Gemm", ["h1", "w2", "b2"], ["y"], name="fc2", transB=1),
    ]
    inits = [_f32(w1, "w1"), _f32(b1, "b1"), _f32(w2, "w2"), _f32(b2, "b2")]
    path = _save(nodes, inits, [_vi("x", [1, 6])], [_vi("y", [1, 4])], tmp_path)

    model = fhe.load_onnx(path)
    cal = rng.uniform(0.0, 16.0, size=(64, 6))
    graph = model.quantize(cal, n_bits=6, act_bits=2, calibration="mse")
    assert [n.op.op_type for n in graph.nodes] == ["Linear", "Requant", "Linear"]

    def float_ref(x):
        h = np.maximum(x @ w1.T + b1, 0.0)
        return h @ w2.T + b2

    # Agreement rate over a batch (individual low-precision samples can flip; the label
    # distribution must track the float model).
    agree = 0
    n = 40
    for x in rng.uniform(0.0, 16.0, size=(n, 6)):
        xq = _quantize_input(model, x)
        logits = evaluate_graph_int(graph, {"x": xq})[graph.outputs[0]]
        if int(np.argmax(logits)) == int(np.argmax(float_ref(x))):
            agree += 1
    assert agree / n >= 0.8, f"quantized argmax tracks float only {agree}/{n} of the time"


def test_relu_lowers_to_relu_activation(tmp_path):
    """The emitted Activation for an ONNX Relu behaves like max(x, 0)."""
    w = np.eye(3)
    nodes = [
        helper.make_node("Gemm", ["x", "w"], ["h"], name="fc", transB=1),
        helper.make_node("Relu", ["h"], ["r"], name="relu"),
        helper.make_node("Gemm", ["r", "w"], ["y"], name="fc2", transB=1),
    ]
    path = _save(nodes, [_f32(w, "w")], [_vi("x", [1, 3])], [_vi("y", [1, 3])], tmp_path)
    model = fhe.load_onnx(path)
    act = model.layers[1]
    assert isinstance(act, Activation)
    assert [act.fn(v) for v in (-2.0, -0.1, 0.0, 3.0)] == [0.0, 0.0, 0.0, 3.0]


def test_input_bits_override(tmp_path):
    """load_onnx defaults input_bits=4 and honors an override."""
    w = np.eye(3)
    nodes = [helper.make_node("Gemm", ["x", "w"], ["y"], name="fc", transB=1)]
    path = _save(nodes, [_f32(w, "w")], [_vi("x", [1, 3])], [_vi("y", [1, 3])], tmp_path)
    assert fhe.load_onnx(path).input_bits == 4
    assert fhe.load_onnx(path, input_bits=6).input_bits == 6
