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
from penumbra.layers import Activation, Conv2d, Pool
from penumbra.reference import evaluate_graph_int

OPSET = 13


def _save(nodes, inits, inputs, outputs, tmp_path, name="m.onnx", opset: int = OPSET) -> str:
    graph = helper.make_graph(nodes, "g", inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", opset)])
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


def test_lowers_conv_maxpool_flatten_gemm(tmp_path):
    """Conv -> MaxPool -> Flatten -> Gemm lowers the MaxPool to a Pool('max') with the right fields.

    Guards the pooling lowering path (`_lower_pool`): a MaxPool's kernel/stride and the input
    feature dims (from shape inference) must land on `layers.Pool`. Conv 1->2ch, 3x3 stride-1 on
    8x8 -> 6x6; MaxPool 2x2 stride-2 -> 3x3; Flatten -> 2*3*3 = 18 features into the Gemm.
    """
    rng = np.random.default_rng(6)
    wc = rng.normal(size=(2, 1, 3, 3))  # (out, in, kh, kw); stride-1 on 8x8 -> 6x6
    wg = rng.normal(size=(4, 2 * 3 * 3))  # 2 channels * 3x3 pooled map = 18 features
    nodes = [
        helper.make_node("Conv", ["x", "wc"], ["c"], name="conv1", strides=[1, 1], group=1),
        helper.make_node(
            "MaxPool", ["c"], ["p"], name="pool1", kernel_shape=[2, 2], strides=[2, 2]
        ),
        helper.make_node("Flatten", ["p"], ["f"], name="flat", axis=1),
        helper.make_node("Gemm", ["f", "wg"], ["y"], name="fc", transB=1),
    ]
    inits = [_f32(wc, "wc"), _f32(wg, "wg")]
    path = _save(nodes, inits, [_vi("x", [1, 1, 8, 8])], [_vi("y", [1, 4])], tmp_path)

    model = fhe.load_onnx(path)
    assert [type(layer).__name__ for layer in model.layers] == ["Conv2d", "Pool", "Linear"]
    pool = model.layers[1]
    assert isinstance(pool, Pool)
    assert pool.mode == "max"
    assert (pool.channels, pool.in_h, pool.in_w) == (2, 6, 6)
    assert (pool.pool_h, pool.pool_w, pool.stride) == (2, 2, 2)


def test_lowers_global_average_pool(tmp_path):
    """GlobalAveragePool lowers to a Pool('avg') whose window is the whole feature map."""
    rng = np.random.default_rng(7)
    wc = rng.normal(size=(3, 1, 3, 3))  # stride-1 on 8x8 -> 6x6, 3 channels
    wg = rng.normal(size=(4, 3))  # GAP collapses each channel's 6x6 map to 1 value -> 3 features
    nodes = [
        helper.make_node("Conv", ["x", "wc"], ["c"], name="conv1", strides=[1, 1], group=1),
        helper.make_node("GlobalAveragePool", ["c"], ["g"], name="gap"),
        helper.make_node("Flatten", ["g"], ["f"], name="flat", axis=1),
        helper.make_node("Gemm", ["f", "wg"], ["y"], name="fc", transB=1),
    ]
    inits = [_f32(wc, "wc"), _f32(wg, "wg")]
    path = _save(nodes, inits, [_vi("x", [1, 1, 8, 8])], [_vi("y", [1, 4])], tmp_path)

    model = fhe.load_onnx(path)
    assert [type(layer).__name__ for layer in model.layers] == ["Conv2d", "Pool", "Linear"]
    pool = model.layers[1]
    assert isinstance(pool, Pool)
    # avg mode; the window spans the whole 6x6 map (kernel = full spatial size) -> a 1x1 output.
    assert pool.mode == "avg"
    assert (pool.channels, pool.in_h, pool.in_w) == (3, 6, 6)
    assert (pool.pool_h, pool.pool_w) == (6, 6)


def test_leading_float_cast_folds_away(tmp_path):
    """A leading Cast(to=FLOAT) — as skl2onnx emits at the input — folds to a layout no-op.

    Exporters routinely insert a dtype-normalizing Cast at the graph input; casting to a float type
    is identity on Penumbra's real-valued wire, so it must emit no layer and leave the dense chain
    intact (Cast -> Gemm lowers to just Linear).
    """
    rng = np.random.default_rng(8)
    w = rng.normal(size=(4, 6))  # (n_out, n_in), transB=1
    nodes = [
        helper.make_node("Cast", ["x"], ["xf"], name="cast", to=TensorProto.FLOAT),
        helper.make_node("Gemm", ["xf", "w"], ["y"], name="fc", transB=1),
    ]
    path = _save(nodes, [_f32(w, "w")], [_vi("x", [1, 6])], [_vi("y", [1, 4])], tmp_path)

    model = fhe.load_onnx(path)
    assert [type(layer).__name__ for layer in model.layers] == ["Linear"]
    assert model.layers[0].weight.shape == (4, 6)


def test_order_preserving_transpose_over_size1_axis_folds_away(tmp_path):
    """A Transpose that only permutes size-1 axes preserves the flat wire and folds to a no-op.

    The input is (1, 1, 6); perm=[1, 0, 2] swaps the two size-1 axes, so the row-major flattening is
    unchanged (a size-1 axis contributes no stride). The loader must fold it away, leaving just the
    dense layer — proving the analytical order-preservation check accepts a genuine layout no-op
    (and, with a size-1 batch, without materializing a prod(dims) index array).
    """
    rng = np.random.default_rng(9)
    w = rng.normal(size=(4, 6))  # (n_out, n_in), transB=1
    nodes = [
        helper.make_node("Transpose", ["x"], ["xt"], name="t", perm=[1, 0, 2]),
        helper.make_node("Reshape", ["xt", "shp"], ["xf"], name="rs"),
        helper.make_node("Gemm", ["xf", "w"], ["y"], name="fc", transB=1),
    ]
    shp = numpy_helper.from_array(np.array([-1, 6], dtype=np.int64), "shp")
    inits = [shp, _f32(w, "w")]
    path = _save(nodes, inits, [_vi("x", [1, 1, 6])], [_vi("y", [1, 4])], tmp_path)

    model = fhe.load_onnx(path)
    assert [type(layer).__name__ for layer in model.layers] == ["Linear"]
    assert model.layers[0].weight.shape == (4, 6)


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


def test_all_seven_activations_lower_accurately(tmp_path):
    """Each of the seven non-ReLU activations lowers to exact Activation.fn."""
    import math

    from penumbra.quantization import activations as ref_act

    probes = [-3.0, -1.0, 0.0, 0.5, 2.0]
    test_cases = [
        ("Tanh", {}, ref_act.tanh, 13),
        ("LeakyRelu", {"alpha": 0.05}, ref_act.activation_fn("LeakyRelu", {"alpha": 0.05}), 13),
        ("HardSwish", {}, ref_act.hardswish, 14),
        ("Gelu", {"approximate": b"none"}, ref_act.gelu, 20),
        (
            "Gelu",
            {"approximate": b"tanh"},
            ref_act.activation_fn("Gelu", {"approximate": "tanh"}),
            20,
        ),
        ("Elu", {"alpha": 1.5}, ref_act.activation_fn("Elu", {"alpha": 1.5}), 13),
        (
            "HardSigmoid",
            {"alpha": 0.15, "beta": 0.6},
            ref_act.activation_fn("HardSigmoid", {"alpha": 0.15, "beta": 0.6}),
            13,
        ),
        ("Sigmoid", {}, ref_act.sigmoid, 13),
    ]

    w = np.eye(3)
    for idx, (op_name, attrs, expected_fn, op_opset) in enumerate(test_cases):
        nodes = [
            helper.make_node("Gemm", ["x", "w"], ["h"], name="fc1", transB=1),
            helper.make_node(op_name, ["h"], ["r"], name=f"act_{idx}", **attrs),
            helper.make_node("Gemm", ["r", "w"], ["y"], name="fc2", transB=1),
        ]
        path = _save(
            nodes,
            [_f32(w, "w")],
            [_vi("x", [1, 3])],
            [_vi("y", [1, 3])],
            tmp_path,
            name=f"act_{idx}.onnx",
            opset=op_opset,
        )
        model = fhe.load_onnx(path)
        assert len(model.layers) == 3
        act_layer = model.layers[1]
        assert isinstance(act_layer, Activation)
        for p in probes:
            assert math.isclose(act_layer.fn(p), expected_fn(p), rel_tol=1e-5, abs_tol=1e-7)


def test_terminal_sigmoid_is_dropped_mid_graph_lowers(tmp_path):
    """A terminal Sigmoid is dropped; a mid-graph Sigmoid lowers to Activation."""
    w = np.eye(3)
    # Terminal Sigmoid: Gemm -> Sigmoid
    nodes_term = [
        helper.make_node("Gemm", ["x", "w"], ["h"], name="fc", transB=1),
        helper.make_node("Sigmoid", ["h"], ["y"], name="sig_term"),
    ]
    path_term = _save(
        nodes_term,
        [_f32(w, "w")],
        [_vi("x", [1, 3])],
        [_vi("y", [1, 3])],
        tmp_path,
        name="term_sig.onnx",
    )
    m_term = fhe.load_onnx(path_term)
    assert len(m_term.layers) == 1, "terminal Sigmoid should be dropped"

    # Mid-graph Sigmoid: Gemm -> Sigmoid -> Gemm
    nodes_mid = [
        helper.make_node("Gemm", ["x", "w"], ["h"], name="fc1", transB=1),
        helper.make_node("Sigmoid", ["h"], ["s"], name="sig_mid"),
        helper.make_node("Gemm", ["s", "w"], ["y"], name="fc2", transB=1),
    ]
    path_mid = _save(
        nodes_mid,
        [_f32(w, "w")],
        [_vi("x", [1, 3])],
        [_vi("y", [1, 3])],
        tmp_path,
        name="mid_sig.onnx",
    )
    m_mid = fhe.load_onnx(path_mid)
    assert len(m_mid.layers) == 3
    assert isinstance(m_mid.layers[1], Activation)


def test_residual_add_lowers_to_add_layer(tmp_path):
    """Residual Add (both inputs activations) lowers to a layers.Add node."""
    w1 = np.eye(4)
    w2 = np.eye(4)
    nodes = [
        helper.make_node("Gemm", ["x", "w1"], ["a"], name="ga", transB=1),
        helper.make_node("Gemm", ["x", "w2"], ["b"], name="gb", transB=1),
        helper.make_node("Add", ["a", "b"], ["y"], name="res"),
    ]
    inits = [_f32(w1, "w1"), _f32(w2, "w2")]
    path = _save(nodes, inits, [_vi("x", [1, 4])], [_vi("y", [1, 4])], tmp_path)
    model = fhe.load_onnx(path)
    assert len(model.nodes) == 3
    add_node = next(n for n in model.nodes if n.name == "res")
    assert isinstance(add_node.layer, fhe.layers.Add)
    assert add_node.inputs == ["a", "b"]
    assert add_node.outputs == ["y"]


def test_concat_lowers_with_right_flat_sizes(tmp_path):
    """Concat along axis=1 lowers to layers.Concat."""
    w1 = np.eye(3)
    w2 = np.eye(5)
    nodes = [
        helper.make_node("Gemm", ["x", "w1"], ["a"], name="ga", transB=1),
        helper.make_node("Gemm", ["x", "w2"], ["b"], name="gb", transB=1),
        helper.make_node("Concat", ["a", "b"], ["y"], name="cat", axis=1),
    ]
    inits = [_f32(w1, "w1"), _f32(w2, "w2")]
    path = _save(nodes, inits, [_vi("x", [1, 4])], [_vi("y", [1, 8])], tmp_path)
    model = fhe.load_onnx(path)
    cat_node = next(n for n in model.nodes if n.name == "cat")
    assert isinstance(cat_node.layer, fhe.layers.Concat)
    assert cat_node.inputs == ["a", "b"]
    assert cat_node.outputs == ["y"]


def test_split_lowers_with_channel_to_flat_sizes_rank4(tmp_path):
    """Split along channel axis converts channel counts to flat element counts on rank-4."""
    wc = np.ones((4, 1, 3, 3))
    sp_split = numpy_helper.from_array(np.array([2, 2], dtype=np.int64), "sp_split")
    nodes = [
        helper.make_node("Conv", ["x", "wc"], ["c"], name="conv", strides=[1, 1]),
        helper.make_node("Split", ["c", "sp_split"], ["s0", "s1"], name="sp", axis=1),
        helper.make_node("Concat", ["s0", "s1"], ["y"], name="cat", axis=1),
    ]
    inits = [_f32(wc, "wc"), sp_split]
    path = _save(nodes, inits, [_vi("x", [1, 1, 8, 8])], [_vi("y", [1, 4, 6, 6])], tmp_path)
    model = fhe.load_onnx(path)
    split_node = next(n for n in model.nodes if n.name == "sp")
    assert isinstance(split_node.layer, fhe.layers.Split)
    assert split_node.layer.sizes == [72, 72]
    assert split_node.outputs == ["s0", "s1"]


def test_batchnorm_folds_into_preceding_conv(tmp_path):
    """BatchNorm immediately following Conv folds into Conv weights/bias with no extra layer."""
    rng = np.random.default_rng(42)
    wc = rng.normal(size=(4, 1, 3, 3))
    bc = rng.normal(size=4)
    scale = rng.uniform(0.5, 2.0, size=4)
    b = rng.normal(size=4)
    mean = rng.normal(size=4)
    var = rng.uniform(0.1, 3.0, size=4)
    eps = 1e-5

    nodes = [
        helper.make_node("Conv", ["x", "wc", "bc"], ["c"], name="conv", strides=[1, 1]),
        helper.make_node(
            "BatchNormalization",
            ["c", "scale", "b", "mean", "var"],
            ["y"],
            name="bn",
            epsilon=eps,
        ),
    ]
    inits = [
        _f32(wc, "wc"),
        _f32(bc, "bc"),
        _f32(scale, "scale"),
        _f32(b, "b"),
        _f32(mean, "mean"),
        _f32(var, "var"),
    ]
    path = _save(nodes, inits, [_vi("x", [1, 1, 8, 8])], [_vi("y", [1, 4, 6, 6])], tmp_path)
    model = fhe.load_onnx(path)

    assert len(model.nodes) == 1
    assert model.nodes[0].name == "conv"
    conv_layer = model.nodes[0].layer
    assert isinstance(conv_layer, Conv2d)

    s = scale / np.sqrt(var + eps)
    expected_w = s[:, None, None, None] * wc
    expected_b = (bc - mean) * s + b
    np.testing.assert_allclose(conv_layer.weight, expected_w, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(conv_layer.bias, expected_b, rtol=1e-6, atol=1e-6)


def test_fanout_tensor_lowers_without_error(tmp_path):
    """A fan-out tensor read by two downstream nodes lowers cleanly in the DAG."""
    w1 = np.eye(3)
    w2 = np.eye(3)
    nodes = [
        helper.make_node("Gemm", ["x", "w1"], ["h"], name="fc1", transB=1),
        helper.make_node("Gemm", ["h", "w2"], ["a"], name="branch_a", transB=1),
        helper.make_node("Gemm", ["h", "w2"], ["b"], name="branch_b", transB=1),
        helper.make_node("Add", ["a", "b"], ["y"], name="add"),
    ]
    inits = [_f32(w1, "w1"), _f32(w2, "w2")]
    path = _save(nodes, inits, [_vi("x", [1, 3])], [_vi("y", [1, 3])], tmp_path)
    model = fhe.load_onnx(path)
    assert len(model.nodes) == 4
    assert [n.name for n in model.nodes] == ["fc1", "branch_a", "branch_b", "add"]


def test_avg_pool_lowering_matches_true_mean(tmp_path):
    """AveragePool lowers to Pool(avg) whose float forward emits the true mean."""
    rng = np.random.default_rng(0)
    wc = rng.normal(size=(3, 1, 3, 3)).astype(np.float32)
    bc = rng.normal(size=3).astype(np.float32)
    # Conv on 6x6 with 3x3 s1 -> 4x4. AvgPool 2x2 s2 -> 2x2. Flatten -> 3*2*2 = 12.
    wg = rng.normal(size=(4, 12)).astype(np.float32)
    bg = (rng.normal(size=4) * 5.0).astype(np.float32)

    nodes = [
        helper.make_node("Conv", ["x", "wc", "bc"], ["c"], name="conv", strides=[1, 1]),
        helper.make_node("Relu", ["c"], ["r"], name="relu"),
        helper.make_node(
            "AveragePool", ["r"], ["p"], name="pool", kernel_shape=[2, 2], strides=[2, 2]
        ),
        helper.make_node("Flatten", ["p"], ["f"], name="flatten"),
        helper.make_node("Gemm", ["f", "wg", "bg"], ["y"], name="gemm", transB=1),
    ]
    inits = [_f32(wc, "wc"), _f32(bc, "bc"), _f32(wg, "wg"), _f32(bg, "bg")]
    path = _save(nodes, inits, [_vi("x", [1, 1, 6, 6])], [_vi("y", [1, 4])], tmp_path)

    model = fhe.load_onnx(path)
    x = rng.uniform(0.0, 1.0, size=(3, 36))
    acts = x
    for layer in model.layers:
        acts = layer.forward(acts)

    # Independent numpy reference:
    xr = x.reshape(3, 1, 6, 6)
    conv_out = np.zeros((3, 3, 4, 4), dtype=np.float64)
    for c_out in range(3):
        for oy in range(4):
            for ox in range(4):
                conv_out[:, c_out, oy, ox] = (
                    xr[:, 0, oy : oy + 3, ox : ox + 3] * wc[c_out, 0]
                ).sum(axis=(1, 2)) + bc[c_out]
    relu_out = np.maximum(conv_out, 0.0)
    pool_out = relu_out.reshape(3, 3, 2, 2, 2, 2).mean(axis=(3, 5))
    flat_out = pool_out.reshape(3, 12)
    ref = flat_out @ wg.T + bg
    assert np.allclose(acts, ref, atol=1e-9)


def test_global_average_pool_lowering_matches_true_mean(tmp_path):
    """GlobalAveragePool lowers to Pool(avg) whose float forward emits the true mean."""
    rng = np.random.default_rng(0)
    wc = rng.normal(size=(3, 1, 3, 3)).astype(np.float32)
    bc = rng.normal(size=3).astype(np.float32)
    # Conv on 6x6 with 3x3 s1 -> 4x4. GlobalAveragePool -> 1x1. Flatten -> 3*1*1 = 3.
    wg = rng.normal(size=(4, 3)).astype(np.float32)
    bg = (rng.normal(size=4) * 5.0).astype(np.float32)

    nodes = [
        helper.make_node("Conv", ["x", "wc", "bc"], ["c"], name="conv", strides=[1, 1]),
        helper.make_node("Relu", ["c"], ["r"], name="relu"),
        helper.make_node("GlobalAveragePool", ["r"], ["p"], name="gap"),
        helper.make_node("Flatten", ["p"], ["f"], name="flatten"),
        helper.make_node("Gemm", ["f", "wg", "bg"], ["y"], name="gemm", transB=1),
    ]
    inits = [_f32(wc, "wc"), _f32(bc, "bc"), _f32(wg, "wg"), _f32(bg, "bg")]
    path = _save(nodes, inits, [_vi("x", [1, 1, 6, 6])], [_vi("y", [1, 4])], tmp_path)

    model = fhe.load_onnx(path)
    x = rng.uniform(0.0, 1.0, size=(3, 36))
    acts = x
    for layer in model.layers:
        acts = layer.forward(acts)

    # Independent numpy reference:
    xr = x.reshape(3, 1, 6, 6)
    conv_out = np.zeros((3, 3, 4, 4), dtype=np.float64)
    for c_out in range(3):
        for oy in range(4):
            for ox in range(4):
                conv_out[:, c_out, oy, ox] = (
                    xr[:, 0, oy : oy + 3, ox : ox + 3] * wc[c_out, 0]
                ).sum(axis=(1, 2)) + bc[c_out]
    relu_out = np.maximum(conv_out, 0.0)
    pool_out = relu_out.mean(axis=(2, 3))
    flat_out = pool_out.reshape(3, 3)
    ref = flat_out @ wg.T + bg
    assert np.allclose(acts, ref, atol=1e-9)


def test_lowers_padded_average_pool(tmp_path):
    """AveragePool with symmetric padding lowers with padding, matches numpy, and quantizes."""
    rng = np.random.default_rng(0)
    wc = rng.normal(size=(2, 1, 3, 3)).astype(np.float32)
    bc = rng.normal(size=2).astype(np.float32)
    # Conv 1->2 on 6x6 -> 2x4x4. AvgPool 3x3 s1 pad 1 -> 2x4x4 = 32 features.
    wg = rng.normal(size=(4, 32)).astype(np.float32)
    bg = (rng.normal(size=4) * 2.0).astype(np.float32)

    nodes = [
        helper.make_node("Conv", ["x", "wc", "bc"], ["c"], name="conv", strides=[1, 1]),
        helper.make_node(
            "AveragePool",
            ["c"],
            ["p"],
            name="pool",
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],
            count_include_pad=1,
        ),
        helper.make_node("Flatten", ["p"], ["f"], name="flatten"),
        helper.make_node("Gemm", ["f", "wg", "bg"], ["y"], name="gemm", transB=1),
    ]
    inits = [_f32(wc, "wc"), _f32(bc, "bc"), _f32(wg, "wg"), _f32(bg, "bg")]
    path = _save(nodes, inits, [_vi("x", [1, 1, 6, 6])], [_vi("y", [1, 4])], tmp_path)

    model = fhe.load_onnx(path)
    pool_layer = model.layers[1]
    assert isinstance(pool_layer, Pool)
    assert pool_layer.padding == 1
    assert pool_layer.pool_h == 3
    assert pool_layer.pool_w == 3
    assert pool_layer.stride == 1
    assert pool_layer.mode == "avg"

    x = rng.uniform(0.0, 1.0, size=(3, 36))
    acts = x
    for layer in model.layers:
        acts = layer.forward(acts)

    # Independent numpy reference:
    xr = x.reshape(3, 1, 6, 6)
    conv_out = np.zeros((3, 2, 4, 4), dtype=np.float64)
    for c_out in range(2):
        for oy in range(4):
            for ox in range(4):
                conv_out[:, c_out, oy, ox] = (
                    xr[:, 0, oy : oy + 3, ox : ox + 3] * wc[c_out, 0]
                ).sum(axis=(1, 2)) + bc[c_out]
    padded = np.pad(conv_out, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=0.0)
    pool_out = np.zeros((3, 2, 4, 4), dtype=np.float64)
    for oy in range(4):
        for ox in range(4):
            pool_out[:, :, oy, ox] = padded[:, :, oy : oy + 3, ox : ox + 3].mean(axis=(2, 3))
    flat_out = pool_out.reshape(3, 32)
    ref = flat_out @ wg.T + bg
    assert np.allclose(acts, ref, atol=1e-9)

    cal = rng.uniform(0.0, 1.0, size=(16, 36))
    graph = model.quantize(cal, n_bits=4)
    pool_specs = [n.op for n in graph.nodes if n.op.op_type == "Pool"]
    assert len(pool_specs) == 1
    assert pool_specs[0].padding == 1

    xq = _quantize_input(model, x[0])
    out = evaluate_graph_int(graph, {"x": xq})
    assert len(out[graph.outputs[0]]) == 4


def test_lowers_padded_max_pool(tmp_path):
    """MaxPool with symmetric padding lowers with padding, matches numpy, and quantizes."""
    rng = np.random.default_rng(1)
    wc = rng.normal(size=(2, 1, 3, 3)).astype(np.float32)
    bc = rng.normal(size=2).astype(np.float32)
    # Conv 1->2 on 6x6 -> 2x4x4. MaxPool 2x2 s2 pad 1 -> (4+2-2)//2 + 1 = 3 -> 2x3x3 = 18 features.
    wg = rng.normal(size=(4, 18)).astype(np.float32)
    bg = (rng.normal(size=4) * 2.0).astype(np.float32)

    nodes = [
        helper.make_node("Conv", ["x", "wc", "bc"], ["c"], name="conv", strides=[1, 1]),
        helper.make_node(
            "MaxPool",
            ["c"],
            ["p"],
            name="pool",
            kernel_shape=[2, 2],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Flatten", ["p"], ["f"], name="flatten"),
        helper.make_node("Gemm", ["f", "wg", "bg"], ["y"], name="gemm", transB=1),
    ]
    inits = [_f32(wc, "wc"), _f32(bc, "bc"), _f32(wg, "wg"), _f32(bg, "bg")]
    path = _save(nodes, inits, [_vi("x", [1, 1, 6, 6])], [_vi("y", [1, 4])], tmp_path)

    model = fhe.load_onnx(path)
    pool_layer = model.layers[1]
    assert isinstance(pool_layer, Pool)
    assert pool_layer.padding == 1
    assert pool_layer.pool_h == 2
    assert pool_layer.pool_w == 2
    assert pool_layer.stride == 2
    assert pool_layer.mode == "max"

    x = rng.uniform(0.0, 1.0, size=(3, 36))
    acts = x
    for layer in model.layers:
        acts = layer.forward(acts)

    # Independent numpy reference:
    xr = x.reshape(3, 1, 6, 6)
    conv_out = np.zeros((3, 2, 4, 4), dtype=np.float64)
    for c_out in range(2):
        for oy in range(4):
            for ox in range(4):
                conv_out[:, c_out, oy, ox] = (
                    xr[:, 0, oy : oy + 3, ox : ox + 3] * wc[c_out, 0]
                ).sum(axis=(1, 2)) + bc[c_out]
    padded = np.pad(conv_out, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    pool_out = np.zeros((3, 2, 3, 3), dtype=np.float64)
    for oy in range(3):
        for ox in range(3):
            pool_out[:, :, oy, ox] = padded[:, :, oy * 2 : oy * 2 + 2, ox * 2 : ox * 2 + 2].max(
                axis=(2, 3)
            )
    flat_out = pool_out.reshape(3, 18)
    ref = flat_out @ wg.T + bg
    assert np.allclose(acts, ref, atol=1e-9)

    cal = rng.uniform(0.0, 1.0, size=(16, 36))
    graph = model.quantize(cal, n_bits=4)
    pool_specs = [n.op for n in graph.nodes if n.op.op_type == "Pool"]
    assert len(pool_specs) == 1
    assert pool_specs[0].padding == 1

    xq = _quantize_input(model, x[0])
    out = evaluate_graph_int(graph, {"x": xq})
    assert len(out[graph.outputs[0]]) == 4
