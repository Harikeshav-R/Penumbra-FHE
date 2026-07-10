"""The loud-failure gate for the ONNX front door (``AGENTS.md`` §1.4), Phase 6.

Unsupported ops, unsupported attributes, and out-of-scope topologies must be caught at
``load_onnx()`` time — never mysteriously at runtime — with actionable messages, and **all at
once** where feasible (not one at a time). This mirrors the IR conformance discipline
(``tests/test_ir_conformance.py::test_from_dict_rejects_unknown_op_type``) and is the ROADMAP
Phase 6 exit criterion "unsupported ops fail loudly at load time with actionable messages".

Hermetic: builds ONNX in memory via ``onnx.helper`` (``onnx`` is a core dep), no framework.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import penumbra as fhe
from penumbra.onnx_loader import UnsupportedModelError

OPSET = 13


def _model(nodes, inits, inputs, outputs, tmp_path, opset=OPSET, check=False):
    graph = helper.make_graph(nodes, "g", inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", opset)])
    if check:
        onnx.checker.check_model(model)
    path = str(tmp_path / "m.onnx")
    onnx.save(model, path)
    return path


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _f32(arr, name):
    return numpy_helper.from_array(arr.astype(np.float32), name)


def test_lists_all_unsupported_problems_at_once(tmp_path):
    """A model with BatchNorm + a non-ReLU activation + a residual Add reports all three at once."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 4))
    scale = np.ones(4)
    b = np.zeros(4)
    mean = np.zeros(4)
    var = np.ones(4)
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"], name="mm"),
        helper.make_node("BatchNormalization", ["h", "s", "bb", "m", "v"], ["bn"], name="bn1"),
        helper.make_node("Tanh", ["bn"], ["t"], name="tanh1"),
        helper.make_node(
            "Add", ["t", "h"], ["y"], name="res"
        ),  # residual: both operands activations
    ]
    inits = [_f32(w, "w"), _f32(scale, "s"), _f32(b, "bb"), _f32(mean, "m"), _f32(var, "v")]
    path = _model(nodes, inits, [_vi("x", [1, 4])], [_vi("y", [1, 4])], tmp_path)

    with pytest.raises(UnsupportedModelError) as exc:
        fhe.load_onnx(path)
    problems = exc.value.problems
    # All three offenders named in one report, each actionable.
    joined = "\n".join(problems)
    assert "BatchNormalization" in joined and "'bn1'" in joined
    assert "Tanh" in joined and "'tanh1'" in joined
    assert "Add" in joined and "'res'" in joined
    assert len(problems) == 3


def test_branching_graph_fails_loudly(tmp_path):
    """A tensor consumed by two nodes (fan-out) is rejected as branching (Phase 8)."""
    rng = np.random.default_rng(1)
    w1 = rng.normal(size=(4, 4))
    w2 = rng.normal(size=(4, 4))
    # x feeds two Gemms — fan-out branching. (Their outputs go to separate graph outputs.)
    nodes = [
        helper.make_node("Gemm", ["x", "w1"], ["y1"], name="a", transB=1),
        helper.make_node("Gemm", ["x", "w2"], ["y2"], name="b", transB=1),
    ]
    inits = [_f32(w1, "w1"), _f32(w2, "w2")]
    path = _model(
        nodes, inits, [_vi("x", [1, 4])], [_vi("y1", [1, 4]), _vi("y2", [1, 4])], tmp_path
    )
    with pytest.raises(UnsupportedModelError, match="output"):
        # Two graph outputs is itself rejected; if single-output, fan-out is caught in the walker.
        fhe.load_onnx(path)


def test_fanout_branching_single_output_fails_loudly(tmp_path):
    """x feeding two nodes that reconverge is branching even with a single graph output."""
    rng = np.random.default_rng(2)
    w1 = rng.normal(size=(4, 4))
    w2 = rng.normal(size=(4, 4))
    nodes = [
        helper.make_node("Gemm", ["x", "w1"], ["a"], name="ga", transB=1),
        helper.make_node("Gemm", ["x", "w2"], ["b"], name="gb", transB=1),  # fan-out on x
        helper.make_node("Add", ["a", "b"], ["y"], name="merge"),  # reconverge (residual)
    ]
    inits = [_f32(w1, "w1"), _f32(w2, "w2")]
    path = _model(nodes, inits, [_vi("x", [1, 4])], [_vi("y", [1, 4])], tmp_path)
    with pytest.raises(UnsupportedModelError, match="branch|fan-out|residual"):
        fhe.load_onnx(path)


def test_opset_out_of_range_fails_loudly(tmp_path):
    """An opset below the supported range is rejected with an actionable message."""
    w = np.eye(4)
    nodes = [helper.make_node("Gemm", ["x", "w"], ["y"], name="fc", transB=1)]
    path = _model(nodes, [_f32(w, "w")], [_vi("x", [1, 4])], [_vi("y", [1, 4])], tmp_path, opset=8)
    with pytest.raises(UnsupportedModelError, match="opset"):
        fhe.load_onnx(path)


def test_conv_group2_attribute_fails_loudly(tmp_path):
    """A grouped conv (group != 1) is rejected at load time (grouped/depthwise conv is Phase 8)."""
    rng = np.random.default_rng(3)
    wc = rng.normal(size=(4, 1, 3, 3))  # group=2 over 2 in-channels
    nodes = [helper.make_node("Conv", ["x", "wc"], ["y"], name="c", group=2, strides=[1, 1])]
    path = _model(
        nodes, [_f32(wc, "wc")], [_vi("x", [1, 2, 8, 8])], [_vi("y", [1, 4, 6, 6])], tmp_path
    )
    with pytest.raises(UnsupportedModelError, match="group=2"):
        fhe.load_onnx(path)


def test_nonterminal_softmax_fails_loudly(tmp_path):
    """A Softmax that is NOT the terminal node is a real activation and is rejected."""
    rng = np.random.default_rng(4)
    w1 = rng.normal(size=(4, 4))
    w2 = rng.normal(size=(4, 4))
    nodes = [
        helper.make_node("Gemm", ["x", "w1"], ["h"], name="fc1", transB=1),
        helper.make_node("Softmax", ["h"], ["s"], name="sm", axis=1),  # mid-graph
        helper.make_node("Gemm", ["s", "w2"], ["y"], name="fc2", transB=1),
    ]
    inits = [_f32(w1, "w1"), _f32(w2, "w2")]
    path = _model(nodes, inits, [_vi("x", [1, 4])], [_vi("y", [1, 4])], tmp_path)
    with pytest.raises(UnsupportedModelError, match="terminal"):
        fhe.load_onnx(path)


def test_terminal_relu_fails_in_quantize(tmp_path):
    """A ReLU on the terminal accumulator has no Requant to fuse into; Model.quantize rejects it.

    The loader lowers it (a ReLU is a valid Activation); the loud failure is Model.quantize's
    terminal-ReLU guard (``model.py``). This documents that the front door defers that specific
    check to the same place the hand-built path does, so the message is uniform.
    """
    w = np.eye(4)
    nodes = [
        helper.make_node("Gemm", ["x", "w"], ["h"], name="fc", transB=1),
        helper.make_node("Relu", ["h"], ["y"], name="relu"),  # terminal ReLU on the logit head
    ]
    path = _model(nodes, [_f32(w, "w")], [_vi("x", [1, 4])], [_vi("y", [1, 4])], tmp_path)
    model = fhe.load_onnx(path)
    with pytest.raises(ValueError, match="terminal ReLU"):
        model.quantize(np.random.default_rng(0).uniform(0, 16, size=(16, 4)), n_bits=4)
