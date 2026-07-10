"""Differential lowering-fidelity oracle for the ONNX front door (Phase 6, appendix S4).

The committed verification chain (float ``Model.forward`` -> quantized integer oracle -> FHE)
shares one blind spot: our lowered float forward **is** the lowering, so a lowering-fidelity bug
(a wrong ``transB``, a Conv layout mismatch, a mis-folded bias) is invisible — both the float and
the quantized sides agree because both came from the same buggy lower. This test closes that gap
with an **independent** oracle: it compares the lowered ``Model``'s chained float forward against
**onnxruntime**'s inference of the *original* ``.onnx``. A mismatch catches exactly the class of
bug the golden invariant structurally cannot see.

Guarded by ``pytest.importorskip("onnxruntime")``: onnxruntime is in the optional ``ml`` extra, so
this **skips in CI** (which installs dev-only) and runs locally under ``uv run --extra ml pytest``.

Comparison-model constraints (baked into the fixtures below so the two sides are directly
comparable): the models are **logit-terminated** (no dropped Softmax tail) and **avg-pool-free**
(our ``Pool('avg')`` emits the window *sum*, not the mean, deferring 1/k to the next Requant —
which has no float-forward counterpart in onnxruntime's true averaging).
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import penumbra as fhe

pytest.importorskip("onnxruntime")
import onnxruntime as ort  # noqa: E402

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


def _lowered_forward(model: fhe.Model, x: np.ndarray) -> np.ndarray:
    """Chain the lowered float layers exactly as ``Model._calibrate_accumulators`` does."""
    acts = x
    for layer in model.layers:
        acts = layer.forward(acts)
    return acts


def _ort_run(path: str, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x.astype(np.float32)})[0]


def test_fidelity_gemm_transb1(tmp_path):
    """Lowered forward of a Gemm(transB=1)->Relu->Gemm(transB=1) matches onnxruntime."""
    rng = np.random.default_rng(0)
    w1 = rng.normal(size=(6, 4)).astype(np.float32)
    b1 = rng.normal(size=6).astype(np.float32)
    w2 = rng.normal(size=(3, 6)).astype(np.float32)
    b2 = rng.normal(size=3).astype(np.float32)
    nodes = [
        helper.make_node("Gemm", ["x", "w1", "b1"], ["h0"], name="fc1", transB=1),
        helper.make_node("Relu", ["h0"], ["h1"], name="relu1"),
        helper.make_node("Gemm", ["h1", "w2", "b2"], ["y"], name="fc2", transB=1),
    ]
    inits = [_f32(w1, "w1"), _f32(b1, "b1"), _f32(w2, "w2"), _f32(b2, "b2")]
    path = _save(nodes, inits, [_vi("x", [1, 4])], [_vi("y", [1, 3])], tmp_path)

    model = fhe.load_onnx(path)
    x = rng.uniform(-2.0, 2.0, size=(1, 4)).astype(np.float32)
    got = _lowered_forward(model, x.astype(np.float64))
    ref = _ort_run(path, x)
    assert np.allclose(got, ref, atol=1e-4), f"lowering diverges from ONNX runtime:\n{got}\n{ref}"


def test_fidelity_gemm_transb0(tmp_path):
    """transB=0 (weight (n_in,n_out)) must be transposed correctly — the classic lowering bug."""
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 6)).astype(np.float32)  # (n_in, n_out)
    b = rng.normal(size=6).astype(np.float32)
    nodes = [helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="fc", transB=0)]
    path = _save(
        nodes, [_f32(w, "w"), _f32(b, "b")], [_vi("x", [1, 4])], [_vi("y", [1, 6])], tmp_path
    )

    model = fhe.load_onnx(path)
    x = rng.uniform(-2.0, 2.0, size=(1, 4)).astype(np.float32)
    assert np.allclose(_lowered_forward(model, x.astype(np.float64)), _ort_run(path, x), atol=1e-4)


def test_fidelity_matmul_add_bias(tmp_path):
    """A mis-folded MatMul+Add bias would show up here against onnxruntime."""
    rng = np.random.default_rng(2)
    w = rng.normal(size=(4, 5)).astype(np.float32)  # (n_in, n_out)
    bias = rng.normal(size=5).astype(np.float32)
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"], name="mm"),
        helper.make_node("Add", ["h", "bias"], ["y"], name="addbias"),
    ]
    path = _save(
        nodes, [_f32(w, "w"), _f32(bias, "bias")], [_vi("x", [1, 4])], [_vi("y", [1, 5])], tmp_path
    )

    model = fhe.load_onnx(path)
    x = rng.uniform(-2.0, 2.0, size=(1, 4)).astype(np.float32)
    assert np.allclose(_lowered_forward(model, x.astype(np.float64)), _ort_run(path, x), atol=1e-4)


def test_fidelity_conv_layout(tmp_path):
    """A Conv layout mismatch (weight/stride/channel-major flattening) is caught vs onnxruntime."""
    rng = np.random.default_rng(3)
    wc = rng.normal(size=(3, 2, 3, 3)).astype(np.float32)  # (out, in, kh, kw), 2 in-channels
    bc = rng.normal(size=3).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["x", "wc", "bc"], ["c"], name="conv", strides=[2, 2], group=1),
        helper.make_node("Relu", ["c"], ["y"], name="relu"),
    ]
    path = _save(
        nodes,
        [_f32(wc, "wc"), _f32(bc, "bc")],
        [_vi("x", [1, 2, 8, 8])],
        [_vi("y", [1, 3, 3, 3])],
        tmp_path,
    )

    model = fhe.load_onnx(path)
    x = rng.uniform(-2.0, 2.0, size=(1, 2, 8, 8)).astype(np.float32)
    got = _lowered_forward(model, x.reshape(1, -1).astype(np.float64))  # flat channel-major
    ref = _ort_run(path, x).reshape(1, -1)  # onnxruntime output is (1,3,3,3) channel-major
    assert np.allclose(got, ref, atol=1e-4), f"conv lowering diverges:\n{got}\n{ref}"
