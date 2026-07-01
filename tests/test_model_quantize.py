"""Tests for the user-facing quantization service: ``Model.quantize`` / ``export`` (Phase 5).

These pin the orchestration in :mod:`penumbra.model`: a float :class:`~penumbra.model.Model`
calibrates on data, quantizes to an int IR graph with auto-inserted (and rescale-calibrated)
Requants, sizes the radix to fit, self-verifies, and round-trips through ``export``. No FHE and
no ML stack (NumPy-only, hermetic) — the FHE bit-for-bit gate lives in the Rust golden tests; a
new ``Model`` golden fixture is added with the real-MNIST example.

The headline contract is that the produced int graph is **self-consistent**: it fits its radix
budget and the integer oracle (:mod:`penumbra.reference`) evaluates it without an out-of-domain
Activation or wiring error — exactly the self-verify the service runs inside ``quantize``.
"""

from __future__ import annotations

import numpy as np
import pytest

from penumbra import Conv2d, Linear, Model, Pool
from penumbra.bitwidth import check_bit_width_budget, radix_capacity_bits
from penumbra.ir import Graph
from penumbra.layers import Activation
from penumbra.reference import evaluate_graph_int


def _relu(x: float) -> float:
    return max(x, 0.0)


def test_quantize_linear_only_model_builds_valid_graph():
    """A single Linear model quantizes to a one-node graph that fits its radix and evaluates."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=(3, 8))
    b = rng.normal(size=3)
    model = Model([Linear(weight=w, bias=b)], input_bits=4)

    cal = rng.uniform(0.0, 16.0, size=(64, 8))
    graph = model.quantize(cal, n_bits=4)

    assert [n.op.op_type for n in graph.nodes] == ["Linear"]
    check_bit_width_budget(graph)  # raises if over budget
    # The integer oracle evaluates a sample without error and returns 3 logits.
    xq = (cal[0] / model.input_scale).round().astype(int).tolist()
    out = evaluate_graph_int(graph, {"x": xq})
    assert len(out[graph.outputs[0]]) == 3


def test_quantize_cnn_inserts_requant_and_fits_budget():
    """Conv -> ReLU -> Pool -> Linear: the conv gets a fused-ReLU Requant; the head stays wide."""
    rng = np.random.default_rng(1)
    conv_w = rng.normal(size=(2, 1, 3, 3))
    head_w = rng.normal(size=(10, 8))
    head_b = rng.normal(size=10)

    model = Model(
        [
            Conv2d(weight=conv_w, in_h=6, in_w=6, in_channels=1),
            Activation(_relu),
            Pool("avg", in_h=4, in_w=4, channels=2, pool_h=2, pool_w=2, stride=2),
            Linear(weight=head_w, bias=head_b),
        ],
        input_bits=4,
    )

    cal = rng.uniform(0.0, 16.0, size=(128, 36))  # 6x6 single-channel inputs, flattened
    graph = model.quantize(cal, n_bits=4, act_bits=2)

    kinds = [n.op.op_type for n in graph.nodes]
    assert kinds == ["Conv2d", "Requant", "Pool", "Linear"], kinds
    # The conv's Requant fuses the ReLU; the terminal head is left wide (decrypted + argmaxed).
    check_bit_width_budget(graph)
    widths = {n.outputs[0]: i for i, n in enumerate(graph.nodes)}  # noqa: F841 (smoke)
    # The oracle evaluates a sample to 10 logits, with every intermediate in range.
    xq = (cal[0] / model.input_scale).round().astype(int).tolist()
    out = evaluate_graph_int(graph, {"x": xq})
    assert len(out[graph.outputs[0]]) == 10


def test_quantize_chooses_minimal_num_blocks():
    """The radix is sized to the model, not left at the probe's generous 64 blocks."""
    rng = np.random.default_rng(2)
    model = Model([Linear(weight=rng.normal(size=(2, 8)), bias=np.zeros(2))], input_bits=4)
    cal = rng.uniform(0.0, 16.0, size=(32, 8))
    graph = model.quantize(cal, n_bits=4)
    # A 2-output Linear over 8 4-bit inputs needs ~14 bits -> 7 blocks, far below 64.
    assert graph.num_blocks < 16
    assert max(__import__("penumbra").propagate_bit_widths(graph).values()) <= radix_capacity_bits(
        graph.num_blocks
    )


def test_export_round_trips(tmp_path):
    """export() writes JSON that re-parses to an equal graph (the runtime's input format)."""
    rng = np.random.default_rng(3)
    model = Model([Linear(weight=rng.normal(size=(2, 4)), bias=np.zeros(2))], input_bits=4)
    model.quantize(rng.uniform(0, 16, size=(16, 4)), n_bits=4)

    path = tmp_path / "model.fhe"
    model.export(str(path))
    restored = Graph.from_json(path.read_text())
    assert restored == model.graph


def test_export_before_quantize_fails():
    model = Model([Linear(weight=np.ones((1, 2)), bias=np.zeros(1))])
    with pytest.raises(RuntimeError, match="quantize"):
        model.export("/tmp/never_written.fhe")


def test_empty_model_rejected():
    with pytest.raises(ValueError, match="at least one layer"):
        Model([])


def test_activation_without_accumulator_fails():
    """A leading Activation (no preceding accumulator) fails loudly — the fused path needs one."""
    model = Model([Activation(_relu), Linear(weight=np.ones((1, 4)), bias=np.zeros(1))])
    with pytest.raises(ValueError, match="does not follow an accumulator"):
        model.quantize(np.random.default_rng(0).uniform(0, 16, size=(8, 4)), n_bits=4)


def _float_forward(layers, x):
    """Plain float forward through a list of penumbra float layers (the reference to track)."""
    acts = x
    for layer in layers:
        if isinstance(layer, Activation):
            acts = np.maximum(acts, 0.0)
        else:
            acts = layer.forward(acts)
    return acts


def test_head_quantized_against_post_requant_scale():
    """Regression guard: a post-Requant head must be quantized in the *activation* scale.

    The bug this pins: Model.quantize once threaded the wide *pre-Requant accumulator* scale into
    the head that consumes a fused-ReLU Requant output, mis-scaling the head bias by the requant
    ratio (~89x on the digit example) and swamping the logits.

    We assert the mechanism directly rather than via a noisy accuracy threshold: the head's int
    bias, dequantized back to floats, must be **within a small factor of the true float bias**. If
    the head were quantized against the pre-Requant accumulator scale, ``b_q`` would be off by the
    requant ratio (many tens x), so its dequantized magnitude would be wildly wrong. A non-trivial
    head bias is essential (else there is nothing to mis-scale), so this would actually bite.
    """
    rng = np.random.default_rng(7)
    conv_w = rng.normal(scale=0.5, size=(6, 1, 3, 3))
    head_w = rng.normal(scale=0.3, size=(5, 6 * 3 * 3))  # 6ch * 3x3 (stride-2 on 8x8) = 54 feats
    head_b = rng.normal(scale=2.0, size=5)  # deliberately non-trivial bias — the bug's target

    layers = [
        Conv2d(weight=conv_w, in_h=8, in_w=8, in_channels=1, stride=2),
        Activation(_relu),
        Linear(weight=head_w, bias=head_b),
    ]
    model = Model(list(layers), input_bits=4)
    cal = rng.uniform(0.0, 16.0, size=(128, 64))
    graph = model.quantize(cal, n_bits=6, act_bits=2, per_channel=True)

    head = next(n for n in graph.nodes if n.op.op_type == "Linear")
    b_q = np.array(head.op.bias, dtype=np.float64)

    # Recover the head's bias scale from the model's threaded scales: the head consumes
    # post-Requant activations at ``act_scale`` and multiplies by per-channel weight scales, so
    # its bias lives in units of ``act_scale * weight_scale`` (per output row for per-channel).
    # We reconstruct act_scale the same way quantize() did — from the calibrated post-ReLU peak —
    # and confirm the dequantized bias ``b_q * act_scale * w_scale`` matches the float bias.
    conv_out = _float_forward(layers[:1], cal)  # pre-ReLU conv accumulator (float)
    from penumbra.quantization.calibration import MinMaxObserver
    from penumbra.quantization.spec import symmetric_spec

    obs = MinMaxObserver()  # default calibration
    obs.update(np.maximum(conv_out, 0.0))
    act_scale = obs.magnitude() / ((1 << 2) - 1)  # peak -> top of the 2-bit activation domain
    head_w_specs = symmetric_spec(head_w, 6, signed=True)  # per-tensor scale for magnitude check
    deq_bias = b_q * act_scale * head_w_specs.scale

    # The dequantized bias must be within a modest factor of the true float bias. A pre-Requant
    # mis-scale would blow this up by the requant ratio (tens x), so a 3x tolerance cleanly
    # separates "correct" from "buggy" while allowing per-channel-vs-per-tensor scale slack.
    ratio = np.abs(deq_bias) / (np.abs(head_b) + 1e-9)
    assert np.median(ratio) < 3.0, (
        f"dequantized head bias is {np.median(ratio):.1f}x the float bias (median) — the head is "
        "likely quantized against the pre-Requant accumulator scale, not the activation scale"
    )


def test_calibration_strategies_all_produce_valid_graphs():
    """minmax / percentile / mse calibration each yield a budget-fitting, evaluable graph."""
    rng = np.random.default_rng(8)
    layers = [
        Conv2d(weight=rng.normal(size=(4, 1, 3, 3)), in_h=8, in_w=8, in_channels=1, stride=2),
        Activation(_relu),
        Linear(weight=rng.normal(size=(10, 4 * 3 * 3)), bias=rng.normal(size=10)),
    ]
    cal = rng.uniform(0.0, 16.0, size=(64, 64))
    for strategy in ("minmax", "percentile", "mse"):
        # Layers are stateless w.r.t. quantization (they only hold float weights), so one fresh
        # Model per strategy over the shared layer list is fine.
        model = Model(layers)
        graph = model.quantize(cal, n_bits=6, act_bits=2, per_channel=True, calibration=strategy)
        check_bit_width_budget(graph)
        xq = np.clip(np.round(cal[0] / model.input_scale), 0, 15).astype(np.int64)
        out = evaluate_graph_int(graph, {"x": xq.tolist()})
        assert len(out[graph.outputs[0]]) == 10, strategy


def test_unknown_calibration_rejected():
    model = Model([Linear(weight=np.ones((2, 4)), bias=np.zeros(2))])
    with pytest.raises(ValueError, match="calibration must be one of"):
        model.quantize(np.random.default_rng(0).uniform(0, 16, size=(8, 4)), calibration="bogus")
