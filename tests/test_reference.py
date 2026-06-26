"""Tests for the integer reference evaluator (``penumbra.reference``, Phase 5).

:func:`penumbra.reference.evaluate_graph_int` is the quantized-cleartext oracle — the function
``Model.quantize`` self-verifies against and the examples commit as ``expected_*``. It must
mirror the Rust runtime's integer semantics exactly (``runtime/src/ops``); the Rust golden tests
pin both to the committed fixtures, while these pin the Python oracle to the *committed Phase-4
CNN fixture* (a known-good graph + expected logits) so the oracle cannot silently drift.

NumPy-only, hermetic, fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from penumbra.ir import (
    SCHEMA_VERSION,
    ActivationSpec,
    ArgmaxSpec,
    Graph,
    LinearSpec,
    Node,
    RequantSpec,
)
from penumbra.reference import evaluate_graph_int

CNN_FIXTURE = (
    Path(__file__).resolve().parent.parent / "examples" / "mnist" / "phase4_cnn_fixture.json"
)


def test_matches_committed_cnn_fixture_logits():
    """The oracle reproduces the committed Phase-4 CNN's expected_logits (drift guard)."""
    fx = json.loads(CNN_FIXTURE.read_text())
    g = Graph.from_dict(fx["graph"])
    for x, expected in zip(fx["test_inputs"], fx["expected_logits"], strict=True):
        out = evaluate_graph_int(g, {"x": x})
        assert out[g.outputs[0]] == expected


def test_linear_then_argmax():
    """Linear -> Argmax: the threshold yields a 0/1 label from the wide logit."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=8,
        input_bits=4,
        inputs=["x"],
        outputs=["label"],
        nodes=[
            Node(
                name="fc",
                inputs=["x"],
                outputs=["logit"],
                op=LinearSpec(weights=[[2, -1, 3]], bias=[-5], weight_bits=4),
            ),
            Node(name="head", inputs=["logit"], outputs=["label"], op=ArgmaxSpec(threshold=0)),
        ],
    )
    # logit = 2*3 + (-1)*1 + 3*2 - 5 = 6 - 1 + 6 - 5 = 6 >= 0 -> label 1.
    assert evaluate_graph_int(g, {"x": [3, 1, 2]})["label"] == [1]
    # logit = 2*0 -1*5 + 3*0 - 5 = -10 < 0 -> label 0.
    assert evaluate_graph_int(g, {"x": [0, 5, 0]})["label"] == [0]


def test_requant_multiply_then_round_shift_matches_formula():
    """The reference Requant applies clamp((max(x,0)*mult + round_bias) >> shift, 0, ceil)."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=8,
        input_bits=10,
        inputs=["x"],
        outputs=["y"],
        nodes=[
            Node(
                name="rq",
                inputs=["x"],
                outputs=["y"],
                op=RequantSpec(shift=5, mult=3, round_bias=16, out_bits=2, clamp_lut=[0, 1, 2, 3]),
            )
        ],
    )

    def cleartext(v: int) -> int:
        return min(max((max(v, 0) * 3 + 16) >> 5, 0), 3)

    xs = [-100, 0, 1, 7, 16, 64, 200]
    assert evaluate_graph_int(g, {"x": xs})["y"] == [cleartext(v) for v in xs]


def test_activation_out_of_domain_fails_loudly():
    """An Activation index outside its LUT domain raises (a quantization/wiring bug)."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=4,
        input_bits=2,
        inputs=["x"],
        outputs=["y"],
        nodes=[
            Node(
                name="act",
                inputs=["x"],
                outputs=["y"],
                op=ActivationSpec(lut=[0, 1, 2, 3], output_bits=2),
            )
        ],
    )
    assert evaluate_graph_int(g, {"x": [0, 3]})["y"] == [0, 3]
    with pytest.raises(ValueError, match="outside the LUT domain"):
        evaluate_graph_int(g, {"x": [4]})  # 4 is past the 4-entry table
