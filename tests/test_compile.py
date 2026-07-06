"""Tests for the automatic ``Requant`` insertion pass (``penumbra.compile``, Phase 4).

The pass owns the "automatic bit-width management" deliverable (`PROJECT.md` §9): the front
end builds the natural layer graph and ``insert_requants`` splices the ``Requant`` nodes that
keep multi-layer accumulators within the radix budget. These tests pin its placement,
idempotency, and loud over-budget failure — none need FHE, so they run instantly.
"""

from __future__ import annotations

import pytest

from penumbra.compile import RequantChannelParams, insert_requants
from penumbra.ir import (
    SCHEMA_VERSION,
    ArgmaxSpec,
    Conv2dSpec,
    Graph,
    LinearSpec,
    Node,
    PoolSpec,
    RequantSpec,
)


def _tiny_cnn(num_blocks: int = 16) -> Graph:
    """Conv(1->2, 3x3) on 6x6 -> Pool(avg 2x2) -> Linear(->10). No Requant nodes yet."""
    return Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=num_blocks,
        input_bits=4,
        inputs=["x"],
        outputs=["logits"],
        nodes=[
            Node(
                name="conv",
                inputs=["x"],
                outputs=["c"],
                op=Conv2dSpec(
                    weights=[[1] * 9, [1] * 9],
                    bias=[0, 0],
                    weight_bits=4,
                    in_h=6,
                    in_w=6,
                    in_channels=1,
                    kernel_h=3,
                    kernel_w=3,
                    stride=1,
                    padding=0,
                ),
            ),
            Node(
                name="pool",
                inputs=["c"],
                outputs=["p"],
                op=PoolSpec(mode="avg", in_h=4, in_w=4, channels=2, pool_h=2, pool_w=2, stride=2),
            ),
            Node(
                name="fc",
                inputs=["p"],
                outputs=["logits"],
                op=LinearSpec(weights=[[1] * 8 for _ in range(10)], bias=[0] * 10, weight_bits=4),
            ),
        ],
    )


def test_inserts_requant_after_consumed_accumulator():
    """A Requant is spliced after the conv (consumed by pool); the terminal head stays wide."""
    g = insert_requants(_tiny_cnn())
    kinds = [n.op.op_type for n in g.nodes]
    assert kinds == ["Conv2d", "Requant", "Pool", "Linear"]

    # The Requant reads the conv's output and the pool now reads the requantized tensor.
    conv, rq, pool, fc = g.nodes
    assert rq.inputs == ["c"]
    assert pool.inputs == rq.outputs, "pool must consume the requantized tensor"
    # The terminal Linear (logits) is NOT requantized — it is decrypted + argmaxed client-side.
    assert fc.outputs == ["logits"]


def test_requant_narrows_below_budget():
    """After insertion every tensor fits the radix (the pass's own budget check passes)."""
    g = insert_requants(_tiny_cnn())
    from penumbra.bitwidth import propagate_bit_widths, radix_capacity_bits

    widths = propagate_bit_widths(g)
    cap = radix_capacity_bits(g.num_blocks)
    assert all(b <= cap for b in widths.values()), widths
    # The requantized tensor is narrowed to a single message block.
    assert widths["c__rq"] == 2


def test_idempotent():
    """Re-running the pass adds nothing (requants only follow non-requantized accumulators)."""
    g1 = insert_requants(_tiny_cnn())
    g2 = insert_requants(g1)
    assert [n.op.op_type for n in g2.nodes] == [n.op.op_type for n in g1.nodes]
    assert g2 == g1


def test_honours_explicit_shift():
    """A calibrated shift passed for a producer node is used verbatim in its Requant."""
    g = insert_requants(_tiny_cnn(), shifts={"conv": 9})
    rq = next(n for n in g.nodes if isinstance(n.op, RequantSpec))
    assert rq.op.shift == 9


def test_over_budget_fails_loudly_naming_layer():
    """A single accumulator wider than the radix raises, naming the offending layer."""
    # num_blocks=2 -> 4-bit radix; the conv's ~14-bit accumulator cannot fit.
    with pytest.raises(ValueError, match="conv"):
        insert_requants(_tiny_cnn(num_blocks=2))


def test_argmax_only_producer_stays_wide():
    """A Linear feeding only an Argmax gets no Requant — the head is left wide (Phase-2 shape)."""
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=16,
        input_bits=4,
        inputs=["x"],
        outputs=["label"],
        nodes=[
            Node(
                name="fc",
                inputs=["x"],
                outputs=["z"],
                op=LinearSpec(weights=[[1] * 8], bias=[0], weight_bits=4),
            ),
            Node(name="amax", inputs=["z"], outputs=["label"], op=ArgmaxSpec(threshold=5)),
        ],
    )
    out = insert_requants(g)
    assert [n.op.op_type for n in out.nodes] == ["Linear", "Argmax"]  # no Requant inserted
    amax = next(n for n in out.nodes if isinstance(n.op, ArgmaxSpec))
    assert amax.inputs == ["z"], "Argmax must read the wide logit"


def test_fanout_argmax_reads_wide_logit_not_requantized():
    """A producer feeding both a narrow head and an Argmax: the Argmax keeps the WIDE logit.

    Regression guard: rewiring used to remap *every* consumer of a requantized producer to `__rq`,
    including the wide-input Argmax — so the Argmax silently thresholded a value clamped to
    `2^act_bits - 1`, making a high threshold unreachable. The narrow consumer must read `z__rq`;
    the Argmax must still read `z`.
    """
    g = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=16,
        input_bits=4,
        inputs=["x"],
        outputs=["head_out", "label"],
        nodes=[
            Node(
                name="fc",
                inputs=["x"],
                outputs=["z"],
                op=LinearSpec(weights=[[1] * 8 for _ in range(4)], bias=[0] * 4, weight_bits=4),
            ),
            # A narrow consumer of z -> forces a Requant on the fc producer.
            Node(
                name="head",
                inputs=["z"],
                outputs=["head_out"],
                op=LinearSpec(weights=[[1] * 4 for _ in range(2)], bias=[0, 0], weight_bits=4),
            ),
            # A wide consumer of the SAME z -> must keep reading the wide logit.
            Node(name="amax", inputs=["z"], outputs=["label"], op=ArgmaxSpec(threshold=5)),
        ],
    )
    out = insert_requants(g)
    assert [n.op.op_type for n in out.nodes] == ["Linear", "Requant", "Linear", "Argmax"]

    rq = next(n for n in out.nodes if isinstance(n.op, RequantSpec))
    head = next(n for n in out.nodes if n.name == "head")
    amax = next(n for n in out.nodes if isinstance(n.op, ArgmaxSpec))
    assert rq.inputs == ["z"]
    assert head.inputs == ["z__rq"], "the narrow head must consume the requantized tensor"
    assert amax.inputs == ["z"], "the Argmax must still read the wide logit, not z__rq"


def test_per_channel_requant_derives_channel_size_from_conv():
    """per_channel params build a per-channel Requant; channel_size = out_h*out_w for the conv."""
    # _tiny_cnn: Conv(1->2, 3x3) on 6x6 -> out 4x4 -> Pool consumes it, so the conv is requantized.
    pc = {"conv": RequantChannelParams(mults=[1, 3], shifts=[6, 5], round_biases=[0, 16])}
    g = insert_requants(_tiny_cnn(), per_channel=pc)
    assert [n.op.op_type for n in g.nodes] == ["Conv2d", "Requant", "Pool", "Linear"]
    rq = next(n for n in g.nodes if isinstance(n.op, RequantSpec))
    assert rq.op.mults == [1, 3]
    assert rq.op.shifts == [6, 5]
    assert rq.op.round_biases == [0, 16]
    assert rq.op.channel_size == 16, "conv out is 4x4 -> 16 elements per output channel"
    # Idempotent: the already-per-channel Requant is not touched on a re-run.
    assert insert_requants(g, per_channel=pc) == g


def test_per_channel_requant_wrong_channel_count_fails_loudly():
    """A per-channel param list whose length != output channels raises, naming the layer."""
    # The conv has 2 output channels; supplying 3 multipliers is a wiring bug.
    pc = {"conv": RequantChannelParams(mults=[1, 3, 5], shifts=[6, 5, 4], round_biases=[0, 0, 0])}
    with pytest.raises(ValueError, match="conv.*per output channel|per output channel"):
        insert_requants(_tiny_cnn(), per_channel=pc)


def test_requant_channel_params_defaults_round_biases_to_zero():
    """Omitting round_biases defaults them to one 0 per channel (truncation) — the common call."""
    pc = RequantChannelParams(mults=[1, 3], shifts=[6, 5])  # no round_biases
    assert pc.round_biases == [0, 0]
    # It drives a valid per-channel Requant (round bias 0 = the scalar API's default).
    g = insert_requants(_tiny_cnn(), per_channel={"conv": pc})
    rq = next(n for n in g.nodes if isinstance(n.op, RequantSpec))
    assert rq.op.round_biases == [0, 0]


def test_requant_channel_params_rejects_mismatched_lengths():
    """mults/shifts (and any supplied round_biases) must match in length, else raise."""
    with pytest.raises(ValueError, match="one shift per mult"):
        RequantChannelParams(mults=[1, 3], shifts=[6])
    with pytest.raises(ValueError, match="round_biases"):
        RequantChannelParams(mults=[1, 3], shifts=[6, 5], round_biases=[0])
