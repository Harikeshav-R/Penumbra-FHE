"""Tests for the automatic ``Requant`` insertion pass (``penumbra.compile``, Phase 4).

The pass owns the "automatic bit-width management" deliverable (`PROJECT.md` §9): the front
end builds the natural layer graph and ``insert_requants`` splices the ``Requant`` nodes that
keep multi-layer accumulators within the radix budget. These tests pin its placement,
idempotency, and loud over-budget failure — none need FHE, so they run instantly.
"""

from __future__ import annotations

import pytest

from penumbra.compile import insert_requants
from penumbra.ir import SCHEMA_VERSION, Conv2dSpec, Graph, LinearSpec, Node, PoolSpec, RequantSpec


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
