"""Property tests for random small-model generation (ROADMAP Phase 11)."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from penumbra.bitwidth import check_bit_width_budget, minimal_num_blocks, propagate_bit_widths
from penumbra.ir import Graph
from penumbra.testing import (
    CKKS_ARGMAX_INPUT_BITS,
    CKKS_COMPARE_INPUT_BITS,
    CKKS_REQUANT_INPUT_BITS,
    FAMILIES,
    MAX_NUM_BLOCKS,
    random_model,
)


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    seed=st.integers(0, 2**31 - 1),
    family=st.sampled_from(FAMILIES),
)
def test_random_model_invariants_hold(seed: int, family: str) -> None:
    """Every generated model satisfies IR, radix, and CKKS budget invariants."""
    m1 = random_model(seed, family)
    m2 = random_model(seed, family)

    # Determinism
    assert m1 == m2, f"random_model not deterministic for {family} seed {seed}"

    g = m1.graph

    # JSON round-trip
    assert Graph.from_json(g.to_json()) == g, f"IR JSON round-trip failed on {m1.name}"

    # Single input and single output
    assert len(g.inputs) == 1 and g.inputs[0] == "x", f"{m1.name} inputs must be ['x']"
    assert len(g.outputs) == 1, f"{m1.name} outputs must have length 1"

    # Minimal num_blocks <= MAX_NUM_BLOCKS
    assert g.num_blocks == minimal_num_blocks(
        g
    ), f"{m1.name} num_blocks {g.num_blocks} != minimal {minimal_num_blocks(g)}"
    assert g.num_blocks <= MAX_NUM_BLOCKS, f"{m1.name} num_blocks {g.num_blocks} > {MAX_NUM_BLOCKS}"

    # Bit-width budget
    check_bit_width_budget(g)

    # CKKS polynomial approximation domain limits
    widths = propagate_bit_widths(g)
    for node in g.nodes:
        in_w = widths[node.inputs[0]]
        if node.op.op_type == "Requant":
            assert (
                in_w <= CKKS_REQUANT_INPUT_BITS
            ), f"{m1.name} node {node.name} Requant input {in_w} > {CKKS_REQUANT_INPUT_BITS}"
        elif node.op.op_type == "Argmax":
            assert (
                in_w <= CKKS_ARGMAX_INPUT_BITS
            ), f"{m1.name} node {node.name} Argmax input {in_w} > {CKKS_ARGMAX_INPUT_BITS}"
        elif node.op.op_type == "Compare":
            assert (
                in_w <= CKKS_COMPARE_INPUT_BITS
            ), f"{m1.name} node {node.name} Compare input {in_w} > {CKKS_COMPARE_INPUT_BITS}"

    # Expected outputs evaluate without error
    outputs = m1.expected_outputs()
    assert len(outputs) == len(m1.test_inputs)


def test_random_model_rejects_unknown_family() -> None:
    """An unknown family raises ValueError naming the family and valid options."""
    with pytest.raises(ValueError, match="unknown family 'transformer'"):
        random_model(42, "transformer")
