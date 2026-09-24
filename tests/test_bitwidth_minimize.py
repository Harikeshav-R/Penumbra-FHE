"""Tests for the Phase-10 bit-plan minimization search (penumbra.quantization.minimize).

NumPy-only and hermetic (no ml extra, no FHE). Verifies the deterministic coordinate
descent algorithm against synthetic models with known accuracy cliffs and budget constraints.
"""

from __future__ import annotations

import pytest

from penumbra.ir import Graph, LinearSpec, Node, RequantSpec
from penumbra.quantization.minimize import (
    BitPlan,
    BitPlanResult,
    format_trace,
    minimize_bit_widths,
)


def _make_synthetic_graph(plan: BitPlan) -> Graph:
    mult = max(1, (1 << plan.max_mult_bits) - 1)
    nodes = [
        Node(
            name="fc",
            op=LinearSpec(
                weights=[[1]],
                bias=[0],
                weight_bits=plan.weight_bits[0],
            ),
            inputs=["x"],
            outputs=["fc_out"],
        ),
        Node(
            name="rq",
            op=RequantSpec(
                shift=plan.max_mult_bits,
                out_bits=2,
                clamp_lut=[0, 1, 2, 3],
                mult=mult,
                round_bias=0,
            ),
            inputs=["fc_out"],
            outputs=["y"],
        ),
    ]
    return Graph(
        schema_version="0.6.0",
        num_blocks=8,
        input_bits=plan.input_bits,
        inputs=["x"],
        outputs=["y"],
        nodes=nodes,
    )


def test_descends_to_cliff_and_stops():
    """Search descends until the accuracy cliff and stops at the last viable plan."""
    start = BitPlan(input_bits=4, weight_bits=(6,), max_mult_bits=5)

    def build(plan: BitPlan) -> tuple[Graph, float]:
        graph = _make_synthetic_graph(plan)
        # Accuracy holds at 0.90 down to weight_bits >= 4, then collapses to 0.50 at 3-bit.
        # Accuracy also drops if input_bits < 3.
        acc = 0.90
        if plan.weight_bits[0] < 4:
            acc = 0.50
        if plan.input_bits < 3:
            acc = 0.50
        return graph, acc

    best, trace = minimize_bit_widths(
        build=build,
        start=start,
        accuracy_tolerance=0.01,
        min_input_bits=2,
        min_weight_bits=2,
        min_mult_bits=1,
    )

    baseline = trace[0]
    assert baseline.reason == "baseline"
    assert best.plan.weight_bits == (4,)
    assert best.plan.input_bits == 3
    assert best.num_blocks < baseline.num_blocks
    assert trace[-1].reason == "accuracy floor"
    assert trace[-1].plan.weight_bits == (3,)
    assert not trace[-1].accepted

    cliff_entries = [
        r for r in trace if r.plan.weight_bits == (3,) and r.reason == "accuracy floor"
    ]
    assert len(cliff_entries) >= 1
    assert all(not r.accepted for r in cliff_entries)


def test_never_returns_plan_below_accuracy_floor():
    """Best plan must always satisfy accuracy >= baseline - tolerance."""
    start = BitPlan(input_bits=4, weight_bits=(4,), max_mult_bits=5)

    def build(plan: BitPlan) -> tuple[Graph, float]:
        graph = _make_synthetic_graph(plan)
        # Continuous linear degradation with each bit dropped
        bits_lost = (4 - plan.input_bits) + (4 - plan.weight_bits[0]) + (5 - plan.max_mult_bits)
        acc = 0.95 - 0.005 * bits_lost
        return graph, acc

    best, trace = minimize_bit_widths(
        build=build,
        start=start,
        accuracy_tolerance=0.01,
    )

    baseline_acc = trace[0].accuracy
    assert best.accuracy >= baseline_acc - 0.01
    for r in trace:
        if r.accepted:
            assert r.accuracy >= baseline_acc - 0.01


def test_build_failure_recorded_as_rejected_trace():
    """A build that raises an exception is recorded as rejected and search continues."""
    start = BitPlan(input_bits=4, weight_bits=(4,), max_mult_bits=3)

    def build(plan: BitPlan) -> tuple[Graph, float]:
        if plan.max_mult_bits == 2:
            raise ValueError("simulated build failure at mult=2")
        return _make_synthetic_graph(plan), 0.90

    best, trace = minimize_bit_widths(
        build=build,
        start=start,
        accuracy_tolerance=0.01,
        knobs=("max_mult_bits", "input_bits"),
    )

    failed_entries = [r for r in trace if "simulated build failure" in r.reason]
    assert len(failed_entries) >= 1
    assert all(not r.accepted for r in failed_entries)
    # Search should have continued to input_bits
    assert any(r.plan.input_bits < 4 for r in trace)


def test_candidate_with_regressed_num_blocks_rejected():
    """Candidate where num_blocks increases is rejected even with high accuracy."""
    start = BitPlan(input_bits=4, weight_bits=(4,), max_mult_bits=3)

    def build(plan: BitPlan) -> tuple[Graph, float]:
        if plan.input_bits == 3:
            # Artificially widen the graph
            graph = _make_synthetic_graph(
                BitPlan(input_bits=10, weight_bits=(10,), max_mult_bits=7)
            )
            return graph, 0.99
        return _make_synthetic_graph(plan), 0.90

    best, trace = minimize_bit_widths(
        build=build,
        start=start,
        accuracy_tolerance=0.01,
        knobs=("input_bits",),
    )

    regressed = [r for r in trace if r.reason == "num_blocks regressed"]
    assert len(regressed) == 1
    assert not regressed[0].accepted
    assert best.plan.input_bits == 4


def test_deterministic_search():
    """Two identical invocations produce identical best plan and trace."""
    start = BitPlan(input_bits=4, weight_bits=(5, 5), max_mult_bits=4)

    def build(plan: BitPlan) -> tuple[Graph, float]:
        graph = _make_synthetic_graph(plan)
        acc = 0.90 - 0.002 * (4 - plan.max_mult_bits)
        if plan.weight_bits[0] < 4 or plan.weight_bits[1] < 4:
            acc = 0.60
        return graph, acc

    best1, trace1 = minimize_bit_widths(build=build, start=start)
    best2, trace2 = minimize_bit_widths(build=build, start=start)

    assert best1 == best2
    assert trace1 == trace2


def test_unknown_knob_raises_value_error():
    """Invalid knob name raises ValueError naming the knob."""
    start = BitPlan(input_bits=4, weight_bits=(4,), max_mult_bits=4)

    def build(plan: BitPlan) -> tuple[Graph, float]:
        return _make_synthetic_graph(plan), 0.90

    with pytest.raises(ValueError, match="bogus_knob"):
        _ = minimize_bit_widths(
            build=build,
            start=start,
            knobs=("bogus_knob",),
        )


def test_format_trace():
    """format_trace produces formatted table output."""
    plan = BitPlan(input_bits=4, weight_bits=(4,), max_mult_bits=3)
    trace = [
        BitPlanResult(plan=plan, num_blocks=8, accuracy=0.92, accepted=True, reason="baseline"),
        BitPlanResult(plan=plan, num_blocks=7, accuracy=0.91, accepted=True, reason="accepted"),
    ]
    table = format_trace(trace)
    assert "Plan" in table
    assert "Blocks" in table
    assert "Accuracy" in table
    assert "ACCEPT" in table
