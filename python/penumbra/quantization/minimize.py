"""Deterministic bit-plan coordinate descent for Phase-10 bit-width minimization.

Penumbra has no compiler/optimizer by design (``PROJECT.md`` §3): graphs are small,
layers are shallow, and radix arithmetic allows deterministic search. This module implements
coordinate descent over the three precision knobs that determine ``num_blocks``:

1. ``max_mult_bits`` — caps the Requant fixed-point multiplier and transient internal peak;
2. ``input_bits`` — graph-level input precision;
3. ``weight_bits`` — per-accumulator-layer weight bit widths.

Lowering any coordinate narrows the accumulator, so ``num_blocks`` is monotone non-increasing
in principle. However, lowering ``weight_bits`` changes the calibrated accumulator scale, which
re-runs ``choose_requant_params`` and can pick a larger multiplier, so the ``num_blocks`` guard
is load-bearing and must not be dropped. A ``build`` failure is treated as a rejected step,
not an error: ``Model.quantize`` legitimately raises when a precision combination is infeasible
(e.g., the ratio > 1 rejection or a bit-width budget check).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from penumbra.bitwidth import minimal_num_blocks
from penumbra.ir import Graph

_ALLOWED_KNOBS = {"max_mult_bits", "input_bits", "weight_bits"}


@dataclass(frozen=True)
class BitPlan:
    """One candidate precision assignment. ``weight_bits`` is per accumulator layer, in order."""

    input_bits: int
    weight_bits: tuple[int, ...]
    max_mult_bits: int


@dataclass(frozen=True)
class BitPlanResult:
    """Outcome of one evaluated candidate plan."""

    plan: BitPlan
    num_blocks: int
    accuracy: float
    accepted: bool
    # "baseline" | "accepted" | "accuracy floor" | "num_blocks regressed" | "build failed: <msg>"
    reason: str


def minimize_bit_widths(
    *,
    build: Callable[[BitPlan], tuple[Graph, float]],
    start: BitPlan,
    knobs: Sequence[str] = ("max_mult_bits", "input_bits", "weight_bits"),
    accuracy_tolerance: float = 0.01,
    min_input_bits: int = 2,
    min_weight_bits: int = 2,
    min_mult_bits: int = 1,
    max_rounds: int = 2,
) -> tuple[BitPlanResult, list[BitPlanResult]]:
    """Search for the smallest-radix bit plan within an accuracy tolerance.

    Parameters
    ----------
    build:
        Callable mapping candidate ``BitPlan`` to ``(Graph, accuracy)``. Scoring must use
        the calibration/training split, never the test split.
    start:
        Initial baseline bit plan.
    knobs:
        Ordered sequence of knob names to optimize (subset of
        ``{"max_mult_bits", "input_bits", "weight_bits"}``).
    accuracy_tolerance:
        Maximum allowed accuracy drop relative to baseline (``baseline.accuracy - tolerance``).
    min_input_bits:
        Floor for ``input_bits``.
    min_weight_bits:
        Floor for each entry of ``weight_bits``.
    min_mult_bits:
        Floor for ``max_mult_bits``.
    max_rounds:
        Maximum coordinate-descent rounds before stopping.

    Returns
    -------
    (best, trace)
        Best accepted ``BitPlanResult`` and the full trace of all evaluated candidates.
    """
    for knob in knobs:
        if knob not in _ALLOWED_KNOBS:
            raise ValueError(f"unknown knob {knob!r}; must be one of {sorted(_ALLOWED_KNOBS)}")

    # 1. Baseline
    base_graph, base_acc = build(start)
    base_blocks = minimal_num_blocks(base_graph)
    baseline = BitPlanResult(
        plan=start,
        num_blocks=base_blocks,
        accuracy=base_acc,
        accepted=True,
        reason="baseline",
    )
    floor = base_acc - accuracy_tolerance
    best = baseline
    trace: list[BitPlanResult] = [baseline]

    # 2. Expand knobs into ordered coordinate descriptors: (knob_name, layer_index_or_None)
    coordinates: list[tuple[str, int | None]] = []
    for knob in knobs:
        if knob == "max_mult_bits":
            coordinates.append(("max_mult_bits", None))
        elif knob == "input_bits":
            coordinates.append(("input_bits", None))
        elif knob == "weight_bits":
            for idx in range(len(start.weight_bits)):
                coordinates.append(("weight_bits", idx))

    # 3. Coordinate descent rounds
    for _ in range(max_rounds):
        accepted_in_round = False
        for coord_name, coord_idx in coordinates:
            while True:
                current_plan = best.plan
                if coord_name == "max_mult_bits":
                    if current_plan.max_mult_bits <= min_mult_bits:
                        break
                    candidate = BitPlan(
                        input_bits=current_plan.input_bits,
                        weight_bits=current_plan.weight_bits,
                        max_mult_bits=current_plan.max_mult_bits - 1,
                    )
                elif coord_name == "input_bits":
                    if current_plan.input_bits <= min_input_bits:
                        break
                    candidate = BitPlan(
                        input_bits=current_plan.input_bits - 1,
                        weight_bits=current_plan.weight_bits,
                        max_mult_bits=current_plan.max_mult_bits,
                    )
                elif coord_name == "weight_bits":
                    assert coord_idx is not None
                    if current_plan.weight_bits[coord_idx] <= min_weight_bits:
                        break
                    new_wb = list(current_plan.weight_bits)
                    new_wb[coord_idx] -= 1
                    candidate = BitPlan(
                        input_bits=current_plan.input_bits,
                        weight_bits=tuple(new_wb),
                        max_mult_bits=current_plan.max_mult_bits,
                    )
                else:
                    raise AssertionError(f"unexpected coordinate {coord_name}")

                try:
                    cand_graph, cand_acc = build(candidate)
                    cand_blocks = minimal_num_blocks(cand_graph)
                except Exception as exc:
                    trace.append(
                        BitPlanResult(
                            plan=candidate,
                            num_blocks=best.num_blocks,
                            accuracy=0.0,
                            accepted=False,
                            reason=f"build failed: {exc}",
                        )
                    )
                    break

                if cand_acc < floor:
                    trace.append(
                        BitPlanResult(
                            plan=candidate,
                            num_blocks=cand_blocks,
                            accuracy=cand_acc,
                            accepted=False,
                            reason="accuracy floor",
                        )
                    )
                    break

                if cand_blocks > best.num_blocks:
                    trace.append(
                        BitPlanResult(
                            plan=candidate,
                            num_blocks=cand_blocks,
                            accuracy=cand_acc,
                            accepted=False,
                            reason="num_blocks regressed",
                        )
                    )
                    break

                # Accepted
                result = BitPlanResult(
                    plan=candidate,
                    num_blocks=cand_blocks,
                    accuracy=cand_acc,
                    accepted=True,
                    reason="accepted",
                )
                trace.append(result)
                best = result
                accepted_in_round = True

        if not accepted_in_round:
            break

    return best, trace


def format_trace(trace: Sequence[BitPlanResult]) -> str:
    """One aligned line per proposal — plan, num_blocks, accuracy, verdict — for export scripts."""
    lines: list[str] = []
    header = f"{'Plan':<40} | {'Blocks':>6} | {'Accuracy':>8} | {'Verdict':<8} | Reason"
    lines.append(header)
    lines.append("-" * len(header) + "-" * 20)
    for r in trace:
        plan_str = (
            f"in={r.plan.input_bits} w={list(r.plan.weight_bits)} mult={r.plan.max_mult_bits}"
        )
        verdict = "ACCEPT" if r.accepted else "REJECT"
        lines.append(
            f"{plan_str:<40} | {r.num_blocks:>6} | {r.accuracy:>8.4f} | {verdict:<8} | {r.reason}"
        )
    return "\n".join(lines)
