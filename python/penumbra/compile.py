"""Compile pass: automatic ``Requant`` insertion + budget check (``PROJECT.md`` §9).

A multi-layer model's accumulators grow ~``log2(N)`` bits per ``Conv2d``/``Linear`` layer, but
a programmable bootstrap is feasible only over a narrow value. So between accumulator layers
the value must be **requantized** back down. Doing this by hand is the manual chore Phase 4
removes: the front end builds the natural layer graph (no ``Requant`` nodes), and this pass
**inserts them automatically** wherever a wide accumulator feeds a downstream op, then checks
the whole graph fits the radix budget — failing loudly and naming the layer if not
(`AGENTS.md` §1.3, §1.4).

## Where a ``Requant`` goes

A ``Requant`` is spliced after every ``Conv2d``/``Linear`` node whose output is **consumed by
another node** (i.e. is not a terminal graph output). That is exactly the "between two
accumulator layers" position: the producer's wide accumulator is narrowed before the next
layer widens it again. A terminal ``Linear`` (the classification head) is left wide — its
logits are decrypted and argmaxed on the client (`PROJECT.md` §11), so they never need to be
LUT-narrow.

## The shift (rescale amount)

``Requant`` is `clamp(max(x >> shift, 0), 0, 2^out_bits - 1)`. The ``shift`` sets the output
*scale*, so for accuracy it should come from the quantization service's calibration (the
*typical* accumulator magnitude), not the worst-case bit-width. Callers that have calibrated
scales pass them via ``shifts`` (keyed by producer node name); otherwise the pass falls back
to a bit-width heuristic (``incoming_bits - out_bits``) that is *exactness*-safe — the
cleartext oracle and FHE use the identical shift, so the golden invariant holds either way —
but may be coarse. Either way ``out_bits`` is ``MESSAGE_BITS`` (a single LUT-able block).

The pass is **idempotent**: a graph that already has its ``Requant``s (e.g. re-run) is
unchanged, because requants are only inserted after ``Conv2d``/``Linear``, never after a
``Requant``.
"""

from __future__ import annotations

from dataclasses import replace

from penumbra.bitwidth import (
    MESSAGE_BITS,
    check_bit_width_budget,
    propagate_bit_widths,
    radix_capacity_bits,
)
from penumbra.ir import Conv2dSpec, Graph, LinearSpec, Node, RequantSpec

# Ops after which a requant is inserted when their output is consumed downstream: the
# accumulator-growing (wide-output) layers.
_ACCUMULATOR_OPS = (Conv2dSpec, LinearSpec)


def _clamp_lut(out_bits: int) -> list[int]:
    """Identity-over-range clamp LUT: ``min(v, 2^out_bits - 1)`` over the message space."""
    ceil = (1 << out_bits) - 1
    return [min(v, ceil) for v in range(1 << MESSAGE_BITS)]


def insert_requants(
    graph: Graph,
    *,
    shifts: dict[str, int] | None = None,
    mults: dict[str, int] | None = None,
    round_biases: dict[str, int] | None = None,
    out_bits: int = MESSAGE_BITS,
) -> Graph:
    """Return a copy of ``graph`` with ``Requant`` nodes auto-inserted between layers.

    A ``Requant`` is spliced after each ``Conv2d``/``Linear`` whose output feeds a downstream
    op. The rescale params are looked up per producer node name (all optional, all defaulting
    to the legacy pure power-of-two shift):

    - ``shifts[name]`` — the calibrated right-shift; absent, falls back to
      ``max(0, producer_output_bits - out_bits)`` (exactness-safe but coarse).
    - ``mults[name]`` — the fixed-point multiplier (numerator of the rescale); default ``1``.
    - ``round_biases[name]`` — the round-to-nearest bias; default ``0`` (truncation).

    The quantization service calibrates ``(mult, shift, round_bias)`` together so the rescale
    approximates the real scale ratio; passing only ``shifts`` keeps the Phase-4 behavior. After
    insertion the graph is re-checked against the radix budget — including each ``Requant``'s
    transient internal peak (`max(x,0)*mult + round_bias`) — raising and naming the layer on
    overflow.

    Idempotent: requants are only added after accumulator ops, so re-running adds nothing.
    """
    shifts = shifts or {}
    mults = mults or {}
    round_biases = round_biases or {}

    # Which tensors are consumed at all, which are graph outputs, and which are *already* read
    # by a Requant. The last makes the pass idempotent: a producer whose output already feeds a
    # Requant must not get a second one (re-running, or a hand-authored graph, is left as-is).
    consumed: set[str] = set()
    already_requantized: set[str] = set()
    for node in graph.nodes:
        consumed.update(node.inputs)
        if isinstance(node.op, RequantSpec):
            already_requantized.update(node.inputs)
    graph_outputs = set(graph.outputs)

    # Per-tensor widths of the *input* graph, to size each inserted requant's shift.
    widths = propagate_bit_widths(graph)
    capacity = radix_capacity_bits(graph.num_blocks)

    new_nodes: list[Node] = []
    # Map a producer's original output tensor name -> the requantized tensor downstream nodes
    # should read instead. Built as we insert; consumers are rewired below.
    rewire: dict[str, str] = {}

    for node in graph.nodes:
        # Rewire this node's inputs to any requantized upstream tensors.
        if any(name in rewire for name in node.inputs):
            node = replace(node, inputs=[rewire.get(name, name) for name in node.inputs])
        new_nodes.append(node)

        out_name = node.outputs[0]
        is_accumulator = isinstance(node.op, _ACCUMULATOR_OPS)
        feeds_downstream = out_name in consumed and out_name not in graph_outputs
        # Skip if already requantized (idempotency) — see `already_requantized` above.
        if not (is_accumulator and feeds_downstream) or out_name in already_requantized:
            continue

        incoming_bits = widths[out_name]
        # A single accumulator wider than the radix can't be narrowed after the fact — the wide
        # value itself overflowed. Fail loudly naming the layer (`AGENTS.md` §1.4).
        if incoming_bits > capacity:
            raise ValueError(
                f"layer {node.name!r} produces a {incoming_bits}-bit accumulator, wider than the "
                f"{capacity}-bit radix ({graph.num_blocks} blocks); reduce precision or widen "
                "num_blocks — a Requant cannot recover bits already overflowed"
            )

        shift = shifts.get(node.name, max(0, incoming_bits - out_bits))
        mult = mults.get(node.name, 1)
        round_bias = round_biases.get(node.name, 0)
        rq_name = f"{out_name}__rq"
        new_nodes.append(
            Node(
                name=f"{node.name}__requant",
                inputs=[out_name],
                outputs=[rq_name],
                op=RequantSpec(
                    shift=shift,
                    mult=mult,
                    round_bias=round_bias,
                    out_bits=out_bits,
                    clamp_lut=_clamp_lut(out_bits),
                ),
            )
        )
        rewire[out_name] = rq_name

    # Graph outputs that were requantized would change name; in our placement the requantized
    # tensors are never graph outputs (we skip terminal producers), so outputs are unchanged.
    out_graph = replace(graph, nodes=new_nodes)

    # Final feasibility gate: every tensor must fit the radix after insertion.
    check_bit_width_budget(out_graph)
    return out_graph
