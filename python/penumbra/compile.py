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

from dataclasses import dataclass, field, replace

from penumbra.bitwidth import (
    MESSAGE_BITS,
    check_bit_width_budget,
    propagate_bit_widths,
    radix_capacity_bits,
)
from penumbra.ir import ArgmaxSpec, Conv2dSpec, Graph, LinearSpec, Node, RequantSpec
from penumbra.quantization.lut import identity_clamp_lut


@dataclass(frozen=True)
class RequantChannelParams:
    """Per-output-channel rescale for a per-channel Requant (one entry per channel).

    The quantization service builds this when an accumulator was quantized per-channel: each
    output channel has its own accumulator scale, so each gets its own fixed-point multiplier
    ``mults[i] / 2**shifts[i]`` (with round bias ``round_biases[i]``). ``insert_requants`` derives
    the flat ``channel_size`` stride from the producer op's shape, so it is not carried here.

    ``round_biases`` may be omitted, in which case it defaults to all-zero (truncation — the same
    default as the scalar API): round bias is the optional part of the rescale, so requiring it
    made the common ``RequantChannelParams(mults=..., shifts=...)`` call fail confusingly. ``mults``
    and ``shifts`` must match in length; a supplied ``round_biases`` must too.
    """

    mults: list[int]
    shifts: list[int]
    round_biases: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.mults) != len(self.shifts):
            raise ValueError(
                f"RequantChannelParams needs one shift per mult: mults={len(self.mults)}, "
                f"shifts={len(self.shifts)}"
            )
        if not self.round_biases:
            # Default missing round biases to 0 (truncation), one per channel — matches the scalar
            # API's `round_bias=0` default so the common (mults, shifts)-only call is valid.
            object.__setattr__(self, "round_biases", [0] * len(self.mults))
        elif len(self.round_biases) != len(self.mults):
            raise ValueError(
                f"RequantChannelParams round_biases must have one entry per mult "
                f"({len(self.mults)}); got {len(self.round_biases)}"
            )


# Ops after which a requant is inserted when their output is consumed downstream: the
# accumulator-growing (wide-output) layers.
_ACCUMULATOR_OPS = (Conv2dSpec, LinearSpec)

# Consumers that operate on a *wide* value and therefore do NOT require their input to be
# narrowed by a Requant. ``Argmax`` is a threshold comparison on a wide logit (the Phase-2
# head); the client decrypts and argmaxes wide logits anyway (`PROJECT.md` §11), so an
# accumulator feeding only an Argmax is effectively terminal and is left wide
# (`docs/SUPPORTED-OPS.md`).
_WIDE_INPUT_OPS = (ArgmaxSpec,)


def _clamp_lut(out_bits: int) -> list[int]:
    """Identity-over-range clamp LUT: ``min(v, 2^out_bits - 1)`` over the message space.

    Delegates to the quantization service's :func:`~penumbra.quantization.lut.identity_clamp_lut`,
    which owns LUT generation (that module is the single source of truth). Reusing it — rather than
    rebuilding the table inline — also runs :func:`~penumbra.quantization.lut.validate_lut` on the
    result, so every auto-inserted Requant's ``clamp_lut`` is backend-validated here in Python
    before it can reach a PBS (``AGENTS.md`` §1.4). The equality is test-pinned
    (``tests/test_quantization_lut.py``).
    """
    return identity_clamp_lut(out_bits)


def _channel_size(op: object) -> int:
    """Flat elements-per-channel stride of an accumulator op's output tensor.

    A per-channel ``Requant`` operates on the flat output tensor, so it must know how many
    consecutive elements belong to one output channel to map ``flat_index -> channel``. This
    mirrors the runtime tensor layouts (``conv2d.rs``, ``linear.rs``): a ``Linear`` emits one
    element per output row (stride ``1``); a ``Conv2d`` emits ``[out_ch][out_h][out_w]``
    channel-major, so a whole ``out_h*out_w`` spatial map belongs to one channel.
    """
    if isinstance(op, LinearSpec):
        return 1
    if isinstance(op, Conv2dSpec):
        out_h = (op.in_h + 2 * op.padding - op.kernel_h) // op.stride + 1
        out_w = (op.in_w + 2 * op.padding - op.kernel_w) // op.stride + 1
        return out_h * out_w
    raise ValueError(f"per-channel Requant not supported after op {type(op).__name__}")


def insert_requants(
    graph: Graph,
    *,
    shifts: dict[str, int] | None = None,
    mults: dict[str, int] | None = None,
    round_biases: dict[str, int] | None = None,
    per_channel: dict[str, RequantChannelParams] | None = None,
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
    - ``per_channel[name]`` — a :class:`RequantChannelParams` for a per-channel-quantized
      accumulator: one ``(mult, shift, round_bias)`` per output channel. When present it takes
      precedence over the scalar ``shifts``/``mults``/``round_biases`` for that node, and the
      emitted ``Requant`` carries the per-channel overlay. The flat ``channel_size`` stride is
      derived from the producer op (``1`` for ``Linear``, ``out_h*out_w`` for ``Conv2d``).

    The quantization service calibrates ``(mult, shift, round_bias)`` together so the rescale
    approximates the real scale ratio; passing only ``shifts`` keeps the Phase-4 behavior. After
    insertion the graph is re-checked against the radix budget — including each ``Requant``'s
    transient internal peak (`max(x,0)*mult + round_bias`) — raising and naming the layer on
    overflow.

    Idempotent: requants are only added after accumulator ops, so re-running adds nothing.
    """
    shifts = shifts or {}
    mults = mults or {}
    per_channel = per_channel or {}
    round_biases = round_biases or {}

    # Which tensors are consumed at all, which are graph outputs, and which are *already* read
    # by a Requant. The last makes the pass idempotent: a producer whose output already feeds a
    # Requant must not get a second one (re-running, or a hand-authored graph, is left as-is).
    # ``consumed_by_narrow`` is the subset of consumed tensors read by at least one op that
    # genuinely needs a narrow input: a producer feeding *only* wide-input ops (e.g. Argmax) is
    # effectively terminal and must stay wide (no Requant).
    consumed: set[str] = set()
    consumed_by_narrow: set[str] = set()
    already_requantized: set[str] = set()
    for node in graph.nodes:
        consumed.update(node.inputs)
        if not isinstance(node.op, _WIDE_INPUT_OPS):
            consumed_by_narrow.update(node.inputs)
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
        # Rewire this node's inputs to any requantized upstream tensors — but ONLY for narrow-input
        # ops. A wide-input op (Argmax) must keep reading the original *wide* logit even when the
        # producer fans out to a narrow consumer that triggered a Requant: rewiring it to the
        # narrowed `__rq` tensor would silently threshold a value clamped to `2^act_bits - 1`,
        # making a high threshold unreachable (a silent classification error). The insertion gate
        # (`consumed_by_narrow`) already excludes Argmax from *forcing* a Requant; this excludes it
        # from *consuming* one.
        if not isinstance(node.op, _WIDE_INPUT_OPS) and any(name in rewire for name in node.inputs):
            node = replace(node, inputs=[rewire.get(name, name) for name in node.inputs])
        new_nodes.append(node)

        out_name = node.outputs[0]
        is_accumulator = isinstance(node.op, _ACCUMULATOR_OPS)
        # Only requant when the output feeds a *narrow-input* op (not just any consumer): a
        # terminal accumulator, or one feeding only wide-input ops like Argmax, stays wide.
        feeds_narrow = out_name in consumed_by_narrow and out_name not in graph_outputs
        # Skip if already requantized (idempotency) — see `already_requantized` above.
        if not (is_accumulator and feeds_narrow) or out_name in already_requantized:
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

        rq_name = f"{out_name}__rq"
        pc = per_channel.get(node.name)
        if pc is not None:
            # Per-channel Requant: each output channel rescaled by its own (mult, shift,
            # round_bias). channel_size is the flat elements-per-channel stride, derived from the
            # producer op's shape so it matches the runtime tensor layout exactly (`conv2d.rs`).
            channel_size = _channel_size(node.op)
            n_channels = len(node.op.weights)
            if not (len(pc.mults) == len(pc.shifts) == len(pc.round_biases) == n_channels):
                raise ValueError(
                    f"layer {node.name!r}: per-channel Requant needs one (mult, shift, round_bias) "
                    f"per output channel ({n_channels}); got mults={len(pc.mults)}, "
                    f"shifts={len(pc.shifts)}, round_biases={len(pc.round_biases)}"
                )
            rq_op = RequantSpec(
                shift=0,
                mult=1,
                round_bias=0,
                out_bits=out_bits,
                clamp_lut=_clamp_lut(out_bits),
                mults=list(pc.mults),
                shifts=list(pc.shifts),
                round_biases=list(pc.round_biases),
                channel_size=channel_size,
            )
        else:
            shift = shifts.get(node.name, max(0, incoming_bits - out_bits))
            mult = mults.get(node.name, 1)
            round_bias = round_biases.get(node.name, 0)
            rq_op = RequantSpec(
                shift=shift,
                mult=mult,
                round_bias=round_bias,
                out_bits=out_bits,
                clamp_lut=_clamp_lut(out_bits),
            )
        new_nodes.append(
            Node(name=f"{node.name}__requant", inputs=[out_name], outputs=[rq_name], op=rq_op)
        )
        rewire[out_name] = rq_name

    # Graph outputs that were requantized would change name; in our placement the requantized
    # tensors are never graph outputs (we skip terminal producers), so outputs are unchanged.
    out_graph = replace(graph, nodes=new_nodes)

    # Final feasibility gate: every tensor must fit the radix after insertion.
    check_bit_width_budget(out_graph)
    return out_graph
