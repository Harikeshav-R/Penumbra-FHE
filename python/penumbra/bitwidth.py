"""Bit-width tracking — the Python mirror of the runtime's growth rules (``PROJECT.md`` §9).

A ``Linear``/``Conv2d`` accumulator grows ~``log2(N)`` bits per layer, so multi-layer models
must be **requantized** back down before each nonlinearity or the radix overflows and results
wrap (`AGENTS.md` §1.3 — accumulator overflow is the #1 multi-layer bug). This module owns,
on the Python side, the per-op ``output_bits`` rule that the compile pass (:mod:`penumbra.compile`)
uses to decide *where* a ``Requant`` is needed.

These formulas **must agree, value-for-value, with the Rust ``Op::output_bits``** in
``runtime/src/ops``. That agreement is the whole point of the cross-language bit-width
conformance test (``tests/test_bitwidth_conformance.py`` and its Rust half
``runtime/tests/bitwidth_conformance.rs``): a shared committed table of
``(op, input_bits) -> output_bits`` cases both languages check, so a drift in either
implementation fails CI rather than silently mis-sizing a radix.

The crypto constants below mirror ``runtime/src/keys.rs`` (the default
``PARAM_MESSAGE_2_CARRY_2_KS_PBS`` profile): ``MESSAGE_BITS = 2`` bits per radix block, so a
radix of ``num_blocks`` blocks holds ``num_blocks * MESSAGE_BITS`` bits.
"""

from __future__ import annotations

from penumbra.ir import (
    ActivationSpec,
    AddSpec,
    ArgmaxSpec,
    Conv2dSpec,
    Graph,
    LinearSpec,
    OpSpec,
    PoolSpec,
    RequantSpec,
)

# Mirror of ``runtime/src/keys.rs`` (the default secure profile). Not user-facing.
MESSAGE_BITS = 2


def radix_capacity_bits(num_blocks: int) -> int:
    """Bits of value a radix of ``num_blocks`` blocks holds (mirror of ``radix_capacity_bits``)."""
    return num_blocks * MESSAGE_BITS


def magnitude_bits(x: int) -> int:
    """Minimum bits to represent the magnitude of ``x`` (0 for ``x == 0``).

    Mirror of ``keys::magnitude_bits`` — the position of the top set bit. Callers add their
    own sign/carry headroom; a zero contributor vanishes to 0 bits (correct for a zero bias).
    """
    return abs(int(x)).bit_length()


def _ceil_log2(n: int) -> int:
    """``ceil(log2 n)`` growth from summing ``n`` terms (0 for ``n <= 1``).

    Matches the Rust ``usize::BITS - (n-1).leading_zeros()`` = ``(n-1).bit_length()`` form.
    """
    return 0 if n <= 1 else (n - 1).bit_length()


def _linear_like_bits(input_bits: int, weight_bits: int, fan_in: int, bias: list[int]) -> int:
    """The shared ``Linear``/``Conv2d`` accumulator rule (`linear.rs` / `conv2d.rs`).

    Two contributors, taken at their max: the summed products
    (``input_bits + weight_bits + ceil(log2 fan_in)``) and the plaintext bias (sized from its
    own magnitude). ``+2`` adds one carry from the bias add and one sign bit — two independent
    guard bits, not one (collapsing them under-counts by a bit and can miss a real overflow).
    """
    sum_bits = input_bits + weight_bits + _ceil_log2(fan_in)
    max_bias = max((abs(int(b)) for b in bias), default=0)
    bias_bits = magnitude_bits(max_bias)
    return max(sum_bits, bias_bits) + 2


def output_bits(op: OpSpec, input_bits: list[int]) -> int:
    """Bit-width of ``op``'s output tensor given its input tensors' widths.

    ``input_bits`` is a list to cover multi-input ops (``Add`` takes two); single-input ops
    receive a one-element list. This mirrors the Rust ``Op::output_bits_n`` dispatch. Raises
    ``ValueError`` on an arity mismatch or an unsupported op, rather than guessing.
    """
    if isinstance(op, LinearSpec):
        _expect_arity(op, input_bits, 1)
        fan_in = len(op.weights[0])
        return _linear_like_bits(input_bits[0], op.weight_bits, fan_in, op.bias)

    if isinstance(op, Conv2dSpec):
        _expect_arity(op, input_bits, 1)
        fan_in = op.in_channels * op.kernel_h * op.kernel_w
        return _linear_like_bits(input_bits[0], op.weight_bits, fan_in, op.bias)

    if isinstance(op, PoolSpec):
        _expect_arity(op, input_bits, 1)
        if op.mode == "avg":
            # Summing k = pool_h*pool_w terms adds ceil(log2 k) bits; values stay signed.
            return input_bits[0] + _ceil_log2(op.pool_h * op.pool_w)
        # max selects one input — never grows the magnitude.
        return input_bits[0]

    if isinstance(op, RequantSpec):
        # Requant consumes a *wide* input and emits out_bits, independent of input width.
        _expect_arity(op, input_bits, 1)
        return op.out_bits

    if isinstance(op, ActivationSpec):
        # Activation needs an already-narrowed (<= MESSAGE_BITS) single-block input; its output
        # width is set by the table. Mirror the Rust assert so an un-requantized wide input
        # fails loudly in the tracker, naming the contract.
        _expect_arity(op, input_bits, 1)
        if input_bits[0] > MESSAGE_BITS:
            raise ValueError(
                f"Activation input is {input_bits[0]} bits, wider than the single "
                f"{MESSAGE_BITS}-bit block it consumes; insert a Requant in front to narrow it"
            )
        return op.output_bits

    if isinstance(op, ArgmaxSpec):
        _expect_arity(op, input_bits, 1)
        return 1  # a single class bit

    if isinstance(op, AddSpec):
        _expect_arity(op, input_bits, 2)
        # One carry from the add; the wider operand's sign bit covers the result.
        return max(input_bits[0], input_bits[1]) + 1

    raise ValueError(f"output_bits: unsupported op {op.op_type!r}")


def _expect_arity(op: OpSpec, input_bits: list[int], n: int) -> None:
    if len(input_bits) != n:
        raise ValueError(
            f"{op.op_type} expects {n} input(s) for bit-width tracking, got {len(input_bits)}"
        )


def propagate_bit_widths(graph: Graph) -> dict[str, int]:
    """Per-tensor bit-widths through ``graph``, seeded by ``graph.input_bits``.

    The Python mirror of ``eval::propagate_bit_widths``: walk nodes in order, resolve each
    node's input widths from a running map, apply :func:`output_bits`, and store the single
    output. Fails loudly (`AGENTS.md` §1.4) on a tensor read before it is produced, a duplicate
    output name, or a node without exactly one output.
    """
    widths: dict[str, int] = {name: graph.input_bits for name in graph.inputs}
    for node in graph.nodes:
        if not node.inputs or len(node.outputs) != 1:
            raise ValueError(
                f"node {node.name!r} ({node.op.op_type}) must have at least one input and "
                "exactly one output"
            )
        in_bits = []
        for name in node.inputs:
            if name not in widths:
                raise ValueError(
                    f"node {node.name!r} reads tensor {name!r}, which no earlier node produced "
                    "and is not a graph input — node order is not a valid topological order"
                )
            in_bits.append(widths[name])
        out_name = node.outputs[0]
        if out_name in widths:
            raise ValueError(
                f"node {node.name!r} writes tensor {out_name!r}, which already exists — "
                "tensor names must be unique"
            )
        widths[out_name] = output_bits(node.op, in_bits)
    return widths


def check_bit_width_budget(graph: Graph) -> None:
    """Raise if any tensor's propagated width exceeds the radix capacity (mirror of Rust).

    Names the offending node and the required-vs-available bits (`AGENTS.md` §1.3, §1.4). Used
    by the compile pass to confirm a model fits *after* automatic ``Requant`` insertion.
    """
    capacity = radix_capacity_bits(graph.num_blocks)
    widths = propagate_bit_widths(graph)
    for node in graph.nodes:
        name = node.outputs[0]
        bits = widths[name]
        if bits > capacity:
            raise ValueError(
                f"bit-width budget exceeded at node {node.name!r} (tensor {name!r}): requires "
                f"{bits} bits but the radix holds only {capacity} ({graph.num_blocks} blocks x "
                f"{MESSAGE_BITS} bits). Reduce precision, widen num_blocks, or requantize earlier."
            )
