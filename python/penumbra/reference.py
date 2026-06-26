"""The quantized-integer reference evaluator — the golden oracle, in plain Python ints.

TFHE is exact, so the FHE output must equal the **quantized-cleartext** output bit-for-bit
(``AGENTS.md`` §1.1). This module walks an IR :class:`~penumbra.ir.Graph` in plain ``int``
arithmetic, mirroring each runtime op's exact integer semantics
(``runtime/src/eval.rs`` / ``runtime/src/ops``). It is the oracle:

* the quantization service's :meth:`penumbra.model.Model.quantize` self-verify step runs it to
  confirm the int graph it just built is internally consistent before committing to FHE;
* the example generators commit its output as ``expected_logits``/``expected_labels``;
* the Rust golden tests recompute the same arithmetic in ``i64`` and assert equality.

Keeping this in one place (instead of re-deriving the per-op math in every test/example) means
there is a single Python source of truth for the oracle, kept honest against the Rust runtime by
the golden tests that compare both to the committed fixtures.

The flat tensor convention matches the runtime ``CtVec``: a channel-major, row-major
``[channels][h][w]`` layout that ``Conv2d`` produces and ``Pool`` consumes (so ``Conv2d → Pool``
needs no reshape). All arithmetic is unbounded Python ``int`` — overflow/bit-width feasibility is
the bit-width tracker's job (:mod:`penumbra.bitwidth`), not the oracle's.
"""

from __future__ import annotations

from penumbra.ir import (
    ActivationSpec,
    AddSpec,
    ArgmaxSpec,
    Conv2dSpec,
    Graph,
    LinearSpec,
    PoolSpec,
    RequantSpec,
)


def _conv2d(op: Conv2dSpec, x: list[int]) -> list[int]:
    """Integer 2-D convolution (correlation, virtual zero padding) — mirrors ``conv2d.rs``."""
    out_h = (op.in_h + 2 * op.padding - op.kernel_h) // op.stride + 1
    out_w = (op.in_w + 2 * op.padding - op.kernel_w) // op.stride + 1
    in_hw = op.in_h * op.in_w
    out: list[int] = []
    for kernel, b in zip(op.weights, op.bias, strict=True):
        for oy in range(out_h):
            for ox in range(out_w):
                acc = 0
                for ic in range(op.in_channels):
                    for ky in range(op.kernel_h):
                        iy = oy * op.stride + ky - op.padding
                        for kx in range(op.kernel_w):
                            ix = ox * op.stride + kx - op.padding
                            if not (0 <= iy < op.in_h and 0 <= ix < op.in_w):
                                continue  # virtual padding: out-of-range taps contribute 0
                            w = kernel[(ic * op.kernel_h + ky) * op.kernel_w + kx]
                            acc += w * x[ic * in_hw + iy * op.in_w + ix]
                out.append(acc + b)
    return out


def _pool(op: PoolSpec, x: list[int]) -> list[int]:
    """Integer pooling — ``avg`` emits the window **sum**, ``max`` the window max (``pool.rs``)."""
    out_h = (op.in_h - op.pool_h) // op.stride + 1
    out_w = (op.in_w - op.pool_w) // op.stride + 1
    out: list[int] = []
    for c in range(op.channels):
        base = c * op.in_h * op.in_w
        for oy in range(out_h):
            for ox in range(out_w):
                vals = [
                    x[base + (oy * op.stride + ky) * op.in_w + (ox * op.stride + kx)]
                    for ky in range(op.pool_h)
                    for kx in range(op.pool_w)
                ]
                out.append(sum(vals) if op.mode == "avg" else max(vals))
    return out


def _requant(op: RequantSpec, x: list[int]) -> list[int]:
    """Integer requant: ``clamp((max(v,0)*mult + round_bias) >> shift, 0, 2^out_bits-1)``.

    Mirrors ``requant.rs`` exactly: ReLU first, then the fixed-point multiply + round bias, then
    the arithmetic right shift (floor — the value is non-negative here), then the clamp.
    """
    ceil = (1 << op.out_bits) - 1
    out: list[int] = []
    for v in x:
        nonneg = max(v, 0)
        shifted = (nonneg * op.mult + op.round_bias) >> op.shift
        out.append(min(max(shifted, 0), ceil))
    return out


def _linear(op: LinearSpec, x: list[int]) -> list[int]:
    """Integer dense layer: ``[sum(w*v) + b]`` per output row (``linear.rs``)."""
    return [
        sum(w * v for w, v in zip(row, x, strict=True)) + b
        for row, b in zip(op.weights, op.bias, strict=True)
    ]


def _activation(op: ActivationSpec, x: list[int]) -> list[int]:
    """Integer activation LUT applied per element (``activation.rs``).

    The runtime applies the LUT to a single radix block, so each input must be a small
    non-negative value in the table's domain ``[0, len(lut))`` (a post-Requant value). An
    out-of-domain index is a graph/quantization bug — fail loudly rather than index past the
    table (`AGENTS.md` §1.4), mirroring how the runtime would read the wrong block.
    """
    out: list[int] = []
    for v in x:
        if not (0 <= v < len(op.lut)):
            raise ValueError(
                f"Activation input {v} is outside the LUT domain [0, {len(op.lut)}); a Requant "
                "must narrow the value into the single-block message space first"
            )
        out.append(op.lut[v])
    return out


def evaluate_graph_int(graph: Graph, inputs: dict[str, list[int]]) -> dict[str, list[int]]:
    """Evaluate ``graph`` in plain integers, returning every graph-output tensor.

    ``inputs`` maps each ``graph.inputs`` name to its integer tensor. Walks nodes in their
    serialized order (a valid topological order, as the runtime requires), dispatching each op to
    its exact integer mirror, and returns ``{name: values}`` for every ``graph.outputs`` tensor.
    This is the cleartext oracle the FHE path must match bit-for-bit (``AGENTS.md`` §1.1).
    """
    env: dict[str, list[int]] = {name: list(inputs[name]) for name in graph.inputs}

    for node in graph.nodes:
        op = node.op
        if isinstance(op, AddSpec):
            a, b = (env[name] for name in node.inputs)
            out = [x + y for x, y in zip(a, b, strict=True)]
        else:
            x = env[node.inputs[0]]
            if isinstance(op, Conv2dSpec):
                out = _conv2d(op, x)
            elif isinstance(op, PoolSpec):
                out = _pool(op, x)
            elif isinstance(op, RequantSpec):
                out = _requant(op, x)
            elif isinstance(op, LinearSpec):
                out = _linear(op, x)
            elif isinstance(op, ActivationSpec):
                out = _activation(op, x)
            elif isinstance(op, ArgmaxSpec):
                # 2-class threshold -> encrypted 0/1 label (a single value).
                out = [1 if x[0] >= op.threshold else 0]
            else:  # pragma: no cover - every OpSpec variant is handled above
                raise ValueError(f"reference evaluator: unsupported op {op.op_type!r}")
        env[node.outputs[0]] = out

    return {name: env[name] for name in graph.outputs}
