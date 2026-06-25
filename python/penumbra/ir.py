"""Intermediate Representation (IR): data structures + (de)serialization.

The IR is the product's backbone (``PROJECT.md`` §7): a directed graph of op nodes that the
Python front end emits and the Rust runtime consumes. A new use case is a new graph, never
a backend edit (``AGENTS.md`` §1.2).

This module **must stay in lockstep** with ``runtime/src/ir.rs`` (``AGENTS.md`` §5). Any IR
change updates both language sides, bumps :data:`SCHEMA_VERSION`, and updates the
cross-language conformance test + ``docs/IR-SPEC.md`` in the **same change**. A
schema-version bump is a breaking change (``AGENTS.md`` §8).

Wire format is JSON (human-inspectable, easy to debug); a compact binary format is a later,
profiling-driven decision (ROADMAP.md Phase 10) and an architectural fork to raise first
(``AGENTS.md`` §3.2).

## Encoding (mirrors the Rust serde format exactly)

A node carries its op as a nested, *internally tagged* object keyed on ``op_type`` — the
JSON the Rust ``#[serde(tag = "op_type")]`` enum expects::

    {"name": "fc", "inputs": ["x"], "outputs": ["logit"],
     "op": {"op_type": "Linear", "weights": [[...]], "bias": [-1478], "weight_bits": 4}}

Field order in the emitted dicts matches the Rust struct declaration order so the committed
JSON diffs cleanly; equality across languages is compared on *parsed* dicts, so order is not
load-bearing for correctness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# IR wire-format version. Hardcoded identically in ``runtime/src/ir.rs``; a mismatch is a
# breaking change caught loudly at load time (``AGENTS.md`` §5, §8).
SCHEMA_VERSION = "0.4.0"


@dataclass(frozen=True)
class OpSpec:
    """Base class for the serializable op payloads (the mirror of Rust ``OpSpec``).

    Each subclass declares its ``op_type`` tag and serializes to a flat dict with that tag
    plus its fields. :meth:`from_dict` dispatches on the tag and fails loudly on an unknown
    op, mirroring the Rust enum's ``unknown variant`` error (``AGENTS.md`` §1.4).
    """

    op_type: str = field(init=False)

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    @staticmethod
    def from_dict(d: dict[str, Any]) -> OpSpec:
        op_type = d.get("op_type")
        if op_type == "Linear":
            return LinearSpec(
                weights=[[int(w) for w in row] for row in d["weights"]],
                bias=[int(b) for b in d["bias"]],
                weight_bits=int(d["weight_bits"]),
            )
        if op_type == "Activation":
            return ActivationSpec(
                lut=[int(v) for v in d["lut"]],
                output_bits=int(d["output_bits"]),
            )
        if op_type == "Argmax":
            return ArgmaxSpec(threshold=int(d["threshold"]))
        if op_type == "Requant":
            return RequantSpec(
                shift=int(d["shift"]),
                out_bits=int(d["out_bits"]),
                clamp_lut=[int(v) for v in d["clamp_lut"]],
            )
        if op_type == "Pool":
            return PoolSpec(
                mode=str(d["mode"]),
                in_h=int(d["in_h"]),
                in_w=int(d["in_w"]),
                channels=int(d["channels"]),
                pool_h=int(d["pool_h"]),
                pool_w=int(d["pool_w"]),
                stride=int(d["stride"]),
            )
        if op_type == "Add":
            return AddSpec()
        raise ValueError(
            f"unknown op_type {op_type!r}; expected one of 'Linear', 'Activation', "
            "'Argmax', 'Requant', 'Pool', 'Add'"
        )


@dataclass(frozen=True)
class LinearSpec(OpSpec):
    """Dense layer / logistic-regression head with plaintext quantized weights.

    ``weights`` is row-major ``[n_out][n_in]``; ``bias`` has one entry per output row.
    ``weight_bits`` is the signed magnitude+sign width feeding the bit-width growth rule.
    """

    weights: list[list[int]]
    bias: list[int]
    weight_bits: int

    op_type: str = field(init=False, default="Linear")

    def __post_init__(self) -> None:
        # Mirror ``OpSpec::build`` in Rust: fail loudly on a malformed layer at construction,
        # not as a cryptic error later (``AGENTS.md`` §1.4).
        if not self.weights:
            raise ValueError("LinearSpec has no weight rows")
        if len(self.weights) != len(self.bias):
            raise ValueError(
                f"LinearSpec has {len(self.weights)} weight rows but {len(self.bias)} biases; "
                "need one bias per row"
            )
        width = len(self.weights[0])
        for i, row in enumerate(self.weights):
            if len(row) != width:
                raise ValueError(
                    f"LinearSpec weight row {i} has width {len(row)} but row 0 has width "
                    f"{width}; all rows must match the input length"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_type": self.op_type,
            "weights": self.weights,
            "bias": self.bias,
            "weight_bits": self.weight_bits,
        }


@dataclass(frozen=True)
class ActivationSpec(OpSpec):
    """Single-input activation realized as a LUT over a narrow integer domain.

    ``lut[v]`` is the output for input value ``v``; ``output_bits`` is the table's output
    width (its bit-width growth declaration). The runtime applies it bit-exactly via PBS.
    """

    lut: list[int]
    output_bits: int

    op_type: str = field(init=False, default="Activation")

    def to_dict(self) -> dict[str, Any]:
        return {"op_type": self.op_type, "lut": self.lut, "output_bits": self.output_bits}


@dataclass(frozen=True)
class ArgmaxSpec(OpSpec):
    """2-class argmax: threshold a single logit into an encrypted 0/1 label."""

    threshold: int

    op_type: str = field(init=False, default="Argmax")

    def to_dict(self) -> dict[str, Any]:
        return {"op_type": self.op_type, "threshold": self.threshold}


@dataclass(frozen=True)
class RequantSpec(OpSpec):
    """Rescale a wide accumulator down to a narrow, LUT-able value.

    Exact semantics (matched bit-for-bit by the runtime, ``AGENTS.md`` §1.1)::

        requant(x) = clamp(max(x >> shift, 0), 0, 2**out_bits - 1)

    ``shift`` is a non-negative power-of-two rescale; the op is a fused ReLU+requant so its
    output is non-negative (required by the single-block PBS path). ``clamp_lut[v]`` is the
    output for the already-saturated input block value ``v``; it must cover the whole
    ``MESSAGE_BITS``-bit message space and saturate at ``2**out_bits - 1``. The quantization
    service owns choosing ``shift`` and building ``clamp_lut`` (Phase 5 / the compile pass).
    """

    shift: int
    out_bits: int
    clamp_lut: list[int]

    op_type: str = field(init=False, default="Requant")

    def __post_init__(self) -> None:
        # Fail loudly at construction (mirrors Rust ``OpSpec::build``), not later (§1.4).
        if self.shift < 0:
            raise ValueError(f"RequantSpec shift must be non-negative, got {self.shift}")
        if self.out_bits < 1:
            raise ValueError(f"RequantSpec out_bits must be >= 1, got {self.out_bits}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_type": self.op_type,
            "shift": self.shift,
            "out_bits": self.out_bits,
            "clamp_lut": self.clamp_lut,
        }


@dataclass(frozen=True)
class PoolSpec(OpSpec):
    """Spatial pooling over a flattened ``[channels][in_h][in_w]`` feature map.

    ``mode`` is ``"avg"`` (window **sum** — the ``1/k`` averaging is folded into the
    downstream ``Requant`` so pooling stays PBS-free) or ``"max"`` (pairwise max, expensive).
    The flat tensor is **channel-major, row-major**: element ``(c, y, x)`` is at
    ``c*in_h*in_w + y*in_w + x`` — the same layout ``Conv2d`` produces, so a ``Conv2d → Pool``
    chain needs no reshape. Output is ``[channels][out_h][out_w]`` in the same layout.
    """

    mode: str
    in_h: int
    in_w: int
    channels: int
    pool_h: int
    pool_w: int
    stride: int

    op_type: str = field(init=False, default="Pool")

    def __post_init__(self) -> None:
        # Fail loudly at construction (mirrors Rust ``OpSpec::build``), not later (§1.4).
        if self.mode not in ("avg", "max"):
            raise ValueError(f'PoolSpec mode must be "avg" or "max"; got {self.mode!r}')
        if min(self.in_h, self.in_w, self.channels, self.pool_h, self.pool_w, self.stride) < 1:
            raise ValueError("PoolSpec dims/window/stride must all be positive")
        if self.pool_h > self.in_h or self.pool_w > self.in_w:
            raise ValueError(
                f"PoolSpec window ({self.pool_h}x{self.pool_w}) must fit the input "
                f"({self.in_h}x{self.in_w})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_type": self.op_type,
            "mode": self.mode,
            "in_h": self.in_h,
            "in_w": self.in_w,
            "channels": self.channels,
            "pool_h": self.pool_h,
            "pool_w": self.pool_w,
            "stride": self.stride,
        }


@dataclass(frozen=True)
class AddSpec(OpSpec):
    """Element-wise addition of two input tensors (residuals).

    The first **multi-input** op: the carrying ``Node`` has two entries in ``inputs`` (the
    operands, in their declared merge order). There is no payload — the operands come from
    the graph wiring, so the serialized object is just ``{"op_type": "Add"}``.
    """

    op_type: str = field(init=False, default="Add")

    def to_dict(self) -> dict[str, Any]:
        return {"op_type": self.op_type}


@dataclass(frozen=True)
class Node:
    """One op in the graph: a name, the tensor names it reads/writes, and its op payload."""

    name: str
    inputs: list[str]
    outputs: list[str]
    op: OpSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "op": self.op.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Node:
        return Node(
            name=d["name"],
            inputs=list(d["inputs"]),
            outputs=list(d["outputs"]),
            op=OpSpec.from_dict(d["op"]),
        )


@dataclass(frozen=True)
class Graph:
    """The root IR object: a directed graph of op nodes in a valid topological order.

    ``num_blocks`` is the central bit-width budget (the shared radix width); ``input_bits``
    is the declared width of the encrypted model input. ``inputs``/``outputs`` name the
    graph's boundary tensors.
    """

    schema_version: str
    num_blocks: int
    input_bits: int
    inputs: list[str]
    outputs: list[str]
    nodes: list[Node]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "num_blocks": self.num_blocks,
            "input_bits": self.input_bits,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": [n.to_dict() for n in self.nodes],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Graph:
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"IR schema-version mismatch: data declares {version!r} but this front end "
                f"emits {SCHEMA_VERSION!r}. A schema-version change is breaking "
                "(AGENTS.md §5, §8)."
            )
        return Graph(
            schema_version=version,
            num_blocks=int(d["num_blocks"]),
            input_bits=int(d["input_bits"]),
            inputs=list(d["inputs"]),
            outputs=list(d["outputs"]),
            nodes=[Node.from_dict(n) for n in d["nodes"]],
        )

    @staticmethod
    def from_json(s: str) -> Graph:
        return Graph.from_dict(json.loads(s))


def build_linear_argmax_graph(
    *,
    num_blocks: int,
    input_bits: int,
    weights: Any,
    bias: Any,
    weight_bits: int,
    threshold: int,
    input_name: str = "x",
    logit_name: str = "logit",
    label_name: str = "label",
) -> Graph:
    """Build the Phase-2 ``Linear → Argmax`` IR graph.

    Accepts NumPy arrays or nested lists for ``weights``/``bias`` and coerces them to plain
    Python ints so the result is JSON-serializable and language-agnostic. This is the single
    place the example exporter and the conformance test construct the canonical graph, so the
    committed fixture and the test cannot drift (``AGENTS.md`` §5).
    """
    weights_i = [[int(w) for w in row] for row in weights]
    bias_i = [int(b) for b in bias]
    return Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=int(num_blocks),
        input_bits=int(input_bits),
        inputs=[input_name],
        outputs=[label_name],
        nodes=[
            Node(
                name="fc",
                inputs=[input_name],
                outputs=[logit_name],
                op=LinearSpec(weights=weights_i, bias=bias_i, weight_bits=int(weight_bits)),
            ),
            Node(
                name="head",
                inputs=[logit_name],
                outputs=[label_name],
                op=ArgmaxSpec(threshold=int(threshold)),
            ),
        ],
    )
