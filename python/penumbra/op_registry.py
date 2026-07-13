"""Supported-op registry: ONNX op -> internal narrow-waist op.

A declarative table is the single source of truth for what Penumbra-FHE accepts. The
ONNX loader (:mod:`penumbra.onnx_loader`) validates every node against it and **fails
loudly at load time** with an actionable message when an op is unsupported (``PROJECT.md``
§10, ``AGENTS.md`` §1.4).

An ONNX op is FHE-viable only if it reduces to the TFHE primitives the runtime already
implements — plaintext-weight arithmetic (``Linear``/``Conv2d``), ciphertext adds, a
single-input LUT (``Activation``), or a pure layout relabel — within the bit-width budget
(``PROJECT.md`` §9). Each rule below documents *why* its op qualifies.

The documented supported-op list (``docs/SUPPORTED-OPS.md``) must always match what this
registry accepts (:func:`supported_onnx_ops`) — this is testable
(``tests/test_supported_ops_doc.py``),
keep it true (``AGENTS.md`` §5).

Scope note (Phase 6): the loader lowers to a **linear chain** of the existing IR ops, so the
registry recognizes exactly the ONNX ops that shape maps onto that chain:

    Gemm / MatMul               -> Linear      (dense accumulator)
    Conv                        -> Conv2d       (2-D conv accumulator)
    Relu                        -> Activation   (ReLU, fused into the preceding Requant)
    MaxPool / AveragePool /     -> Pool
      GlobalAveragePool
    Add                         -> Linear bias  (only a constant-initializer add after MatMul;
                                                 a residual/branching Add is Phase 8)
    Reshape / Flatten /         -> (layout no-op, folded away)
      Transpose
    Softmax / LogSoftmax /      -> (terminal classifier tail, dropped; client argmaxes logits)
      Sigmoid / ArgMax

Branching (residual ``Add``, ``Concat``), non-ReLU activations, and ``BatchNormalization``
are deferred to Phase 8 and rejected loudly (``ROADMAP.md`` Phase 6/8).
"""

from __future__ import annotations

from dataclasses import dataclass

# Supported opset range for the default (ai.onnx, domain "") operator set. The ops we lower
# (Conv, Gemm, MatMul, the pools, the shape/terminal ops) are stable across this window; we pin
# a range rather than accept anything so a model exported against an opset with different op
# semantics fails loudly at load time instead of lowering to a subtly-wrong graph (``ROADMAP.md``
# Phase 6 pitfalls: "pin a supported opset range").
SUPPORTED_OPSET_MIN = 11
SUPPORTED_OPSET_MAX = 22

# Machine-readable categories the loader dispatches on (the ``internal_op`` string is the
# human/doc-facing target shown in ``docs/SUPPORTED-OPS.md``).
CAT_ACCUMULATOR = "accumulator"  # Gemm / MatMul / Conv -> Linear / Conv2d
CAT_ACTIVATION = "activation"  # Relu -> Activation (fused ReLU)
CAT_POOL = "pool"  # MaxPool / AveragePool / GlobalAveragePool -> Pool
CAT_BIAS_ADD = "bias_add"  # Add -> folded into a preceding MatMul's Linear bias (or rejected)
CAT_SHAPE = "shape"  # Reshape / Flatten / Transpose -> layout no-op, folded away
CAT_TERMINAL = "terminal"  # Softmax / LogSoftmax / Sigmoid / ArgMax -> dropped terminal tail


@dataclass(frozen=True)
class OnnxOpRule:
    """One row of the supported-op table: how an ONNX op maps into the internal narrow waist.

    ``onnx_op`` is the ONNX ``op_type``; ``internal_op`` is the doc-facing target (an IR op name
    like ``"Linear"``, or a bracketed marker like ``"(layout no-op)"``); ``category`` is the
    machine tag the loader dispatches on; ``rationale`` records why the op is FHE-viable (the
    ``docs/SUPPORTED-OPS.md`` "why" column); ``attribute_constraints`` documents the attribute
    restrictions :func:`check_attributes` enforces (empty string when the op takes no constrained
    attributes).
    """

    onnx_op: str
    internal_op: str
    category: str
    rationale: str
    attribute_constraints: str


REGISTRY: dict[str, OnnxOpRule] = {
    "Gemm": OnnxOpRule(
        onnx_op="Gemm",
        internal_op="Linear",
        category=CAT_ACCUMULATOR,
        rationale=(
            "General matrix multiply Y = alpha*(A@B) + beta*C is a dense layer: a sum of "
            "ciphertext-times-plaintext-weight products plus a plaintext bias — scalar-mul + adds, "
            "no PBS (`Linear`, docs/SUPPORTED-OPS.md)."
        ),
        attribute_constraints="transA=0; alpha=1.0; beta=1.0 (transB handled by the loader).",
    ),
    "MatMul": OnnxOpRule(
        onnx_op="MatMul",
        internal_op="Linear",
        category=CAT_ACCUMULATOR,
        rationale=(
            "Bare x@W is a dense layer without bias; same plaintext-weight arithmetic as Gemm. A "
            "following constant-bias Add folds into the Linear bias."
        ),
        attribute_constraints="none (2-D operands; the constant weight is a graph initializer).",
    ),
    "Conv": OnnxOpRule(
        onnx_op="Conv",
        internal_op="Conv2d",
        category=CAT_ACCUMULATOR,
        rationale=(
            "2-D convolution is the Linear pattern shared across spatial positions: a sum of "
            "ciphertext-times-plaintext-kernel products + bias at each output pixel — scalar-mul + "
            "adds, no PBS (`Conv2d`)."
        ),
        attribute_constraints=(
            "group=1; dilations=[1,1]; symmetric equal pads; square strides (sh==sw); 2-D kernel."
        ),
    ),
    "Relu": OnnxOpRule(
        onnx_op="Relu",
        internal_op="Activation",
        category=CAT_ACTIVATION,
        rationale=(
            "ReLU max(x,0) is the exact hard-clip the fused Requant already applies "
            "(`runtime/src/ops/requant.rs`), so it costs no extra op — the accumulator's Requant "
            "realizes it. Must follow an accumulator (Conv/Gemm/MatMul), not be terminal."
        ),
        attribute_constraints="none.",
    ),
    "MaxPool": OnnxOpRule(
        onnx_op="MaxPool",
        internal_op="Pool",
        category=CAT_POOL,
        rationale=(
            "Max pooling is a per-channel window reduction realized as pairwise ciphertext max "
            "(comparison PBS) — `Pool` mode 'max'."
        ),
        attribute_constraints="pads=0; ceil_mode=0; dilations=[1,1]; uniform kernel/stride.",
    ),
    "AveragePool": OnnxOpRule(
        onnx_op="AveragePool",
        internal_op="Pool",
        category=CAT_POOL,
        rationale=(
            "Average pooling is a per-channel window sum (`add_parallelized`, no PBS); the 1/k "
            "averaging folds into the next Requant's rescale — `Pool` mode 'avg'."
        ),
        attribute_constraints=(
            "pads=0; ceil_mode=0; count_include_pad moot (pads=0); uniform kernel/stride."
        ),
    ),
    "GlobalAveragePool": OnnxOpRule(
        onnx_op="GlobalAveragePool",
        internal_op="Pool",
        category=CAT_POOL,
        rationale=(
            "Global average pooling is AveragePool over the whole feature map (kernel = spatial "
            "size) — the same PBS-free window sum, 1/k folded into the next Requant."
        ),
        attribute_constraints="none (kernel = full input spatial size).",
    ),
    "Add": OnnxOpRule(
        onnx_op="Add",
        internal_op="Linear (bias fold)",
        category=CAT_BIAS_ADD,
        rationale=(
            "A constant-initializer Add right after a MatMul is that dense layer's bias — folded "
            "into `Linear.bias`. A residual/branching Add (both operands are activations) needs "
            "multi-input topological eval and is deferred to Phase 8."
        ),
        attribute_constraints=(
            "one operand must be a constant initializer (else branching -> Phase 8)."
        ),
    ),
    "Reshape": OnnxOpRule(
        onnx_op="Reshape",
        internal_op="(layout no-op)",
        category=CAT_SHAPE,
        rationale=(
            "The runtime carries a flat channel-major vector and is shape-blind, so a Reshape "
            "between a Conv/Pool and a dense layer is identity on the wire — folded away, emits no "
            "IR node."
        ),
        attribute_constraints="must not reorder flat elements (a pure flatten/reshape).",
    ),
    "Flatten": OnnxOpRule(
        onnx_op="Flatten",
        internal_op="(layout no-op)",
        category=CAT_SHAPE,
        rationale="Flatten to (N, features) is identity on the already-flat wire — folded away.",
        attribute_constraints="none (the flat vector is unchanged).",
    ),
    "Transpose": OnnxOpRule(
        onnx_op="Transpose",
        internal_op="(layout no-op)",
        category=CAT_SHAPE,
        rationale=(
            "A Transpose that does not permute the flat element order is a no-op and is folded "
            "away; one that genuinely permutes must be baked into the following weight matrix or "
            "rejected."
        ),
        attribute_constraints="perm must not change flat element order (else rejected).",
    ),
    "Cast": OnnxOpRule(
        onnx_op="Cast",
        internal_op="(layout no-op)",
        category=CAT_SHAPE,
        rationale=(
            "A Cast to a floating type is an identity on Penumbra's already-real wire (the runtime "
            "quantizes from floats regardless of the source float width), so it folds away and "
            "emits no IR node. Exporters routinely insert one at the input (skl2onnx casts the "
            "input to float; torch/tf2onnx emit dtype-normalizing Casts). A Cast to an integer or "
            "boolean type would change the represented value and is rejected."
        ),
        attribute_constraints="to must be a floating type (FLOAT/FLOAT16/DOUBLE/BFLOAT16).",
    ),
    "Softmax": OnnxOpRule(
        onnx_op="Softmax",
        internal_op="(terminal, dropped)",
        category=CAT_TERMINAL,
        rationale=(
            "Softmax is monotone in each logit and argmax-invariant "
            "(argmax(softmax(z))==argmax(z)); Penumbra leaves logits wide and argmaxes "
            "client-side, so a terminal Softmax is dropped (`PROJECT.md` §11)."
        ),
        attribute_constraints="must be the terminal (graph-output) node.",
    ),
    "LogSoftmax": OnnxOpRule(
        onnx_op="LogSoftmax",
        internal_op="(terminal, dropped)",
        category=CAT_TERMINAL,
        rationale="Log-softmax is monotone and argmax-invariant — dropped as a terminal tail.",
        attribute_constraints="must be the terminal (graph-output) node.",
    ),
    "Sigmoid": OnnxOpRule(
        onnx_op="Sigmoid",
        internal_op="(terminal, dropped)",
        category=CAT_TERMINAL,
        rationale=(
            "A terminal Sigmoid is monotone; the thresholded label is argmax-invariant, so it is "
            "dropped (a non-terminal Sigmoid is a real activation and is not supported)."
        ),
        attribute_constraints="must be the terminal (graph-output) node.",
    ),
    "ArgMax": OnnxOpRule(
        onnx_op="ArgMax",
        internal_op="(terminal, dropped)",
        category=CAT_TERMINAL,
        rationale=(
            "A terminal ArgMax is the client-side classification step Penumbra already performs on "
            "the decrypted logits — dropped so the graph output stays the wide logit vector."
        ),
        attribute_constraints="must be the terminal (graph-output) node.",
    ),
}


def is_supported(op_type: str) -> bool:
    """True if ``op_type`` is a recognized ONNX op the loader can lower/fold/drop."""
    return op_type in REGISTRY


def supported_onnx_ops() -> list[str]:
    """The sorted list of recognized ONNX op types — the contract the docs table mirrors.

    ``docs/SUPPORTED-OPS.md``'s ONNX front-door mapping table must list exactly these ops
    (``tests/test_supported_ops_doc.py`` enforces it, ``AGENTS.md`` §5).
    """
    return sorted(REGISTRY)


def rule_for(op_type: str) -> OnnxOpRule:
    """Return the :class:`OnnxOpRule` for ``op_type`` (raises ``KeyError`` if unsupported)."""
    return REGISTRY[op_type]


def opset_problem(opset: int) -> str | None:
    """Return an actionable message if ``opset`` is outside the supported range, else ``None``."""
    if not SUPPORTED_OPSET_MIN <= opset <= SUPPORTED_OPSET_MAX:
        return (
            f"ONNX opset {opset} is outside the supported range "
            f"[{SUPPORTED_OPSET_MIN}, {SUPPORTED_OPSET_MAX}]; re-export the model against an opset "
            "in that range"
        )
    return None


def check_attributes(op_type: str, attrs: dict[str, object], node_name: str) -> list[str]:
    """Validate an op's *attributes* against its rule; return a list of actionable problems.

    This checks only what is decidable from the node's attributes alone (group, pads, strides,
    transA, ...). Structural constraints that need graph context — whether an ``Add`` operand is a
    constant initializer, whether a tensor is 2-D, whether a terminal op is really terminal — are
    the loader's job (:mod:`penumbra.onnx_loader`), since they need the initializer set and the
    inferred shapes. An unrecognized ``op_type`` is not this function's concern (the loader reports
    it as an unsupported op); an unknown op returns no attribute problems.
    """
    problems: list[str] = []
    if op_type == "Conv":
        group = attrs.get("group", 1)
        if group != 1:
            problems.append(
                f"Conv (node {node_name!r}): group={group} not supported (only group=1; grouped/"
                "depthwise conv is Phase 8)"
            )
        dilations = attrs.get("dilations")
        if dilations is not None and list(dilations) != [1, 1]:
            problems.append(
                f"Conv (node {node_name!r}): dilations={list(dilations)} not supported "
                "(only [1, 1])"
            )
        pads = attrs.get("pads")
        if pads is not None:
            pads = list(pads)
            if len(pads) != 4 or len(set(pads)) != 1:
                problems.append(
                    f"Conv (node {node_name!r}): pads={pads} not supported (only symmetric equal "
                    "padding on all sides, e.g. [p, p, p, p])"
                )
        strides = attrs.get("strides")
        if strides is not None:
            strides = list(strides)
            if len(strides) != 2 or strides[0] != strides[1]:
                problems.append(
                    f"Conv (node {node_name!r}): strides={strides} not supported (only square "
                    "strides, sh==sw)"
                )
        auto_pad = attrs.get("auto_pad", b"NOTSET")
        if _as_str(auto_pad) not in ("NOTSET", ""):
            problems.append(
                f"Conv (node {node_name!r}): auto_pad={_as_str(auto_pad)!r} not supported (use "
                "explicit symmetric pads)"
            )
    elif op_type == "Gemm":
        trans_a = attrs.get("transA", 0)
        if trans_a != 0:
            problems.append(
                f"Gemm (node {node_name!r}): transA={trans_a} not supported (only transA=0)"
            )
        alpha = attrs.get("alpha", 1.0)
        beta = attrs.get("beta", 1.0)
        if abs(float(alpha) - 1.0) > 1e-6:
            problems.append(f"Gemm (node {node_name!r}): alpha={alpha} not supported (only 1.0)")
        if abs(float(beta) - 1.0) > 1e-6:
            problems.append(f"Gemm (node {node_name!r}): beta={beta} not supported (only 1.0)")
    elif op_type in ("MaxPool", "AveragePool"):
        pads = attrs.get("pads")
        if pads is not None and any(p != 0 for p in pads):
            problems.append(
                f"{op_type} (node {node_name!r}): pads={list(pads)} not supported (only pads=0; "
                "padded pooling is Phase 8)"
            )
        ceil_mode = attrs.get("ceil_mode", 0)
        if ceil_mode != 0:
            problems.append(
                f"{op_type} (node {node_name!r}): ceil_mode={ceil_mode} not supported (only 0)"
            )
        dilations = attrs.get("dilations")
        if dilations is not None and any(d != 1 for d in dilations):
            problems.append(
                f"{op_type} (node {node_name!r}): dilations={list(dilations)} not supported "
                "(only 1)"
            )
        if op_type == "AveragePool":
            cip = attrs.get("count_include_pad", 0)
            if cip not in (0, 1):  # value is irrelevant with pads=0, but reject a garbage value
                problems.append(
                    f"AveragePool (node {node_name!r}): count_include_pad={cip} not understood"
                )
    elif op_type == "Cast":
        # A Cast folds away only if it preserves the represented value. Casting to a float type is
        # an identity on Penumbra's real-valued wire; casting to an int/bool type truncates and is
        # rejected. ONNX TensorProto dtype codes: FLOAT=1, FLOAT16=10, DOUBLE=11, BFLOAT16=16.
        _FLOAT_DTYPES = {1, 10, 11, 16}
        to = attrs.get("to")
        if to is None or int(to) not in _FLOAT_DTYPES:
            problems.append(
                f"Cast (node {node_name!r}): to={to} not supported (only a cast to a floating type "
                "is a value-preserving no-op; an int/bool cast changes the value — Phase 8)"
            )
    return problems


def _as_str(v: object) -> str:
    """ONNX string attributes arrive as bytes; normalize to ``str`` for comparison."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)
