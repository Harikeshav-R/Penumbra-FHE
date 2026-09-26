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

Branching (residual ``Add``, ``Concat``, ``Split``), non-ReLU activations (via affine LUTs),
and inference-mode ``BatchNormalization`` (folded into preceding accumulators) are supported
as of Phase 8.
"""

from __future__ import annotations

import math
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
CAT_MERGE = "merge"  # Concat -> Concat; a residual Add -> Add
CAT_SPLIT = "split"  # Split -> Split
CAT_BN_FOLD = "bn_fold"  # BatchNormalization -> folded into preceding accumulator


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
            "ReLU max(x,0) is realized via the fused-Requant special case: the preceding layer's "
            "Requant applies max(x,0) directly with zero extra op cost "
            "(`runtime/src/ops/requant.rs`). Must follow an accumulator, not be terminal."
        ),
        attribute_constraints="none.",
    ),
    "Tanh": OnnxOpRule(
        onnx_op="Tanh",
        internal_op="Activation",
        category=CAT_ACTIVATION,
        rationale=(
            "Hyperbolic tangent is realized as a single-input LUT over the narrow post-Requant "
            "activation domain, with signed pre-rescale floor and output zero-point folded into "
            "the downstream layer's bias. Must follow an accumulator and feed one."
        ),
        attribute_constraints="none.",
    ),
    "LeakyRelu": OnnxOpRule(
        onnx_op="LeakyRelu",
        internal_op="Activation",
        category=CAT_ACTIVATION,
        rationale=(
            "Leaky ReLU is realized as a single-input LUT over the narrow post-Requant "
            "activation domain, with signed pre-rescale floor and output zero-point folded into "
            "the downstream layer's bias. Must follow an accumulator and feed one."
        ),
        attribute_constraints="alpha (default 0.01).",
    ),
    "HardSwish": OnnxOpRule(
        onnx_op="HardSwish",
        internal_op="Activation",
        category=CAT_ACTIVATION,
        rationale=(
            "HardSwish is realized as a single-input LUT over the narrow post-Requant "
            "activation domain, with signed pre-rescale floor and output zero-point folded into "
            "the downstream layer's bias. Must follow an accumulator and feed one."
        ),
        attribute_constraints="none.",
    ),
    "Gelu": OnnxOpRule(
        onnx_op="Gelu",
        internal_op="Activation",
        category=CAT_ACTIVATION,
        rationale=(
            "GELU is realized as a single-input LUT over the narrow post-Requant "
            "activation domain, with signed pre-rescale floor and output zero-point folded into "
            "the downstream layer's bias. Must follow an accumulator and feed one."
        ),
        attribute_constraints="approximate in {'none', 'tanh'}.",
    ),
    "Elu": OnnxOpRule(
        onnx_op="Elu",
        internal_op="Activation",
        category=CAT_ACTIVATION,
        rationale=(
            "ELU is realized as a single-input LUT over the narrow post-Requant "
            "activation domain, with signed pre-rescale floor and output zero-point folded into "
            "the downstream layer's bias. Must follow an accumulator and feed one."
        ),
        attribute_constraints="alpha (default 1.0).",
    ),
    "HardSigmoid": OnnxOpRule(
        onnx_op="HardSigmoid",
        internal_op="Activation",
        category=CAT_ACTIVATION,
        rationale=(
            "HardSigmoid is realized as a single-input LUT over the narrow post-Requant "
            "activation domain, with signed pre-rescale floor and output zero-point folded into "
            "the downstream layer's bias. Must follow an accumulator and feed one."
        ),
        attribute_constraints="alpha (default 0.2), beta (default 0.5).",
    ),
    "Sigmoid": OnnxOpRule(
        onnx_op="Sigmoid",
        internal_op="Activation",
        category=CAT_ACTIVATION,
        rationale=(
            "Sigmoid is realized as a single-input LUT over the narrow post-Requant activation "
            "domain; a terminal Sigmoid is still dropped (argmax-invariant)."
        ),
        attribute_constraints=(
            "none; a terminal Sigmoid is dropped (argmax-invariant), a mid-graph one lowers to "
            "an Activation LUT."
        ),
    ),
    "MaxPool": OnnxOpRule(
        onnx_op="MaxPool",
        internal_op="Pool",
        category=CAT_POOL,
        rationale=(
            "Max pooling is a per-channel window reduction realized as pairwise ciphertext max "
            "(comparison PBS) — `Pool` mode 'max'."
        ),
        attribute_constraints=(
            "symmetric pads [p,p,p,p] with p < kernel (padded taps ignored); no auto_pad; "
            "ceil_mode=0; dilations=[1,1]; uniform kernel/stride."
        ),
    ),
    "AveragePool": OnnxOpRule(
        onnx_op="AveragePool",
        internal_op="Pool",
        category=CAT_POOL,
        rationale=(
            "Average pooling is a per-channel window sum (`add_parallelized`, no PBS); the 1/k "
            "is carried in the output quantization scale — `Pool` mode 'avg'."
        ),
        attribute_constraints=(
            "symmetric pads [p,p,p,p] with p < kernel; count_include_pad=1 when p > 0; "
            "no auto_pad; ceil_mode=0; uniform kernel/stride."
        ),
    ),
    "GlobalAveragePool": OnnxOpRule(
        onnx_op="GlobalAveragePool",
        internal_op="Pool",
        category=CAT_POOL,
        rationale=(
            "Global average pooling is AveragePool over the whole feature map (kernel = spatial "
            "size) — the same PBS-free window sum, 1/k is carried in the output quantization scale."
        ),
        attribute_constraints="none (kernel = full input spatial size).",
    ),
    "Add": OnnxOpRule(
        onnx_op="Add",
        internal_op="Add / Linear bias fold",
        category=CAT_BIAS_ADD,
        rationale=(
            "A constant-initializer Add after MatMul/Conv folds into its bias; a residual Add "
            "(both operands activations) lowers to internal Add. Operands must share one "
            "quantization scale, which the merge-scale pass enforces."
        ),
        attribute_constraints="none; both operands must have matching shapes.",
    ),
    "Concat": OnnxOpRule(
        onnx_op="Concat",
        internal_op="Concat",
        category=CAT_MERGE,
        rationale=(
            "Concatenation along the channel axis is a relabelling of the flat channel-major "
            "wire — free under TFHE (a ciphertext move, no PBS) and a 0/1 selection linear map "
            "under CKKS. Operands must share one quantization scale, which the merge-scale "
            "pass enforces."
        ),
        attribute_constraints=(
            "axis must resolve to 1 (the channel/feature axis); all inputs must share their "
            "non-concatenated dims."
        ),
    ),
    "Split": OnnxOpRule(
        onnx_op="Split",
        internal_op="Split",
        category=CAT_SPLIT,
        rationale=(
            "Splitting along the channel axis is a contiguous segmentation of the flat wire — "
            "free under TFHE, a 0/1 window map under CKKS. Scale and zero-point pass through "
            "unchanged."
        ),
        attribute_constraints=(
            "axis must resolve to 1; split sizes come from the 'split' attribute (opset < 13), "
            "a constant 'split' input (opset >= 13), or an equal division."
        ),
    ),
    "BatchNormalization": OnnxOpRule(
        onnx_op="BatchNormalization",
        internal_op="(folded into the preceding Conv2d/Linear)",
        category=CAT_BN_FOLD,
        rationale=(
            "Inference-time BN is a per-channel affine map that composes exactly with the "
            "preceding accumulator's weights and bias, so it is folded at load time and costs "
            "nothing at runtime."
        ),
        attribute_constraints=(
            "training_mode must be 0/absent; exactly one output; scale/B/mean/var must be "
            "constant initializers; must directly follow a Conv/Gemm/MatMul."
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
        kernel = [int(k) for k in attrs.get("kernel_shape", [])]
        pads = attrs.get("pads")
        pad = 0
        if pads is not None:
            pads = [int(p) for p in pads]
            if len(pads) != 4 or len(set(pads)) != 1:
                problems.append(
                    f"{op_type} (node {node_name!r}): pads={pads} not supported (only symmetric "
                    "equal padding on all sides, e.g. [p, p, p, p])"
                )
            else:
                pad = pads[0]
                if len(kernel) == 2 and pad >= min(kernel):
                    problems.append(
                        f"{op_type} (node {node_name!r}): pads={pads} must be smaller than "
                        f"kernel_shape={kernel} (every window must cover at least one real input)"
                    )
        auto_pad = attrs.get("auto_pad", b"NOTSET")
        if _as_str(auto_pad) not in ("NOTSET", ""):
            problems.append(
                f"{op_type} (node {node_name!r}): auto_pad={_as_str(auto_pad)!r} not supported "
                "(use explicit symmetric pads)"
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
            if cip not in (0, 1):
                problems.append(
                    f"AveragePool (node {node_name!r}): count_include_pad={cip} not understood"
                )
            elif pad > 0 and cip != 1:
                problems.append(
                    f"AveragePool (node {node_name!r}): count_include_pad={cip} with pads={pads} "
                    "not supported (border windows would divide by a per-position count, which "
                    "no single quantization scale can carry); export with count_include_pad=1 "
                    "(PyTorch's default)"
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
    elif op_type == "Gelu":
        approx = _as_str(attrs.get("approximate", "none")).lower()
        if not approx:
            approx = "none"
        if approx not in ("none", "tanh"):
            problems.append(
                f"Gelu (node {node_name!r}): approximate={approx!r} not supported "
                "(only 'none' or 'tanh')"
            )
    elif op_type in ("LeakyRelu", "Elu"):
        alpha = attrs.get("alpha")
        if alpha is not None:
            try:
                val = float(alpha)  # type: ignore[arg-type]
                if not math.isfinite(val):
                    problems.append(f"{op_type} (node {node_name!r}): alpha={alpha} must be finite")
            except (TypeError, ValueError):
                problems.append(
                    f"{op_type} (node {node_name!r}): alpha={alpha} is not a valid float"
                )
    elif op_type == "HardSigmoid":
        alpha = attrs.get("alpha")
        if alpha is not None:
            try:
                val = float(alpha)  # type: ignore[arg-type]
                if not math.isfinite(val):
                    problems.append(
                        f"HardSigmoid (node {node_name!r}): alpha={alpha} must be finite"
                    )
            except (TypeError, ValueError):
                problems.append(
                    f"HardSigmoid (node {node_name!r}): alpha={alpha} is not a valid float"
                )
        beta = attrs.get("beta")
        if beta is not None:
            try:
                val = float(beta)  # type: ignore[arg-type]
                if not math.isfinite(val):
                    problems.append(f"HardSigmoid (node {node_name!r}): beta={beta} must be finite")
            except (TypeError, ValueError):
                problems.append(
                    f"HardSigmoid (node {node_name!r}): beta={beta} is not a valid float"
                )
    elif op_type in ("Concat", "Split"):
        axis = attrs.get("axis")
        if axis is None and op_type == "Concat":
            problems.append(f"Concat (node {node_name!r}): missing required attribute 'axis'")
        elif axis is not None:
            try:
                ax = int(axis)  # type: ignore[arg-type]
                if ax != 1 and ax >= 0:
                    problems.append(
                        f"{op_type} (node {node_name!r}): axis={axis} not supported "
                        "(only channel axis 1)"
                    )
            except (TypeError, ValueError):
                problems.append(f"{op_type} (node {node_name!r}): axis={axis!r} is not a valid int")
    elif op_type == "BatchNormalization":
        training_mode = attrs.get("training_mode", 0)
        if training_mode != 0:
            problems.append(
                f"BatchNormalization (node {node_name!r}): training_mode={training_mode} not "
                "supported (only inference-mode BN with training_mode=0)"
            )
    return problems


def _as_str(v: object) -> str:
    """ONNX string attributes arrive as bytes; normalize to ``str`` for comparison."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)
