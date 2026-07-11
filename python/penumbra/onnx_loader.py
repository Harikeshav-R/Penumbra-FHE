"""The ONNX front door: parse -> validate -> lower to a float :class:`~penumbra.model.Model`.

ONNX is the universal export format (PyTorch, sklearn, Keras, XGBoost all emit it):
"train anywhere, run encrypted here" (``PROJECT.md`` §10).

:func:`load_onnx` runs entirely host-side (NumPy + the ``onnx`` package, no crypto):

    1. **Parse** the ONNX graph (``onnx.load`` + ``onnx.checker.check_model``) and pin the opset
       to a supported range (:mod:`penumbra.op_registry`).
    2. **Shape-infer** (``onnx.shape_inference``) so every tensor has a resolved shape, and read
       initializers into NumPy arrays.
    3. **Validate every node against the registry and fail loudly at load time**, listing *all*
       problems at once (``AGENTS.md`` §1.4) via :class:`UnsupportedModelError`.
    4. **Identify the linear chain** input -> output (this is exactly what ``Model.quantize``
       compiles — branching is Phase 8), dropping a terminal Softmax/Sigmoid/ArgMax tail and
       folding away layout-only Reshape/Flatten/Transpose nodes.
    5. **Lower** each surviving node to a :mod:`penumbra.layers` float layer and return
       ``Model(layers, input_bits=...)``.

The returned :class:`~penumbra.model.Model` flows through the **existing** Phase-5 quantization
service unchanged — the caller does ``model.quantize(calibration_data, ...)`` then
``model.export(path)``. There is deliberately no separate ``.compile()``: ONNX validation *is*
the compile step (the ``PROJECT.md`` §12 sketch predates the Phase-5 quantize-lowers-to-IR
design). This front end emits nothing new at the IR layer — every layer it builds already has an
``OpSpec`` — so the golden invariant is preserved by construction (``AGENTS.md`` §1.1, §1.2).

"Any ONNX model" is bounded: only the supported ops
(:func:`penumbra.op_registry.supported_onnx_ops`),
only a **linear chain** of them (a chain of Conv/Gemm/MatMul accumulators, each optionally ReLU'd,
plus pooling, with a wide logit head), only models that quantize acceptably, and only sizes that
run in reasonable time (``PROJECT.md`` §10, §16). Anything else fails loudly here.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from penumbra import op_registry
from penumbra.layers import Activation, Conv2d, Layer, Linear, Pool
from penumbra.model import Model


class UnsupportedModelError(ValueError):
    """Raised at load time when an ONNX model is outside Penumbra-FHE's supported subset.

    Carries the full list of problems (:attr:`problems`) so the user sees *every* unsupported op /
    attribute / structural issue in one shot rather than fixing them one at a time (``AGENTS.md``
    §1.4). ``str(err)`` is the joined, actionable report.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        header = (
            f"model is not supported ({len(self.problems)} problem(s) — Penumbra-FHE accepts a "
            "linear chain of supported ops; see docs/SUPPORTED-OPS.md):"
        )
        super().__init__(header + "".join(f"\n  - {p}" for p in self.problems))


def load_onnx(path: str, *, input_bits: int = 4) -> Model:
    """Parse and lower an ONNX model to a quantizable :class:`~penumbra.model.Model`.

    Args:
        path: filesystem path to a ``.onnx`` file.
        input_bits: bit-width the input tensor is quantized to (project convention: 4).

    Returns:
        A float :class:`~penumbra.model.Model` (a list of :mod:`penumbra.layers` layers) ready for
        ``.quantize(calibration_data, ...)`` / ``.export(path)``.

    Raises:
        UnsupportedModelError: if any node is unsupported, has an unsupported attribute, or the
            graph is not a single linear chain of supported ops. All problems are listed at once.
    """
    model = onnx.load(str(path))
    # Check the opset range *before* onnx's schema checker: an out-of-range opset makes
    # check_model reject nodes with cryptic per-node schema errors (a node's schema differs across
    # opsets), which would bury our actionable "re-export against a supported opset" message. This
    # is the one global gate we surface on its own (per-node checks against a wrong-opset model are
    # misleading noise); everything else is bundled into one report below (§1.4).
    _check_opset(model)
    # check_model gives a clean, specific parse/structural error rather than a downstream crash.
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph

    consts = _read_constants(graph)
    shapes = _read_shapes(graph)

    # Validate everything decidable up front and raise ONE error listing all problems (§1.4).
    problems = _validate(graph, consts)
    if problems:
        raise UnsupportedModelError(problems)

    # Single non-initializer graph input.
    graph_input = _sole_input(graph, consts)
    # Single graph output.
    if len(graph.output) != 1:
        raise UnsupportedModelError(
            [f"graph has {len(graph.output)} outputs; only single-output models are supported"]
        )
    graph_output = graph.output[0].name

    chain = _identify_chain(graph, consts, graph_input, graph_output)
    layers = _lower_chain(chain, consts, shapes, graph_input)
    if not layers:
        raise UnsupportedModelError(
            ["model has no computational layers after folding shape/terminal ops"]
        )
    return Model(layers, input_bits=input_bits)


# --- parsing helpers ----------------------------------------------------------------------


def _read_constants(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    """All constant tensors: graph initializers plus ``Constant`` node outputs, as NumPy arrays.

    These are the non-activation tensors — weights, biases, reshape target shapes, folded scalars.
    The chain walker treats a node input as an *activation* iff it is not in this map, which is how
    a MatMul's constant weight or a bias Add's constant operand is told apart from the running
    tensor.
    """
    consts: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        consts[init.name] = numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    consts[node.output[0]] = numpy_helper.to_array(attr.t)
    return consts


def _read_shapes(graph: onnx.GraphProto) -> dict[str, tuple[int | None, ...]]:
    """Map every tensor name to its (shape-inferred) dims, ``None`` for an unknown/symbolic dim."""
    shapes: dict[str, tuple[int | None, ...]] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        dims: list[int | None] = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.HasField("dim_value") else None)
        shapes[vi.name] = tuple(dims)
    return shapes


def _attrs(node: onnx.NodeProto) -> dict[str, object]:
    """Node attributes as a plain dict (``onnx.helper.get_attribute_value`` per attribute)."""
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _sole_input(graph: onnx.GraphProto, consts: dict[str, np.ndarray]) -> str:
    """The single non-initializer graph input name (raises loudly on 0 or >1)."""
    real = [i.name for i in graph.input if i.name not in consts]
    if len(real) != 1:
        raise UnsupportedModelError(
            [
                f"graph has {len(real)} non-initializer inputs {real}; only single-input models "
                "are supported"
            ]
        )
    return real[0]


# --- validation (collect ALL problems) ----------------------------------------------------


def _check_opset(model: onnx.ModelProto) -> None:
    """Reject an unsupported default-domain opset up front, before onnx's per-node schema checker.

    Raises :class:`UnsupportedModelError` with the registry's actionable message if the ai.onnx
    (domain "") opset is outside the supported range.
    """
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            msg = op_registry.opset_problem(opset.version)
            if msg:
                raise UnsupportedModelError([msg])


def _validate(graph: onnx.GraphProto, consts: dict[str, np.ndarray]) -> list[str]:
    """Collect every decidable problem at once: unsupported ops, bad attributes, residual Adds.

    Structural problems that need the full traversal (fan-out branching, disconnected nodes,
    non-foldable transpose, a non-terminal classifier) are raised later by the chain walker /
    lowerer with their own clear messages; those are genuinely different topologies, not a list of
    independent node defects.
    """
    problems: list[str] = []

    for node in graph.node:
        if node.op_type == "Constant":
            continue  # folded into `consts`, never lowered
        name = node.name or f"<{node.op_type}>"
        if not op_registry.is_supported(node.op_type):
            problems.append(
                f"operator {node.op_type} (node {name!r}) not supported (deferred to Phase 8 or "
                "out of scope; see docs/SUPPORTED-OPS.md)"
            )
            continue
        problems.extend(op_registry.check_attributes(node.op_type, _attrs(node), name))
        # A residual/branching Add (both operands are activations, not a constant bias) needs
        # multi-input topological eval — surfaced here so it lands in the same all-at-once report
        # as unsupported ops (the registry lists Add as the constant-bias-fold case only).
        if node.op_type == "Add":
            act_inputs = [i for i in node.input if i not in consts]
            if len(act_inputs) != 1:
                problems.append(
                    f"Add (node {name!r}): residual/branching Add not supported (needs exactly one "
                    "constant-bias operand; both operands are activations here — deferred to "
                    "Phase 8)"
                )

    return problems


# --- linear-chain identification ----------------------------------------------------------


def _identify_chain(
    graph: onnx.GraphProto,
    consts: dict[str, np.ndarray],
    graph_input: str,
    graph_output: str,
) -> list[onnx.NodeProto]:
    """Walk the single input->output path; fail loudly on any branching or disconnection.

    Each step: the current activation tensor must be consumed by exactly one node (else fan-out
    branching), and that node must have exactly one activation input (its other inputs are
    constants — else fan-in branching, already reported for Add). Returns the ordered node list.
    """
    producers = _nonconstant_nodes(graph)  # nodes we must account for (Constants excluded)
    by_input: dict[str, list[onnx.NodeProto]] = {}
    for node in producers:
        for name in node.input:
            if name not in consts:
                by_input.setdefault(name, []).append(node)

    chain: list[onnx.NodeProto] = []
    seen: set[int] = set()
    current = graph_input
    while True:
        consumers = by_input.get(current, [])
        if not consumers:
            break  # reached a graph output (a dead-end tensor)
        if len(consumers) > 1:
            names = [n.name or f"<{n.op_type}>" for n in consumers]
            raise UnsupportedModelError(
                [
                    f"tensor {current!r} feeds {len(consumers)} nodes {names}: branching/fan-out "
                    "graphs are not supported (linear chain only; Phase 8)"
                ]
            )
        node = consumers[0]
        act_inputs = [i for i in node.input if i not in consts]
        if len(act_inputs) != 1:
            name = node.name or f"<{node.op_type}>"
            raise UnsupportedModelError(
                [
                    f"node {name!r} ({node.op_type}) has {len(act_inputs)} activation inputs "
                    f"{act_inputs}: multi-input/branching nodes are not supported (Phase 8)"
                ]
            )
        chain.append(node)
        seen.add(id(node))
        current = node.output[0]

    if current != graph_output:
        raise UnsupportedModelError(
            [
                f"the linear chain ends at tensor {current!r}, not the graph output "
                f"{graph_output!r}: the graph is not a single input->output path (Phase 8)"
            ]
        )
    unreached = [n.name or f"<{n.op_type}>" for n in producers if id(n) not in seen]
    if unreached:
        raise UnsupportedModelError(
            [
                f"nodes {unreached} are not on the input->output path: disconnected/branching "
                "graphs are not supported (Phase 8)"
            ]
        )
    return chain


def _nonconstant_nodes(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    return [n for n in graph.node if n.op_type != "Constant"]


# --- lowering -----------------------------------------------------------------------------


def _lower_chain(
    chain: list[onnx.NodeProto],
    consts: dict[str, np.ndarray],
    shapes: dict[str, tuple[int | None, ...]],
    graph_input: str,
) -> list[Layer]:
    """Lower an ordered node chain to float :mod:`penumbra.layers` layers.

    Shape ops (Reshape/Flatten/Transpose) fold to nothing; a terminal classifier tail
    (Softmax/LogSoftmax/Sigmoid/ArgMax) is dropped; a constant-bias Add folds into the preceding
    accumulator's bias; everything else lowers to its layer.
    """
    layers: list[Layer] = []
    n = len(chain)
    for idx, node in enumerate(chain):
        cat = op_registry.rule_for(node.op_type).category
        is_last = idx == n - 1
        act_in = _activation_input(node, consts)

        if cat == op_registry.CAT_TERMINAL:
            if not is_last:
                raise UnsupportedModelError(
                    [
                        f"{node.op_type} (node {node.name or '<?>'!r}) is not the terminal node: a "
                        "non-terminal Softmax/Sigmoid/ArgMax is a real activation and is not "
                        "supported (only a terminal classifier tail is dropped; Phase 8)"
                    ]
                )
            break  # drop the terminal tail: logits are the graph output, client argmaxes them
        elif cat == op_registry.CAT_SHAPE:
            _check_shape_op_is_noop(node, shapes, act_in)
            continue  # layout no-op on the flat wire
        elif cat == op_registry.CAT_BIAS_ADD:
            _fold_bias_add(node, consts, layers)
            continue
        elif node.op_type == "Conv":
            layers.append(_lower_conv(node, consts, shapes, act_in))
        elif node.op_type in ("Gemm", "MatMul"):
            layers.append(_lower_linear(node, consts, act_in))
        elif cat == op_registry.CAT_ACTIVATION:  # Relu
            layers.append(Activation(lambda v: max(v, 0.0)))
        elif cat == op_registry.CAT_POOL:
            layers.append(_lower_pool(node, shapes, act_in))
        else:  # pragma: no cover - registry/loader drift guard
            raise UnsupportedModelError(
                [f"operator {node.op_type} (node {node.name or '<?>'!r}) has no lowering rule"]
            )
    return layers


def _activation_input(node: onnx.NodeProto, consts: dict[str, np.ndarray]) -> str:
    """The node's single activation (non-constant) input tensor name."""
    act = [i for i in node.input if i not in consts]
    # Chain identification guarantees exactly one; guard anyway for a clear message.
    if len(act) != 1:
        raise UnsupportedModelError(
            [
                f"node {node.name or '<?>'!r} ({node.op_type}) does not have a single "
                "activation input"
            ]
        )
    return act[0]


def _nonbatch(
    name: str, shapes: dict[str, tuple[int | None, ...]], node_name: str, *, expect: int
) -> tuple[int, ...]:
    """The ``expect`` non-batch dims of a tensor, resolved to concrete ints (else raises loudly).

    ``expect`` is the required non-batch rank (3 = NCHW for both Conv and 2-D Pool). A tensor with
    a different rank — e.g. a 1-D (NCL) or 3-D (NCDHW) pool, both valid ONNX that pass
    ``onnx.checker`` — is rejected here with an actionable message (``AGENTS.md`` §1.4) rather than
    crashing on the caller's fixed-arity tuple unpack.
    """
    shape = shapes.get(name)
    if shape is None or len(shape) < 2:
        raise UnsupportedModelError(
            [
                f"node {node_name!r}: input tensor {name!r} has no inferred shape "
                f"({shape}); shape inference could not resolve it"
            ]
        )
    nonbatch = shape[1:]
    if len(nonbatch) != expect:
        raise UnsupportedModelError(
            [
                f"node {node_name!r}: input tensor {name!r} has shape {shape} with {len(nonbatch)} "
                f"non-batch dims; only {expect}-D feature maps (NCHW, i.e. 2-D conv/pool) are "
                "supported"
            ]
        )
    if any(d is None for d in nonbatch):
        raise UnsupportedModelError(
            [
                f"node {node_name!r}: input tensor {name!r} has an unresolved non-batch dim in "
                f"{shape}; only statically-shaped feature maps are supported"
            ]
        )
    return tuple(int(d) for d in nonbatch)  # type: ignore[arg-type]


def _lower_conv(
    node: onnx.NodeProto,
    consts: dict[str, np.ndarray],
    shapes: dict[str, tuple[int | None, ...]],
    act_in: str,
) -> Conv2d:
    """Conv -> layers.Conv2d. Weight (out,in,kh,kw) passes through; stride/padding are scalars."""
    name = node.name or "<Conv>"
    attrs = _attrs(node)
    weight = np.asarray(consts[node.input[1]], dtype=np.float64)
    if weight.ndim != 4:
        raise UnsupportedModelError(
            [f"Conv (node {name!r}): only 2-D conv is supported, weight has {weight.ndim} dims"]
        )
    bias = None
    if len(node.input) >= 3 and node.input[2]:
        bias = np.asarray(consts[node.input[2]], dtype=np.float64)

    in_ch, in_h, in_w = _nonbatch(act_in, shapes, name, expect=3)
    if weight.shape[1] != in_ch:
        raise UnsupportedModelError(
            [
                f"Conv (node {name!r}): weight in-channels {weight.shape[1]} != input channels "
                f"{in_ch}"
            ]
        )
    strides = list(attrs.get("strides", [1, 1]))  # attributes validated square in the registry
    stride = int(strides[0])
    pads = list(attrs.get("pads", [0, 0, 0, 0]))  # validated symmetric-equal in the registry
    padding = int(pads[0]) if pads else 0
    return Conv2d(
        weight=weight,
        in_h=in_h,
        in_w=in_w,
        in_channels=in_ch,
        stride=stride,
        padding=padding,
        bias=bias,
    )


def _lower_linear(node: onnx.NodeProto, consts: dict[str, np.ndarray], act_in: str) -> Linear:
    """Gemm/MatMul -> layers.Linear with weight resolved to (n_out, n_in) and optional bias.

    The weight is the *constant* matrix operand and the activation is ``act_in``. A dense layer
    ``x @ W`` exports with the activation as the first operand (``input[0]``), which is the case
    the lowering supports. ``MatMul`` is operand-symmetric, so a weight-first ``W @ x`` (constant at
    ``input[0]``, activation at ``input[1]``) is also valid ONNX but lowers to a different layout
    (``y = W @ x`` is not ``x @ W.T``); rather than index ``input[1]`` blindly (a raw ``KeyError``
    when the weight is at ``input[0]``), reject it loudly (``AGENTS.md`` §1.4).
    """
    name = node.name or f"<{node.op_type}>"
    if not node.input or node.input[0] != act_in:
        raise UnsupportedModelError(
            [
                f"{node.op_type} (node {name!r}): the activation must be the first operand "
                f"(x @ W); a weight-first {node.op_type} (W @ x) lowers to a different layout and "
                "is not supported (transpose the export, or Phase 8)"
            ]
        )
    b = np.asarray(consts[node.input[1]], dtype=np.float64)
    if b.ndim != 2:
        raise UnsupportedModelError(
            [f"{node.op_type} (node {name!r}): weight must be 2-D, got shape {b.shape}"]
        )
    # Gemm computes A@B (transB=0) or A@B^T (transB=1); MatMul is A@B. Linear wants W with
    # y = x @ W.T and W = (n_out, n_in). transB=1 -> B is already (n_out, n_in); otherwise B is
    # (n_in, n_out) and we transpose. (transA/alpha/beta constrained to identity by the registry.)
    trans_b = int(_attrs(node).get("transB", 0)) if node.op_type == "Gemm" else 0
    weight = b if trans_b == 1 else b.T
    bias = None
    if node.op_type == "Gemm" and len(node.input) >= 3 and node.input[2]:
        bias = np.asarray(consts[node.input[2]], dtype=np.float64).reshape(-1)
        if bias.shape[0] != weight.shape[0]:
            raise UnsupportedModelError(
                [
                    f"Gemm (node {name!r}): bias length {bias.shape[0]} != output size "
                    f"{weight.shape[0]}"
                ]
            )
    return Linear(weight=np.ascontiguousarray(weight), bias=bias)


def _lower_pool(
    node: onnx.NodeProto,
    shapes: dict[str, tuple[int | None, ...]],
    act_in: str,
) -> Pool:
    """MaxPool/AveragePool/GlobalAveragePool -> layers.Pool (avg emits the window sum)."""
    name = node.name or f"<{node.op_type}>"
    channels, in_h, in_w = _nonbatch(act_in, shapes, name, expect=3)
    mode = "max" if node.op_type == "MaxPool" else "avg"

    if node.op_type == "GlobalAveragePool":
        pool_h, pool_w, stride = in_h, in_w, 1  # whole-map window -> 1x1 output
    else:
        attrs = _attrs(node)
        kernel = list(attrs.get("kernel_shape", []))
        if len(kernel) != 2:
            raise UnsupportedModelError(
                [
                    f"{node.op_type} (node {name!r}): only 2-D pooling is supported "
                    f"(kernel_shape={kernel})"
                ]
            )
        pool_h, pool_w = int(kernel[0]), int(kernel[1])
        strides = list(attrs.get("strides", [1, 1]))  # ONNX default stride is 1 per axis
        if len(strides) != 2 or strides[0] != strides[1]:
            raise UnsupportedModelError(
                [
                    f"{node.op_type} (node {name!r}): only square strides are supported "
                    f"(strides={strides})"
                ]
            )
        stride = int(strides[0])
    return Pool(
        mode=mode,
        in_h=in_h,
        in_w=in_w,
        channels=channels,
        pool_h=pool_h,
        pool_w=pool_w,
        stride=stride,
    )


def _fold_bias_add(
    node: onnx.NodeProto, consts: dict[str, np.ndarray], layers: list[Layer]
) -> None:
    """Fold a constant-operand Add into the preceding accumulator layer's bias.

    A dense layer commonly exports as MatMul then Add-of-a-constant; that Add *is* the layer's
    bias. (A residual Add — both operands activations — was already rejected in validation.)
    """
    name = node.name or "<Add>"
    const_inputs = [i for i in node.input if i in consts]
    if len(const_inputs) != 1:
        raise UnsupportedModelError(
            [f"Add (node {name!r}): expected exactly one constant operand to fold as a bias"]
        )
    addend = np.asarray(consts[const_inputs[0]], dtype=np.float64).reshape(-1)
    if not layers or not isinstance(layers[-1], (Linear, Conv2d)):
        raise UnsupportedModelError(
            [
                f"Add (node {name!r}): a constant-bias Add must directly follow a "
                "Gemm/MatMul/Conv accumulator to fold into its bias"
            ]
        )
    acc = layers[-1]
    n_out = acc.weight.shape[0]
    if addend.shape[0] != n_out:
        raise UnsupportedModelError(
            [
                f"Add (node {name!r}): constant operand length {addend.shape[0]} != preceding "
                f"layer output size {n_out}; cannot fold as a bias"
            ]
        )
    acc.bias = addend if acc.bias is None else np.asarray(acc.bias, dtype=np.float64) + addend


def _check_shape_op_is_noop(
    node: onnx.NodeProto,
    shapes: dict[str, tuple[int | None, ...]],
    act_in: str,
) -> None:
    """Confirm a Reshape/Flatten/Transpose does not reorder the flat channel-major wire.

    Reshape/Flatten only reinterpret a row-major buffer, so they never reorder the flat elements —
    always foldable. A Transpose *does* permute; it is a no-op only when its perm leaves the
    row-major flattening unchanged (identity, or permuting only size-1 axes). A genuinely
    reordering Transpose is rejected loudly (baking it into the next weight is Phase-8 work).
    """
    if node.op_type in ("Reshape", "Flatten"):
        return
    # Transpose: decide from the concrete input shape whether the flat order is preserved.
    name = node.name or "<Transpose>"
    shape = shapes.get(act_in)
    perm = _attrs(node).get("perm")
    if shape is None or any(d is None for d in shape):
        # Can't prove it's a no-op without a concrete shape; only an identity perm is safe.
        if perm is not None and list(perm) != list(range(len(list(perm)))):
            raise UnsupportedModelError(
                [
                    f"Transpose (node {name!r}): cannot prove it preserves flat order without a "
                    f"resolved input shape (shape={shape}, perm={list(perm)})"
                ]
            )
        return
    dims = [int(d) for d in shape]  # type: ignore[arg-type]
    perm = list(perm) if perm is not None else list(reversed(range(len(dims))))
    flat = np.arange(int(np.prod(dims))).reshape(dims).transpose(perm).reshape(-1)
    if not np.array_equal(flat, np.arange(flat.size)):
        raise UnsupportedModelError(
            [
                f"Transpose (node {name!r}): perm={perm} reorders the flat channel-major vector "
                "and cannot be folded away (bake it into the following weight, or Phase 8)"
            ]
        )
