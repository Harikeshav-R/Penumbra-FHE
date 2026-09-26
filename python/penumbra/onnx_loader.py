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

import heapq

import numpy as np
import onnx
from onnx import numpy_helper

from penumbra import op_registry
from penumbra.layers import (
    Activation,
    Add,
    Concat,
    Conv2d,
    LayerNode,
    Linear,
    Pool,
    Split,
)
from penumbra.model import Model
from penumbra.quantization.activations import activation_fn
from penumbra.quantization.batchnorm import fold_batchnorm


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
            "directed acyclic graph of supported ops; see docs/SUPPORTED-OPS.md):"
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

    topo_nodes = _topological_nodes(graph, consts, graph_input, graph_output)
    layer_nodes, effective_output = _lower_graph(
        topo_nodes, consts, shapes, graph_input, graph_output
    )
    if not layer_nodes:
        raise UnsupportedModelError(
            ["model has no computational layers after folding shape/terminal ops"]
        )
    return Model(
        layer_nodes,
        input_bits=input_bits,
        input_name=graph_input,
        output_name=effective_output,
    )


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

    return problems


# --- linear-chain identification ----------------------------------------------------------


def _nonconstant_nodes(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    return [n for n in graph.node if n.op_type != "Constant"]


def _topological_nodes(
    graph: onnx.GraphProto,
    consts: dict[str, np.ndarray],
    graph_input: str,
    graph_output: str,
) -> list[onnx.NodeProto]:
    """Every non-Constant node in a valid evaluation order (stable Kahn).

    Raises loudly on a cycle, on a node whose activation input no node produces, and on a
    node that cannot reach the graph output (dead code).
    """
    producers = _nonconstant_nodes(graph)
    if not producers:
        return []

    producer_map: dict[str, int] = {}
    for idx, node in enumerate(producers):
        for out in node.output:
            producer_map[out] = idx

    for node in producers:
        name = node.name or f"<{node.op_type}>"
        for inp in node.input:
            if inp not in consts and inp != graph_input and inp not in producer_map:
                raise UnsupportedModelError(
                    [
                        f"node {name!r} ({node.op_type}) reads tensor {inp!r}, which no node "
                        "produces and is not a graph input"
                    ]
                )

    forward_reachable_tensors: set[str] = {graph_input}
    backward_reachable_tensors: set[str] = {graph_output}

    node_act_inputs: list[list[str]] = [
        [i for i in node.input if i not in consts] for node in producers
    ]

    changed = True
    while changed:
        changed = False
        for idx, node in enumerate(producers):
            ins = node_act_inputs[idx]
            if ins and all(i in forward_reachable_tensors for i in ins):
                for o in node.output:
                    if o not in forward_reachable_tensors:
                        forward_reachable_tensors.add(o)
                        changed = True

    changed = True
    while changed:
        changed = False
        for idx, node in enumerate(producers):
            if any(o in backward_reachable_tensors for o in node.output):
                for i in node_act_inputs[idx]:
                    if i not in backward_reachable_tensors:
                        backward_reachable_tensors.add(i)
                        changed = True

    unreached = []
    for idx, node in enumerate(producers):
        name = node.name or f"<{node.op_type}>"
        is_forward = (
            all(i in forward_reachable_tensors for i in node_act_inputs[idx])
            if node_act_inputs[idx]
            else False
        )
        is_backward = any(o in backward_reachable_tensors for o in node.output)
        if not (is_forward and is_backward):
            unreached.append(name)

    if unreached:
        raise UnsupportedModelError(
            [
                f"nodes {unreached} are not on the input->output path: disconnected/branching "
                "graphs are not supported (Phase 8)"
            ]
        )

    num_nodes = len(producers)
    in_deps: list[set[int]] = [set() for _ in range(num_nodes)]
    dependents: list[list[int]] = [[] for _ in range(num_nodes)]

    for c_idx, ins in enumerate(node_act_inputs):
        for inp in ins:
            if inp in producer_map:
                p_idx = producer_map[inp]
                if p_idx not in in_deps[c_idx]:
                    in_deps[c_idx].add(p_idx)
                    dependents[p_idx].append(c_idx)

    ready = [idx for idx, deps in enumerate(in_deps) if not deps]
    heapq.heapify(ready)

    order: list[int] = []
    while ready:
        idx = heapq.heappop(ready)
        order.append(idx)
        for dep in dependents[idx]:
            in_deps[dep].remove(idx)
            if not in_deps[dep]:
                heapq.heappush(ready, dep)

    if len(order) < num_nodes:
        visited = set(order)
        cycle_names = [
            producers[i].name or f"<{producers[i].op_type}>"
            for i in range(num_nodes)
            if i not in visited
        ]
        raise UnsupportedModelError(
            [
                f"graph has a cycle: node(s) {cycle_names} are never ready — their inputs depend "
                "on their own outputs"
            ]
        )

    return [producers[i] for i in order]


def _lower_graph(
    topo_nodes: list[onnx.NodeProto],
    consts: dict[str, np.ndarray],
    shapes: dict[str, tuple[int | None, ...]],
    graph_input: str,
    graph_output: str,
) -> tuple[list[LayerNode], str]:
    """Lower an ordered node DAG to a list of LayerNodes.

    Returns (layer_nodes, effective_graph_output).
    """
    layer_nodes: list[LayerNode] = []
    produced: dict[str, str] = {graph_input: graph_input}
    effective_output = graph_output

    for node in topo_nodes:
        rule = op_registry.rule_for(node.op_type)
        cat = rule.category
        name = node.name or f"<{node.op_type}>"

        if cat == op_registry.CAT_TERMINAL:
            if graph_output not in node.output:
                raise UnsupportedModelError(
                    [
                        f"{node.op_type} (node {name!r}) is not the terminal node: a "
                        "non-terminal Softmax/Sigmoid/ArgMax is a real activation and is not "
                        "supported (only a terminal classifier tail is dropped; Phase 8)"
                    ]
                )
            act_ins = [i for i in node.input if i not in consts]
            if len(act_ins) != 1:
                raise UnsupportedModelError(
                    [f"node {name!r} ({node.op_type}) does not have a single activation input"]
                )
            effective_output = produced.get(act_ins[0], act_ins[0])
            continue

        if cat == op_registry.CAT_SHAPE:
            act_ins = [i for i in node.input if i not in consts]
            if len(act_ins) != 1:
                raise UnsupportedModelError(
                    [f"node {name!r} ({node.op_type}) does not have a single activation input"]
                )
            act_in = act_ins[0]
            _check_shape_op_is_noop(node, shapes, act_in)
            produced[node.output[0]] = produced.get(act_in, act_in)
            continue

        if cat == op_registry.CAT_BIAS_ADD:
            const_inputs = [i for i in node.input if i in consts]
            if const_inputs:
                _fold_bias_add(node, consts, produced, layer_nodes, topo_nodes)
                continue
            cat = op_registry.CAT_MERGE

        if cat == op_registry.CAT_BN_FOLD:
            _fold_batchnorm(node, consts, produced, layer_nodes, topo_nodes)
            continue

        if cat == op_registry.CAT_MERGE:
            if node.op_type == "Concat":
                act_ins = [i for i in node.input if i not in consts]
                rank = len(shapes[act_ins[0]]) if act_ins and act_ins[0] in shapes else 2
                raw_axis = int(_attrs(node).get("axis", 1))  # type: ignore[arg-type]
                resolved_axis = raw_axis % rank
                if resolved_axis != 1:
                    raise UnsupportedModelError(
                        [
                            f"Concat (node {name!r}): axis={raw_axis} resolved to {resolved_axis}; "
                            "only axis 1 is supported"
                        ]
                    )
                if rank == 4 and act_ins[0] in shapes:
                    h0, w0 = shapes[act_ins[0]][2], shapes[act_ins[0]][3]
                    for inp in act_ins[1:]:
                        if inp in shapes:
                            h, w = shapes[inp][2], shapes[inp][3]
                            if (h, w) != (h0, w0):
                                raise UnsupportedModelError(
                                    [
                                        f"Concat (node {name!r}): all inputs must share spatial "
                                        f"dimensions (H, W); got ({h0}, {w0}) vs ({h}, {w})"
                                    ]
                                )
                inputs = [produced.get(i, i) for i in node.input]
                ln = LayerNode(name=name, layer=Concat(), inputs=inputs, outputs=[node.output[0]])
                layer_nodes.append(ln)
                produced[node.output[0]] = node.output[0]
            elif node.op_type == "Add":
                inputs = [produced.get(i, i) for i in node.input]
                ln = LayerNode(name=name, layer=Add(), inputs=inputs, outputs=[node.output[0]])
                layer_nodes.append(ln)
                produced[node.output[0]] = node.output[0]
            continue

        if cat == op_registry.CAT_SPLIT:
            act_ins = [i for i in node.input if i not in consts]
            if len(act_ins) != 1:
                raise UnsupportedModelError(
                    [f"Split (node {name!r}): must have a single activation input"]
                )
            act_in = act_ins[0]
            rank = len(shapes[act_in]) if act_in in shapes else 2
            node_attrs = _attrs(node)
            axis_attr = node_attrs.get("axis", 0)
            raw_axis = int(axis_attr)  # type: ignore[arg-type]
            resolved_axis = raw_axis % rank
            if resolved_axis != 1:
                raise UnsupportedModelError(
                    [
                        f"Split (node {name!r}): axis={raw_axis} resolved to {resolved_axis}; "
                        "only axis 1 is supported"
                    ]
                )

            attrs = _attrs(node)
            if "split" in attrs:
                split_channels = [int(v) for v in attrs["split"]]  # type: ignore[union-attr]
            elif len(node.input) > 1 and node.input[1] in consts:
                split_channels = [int(v) for v in consts[node.input[1]].reshape(-1)]
            else:
                total_c = (
                    shapes[act_in][1]
                    if act_in in shapes and shapes[act_in][1] is not None
                    else None
                )
                if total_c is None:
                    raise UnsupportedModelError(
                        [
                            f"Split (node {name!r}): cannot infer split sizes without shape or "
                            "split attribute"
                        ]
                    )
                n_splits = len(node.output)
                if total_c % n_splits != 0:
                    raise UnsupportedModelError(
                        [
                            f"Split (node {name!r}): cannot split {total_c} channels into "
                            f"{n_splits} equal parts"
                        ]
                    )
                split_channels = [total_c // n_splits] * n_splits

            if rank == 4 and act_in in shapes:
                h, w = shapes[act_in][2], shapes[act_in][3]
                spatial = (h or 1) * (w or 1)
            else:
                spatial = 1
            flat_sizes = [c * spatial for c in split_channels]

            ln = LayerNode(
                name=name,
                layer=Split(sizes=flat_sizes),
                inputs=[produced.get(act_in, act_in)],
                outputs=list(node.output),
            )
            layer_nodes.append(ln)
            for out in node.output:
                produced[out] = out
            continue

        act_ins = [i for i in node.input if i not in consts]
        act_in = act_ins[0]
        in_name = produced.get(act_in, act_in)

        if node.op_type == "Conv":
            ly = _lower_conv(node, consts, shapes, act_in)
        elif node.op_type in ("Gemm", "MatMul"):
            ly = _lower_linear(node, consts, act_in)
        elif cat == op_registry.CAT_ACTIVATION:
            if node.op_type == "Sigmoid" and node.output[0] == graph_output:
                effective_output = in_name
                continue
            ly = Activation(activation_fn(node.op_type, _attrs(node)))
        elif cat == op_registry.CAT_POOL:
            ly = _lower_pool(node, shapes, act_in)
        else:
            raise UnsupportedModelError(
                [f"operator {node.op_type} (node {name!r}) has no lowering rule"]
            )

        ln = LayerNode(name=name, layer=ly, inputs=[in_name], outputs=[node.output[0]])
        layer_nodes.append(ln)
        produced[node.output[0]] = node.output[0]

    return layer_nodes, effective_output


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
    """MaxPool/AveragePool/GlobalAveragePool -> layers.Pool (float avg is the true mean)."""
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


def _fold_into_producer(
    produced: dict[str, str],
    layer_nodes: list[LayerNode],
    topo_nodes: list[onnx.NodeProto],
    act_in: str,
    node_name: str,
    what: str,
) -> tuple[Linear | Conv2d, LayerNode]:
    prod_tensor = produced.get(act_in, act_in)
    producer_ln = None
    for ln in layer_nodes:
        if prod_tensor in ln.outputs:
            producer_ln = ln
            break

    consumers = [n for n in topo_nodes if any(i == act_in for i in n.input)]
    if len(consumers) != 1:
        raise UnsupportedModelError(
            [
                f"{what} (node {node_name!r}): must directly follow a Conv/Gemm/MatMul to fold "
                f"into its weights; its input is consumed by {len(consumers)} nodes. Branching an "
                "un-folded accumulator is not supported."
            ]
        )

    if producer_ln is None or not isinstance(producer_ln.layer, (Linear, Conv2d)):
        producer_desc = type(producer_ln.layer).__name__ if producer_ln else "nothing"
        raise UnsupportedModelError(
            [
                f"{what} (node {node_name!r}): must directly follow a Conv/Gemm/MatMul to "
                f"fold into its weights; its input is produced by {producer_desc}. Re-export "
                f"with the {what} adjacent to its accumulator, or remove it."
            ]
        )
    return producer_ln.layer, producer_ln


def _fold_bias_add(
    node: onnx.NodeProto,
    consts: dict[str, np.ndarray],
    produced: dict[str, str],
    layer_nodes: list[LayerNode],
    topo_nodes: list[onnx.NodeProto],
) -> None:
    """Fold a constant-operand Add into the preceding accumulator layer's bias."""
    name = node.name or "<Add>"
    const_inputs = [i for i in node.input if i in consts]
    act_inputs = [i for i in node.input if i not in consts]
    if len(const_inputs) != 1 or len(act_inputs) != 1:
        raise UnsupportedModelError(
            [f"Add (node {name!r}): expected exactly one constant operand to fold as a bias"]
        )
    act_in = act_inputs[0]
    addend = np.asarray(consts[const_inputs[0]], dtype=np.float64).reshape(-1)

    producer_layer, _ = _fold_into_producer(produced, layer_nodes, topo_nodes, act_in, name, "Add")
    n_out = producer_layer.weight.shape[0]
    if addend.shape[0] != n_out:
        raise UnsupportedModelError(
            [
                f"Add (node {name!r}): constant operand length {addend.shape[0]} != preceding "
                f"layer output size {n_out}; cannot fold as a bias"
            ]
        )
    producer_layer.bias = (
        addend
        if producer_layer.bias is None
        else np.asarray(producer_layer.bias, dtype=np.float64) + addend
    )
    produced[node.output[0]] = produced.get(act_in, act_in)


def _fold_batchnorm(
    node: onnx.NodeProto,
    consts: dict[str, np.ndarray],
    produced: dict[str, str],
    layer_nodes: list[LayerNode],
    topo_nodes: list[onnx.NodeProto],
) -> None:
    """Fold inference-time BatchNorm into the preceding accumulator layer."""
    name = node.name or "<BatchNormalization>"
    if len(node.output) != 1:
        raise UnsupportedModelError(
            [
                f"BatchNormalization (node {name!r}): has {len(node.output)} outputs; "
                "only inference-mode BN (1 output) is supported"
            ]
        )
    if len(node.input) < 5 or any(node.input[i] not in consts for i in range(1, 5)):
        raise UnsupportedModelError(
            [
                f"BatchNormalization (node {name!r}): scale/B/mean/var must be constant "
                "initializers (a runtime-computed BN cannot be folded)"
            ]
        )

    act_in = node.input[0]
    producer_layer, _ = _fold_into_producer(
        produced, layer_nodes, topo_nodes, act_in, name, "BatchNormalization"
    )

    scale = np.asarray(consts[node.input[1]], dtype=np.float64)
    beta = np.asarray(consts[node.input[2]], dtype=np.float64)
    mean = np.asarray(consts[node.input[3]], dtype=np.float64)
    var = np.asarray(consts[node.input[4]], dtype=np.float64)
    epsilon = float(_attrs(node).get("epsilon", 1e-5))

    folded_w, folded_b = fold_batchnorm(
        producer_layer.weight,
        producer_layer.bias,
        scale=scale,
        beta=beta,
        mean=mean,
        var=var,
        epsilon=epsilon,
    )
    producer_layer.weight = folded_w
    producer_layer.bias = folded_b
    produced[node.output[0]] = produced.get(act_in, act_in)


def _check_shape_op_is_noop(
    node: onnx.NodeProto,
    shapes: dict[str, tuple[int | None, ...]],
    act_in: str,
) -> None:
    """Confirm a Reshape/Flatten/Transpose does not reorder the flat channel-major wire.

    Reshape/Flatten only reinterpret a row-major buffer, so they never reorder the flat elements —
    always foldable. A Cast to a float type is an identity on the real-valued wire (the registry
    rejects a non-float ``to``), so it too is always foldable. A Transpose *does* permute; it is a
    no-op only when its perm leaves the row-major flattening unchanged (identity, or permuting only
    size-1 axes). A genuinely reordering Transpose is rejected loudly (baking it into the next
    weight is Phase-8 work).
    """
    if node.op_type in ("Reshape", "Flatten", "Cast"):
        return
    # Transpose: fold it away iff its permutation leaves the row-major flat order unchanged.
    name = node.name or "<Transpose>"
    shape = shapes.get(act_in)
    perm_attr = _attrs(node).get("perm")
    perm = list(perm_attr) if perm_attr is not None else None

    # Resolve the rank so we can materialize ONNX's *default* perm — an absent `perm` means
    # "reverse all axes" (NOT identity), a genuine reorder for rank >= 2. (The prior code treated a
    # missing perm as safe when the shape had unresolved dims, silently folding a real reversal.)
    if shape is not None:
        rank = len(shape)
    elif perm is not None:
        rank = len(perm)
    else:
        # Neither a shape nor an explicit perm: we cannot even determine the rank to reason about
        # the reversal default, so we cannot prove it is a no-op — fail loudly (`AGENTS.md` §1.4).
        raise UnsupportedModelError(
            [
                f"Transpose (node {name!r}): no perm and no resolved input shape; cannot prove it "
                "preserves flat order"
            ]
        )
    if perm is None:
        perm = list(reversed(range(rank)))  # ONNX default

    # A transpose preserves the row-major flattening iff the axes that actually iterate — those
    # *not known* to be size 1 — keep their ascending relative order (a size-1 axis contributes no
    # stride, so moving it never reorders elements). Decided analytically in O(rank): no need to
    # materialize a prod(dims)-sized index array. Unknown/dynamic dims (e.g. a symbolic batch) are
    # treated as order-constraining, so a genuinely reordering transpose is never silently folded.
    if shape is not None:
        constraining = {ax for ax, d in enumerate(shape) if d != 1}  # None != 1 -> constraining
    else:
        constraining = set(range(rank))  # nothing is known to be size 1
    ordered_axes = [ax for ax in perm if ax in constraining]
    if ordered_axes != sorted(ordered_axes):
        raise UnsupportedModelError(
            [
                f"Transpose (node {name!r}): perm={perm} reorders the flat channel-major vector "
                "and cannot be folded away (bake it into the following weight, or Phase 8)"
            ]
        )
