"""The user-facing model: assemble float layers, ``quantize`` to IR, ``export`` for the runtime.

This is the entry point ``PROJECT.md`` §7/§12 sketch::

    model = fhe.Model([
        fhe.Conv2d(weights=w1, in_h=6, in_w=6, in_channels=1),
        fhe.Activation(relu),
        fhe.Pool("avg", ...),
        fhe.Linear(weights=w2, bias=b2),
    ])
    model.quantize(calibration_data, n_bits=4)
    model.export("model.fhe")

``quantize`` turns the float layers into the int IR graph the runtime walks, with **no manual
scale math** (quantization is a library service — ``PROJECT.md`` §8, §12). It:

1. **calibrates** — runs ``calibration_data`` through the float layers, observing each
   accumulator's output range so the Requant rescale targets the *typical* magnitude, not the
   worst case;
2. **quantizes** each layer's weights/bias (:mod:`penumbra.quantization.ptq`) into the int IR;
3. **fuses activations into Requant** — a ``Conv2d``/``Linear`` followed by a ReLU ``Activation``
   becomes ``accumulator → Requant(ReLU + rescale)``: the Requant is a *fused ReLU+rescale*
   (``runtime/src/ops/requant.rs``), so the ReLU costs no extra op. The rescale ``(mult, shift,
   round_bias)`` is chosen by :func:`penumbra.quantization.ptq.choose_requant_params` from the
   calibrated accumulator scale vs. the activation scale;
4. **inserts Requants + sizes the radix** — runs :func:`penumbra.compile.insert_requants` with
   those calibrated params, then searches the **minimal ``num_blocks``** that fits every tensor
   *and* every Requant's internal multiply peak, and runs the budget check;
5. **self-verifies** — runs the quantized-integer reference (:mod:`penumbra.reference`, the
   golden oracle) on the calibration samples to confirm the graph it just built actually
   evaluates, catching a scale/wiring bug *inside* ``quantize`` rather than three test files
   later (``AGENTS.md`` §1.1, §1.4).

Activations: a ReLU is fused into the preceding ``Requant``. A non-ReLU activation (tanh,
GELU, leaky ReLU, hardswish, elu, hard sigmoid, sigmoid) is materialized as a signed
``Requant`` + standalone affine ``Activation`` LUT node, folding its output zero-point into the
downstream ``Conv2d``/``Linear``'s bias. A terminal ``Linear`` head is left wide — its logits are
decrypted and argmaxed on the client (``PROJECT.md`` §11), so they never need to be LUT-narrow.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from penumbra.bitwidth import (
    MESSAGE_BITS,
    minimal_num_blocks,
)
from penumbra.client import CryptoProfile, KeySet, run_encrypted
from penumbra.compile import RequantChannelParams, insert_requants
from penumbra.ir import (
    SCHEMA_VERSION,
    ActivationSpec,
    AddSpec,
    ArgmaxSpec,
    ConcatSpec,
    Graph,
    Node,
    SplitSpec,
)
from penumbra.layers import (
    Activation,
    Add,
    Concat,
    Conv2d,
    Layer,
    LayerContext,
    LayerNode,
    Linear,
    QuantConfig,
    Split,
    topological_layer_order,
)
from penumbra.quantization.calibration import (
    MinMaxObserver,
    MSEObserver,
    Observer,
    PercentileObserver,
)
from penumbra.quantization.lut import (
    affine_activation_codomain,
    lut_output_bits,
    make_affine_activation_lut,
)
from penumbra.quantization.ptq import choose_requant_params
from penumbra.quantization.spec import QuantSpec, symmetric_spec
from penumbra.reference import evaluate_graph_int

# Accumulator layer types whose output is rescaled by a (possibly ReLU-fused) Requant.
_ACCUMULATOR_LAYERS = (Conv2d, Linear)


def _is_relu_like(fn: object) -> bool:
    """True if ``fn`` behaves like a ReLU (``max(x, 0)``) on a sampled probe.

    The fused-requant path realizes an accumulator's following ``Activation`` as the Requant's
    hard ``max(x, 0)`` (``runtime/src/ops/requant.rs``), so fusing a *non*-ReLU activation
    (sigmoid, tanh, ...) would silently compute the wrong function — the exported graph diverges
    from the float model even though the integer oracle (also ReLU) still agrees, hiding the bug.
    ``Activation.fn`` is an opaque ``Callable`` (production ReLUs are lambdas, so a name check is
    useless), so we verify it *behaviorally*: negatives clamp to 0 and non-negatives pass through
    unchanged. Sampling a handful of points is enough to reject the common non-ReLU activations
    while accepting every ReLU form the examples use.
    """
    probes = [-100.0, -3.5, -1.0, -1e-6, 0.0, 1e-6, 1.0, 2.5, 7.0, 100.0]
    try:
        for x in probes:
            if abs(float(fn(x)) - max(x, 0.0)) > 1e-9:  # type: ignore[operator]
                return False
    except (TypeError, ValueError):
        return False
    return True


# Named calibration strategies for the post-ReLU activation range an accumulator's Requant
# targets. MinMax (no clipping) is the reproducible default; percentile/MSE clip outliers, which
# can materially help accuracy when activations are heavy-tailed (``PROJECT.md`` §8).
_OBSERVERS: dict[str, type[Observer]] = {
    "minmax": MinMaxObserver,
    "percentile": PercentileObserver,
    "mse": MSEObserver,
}


class Model:
    """An ordered list of float :class:`~penumbra.layers.Layer`\\s, quantizable to an IR graph.

    Construct with the layers in evaluation order. ``quantize`` produces the int IR graph
    (available as :attr:`graph` afterwards); ``export`` serializes it for the runtime.
    """

    def __init__(
        self,
        layers: Sequence[Layer] | Sequence[LayerNode],
        *,
        input_bits: int = 4,
        input_name: str = "x",
        output_name: str | None = None,
    ) -> None:
        if not layers:
            raise ValueError("Model needs at least one layer")
        self.input_bits = int(input_bits)
        self.input_name = input_name
        self.graph: Graph | None = None
        # Populated by quantize(): the input scale and per-layer scales, for accuracy reporting
        # and for callers that want to dequantize results.
        self.input_scale: float | None = None
        # Populated by quantize(): per-accumulator-layer weight bit-widths in evaluation order.
        self.weight_bits: list[int] = []

        if isinstance(layers[0], LayerNode):
            node_list: list[LayerNode] = list(layers)  # type: ignore[arg-type]
            order = topological_layer_order(node_list, input_name)
            self.nodes: list[LayerNode] = [node_list[i] for i in order]
            self.layers: list[Layer] = [n.layer for n in self.nodes]
            self.output_name = output_name if output_name is not None else self.nodes[-1].outputs[0]
        else:
            self.layers = list(layers)  # type: ignore[arg-type]
            self.nodes = []
            for i, layer in enumerate(self.layers):
                inp = input_name if i == 0 else f"t{i - 1}"
                out = f"t{i}"
                self.nodes.append(LayerNode(name=f"l{i}", layer=layer, inputs=[inp], outputs=[out]))
            self.output_name = output_name if output_name is not None else self.nodes[-1].outputs[0]

    # -- calibration -----------------------------------------------------------------------

    def _calibrate_input(self, x: np.ndarray) -> float:
        """Symmetric input scale from the calibration batch (unsigned: pixel-like inputs)."""
        return symmetric_spec(x, self.input_bits, signed=False).scale

    def _calibrate_graph(
        self, x: np.ndarray, observer_cls: type[Observer], act_bits: int
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, tuple[float, float]]]:
        """Observe each accumulator node's output range over the calibration batch.

        Returns ``(acts, peaks, ranges)``:
        - ``acts``: ``{tensor_name: activation_array}``
        - ``peaks``: ``{node_name: clip_magnitude}`` for the fused-ReLU path.
        - ``ranges``: ``{node_name: (min, max)}`` of raw pre-activation outputs for the
          signed-activation path.
        """
        acts: dict[str, np.ndarray] = {self.input_name: x}
        peaks: dict[str, float] = {}
        ranges: dict[str, tuple[float, float]] = {}

        for node in self.nodes:
            ins = [acts[name] for name in node.inputs]
            if isinstance(node.layer, (Add, Concat, Split)):
                outs = node.layer.forward_multi(*ins)
            else:
                outs = [node.layer.forward(ins[0])]

            for out_name, out in zip(node.outputs, outs, strict=True):
                acts[out_name] = out

            if isinstance(node.layer, _ACCUMULATOR_LAYERS):
                out = outs[0]
                ranges[node.name] = (float(np.min(out)), float(np.max(out)))
                obs = observer_cls()
                obs.update(np.maximum(out, 0.0))
                obs.spec(act_bits, signed=False)
                peaks[node.name] = obs.magnitude()

        return acts, peaks, ranges

    def _merge_scale_classes(self) -> dict[str, str]:
        """Map each float-graph tensor to its scale-class representative.

        ``Add``/``Concat`` union every operand with their output; ``Split`` unions its input
        with every output. All members of a class must share one quantization scale, because
        integer addition/concatenation is only meaningful in common units.
        """
        parent: dict[str, str] = {}

        def find(t: str) -> str:
            root = parent.setdefault(t, t)
            if root != t:
                parent[t] = find(root)
            return parent[t]

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for node in self.nodes:
            if isinstance(node.layer, (Add, Concat)):
                for inp in node.inputs:
                    union(inp, node.outputs[0])
            elif isinstance(node.layer, Split):
                for out in node.outputs:
                    union(node.inputs[0], out)

        return {t: find(t) for t in parent}

    # -- quantization ----------------------------------------------------------------------

    def quantize(
        self,
        calibration_data: np.ndarray,
        *,
        n_bits: int | Sequence[int] = 4,
        act_bits: int = MESSAGE_BITS,
        per_channel: bool = False,
        max_mult_bits: int = 5,
        calibration: str = "minmax",
        verify: bool = True,
    ) -> Graph:
        """Quantize the float model to an IR graph using ``calibration_data`` (no manual scales).

        ``calibration_data`` is a float batch ``(N, ...)`` of representative inputs (flattened to
        ``(N, feature_len)`` per the input tensor's layout). ``n_bits`` accepts either a single int
        for all accumulator layers or one entry per accumulator layer (Conv2d/Linear) in evaluation
        order; per-layer widths are the Phase-10 bit-width minimization lever. ``calibration``
        selects the activation-range strategy: ``"minmax"`` (the peak, no clipping — the default,
        reproducible and safe), ``"percentile"`` (clip the extreme tail — outlier-robust), or
        ``"mse"`` (the clip minimizing round-trip quantization MSE at ``act_bits``).
        Percentile/MSE can help accuracy when activations are heavy-tailed (``PROJECT.md`` §8).
        Returns the IR :class:`Graph` and stores it on :attr:`graph`. See the module docstring for
        the pipeline.
        """
        # A Requant output (post-activation value) must fit a SINGLE radix block, so act_bits
        # cannot exceed MESSAGE_BITS — the Rust runtime rejects a wider Requant at load, and the
        # integer oracle's single-block clamp would otherwise disagree with FHE. Fail loudly here
        # (`AGENTS.md` §1.4) rather than emit a graph that silently violates the golden invariant.
        if not 1 <= act_bits <= MESSAGE_BITS:
            raise ValueError(
                f"act_bits must be in [1, MESSAGE_BITS={MESSAGE_BITS}]; got {act_bits}. A "
                "post-Requant activation must fit one shortint block — wider activations are not "
                "representable (raise n_bits for weights/inputs instead, which is independent)."
            )
        x = np.asarray(calibration_data, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(
                f"calibration_data must be a 2D float batch (N, feature_len); got shape {x.shape}"
            )
        if len(x) == 0:
            raise ValueError("calibration_data is empty")

        if calibration not in _OBSERVERS:
            raise ValueError(
                f"calibration must be one of {sorted(_OBSERVERS)}; got {calibration!r}"
            )
        obs_cls = _OBSERVERS[calibration]
        acc_nodes = [n for n in self.nodes if isinstance(n.layer, _ACCUMULATOR_LAYERS)]
        n_accumulators = len(acc_nodes)

        if isinstance(n_bits, int):
            layer_bits = [int(n_bits)] * n_accumulators
        else:
            layer_bits = [int(b) for b in n_bits]
            if len(layer_bits) != n_accumulators:
                raise ValueError(
                    f"n_bits has {len(layer_bits)} entries but the model has {n_accumulators} "
                    f"accumulator layer(s) (Conv2d/Linear); provide either a single int or one "
                    "entry per accumulator layer in topological order"
                )
        self.weight_bits = list(layer_bits)
        if any(b < 1 for b in layer_bits):
            raise ValueError(f"every n_bits entry must be >= 1, got {layer_bits}")

        self.input_scale = self._calibrate_input(x)
        _cal_acts, acc_peaks, acc_ranges = self._calibrate_graph(x, obs_cls, act_bits)

        scale_classes = self._merge_scale_classes()

        cfg = QuantConfig(
            n_bits=4,
            act_bits=act_bits,
            per_channel=per_channel,
            max_mult_bits=max_mult_bits,
        )
        act_ceiling = (1 << cfg.act_bits) - 1

        # Precompute class target scales for merge classes:
        class_scales: dict[str, float] = {}
        for node in acc_nodes:
            consumers = [n for n in self.nodes if node.outputs[0] in n.inputs]
            if (
                len(consumers) == 1
                and isinstance(consumers[0].layer, Activation)
                and _is_relu_like(consumers[0].layer.fn)
            ):
                act_node = consumers[0]
                peak = acc_peaks.get(node.name, 0.0)
                root = scale_classes.get(act_node.outputs[0], act_node.outputs[0])
                natural_scale = (peak / act_ceiling) if peak > 0 else 0.0
                class_scales[root] = max(class_scales.get(root, 0.0), natural_scale)

        ir_tensor: dict[str, str] = {self.input_name: self.input_name}
        scale: dict[str, float] = {self.input_name: self.input_scale}
        zero_point: dict[str, int] = {self.input_name: 0}
        length: dict[str, int] = {self.input_name: x.shape[1]}
        producer_map: dict[str, LayerNode] = {}

        nodes: list[Node] = []
        shifts: dict[str, int] = {}
        mults: dict[str, int] = {}
        round_biases: dict[str, int] = {}
        clamp_los: dict[str, int] = {}
        zero_points: dict[str, int] = {}
        per_channel_params: dict[str, RequantChannelParams] = {}

        handled_nodes: set[str] = set()

        for idx, node in enumerate(self.nodes):
            if node.name in handled_nodes:
                continue

            layer = node.layer

            if isinstance(layer, Activation):
                loc = f"layer {idx}" if self.nodes[0].name == "l0" else f"node {node.name!r}"
                raise ValueError(
                    f"Activation at {loc} does not follow an accumulator "
                    "(Conv2d/Linear); a standalone Activation without a preceding accumulator "
                    "is not supported"
                )

            if isinstance(layer, (Add, Concat)):
                for inp_t in node.inputs:
                    prod = producer_map.get(inp_t)
                    is_valid = False
                    if prod is not None:
                        if isinstance(prod.layer, (Add, Concat, Split)):
                            is_valid = True
                        elif isinstance(prod.layer, Activation) and _is_relu_like(prod.layer.fn):
                            is_valid = True
                    if not is_valid:
                        if inp_t == self.input_name:
                            prod_desc = "the graph input"
                        elif prod is not None:
                            prod_desc = f"{type(prod.layer).__name__} at node {prod.name!r}"
                        else:
                            prod_desc = "an unknown producer"
                        raise ValueError(
                            f"{type(layer).__name__} at node {node.name!r}: operand {inp_t!r} is "
                            f"produced by {prod_desc}, but a merge operand must be a "
                            "ReLU-activated layer output (zero_point 0) so both operands share "
                            "one integer scale. Insert a ReLU after that layer, or move the skip "
                            "connection."
                        )
                    if zero_point.get(inp_t, 0) != 0:
                        raise ValueError(
                            f"{type(layer).__name__} at node {node.name!r}: operand {inp_t!r} has "
                            f"non-zero zero_point {zero_point[inp_t]}; merge operands must have "
                            "zero_point 0"
                        )

                root = scale_classes.get(node.inputs[0], node.inputs[0])
                target_scale = class_scales.get(root, scale[node.inputs[0]])

                if isinstance(layer, Add):
                    if len(node.inputs) != 2:
                        raise ValueError(
                            f"Add at node {node.name!r} expects 2 inputs, got {len(node.inputs)}"
                        )
                    t_a, t_b = node.inputs
                    if length[t_a] != length[t_b]:
                        raise ValueError(
                            f"Add at node {node.name!r}: operands {t_a!r} (length {length[t_a]}) "
                            f"and {t_b!r} (length {length[t_b]}) must have equal length"
                        )
                    out_tensor_name = f"{node.name}_out"
                    ir_node = Node(
                        name=node.name,
                        inputs=[ir_tensor[t_a], ir_tensor[t_b]],
                        outputs=[out_tensor_name],
                        op=AddSpec(),
                    )
                    nodes.append(ir_node)
                    out_t = node.outputs[0]
                    ir_tensor[out_t] = out_tensor_name
                    scale[out_t] = target_scale
                    zero_point[out_t] = 0
                    length[out_t] = length[t_a]
                    producer_map[out_t] = node
                else:
                    sizes = [length[t] for t in node.inputs]
                    out_tensor_name = f"{node.name}_out"
                    ir_node = Node(
                        name=node.name,
                        inputs=[ir_tensor[t] for t in node.inputs],
                        outputs=[out_tensor_name],
                        op=ConcatSpec(sizes=sizes),
                    )
                    nodes.append(ir_node)
                    out_t = node.outputs[0]
                    ir_tensor[out_t] = out_tensor_name
                    scale[out_t] = target_scale
                    zero_point[out_t] = 0
                    length[out_t] = sum(sizes)
                    producer_map[out_t] = node

                continue

            if isinstance(layer, Split):
                in_t = node.inputs[0]
                if sum(layer.sizes) != length[in_t]:
                    raise ValueError(
                        f"Split at node {node.name!r}: sum of sizes {sum(layer.sizes)} does not "
                        f"match input tensor {in_t!r} length {length[in_t]}"
                    )
                out_tensor_names = [f"{node.name}_out{j}" for j in range(len(layer.sizes))]
                ir_node = Node(
                    name=node.name,
                    inputs=[ir_tensor[in_t]],
                    outputs=out_tensor_names,
                    op=SplitSpec(sizes=list(layer.sizes)),
                )
                nodes.append(ir_node)
                for out_t, sz, ir_out in zip(
                    node.outputs, layer.sizes, out_tensor_names, strict=True
                ):
                    ir_tensor[out_t] = ir_out
                    scale[out_t] = scale[in_t]
                    zero_point[out_t] = zero_point[in_t]
                    length[out_t] = sz
                    producer_map[out_t] = node
                continue

            # Pool or Accumulator
            in_t = node.inputs[0]
            ctx = LayerContext(
                tensor=ir_tensor[in_t],
                scale=scale[in_t],
                config=cfg,
                index=idx,
                zero_point=zero_point[in_t],
            )

            if isinstance(layer, _ACCUMULATOR_LAYERS):
                ctx.config = replace(cfg, n_bits=layer_bits[acc_nodes.index(node)])

            layer_nodes, out_scale, out_len, ch_scales = layer.quantize(ctx)
            nodes.extend(layer_nodes)
            acc_out_name = layer_nodes[-1].outputs[0]
            out_t = node.outputs[0]
            ir_tensor[out_t] = acc_out_name
            scale[out_t] = out_scale
            zero_point[out_t] = 0
            length[out_t] = out_len
            producer_map[out_t] = node

            consumers = [n for n in self.nodes if node.outputs[0] in n.inputs]
            has_act = any(isinstance(c.layer, Activation) for c in consumers)
            if has_act:
                if len(consumers) != 1:
                    act_c = [c.name for c in consumers if isinstance(c.layer, Activation)][0]
                    other_c = [c.name for c in consumers if not isinstance(c.layer, Activation)]
                    raise ValueError(
                        f"accumulator {node.name!r} feeds Activation {act_c!r} but also feeds "
                        f"other consumer(s): {', '.join(other_c)}; branching an un-activated "
                        "accumulator is not supported"
                    )
                act_node = consumers[0]
                act = act_node.layer
                assert isinstance(act, Activation)

                if _is_relu_like(act.fn):
                    act_loc = (
                        f"layer {self.nodes.index(act_node)}"
                        if self.nodes[0].name == "l0"
                        else f"node {act_node.name!r}"
                    )
                    node_loc = (
                        f"layer {self.nodes.index(node)}"
                        if self.nodes[0].name == "l0"
                        else f"node {node.name!r}"
                    )
                    if act_node.outputs[0] == self.output_name:
                        raise ValueError(
                            f"the terminal ReLU at {act_loc} cannot be fused: it "
                            f"follows the final accumulator ({node_loc}), whose output "
                            "is the model's wide logit head (left un-narrowed for "
                            "client-side argmax, `PROJECT.md` §11). A trailing ReLU has no "
                            "Requant to fuse into — drop it (argmax is unaffected by a "
                            "monotonic ReLU on the logits), or add a layer after it."
                        )
                    peak = acc_peaks.get(node.name, 0.0)
                    root = scale_classes.get(act_node.outputs[0], act_node.outputs[0])
                    natural_scale = (peak / act_ceiling) if peak > 0 else out_scale
                    act_scale = class_scales.get(root, natural_scale)
                    if act_scale == 0.0:
                        act_scale = natural_scale
                    acc_name = layer_nodes[-1].name

                    if ch_scales is not None:
                        ch_mults, ch_shifts, ch_rbs = [], [], []
                        for acc_scale_i in ch_scales:
                            m_i, s_i, rb_i = choose_requant_params(
                                acc_scale_i,
                                act_scale,
                                out_bits=cfg.act_bits,
                                max_mult_bits=cfg.max_mult_bits,
                            )
                            ch_mults.append(m_i)
                            ch_shifts.append(s_i)
                            ch_rbs.append(rb_i)
                        per_channel_params[acc_name] = RequantChannelParams(
                            mults=ch_mults, shifts=ch_shifts, round_biases=ch_rbs
                        )
                    else:
                        mult, shift, round_bias = choose_requant_params(
                            out_scale,
                            act_scale,
                            out_bits=cfg.act_bits,
                            max_mult_bits=cfg.max_mult_bits,
                        )
                        shifts[acc_name] = shift
                        mults[acc_name] = mult
                        round_biases[acc_name] = round_bias

                    act_out_t = act_node.outputs[0]
                    scale[act_out_t] = act_scale
                    zero_point[act_out_t] = 0
                    length[act_out_t] = out_len
                    ir_tensor[act_out_t] = acc_out_name
                    producer_map[act_out_t] = act_node
                    handled_nodes.add(act_node.name)
                else:
                    act_loc = (
                        f"layer {self.nodes.index(act_node)}"
                        if self.nodes[0].name == "l0"
                        else f"node {act_node.name!r}"
                    )
                    node_loc = (
                        f"layer {self.nodes.index(node)}"
                        if self.nodes[0].name == "l0"
                        else f"node {node.name!r}"
                    )
                    if act_node.outputs[0] == self.output_name:
                        raise ValueError(
                            f"the terminal activation at {act_loc} cannot be lowered: "
                            f"it follows the final accumulator ({node_loc}), whose output "
                            "is the model's wide logit head (left un-narrowed for client-side "
                            "argmax, `PROJECT.md` §11). A trailing activation has nowhere to go — "
                            "drop it, or add a layer after it."
                        )
                    act_consumers = [n for n in self.nodes if act_node.outputs[0] in n.inputs]
                    if not act_consumers or not all(
                        isinstance(c.layer, _ACCUMULATOR_LAYERS) for c in act_consumers
                    ):
                        next_desc = (
                            type(act_consumers[0].layer).__name__ if act_consumers else "nothing"
                        )
                        raise ValueError(
                            f"the activation at node {act_node.name!r} is consumed by {next_desc}; "
                            "a non-ReLU activation must feed a Conv2d/Linear (its output "
                            "zero-point folds into that layer's bias). Pool/other consumers are "
                            "not supported."
                        )
                    lo, hi = acc_ranges[node.name]
                    lo = min(lo, 0.0)
                    hi = max(hi, 0.0)
                    levels = 1 << cfg.act_bits
                    act_scale = (hi - lo) / (levels - 1) if hi > lo else 1.0
                    acc_name = layer_nodes[-1].name

                    if ch_scales is not None:
                        clamp_lo = min(0, min(math.floor(lo / s) for s in ch_scales))
                        ch_mults, ch_shifts, ch_rbs = [], [], []
                        for acc_scale_i in ch_scales:
                            m_i, s_i, rb_i = choose_requant_params(
                                acc_scale_i,
                                act_scale,
                                out_bits=cfg.act_bits,
                                max_mult_bits=cfg.max_mult_bits,
                                clamp_lo=clamp_lo,
                            )
                            ch_mults.append(m_i)
                            ch_shifts.append(s_i)
                            ch_rbs.append(rb_i)
                        per_channel_params[acc_name] = RequantChannelParams(
                            mults=ch_mults, shifts=ch_shifts, round_biases=ch_rbs
                        )
                        u_min = min(
                            (clamp_lo * m + rb) >> s
                            for m, s, rb in zip(ch_mults, ch_shifts, ch_rbs, strict=True)
                        )
                        zero_point_val = max(0, -u_min)
                        clamp_los[acc_name] = clamp_lo
                        zero_points[acc_name] = zero_point_val
                    else:
                        clamp_lo = min(0, math.floor(lo / out_scale))
                        mult, shift, round_bias = choose_requant_params(
                            out_scale,
                            act_scale,
                            out_bits=cfg.act_bits,
                            max_mult_bits=cfg.max_mult_bits,
                            clamp_lo=clamp_lo,
                        )
                        u_min = (clamp_lo * mult + round_bias) >> shift
                        zero_point_val = max(0, -u_min)
                        shifts[acc_name] = shift
                        mults[acc_name] = mult
                        round_biases[acc_name] = round_bias
                        clamp_los[acc_name] = clamp_lo
                        zero_points[acc_name] = zero_point_val

                    out_act_scale, out_zp = affine_activation_codomain(
                        act.fn,
                        in_scale=act_scale,
                        in_zero_point=zero_point_val,
                        act_bits=cfg.act_bits,
                    )
                    lut = make_affine_activation_lut(
                        act.fn,
                        in_scale=act_scale,
                        in_zero_point=zero_point_val,
                        out_scale=out_act_scale,
                        out_zero_point=out_zp,
                        out_bits=cfg.act_bits,
                    )
                    act_name = f"act{self.nodes.index(act_node)}"
                    act_out_tensor = f"{act_name}_out"
                    nodes.append(
                        Node(
                            name=act_name,
                            inputs=[acc_out_name],
                            outputs=[act_out_tensor],
                            op=ActivationSpec(lut=lut, output_bits=lut_output_bits(lut)),
                        )
                    )
                    act_out_t = act_node.outputs[0]
                    scale[act_out_t] = out_act_scale
                    zero_point[act_out_t] = out_zp
                    length[act_out_t] = out_len
                    ir_tensor[act_out_t] = act_out_tensor
                    producer_map[act_out_t] = act_node
                    handled_nodes.add(act_node.name)

        outputs = [ir_tensor[self.output_name]]

        probe = Graph(
            schema_version=SCHEMA_VERSION,
            num_blocks=64,
            input_bits=self.input_bits,
            inputs=[self.input_name],
            outputs=outputs,
            nodes=nodes,
        )
        probed = insert_requants(
            probe,
            shifts=shifts,
            mults=mults,
            round_biases=round_biases,
            clamp_los=clamp_los,
            zero_points=zero_points,
            per_channel=per_channel_params,
            out_bits=cfg.act_bits,
        )
        num_blocks = self._minimal_num_blocks(probed)

        graph = insert_requants(
            Graph(
                schema_version=SCHEMA_VERSION,
                num_blocks=num_blocks,
                input_bits=self.input_bits,
                inputs=[self.input_name],
                outputs=outputs,
                nodes=nodes,
            ),
            shifts=shifts,
            mults=mults,
            round_biases=round_biases,
            clamp_los=clamp_los,
            zero_points=zero_points,
            per_channel=per_channel_params,
            out_bits=cfg.act_bits,
        )
        if verify:
            self._self_verify(graph, x)

        self.graph = graph
        return graph

    @staticmethod
    def _minimal_num_blocks(graph: Graph) -> int:
        """Smallest ``num_blocks`` whose radix holds every tensor width and Requant internal peak.

        The radix must fit not just each tensor's propagated width but each ``Requant``'s
        transient multiply peak (``max(x,0)*mult + round_bias``) — the internal-peak budget. We
        take the max of both over the graph and round up to whole ``MESSAGE_BITS`` blocks.
        """
        return minimal_num_blocks(graph)

    def _self_verify(self, graph: Graph, x: np.ndarray) -> None:
        """Run the integer oracle on the calibration inputs to confirm the graph evaluates.

        This is the in-``quantize`` guard the quantization service owes (``AGENTS.md`` §1.1): a
        scale or wiring mistake that makes an Activation index out of its LUT domain, or a tensor
        overflow the radix, surfaces here with an actionable message — not as a confusing Rust
        golden violation later. It evaluates the *quantized-integer* graph (the golden oracle)
        over the quantized calibration inputs; the FHE path must then match it bit-for-bit.
        """
        from penumbra.bitwidth import check_bit_width_budget

        # Budget must fit (also re-checks the internal peak); raises naming the layer if not.
        check_bit_width_budget(graph)

        # Quantize a few calibration inputs and run the integer oracle — it raises loudly on an
        # out-of-domain Activation index or a wiring error. A handful of samples is enough to
        # exercise the op chain (this is a smoke check, not an accuracy measurement).
        assert self.input_scale is not None
        in_spec = symmetric_spec(x, self.input_bits, signed=False)
        sample = x[: min(4, len(x))]
        for row in sample:
            xq = in_spec.quantize(row).tolist()
            evaluate_graph_int(graph, {self.input_name: xq})  # raises on any inconsistency

    # -- export ----------------------------------------------------------------------------

    def export(self, path: str) -> None:
        """Serialize the quantized IR graph to ``path`` (JSON). Requires :meth:`quantize` first."""
        if self.graph is None:
            raise RuntimeError("call quantize() before export()")
        with open(path, "w") as f:
            f.write(self.graph.to_json())

    # -- encrypted inference ---------------------------------------------------------------

    def predict_encrypted(
        self,
        x: np.ndarray,
        *,
        return_logits: bool = False,
        keys: KeySet | None = None,
        backend: str = "tfhe",
        profile: CryptoProfile | None = None,
    ):
        """Run the encrypted forward pass on ``x`` and return the prediction(s).

        The one-call round trip (``PROJECT.md`` §12): quantize ``x`` to the graph's integer
        input domain, run encrypted inference in-process via compiled PyO3 bindings
        (:mod:`penumbra.client`), and decode the decrypted outputs client-side (argmax,
        ``PROJECT.md`` §11). Requires :meth:`quantize` first.

        ``x`` is a float array: a single sample ``(feature_len,)`` or a batch ``(N, feature_len)``.
        Returns the predicted class label (an ``int``) for a single sample or a ``list[int]`` for a
        batch. With ``return_logits=True`` it returns ``(labels, logits)`` where ``logits`` are the
        raw decrypted output tensors (a single row / a list of rows to match ``x``) — useful for
        inspection and for the golden cross-check against the
        :func:`penumbra.reference.evaluate_graph_int` oracle.

        ``backend`` selects the cryptographic backend (``"tfhe"`` or ``"ckks"``). Under TFHE,
        evaluation is exact and matches the quantized-cleartext oracle bit-for-bit (``AGENTS.md``
        §1.1). Under CKKS, evaluation is approximate (bounded within the model's declared error
        bound).

        ``profile`` allows overriding the single backend crypto parameter profile
        (:class:`~penumbra.client.CryptoProfile`).

        ``keys`` selects the round-trip path (:func:`penumbra.client.run_encrypted`): omit it for
        the convenient in-process all-in-one execution (ephemeral keys), or pass a
        :class:`~penumbra.client.KeySet` to **reuse** persisted keys and run the faithful
        client/server split (the server evaluates with only the public key). The keys must match
        this model's radix width and backend or the call fails loudly (``AGENTS.md`` §1.4).
        """
        if self.graph is None or self.input_scale is None:
            raise RuntimeError("call quantize() before predict_encrypted()")

        arr = np.asarray(x, dtype=np.float64)
        single = arr.ndim == 1
        batch = arr[None, :] if single else arr

        # Reconstruct the input spec from the stored scale (quantize() keeps only the scalar
        # input_scale, not the QuantSpec) and quantize each row into the integer input domain.
        # The input is unsigned (pixel-like), matching _calibrate_input.
        in_spec = QuantSpec(scale=self.input_scale, bits=self.input_bits, signed=False)
        int_inputs = [in_spec.quantize(row).tolist() for row in batch]

        outputs = run_encrypted(self.graph, int_inputs, keys=keys, backend=backend, profile=profile)
        return self._decode(outputs, single=single, return_logits=return_logits)

    def _decode(self, outputs: list[list[int]], *, single: bool, return_logits: bool):
        """Turn the runtime's raw output rows into class labels (``PROJECT.md`` §11).

        The output shape depends on the graph's terminal op: a 2-class ``Argmax`` head already
        emits the label bit (return it as-is), while a wide multi-logit ``Linear`` head is
        argmaxed client-side (NumPy's first-max tie-break matches the Rust ``max_by`` in the
        golden tests, so client and runtime agree). ``single`` unwraps the batch of one back to a
        scalar to mirror the caller's input shape.
        """
        assert self.graph is not None
        terminal_is_argmax = bool(self.graph.nodes) and isinstance(
            self.graph.nodes[-1].op, ArgmaxSpec
        )
        labels = [int(row[0]) if terminal_is_argmax else int(np.argmax(row)) for row in outputs]

        result_labels = labels[0] if single else labels
        if not return_logits:
            return result_labels
        result_logits = outputs[0] if single else outputs
        return result_labels, result_logits
