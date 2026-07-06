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

A note on scope: the float ``Activation`` must be a **ReLU** (the fused-requant path); a
non-ReLU activation after a Requant (a standalone ``Activation`` LUT node) is a follow-on. A
terminal ``Linear`` head is left wide — its logits are decrypted and argmaxed on the client
(``PROJECT.md`` §11), so they never need to be LUT-narrow.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from penumbra.bitwidth import (
    MESSAGE_BITS,
    propagate_bit_widths,
    requant_internal_bits,
)
from penumbra.compile import insert_requants
from penumbra.ir import SCHEMA_VERSION, Graph
from penumbra.layers import Activation, Conv2d, Layer, LayerContext, Linear, QuantConfig
from penumbra.quantization.calibration import (
    MinMaxObserver,
    MSEObserver,
    Observer,
    PercentileObserver,
)
from penumbra.quantization.ptq import choose_requant_params
from penumbra.quantization.spec import symmetric_spec
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

    def __init__(self, layers: Sequence[Layer], *, input_bits: int = 4) -> None:
        if not layers:
            raise ValueError("Model needs at least one layer")
        self.layers: list[Layer] = list(layers)
        self.input_bits = int(input_bits)
        self.graph: Graph | None = None
        # Populated by quantize(): the input scale and per-layer scales, for accuracy reporting
        # and for callers that want to dequantize results.
        self.input_scale: float | None = None

    # -- calibration -----------------------------------------------------------------------

    def _calibrate_input(self, x: np.ndarray) -> float:
        """Symmetric input scale from the calibration batch (unsigned: pixel-like inputs)."""
        return symmetric_spec(x, self.input_bits, signed=False).scale

    def _calibrate_accumulators(
        self, x: np.ndarray, observer_cls: type[Observer], act_bits: int
    ) -> dict[int, float]:
        """Observe each accumulator layer's post-ReLU output magnitude over the calibration batch.

        Returns ``{layer_index: clip_magnitude}`` — the clipping magnitude the layer's following
        Requant should map to the top of the activation domain (the calibrated magnitude, not the
        worst-case bit-width — ``penumbra.compile`` docstring). ``observer_cls`` selects the
        strategy: :class:`MinMaxObserver` (the peak, no clipping — reproducible default),
        :class:`PercentileObserver`, or :class:`MSEObserver` (both clip outliers, which helps when
        activations are heavy-tailed). The magnitude is read at ``act_bits`` (MSE's optimal clip
        is bit-width dependent); a signed=False spec matches the non-negative post-ReLU domain.
        """
        peaks: dict[int, float] = {}
        acts = x
        for i, layer in enumerate(self.layers):
            out = layer.forward(acts)
            if isinstance(layer, _ACCUMULATOR_LAYERS):
                obs = observer_cls()
                # Post-ReLU magnitude: the Requant fuses a ReLU, so only non-negative values
                # survive to the activation domain. Observe max(out, 0).
                obs.update(np.maximum(out, 0.0))
                # spec(act_bits) drives MSE's bit-width-dependent clip search and updates the
                # observer's chosen magnitude; magnitude() then returns the clip (== peak for
                # MinMax, so the default path is unchanged and the committed fixtures reproduce).
                obs.spec(act_bits, signed=False)
                peaks[i] = obs.magnitude()
            acts = out
        return peaks

    # -- quantization ----------------------------------------------------------------------

    def quantize(
        self,
        calibration_data: np.ndarray,
        *,
        n_bits: int = 4,
        act_bits: int = MESSAGE_BITS,
        per_channel: bool = False,
        max_mult_bits: int = 5,
        calibration: str = "minmax",
        verify: bool = True,
    ) -> Graph:
        """Quantize the float model to an IR graph using ``calibration_data`` (no manual scales).

        ``calibration_data`` is a float batch ``(N, ...)`` of representative inputs (flattened to
        ``(N, feature_len)`` per the input tensor's layout). ``calibration`` selects the
        activation-range strategy: ``"minmax"`` (the peak, no clipping — the default, reproducible
        and safe), ``"percentile"`` (clip the extreme tail — outlier-robust), or ``"mse"`` (the
        clip minimizing round-trip quantization MSE at ``act_bits``). Percentile/MSE can help
        accuracy when activations are heavy-tailed (``PROJECT.md`` §8). Returns the IR
        :class:`Graph` and stores it on :attr:`graph`. See the module docstring for the pipeline.
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
        if calibration not in _OBSERVERS:
            raise ValueError(
                f"calibration must be one of {sorted(_OBSERVERS)}; got {calibration!r}"
            )
        observer_cls = _OBSERVERS[calibration]
        cfg = QuantConfig(
            n_bits=n_bits, act_bits=act_bits, per_channel=per_channel, max_mult_bits=max_mult_bits
        )
        x = np.asarray(calibration_data, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]

        self.input_scale = self._calibrate_input(x)
        acc_peaks = self._calibrate_accumulators(x, observer_cls, cfg.act_bits)

        # Walk layers, emitting natural IR nodes (no Requant yet). An accumulator (Conv2d/Linear)
        # immediately followed by a ReLU Activation is **requantized**: the fused-ReLU Requant
        # narrows its wide accumulator to the `act_bits` activation domain. That changes the scale
        # the *downstream* layer reads — it consumes post-Requant activations at `act_scale`, NOT
        # the accumulator's wide `acc_scale`. So we must choose the Requant rescale and switch the
        # threaded scale to `act_scale` **before** quantizing the downstream layer, or the
        # downstream layer's bias is mis-scaled by the requant ratio (a silent accuracy killer).
        ctx = LayerContext(tensor="x", scale=self.input_scale, config=cfg, index=0)
        nodes = []
        shifts: dict[str, int] = {}
        mults: dict[str, int] = {}
        round_biases: dict[str, int] = {}
        act_ceiling = (1 << cfg.act_bits) - 1

        i = 0
        n_layers = len(self.layers)
        while i < n_layers:
            layer = self.layers[i]
            ctx.index = i

            if isinstance(layer, Activation):
                # Reached standalone: an accumulator+ReLU pair is consumed together below (i += 2),
                # so hitting an Activation here means it does not follow an accumulator.
                raise ValueError(
                    f"Activation at layer {i} does not follow an accumulator (Conv2d/Linear); "
                    "a standalone post-Requant Activation LUT is not yet supported by Model"
                )

            layer_nodes, out_scale, _out_len = layer.quantize(ctx)
            nodes.extend(layer_nodes)
            ctx.tensor = layer_nodes[-1].outputs[0]
            ctx.scale = out_scale

            followed_by_activation = (
                isinstance(layer, _ACCUMULATOR_LAYERS)
                and i + 1 < n_layers
                and isinstance(self.layers[i + 1], Activation)
            )
            if followed_by_activation:
                # The fused-requant path realizes the Activation as the Requant's hard max(x, 0)
                # (`runtime/src/ops/requant.rs`), so it is only correct for a ReLU. Verify the
                # activation behaves like a ReLU before fusing — a sigmoid/tanh would otherwise be
                # silently replaced by a ReLU (`AGENTS.md` §1.4). See `_is_relu_like`.
                act = self.layers[i + 1]
                assert isinstance(act, Activation)
                if not _is_relu_like(act.fn):
                    raise ValueError(
                        f"Activation at layer {i + 1} (following the accumulator at layer {i}) is "
                        "not a ReLU. Model only supports fusing a ReLU into the preceding layer's "
                        "Requant (it applies max(x, 0)); a non-ReLU activation would be silently "
                        "computed as a ReLU. Use a ReLU here, or drop the activation."
                    )
                # A ReLU on the *terminal* accumulator cannot be fused: the head is left wide (its
                # accumulator output is a graph output, so `insert_requants` inserts no Requant —
                # logits are decrypted and argmaxed client-side, `PROJECT.md` §11). Fusing here
                # would need a terminal Requant that narrows the logits to act_bits, which is wrong
                # for a classification head. Rather than silently drop the ReLU, fail loudly
                # (`AGENTS.md` §1.4): the ReLU has nowhere to go.
                if i + 2 >= n_layers:
                    raise ValueError(
                        f"the terminal ReLU at layer {i + 1} cannot be fused: it follows the final "
                        f"accumulator (layer {i}), whose output is the model's wide logit head "
                        "(left un-narrowed for client-side argmax, `PROJECT.md` §11). A trailing "
                        "ReLU has no Requant to fuse into — drop it (argmax is unaffected by a "
                        "monotonic ReLU on the logits), or add a layer after it."
                    )
                # This accumulator will be requantized (fused ReLU). Choose the rescale that maps
                # the calibrated post-ReLU peak to the top of the act_bits domain, and thread the
                # post-Requant activation scale to the downstream layer.
                acc_scale = out_scale
                peak = acc_peaks.get(i, 0.0)
                act_scale = (peak / act_ceiling) if peak > 0 else acc_scale
                mult, shift, round_bias = choose_requant_params(
                    acc_scale, act_scale, out_bits=cfg.act_bits, max_mult_bits=cfg.max_mult_bits
                )
                acc_name = layer_nodes[-1].name
                shifts[acc_name] = shift
                mults[acc_name] = mult
                round_biases[acc_name] = round_bias
                ctx.scale = act_scale  # downstream reads post-Requant activations
                i += 2  # consume the fused ReLU Activation with its accumulator
                continue
            i += 1

        outputs = [nodes[-1].outputs[0]]

        # Build the natural graph at a generous radix to learn widths, insert requants, then
        # search the minimal num_blocks that fits every tensor AND every Requant internal peak.
        probe = Graph(
            schema_version=SCHEMA_VERSION,
            num_blocks=64,
            input_bits=self.input_bits,
            inputs=["x"],
            outputs=outputs,
            nodes=nodes,
        )
        probed = insert_requants(
            probe, shifts=shifts, mults=mults, round_biases=round_biases, out_bits=cfg.act_bits
        )
        num_blocks = self._minimal_num_blocks(probed)

        graph = insert_requants(
            Graph(
                schema_version=SCHEMA_VERSION,
                num_blocks=num_blocks,
                input_bits=self.input_bits,
                inputs=["x"],
                outputs=outputs,
                nodes=nodes,
            ),
            shifts=shifts,
            mults=mults,
            round_biases=round_biases,
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
        widths = propagate_bit_widths(graph)
        peak_bits = max(widths.values())
        for node in graph.nodes:
            op = node.op
            if hasattr(op, "mult"):  # RequantSpec
                in_bits = widths[node.inputs[0]]
                peak_bits = max(peak_bits, requant_internal_bits(in_bits, op.mult, op.round_bias))
        return max(2, (peak_bits + MESSAGE_BITS - 1) // MESSAGE_BITS)

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
            evaluate_graph_int(graph, {"x": xq})  # raises on any inconsistency

    # -- export ----------------------------------------------------------------------------

    def export(self, path: str) -> None:
        """Serialize the quantized IR graph to ``path`` (JSON). Requires :meth:`quantize` first."""
        if self.graph is None:
            raise RuntimeError("call quantize() before export()")
        with open(path, "w") as f:
            f.write(self.graph.to_json())
