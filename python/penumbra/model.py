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
from penumbra.quantization.calibration import MinMaxObserver
from penumbra.quantization.ptq import choose_requant_params
from penumbra.quantization.spec import symmetric_spec
from penumbra.reference import evaluate_graph_int

# Accumulator layer types whose output is rescaled by a (possibly ReLU-fused) Requant.
_ACCUMULATOR_LAYERS = (Conv2d, Linear)


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

    def _calibrate_accumulators(self, x: np.ndarray) -> dict[int, float]:
        """Observe each accumulator layer's float output range over the calibration batch.

        Returns ``{layer_index: post_relu_peak}`` — the largest non-negative accumulator value
        each Conv/Linear produces, which sets how aggressively its following Requant must rescale
        (the calibrated magnitude, not the worst-case bit-width — ``penumbra.compile`` docstring).
        """
        peaks: dict[int, float] = {}
        acts = x
        for i, layer in enumerate(self.layers):
            out = layer.forward(acts)
            if isinstance(layer, _ACCUMULATOR_LAYERS):
                obs = MinMaxObserver()
                # Post-ReLU magnitude: the Requant fuses a ReLU, so only non-negative values
                # survive to the activation domain. Observe max(out, 0).
                obs.update(np.maximum(out, 0.0))
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
        verify: bool = True,
    ) -> Graph:
        """Quantize the float model to an IR graph using ``calibration_data`` (no manual scales).

        ``calibration_data`` is a float batch ``(N, ...)`` of representative inputs (flattened to
        ``(N, feature_len)`` per the input tensor's layout). Returns the IR :class:`Graph` and
        stores it on :attr:`graph`. See the module docstring for the pipeline.
        """
        cfg = QuantConfig(
            n_bits=n_bits, act_bits=act_bits, per_channel=per_channel, max_mult_bits=max_mult_bits
        )
        x = np.asarray(calibration_data, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]

        self.input_scale = self._calibrate_input(x)
        acc_peaks = self._calibrate_accumulators(x)

        # Walk layers, emitting natural IR nodes (no Requant yet). Activations are *fused* into
        # the preceding accumulator's Requant, so we record where a ReLU follows an accumulator
        # and skip emitting it as a node; the Requant params are calibrated below.
        ctx = LayerContext(tensor="x", scale=self.input_scale, config=cfg, index=0)
        nodes = []
        # accumulator node name -> (acc_scale, post_relu_peak) for choosing its Requant rescale.
        requant_targets: dict[str, tuple[float, float]] = {}
        last_acc_node: str | None = None

        for i, layer in enumerate(self.layers):
            ctx.index = i
            if isinstance(layer, Activation):
                # A ReLU activation is realized by the preceding accumulator's fused-ReLU Requant.
                if last_acc_node is None:
                    raise ValueError(
                        f"Activation at layer {i} does not follow an accumulator (Conv2d/Linear); "
                        "a standalone post-Requant Activation LUT is not yet supported by Model"
                    )
                # Mark that the accumulator's Requant must narrow to act_bits (already the
                # default); nothing more to emit — the fused ReLU lives in the Requant.
                continue

            layer_nodes, out_scale, _out_len = layer.quantize(ctx)
            nodes.extend(layer_nodes)
            produced = layer_nodes[-1].outputs[0]
            if isinstance(layer, _ACCUMULATOR_LAYERS):
                last_acc_node = layer_nodes[-1].name
                requant_targets[last_acc_node] = (out_scale, acc_peaks.get(i, 0.0))
            else:
                last_acc_node = None
            ctx.tensor = produced
            ctx.scale = out_scale

        outputs = [nodes[-1].outputs[0]]

        # Choose the Requant rescale for each accumulator that will be requantized. The target
        # activation scale maps the calibrated post-ReLU peak to the top of the act_bits domain:
        #   act_scale = post_relu_peak / (2^act_bits - 1)
        # and the rescale ratio is acc_scale / act_scale. We compute params per accumulator and
        # hand them to insert_requants, which decides *where* a Requant actually goes.
        act_ceiling = (1 << cfg.act_bits) - 1
        shifts: dict[str, int] = {}
        mults: dict[str, int] = {}
        round_biases: dict[str, int] = {}
        for acc_name, (acc_scale, peak) in requant_targets.items():
            act_scale = (peak / act_ceiling) if peak > 0 else acc_scale
            mult, shift, round_bias = choose_requant_params(
                acc_scale, act_scale, out_bits=cfg.act_bits, max_mult_bits=cfg.max_mult_bits
            )
            shifts[acc_name] = shift
            mults[acc_name] = mult
            round_biases[acc_name] = round_bias

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
