"""Float layer builders — the Layer-3 vocabulary a use case assembles into a model.

A model is a list of these float layers (:class:`penumbra.model.Model`); the quantization
service turns them into the int IR graph the runtime walks. Each layer holds **float** weights
and knows two things:

1. :meth:`Layer.forward` — a plain-float forward pass over a batch, used during calibration to
   observe activation ranges (the service never asks the user for scales — ``PROJECT.md`` §12).
2. :meth:`Layer.quantize` — given the (already-chosen) input scale and the quantization config,
   produce the IR :class:`~penumbra.ir.Node`\\(s) for this layer and return the output tensor's
   scale, so the next layer can quantize against it.

These layers contain **no cryptography** (``PROJECT.md`` §4): they only build a graph of
narrow-waist ops. The naming is chosen to not collide with the int ``*Spec`` IR payloads — a
``layers.Conv2d`` is the float front end, an ``ir.Conv2dSpec`` is its quantized IR form.

Quantization conventions (mirrors :mod:`penumbra.quantization.ptq`):
* Weights are symmetric, signed, per-tensor by default (per-channel optional).
* A ``Linear``/``Conv2d`` accumulator lives in units ``input_scale * weight_scale``; its bias is
  quantized into those accumulator units. The accumulator is *wide* — the service inserts a
  ``Requant`` after it (when consumed downstream) to narrow it back, choosing the rescale from
  the calibrated accumulator range vs. the chosen activation scale.
* An ``Activation`` consumes a single narrow post-Requant block, so its LUT is generated in that
  block's integer domain (:func:`penumbra.quantization.lut.make_activation_lut`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from penumbra.ir import Conv2dSpec, LinearSpec, Node, PoolSpec
from penumbra.quantization.ptq import quantize_conv, quantize_linear


@dataclass
class QuantConfig:
    """Knobs for one quantization pass (the only quantization choices a user makes).

    ``n_bits`` is the working integer width for weights/inputs (kept small — ``PROJECT.md`` §9);
    ``act_bits`` is the post-Requant activation width (a single radix block, ``<= MESSAGE_BITS``);
    ``per_channel`` selects per-output-channel weight scales (better accuracy — ROADMAP P5);
    ``max_mult_bits`` caps the Requant fixed-point multiplier so its internal peak stays in budget.
    """

    n_bits: int = 4
    act_bits: int = 2
    per_channel: bool = False
    max_mult_bits: int = 5


@dataclass
class LayerContext:
    """Threaded state while quantizing a model layer by layer.

    Carries the running tensor name + its quantization scale (so each layer quantizes its bias
    against the right accumulator units and the next layer knows its input scale), a monotonic
    counter for unique node names, and the shared :class:`QuantConfig`.
    """

    tensor: str  # current tensor name flowing between layers
    scale: float  # quantization scale of the current tensor (float = scale * int)
    config: QuantConfig
    index: int = 0  # layer index, for unique node names
    zero_point: int = 0  # incoming tensor's integer offset (for affine activation LUTs)


class Layer:
    """Base class: a float layer that can run a forward pass and emit quantized IR nodes."""

    def forward(self, *inputs: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        """Float forward over a batch ``(N, ...)`` -> ``(N, ...)`` (for calibration)."""
        raise NotImplementedError

    def forward_multi(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """Float forward returning all output tensors."""
        return [self.forward(*inputs)]

    def quantize(
        self, ctx: LayerContext
    ) -> tuple[list[Node], float, int, list[float] | None]:  # pragma: no cover
        """Emit this layer's IR node(s); return ``(nodes, output_scale, output_len, ch_scales)``.

        ``output_len`` is the flat length of this layer's output tensor (the model uses it only
        for bookkeeping / shape sanity). ``ch_scales`` is the list of per-output-channel
        accumulator scales when this is a per-channel accumulator layer, else ``None`` — the model
        uses it to choose a per-channel Requant rescale (each channel has its own accumulator
        scale). The returned nodes are *natural* (no ``Requant``): the service inserts requants
        between accumulator layers afterwards (:mod:`penumbra.compile`).
        """
        raise NotImplementedError


@dataclass
class Linear(Layer):
    """Dense layer ``y = W x + b`` with float weights ``(n_out, n_in)`` and bias ``(n_out,)``."""

    weight: np.ndarray
    bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x @ np.asarray(self.weight, dtype=np.float64).T
        if self.bias is not None:
            y = y + np.asarray(self.bias, dtype=np.float64)
        return y

    def quantize(self, ctx: LayerContext) -> tuple[list[Node], float, int, list[float] | None]:
        cfg = ctx.config
        w_q, b_q, spec = quantize_linear(
            self.weight,
            self.bias,
            ctx.scale,
            bits=cfg.n_bits,
            per_channel=cfg.per_channel,
            in_zero_point=ctx.zero_point,
        )
        # Per-channel returns one scale per output row; each row's accumulator lives in its own
        # units (in_scale * row_scale). We surface those per-channel accumulator scales so the
        # model can build a per-channel Requant (one rescale per channel). `out_scale` is the
        # single nominal scale threaded to the *next* layer / used for a wide head; the max row
        # scale bounds the accumulator magnitude (`_representative_scale`).
        ch_scales = None
        if cfg.per_channel:
            w_scale = _representative_scale(spec)
            ch_scales = [ctx.scale * s.scale for s in spec]
        else:
            w_scale = spec.scale
        out_scale = ctx.scale * w_scale
        name = f"linear{ctx.index}"
        node = Node(
            name=name,
            inputs=[ctx.tensor],
            outputs=[f"{name}_out"],
            op=LinearSpec(weights=w_q.tolist(), bias=b_q.tolist(), weight_bits=cfg.n_bits),
        )
        return [node], out_scale, w_q.shape[0], ch_scales


@dataclass
class Conv2d(Layer):
    """2-D conv with float kernels ``(out_ch, in_ch, kh, kw)`` over a ``(N, in_ch, H, W)`` input."""

    weight: np.ndarray
    in_h: int
    in_w: int
    in_channels: int
    stride: int = 1
    padding: int = 0
    bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        w = np.asarray(self.weight, dtype=np.float64)
        out_ch, _, kh, kw = w.shape
        out_h = (self.in_h + 2 * self.padding - kh) // self.stride + 1
        out_w = (self.in_w + 2 * self.padding - kw) // self.stride + 1
        n = x.shape[0]
        xr = x.reshape(n, self.in_channels, self.in_h, self.in_w)
        if self.padding:
            xr = np.pad(xr, ((0, 0), (0, 0), (self.padding,) * 2, (self.padding,) * 2))
        out = np.zeros((n, out_ch, out_h, out_w), dtype=np.float64)
        for oc in range(out_ch):
            for oy in range(out_h):
                for ox in range(out_w):
                    patch = xr[
                        :,
                        :,
                        oy * self.stride : oy * self.stride + kh,
                        ox * self.stride : ox * self.stride + kw,
                    ]
                    out[:, oc, oy, ox] = np.einsum("nchw,chw->n", patch, w[oc])
                    if self.bias is not None:
                        out[:, oc, oy, ox] += float(self.bias[oc])
        return out.reshape(n, -1)

    def quantize(self, ctx: LayerContext) -> tuple[list[Node], float, int, list[float] | None]:
        cfg = ctx.config
        w = np.asarray(self.weight, dtype=np.float64)
        out_ch, _, kh, kw = w.shape
        w_q, b_q, spec = quantize_conv(
            w,
            bits=cfg.n_bits,
            in_scale=ctx.scale if (self.bias is not None or ctx.zero_point != 0) else None,
            b_f=self.bias,
            per_channel=cfg.per_channel,
            in_zero_point=ctx.zero_point,
        )
        # Per-channel: one scale per output channel; surface each channel's accumulator scale for
        # a per-channel Requant. `out_scale` (the max row scale) is the nominal downstream scale.
        ch_scales = None
        if cfg.per_channel:
            w_scale = _representative_scale(spec)
            ch_scales = [ctx.scale * s.scale for s in spec]
        else:
            w_scale = spec.scale
        out_scale = ctx.scale * w_scale
        out_h = (self.in_h + 2 * self.padding - kh) // self.stride + 1
        out_w = (self.in_w + 2 * self.padding - kw) // self.stride + 1
        name = f"conv{ctx.index}"
        node = Node(
            name=name,
            inputs=[ctx.tensor],
            outputs=[f"{name}_out"],
            op=Conv2dSpec(
                weights=w_q.tolist(),
                bias=(b_q.tolist() if b_q is not None else [0] * out_ch),
                weight_bits=cfg.n_bits,
                in_h=self.in_h,
                in_w=self.in_w,
                in_channels=self.in_channels,
                kernel_h=kh,
                kernel_w=kw,
                stride=self.stride,
                padding=self.padding,
            ),
        )
        return [node], out_scale, out_ch * out_h * out_w, ch_scales


@dataclass
class Pool(Layer):
    """Average/max pool over a [channels][in_h][in_w] feature map (float avg = true mean;
    IR avg = window sum at scale in_scale/k).
    """

    mode: str
    in_h: int
    in_w: int
    channels: int
    pool_h: int
    pool_w: int
    stride: int
    padding: int = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        xr = x.reshape(n, self.channels, self.in_h, self.in_w)
        if self.padding:
            fill = 0.0 if self.mode == "avg" else -np.inf
            p = self.padding
            xr = np.pad(xr, ((0, 0), (0, 0), (p, p), (p, p)), constant_values=fill)
        out_h = (self.in_h + 2 * self.padding - self.pool_h) // self.stride + 1
        out_w = (self.in_w + 2 * self.padding - self.pool_w) // self.stride + 1
        out = np.zeros((n, self.channels, out_h, out_w), dtype=np.float64)
        for oy in range(out_h):
            for ox in range(out_w):
                window = xr[
                    :,
                    :,
                    oy * self.stride : oy * self.stride + self.pool_h,
                    ox * self.stride : ox * self.stride + self.pool_w,
                ]
                # avg is the true mean (ONNX/PyTorch semantics); the integer op emits the window
                # sum and the 1/k lives in the output scale (see quantize).
                out[:, :, oy, ox] = (
                    window.mean(axis=(2, 3)) if self.mode == "avg" else window.max(axis=(2, 3))
                )
        return out.reshape(n, -1)

    def quantize(self, ctx: LayerContext) -> tuple[list[Node], float, int, list[float] | None]:
        out_h = (self.in_h + 2 * self.padding - self.pool_h) // self.stride + 1
        out_w = (self.in_w + 2 * self.padding - self.pool_w) // self.stride + 1
        name = f"pool{ctx.index}"
        node = Node(
            name=name,
            inputs=[ctx.tensor],
            outputs=[f"{name}_out"],
            op=PoolSpec(
                mode=self.mode,
                in_h=self.in_h,
                in_w=self.in_w,
                channels=self.channels,
                pool_h=self.pool_h,
                pool_w=self.pool_w,
                stride=self.stride,
                padding=self.padding,
            ),
        )
        # avg: the IR op emits the integer window SUM of k = pool_h*pool_w taps, so float
        # mean = (in_scale / k) * sum — the 1/k is carried in the output scale and folds into
        # the next layer's weight/bias quantization (PBS-free). max selects one value:
        # scale-preserving.
        out_scale = ctx.scale / (self.pool_h * self.pool_w) if self.mode == "avg" else ctx.scale
        return [node], out_scale, self.channels * out_h * out_w, None


@dataclass
class Activation(Layer):
    """Single-input activation (ReLU, tanh, GELU, leaky ReLU, hardswish, elu, ...).

    Holds the float function ``fn``. Realized either as a fused ReLU inside ``Requant``,
    or as a standalone affine ``Activation`` LUT whose input and output zero-points are
    handled by the signed ``Requant`` and folded into the downstream consumer's bias.
    """

    fn: Callable[[float], float]

    def forward(self, x: np.ndarray) -> np.ndarray:
        vf = np.vectorize(self.fn, otypes=[np.float64])
        return vf(x)

    def quantize(self, ctx: LayerContext) -> tuple[list[Node], float, int]:
        # The Activation LUT is built by the Model during requant insertion (it needs the
        # post-Requant input scale, which the requant rescale sets). The Model handles this op
        # specially; reaching here means it was not preceded by an accumulator+requant.
        raise NotImplementedError(
            "Activation is materialized by Model.quantize after Requant insertion; it cannot be "
            "quantized standalone"
        )


@dataclass(frozen=True)
class LayerNode:
    """One node of a branching float model: a layer plus the float-graph tensor names it
    reads and writes. A plain ``Model([...])`` list is the degenerate chain case."""

    name: str
    layer: Layer
    inputs: list[str]
    outputs: list[str]


@dataclass
class Add(Layer):
    """Element-wise residual add of two same-length tensors (IR ``AddSpec``)."""

    def forward(self, *inputs: np.ndarray) -> np.ndarray:
        if len(inputs) != 2:
            raise ValueError(f"Add takes exactly 2 inputs, got {len(inputs)}")
        return inputs[0] + inputs[1]

    def forward_multi(self, *xs: np.ndarray) -> list[np.ndarray]:
        return [self.forward(*xs)]

    def quantize(self, ctx: LayerContext):
        raise NotImplementedError("Add is materialized by Model.quantize directly")


@dataclass
class Concat(Layer):
    """Channel-axis concatenation of N tensors on the flat wire (IR ``ConcatSpec``)."""

    def forward(self, *inputs: np.ndarray) -> np.ndarray:
        if len(inputs) < 2:
            raise ValueError(f"Concat takes at least 2 inputs, got {len(inputs)}")
        return np.concatenate(inputs, axis=1)

    def forward_multi(self, *xs: np.ndarray) -> list[np.ndarray]:
        return [self.forward(*xs)]

    def quantize(self, ctx: LayerContext):
        raise NotImplementedError("Concat is materialized by Model.quantize directly")


@dataclass
class Split(Layer):
    """Contiguous segmentation into ``sizes`` flat parts (IR ``SplitSpec``)."""

    sizes: list[int]

    def __post_init__(self) -> None:
        if len(self.sizes) < 2:
            raise ValueError(f"Split needs at least 2 output segments; got {len(self.sizes)}")
        if any(n <= 0 for n in self.sizes):
            raise ValueError(f"Split segment sizes must be positive; got {self.sizes!r}")

    def forward(self, *inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Split produces multiple outputs; use forward_multi")

    def forward_multi(self, *xs: np.ndarray) -> list[np.ndarray]:
        if len(xs) != 1:
            raise ValueError(f"Split takes exactly 1 input, got {len(xs)}")
        x = xs[0]
        split_points = list(np.cumsum(self.sizes)[:-1])
        return list(np.split(x, split_points, axis=1))

    def quantize(self, ctx: LayerContext):
        raise NotImplementedError("Split is materialized by Model.quantize directly")


def _representative_scale(specs: list) -> float:
    """A single nominal scale for a per-channel-quantized layer (the max row scale).

    Per-channel quantizes each output row with its own scale, but code that needs *one* number
    for the accumulator's units — the next layer's input scale, or a *wide* head/terminal
    accumulator left un-requantized — uses this nominal scale. The largest row scale is the
    conservative choice: it bounds the accumulator's float magnitude.

    Note: as of 0.6.0 this no longer feeds the fused Requant rescale — a per-channel accumulator
    followed by a ReLU gets a **per-channel** Requant built from each row's own accumulator scale
    (:class:`~penumbra.ir.RequantSpec` per-channel overlay), not a single rescale at this max.
    """
    return max(s.scale for s in specs)
