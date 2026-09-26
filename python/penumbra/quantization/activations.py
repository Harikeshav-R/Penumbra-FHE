"""Float activation functions and ONNX attribute binding.

Exact float semantics matching ONNX operators for table generation.
This is the single source of truth for activation formulas across Penumbra.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping


def relu(x: float) -> float:
    return max(x, 0.0)


def tanh(x: float) -> float:
    return math.tanh(x)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    return x if x >= 0.0 else alpha * x


def elu(x: float, alpha: float = 1.0) -> float:
    return x if x >= 0.0 else alpha * (math.exp(x) - 1.0)


def hard_sigmoid(x: float, alpha: float = 0.2, beta: float = 0.5) -> float:
    return max(0.0, min(1.0, alpha * x + beta))


def hardswish(x: float) -> float:
    return x * max(0.0, min(1.0, x / 6.0 + 0.5))


def gelu(x: float, approximate: str = "none") -> float:
    if approximate == "tanh":
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def _as_str(v: object) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v) if v is not None else ""


def _as_float(v: object, default: float) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float, str, bytes)):
        return float(v)
    return default


def activation_fn(
    op_type: str, attrs: Mapping[str, object] | None = None
) -> Callable[[float], float]:
    """ONNX op type + attributes -> the single-argument float function its LUT tabulates.

    Raises ValueError for an op type that is not a supported single-input activation.
    """
    if attrs is None:
        attrs = {}
    if op_type == "Relu":
        return relu
    if op_type == "Tanh":
        return tanh
    if op_type == "Sigmoid":
        return sigmoid
    if op_type == "HardSwish":
        return hardswish
    if op_type == "LeakyRelu":
        alpha = _as_float(attrs.get("alpha"), 0.01)
        return functools.partial(leaky_relu, alpha=alpha)
    if op_type == "Elu":
        alpha = _as_float(attrs.get("alpha"), 1.0)
        return functools.partial(elu, alpha=alpha)
    if op_type == "HardSigmoid":
        alpha = _as_float(attrs.get("alpha"), 0.2)
        beta = _as_float(attrs.get("beta"), 0.5)
        return functools.partial(hard_sigmoid, alpha=alpha, beta=beta)
    if op_type == "Gelu":
        approx = _as_str(attrs.get("approximate", "none")).lower()
        if not approx:
            approx = "none"
        if approx not in ("none", "tanh"):
            raise ValueError(f"Gelu approximate attribute must be 'none' or 'tanh', got {approx!r}")
        return functools.partial(gelu, approximate=approx)
    raise ValueError(f"unsupported activation op_type {op_type!r}")
