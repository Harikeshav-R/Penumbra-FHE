"""Inference-time BatchNorm folding into preceding accumulator weights and bias."""

from __future__ import annotations

import numpy as np


def fold_batchnorm(
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    scale: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold inference-time BatchNorm into the preceding accumulator's weights and bias.

    BN at inference is the per-output-channel affine map
    ``y = scale * (x - mean) / sqrt(var + epsilon) + beta``. Composing it with
    ``x = W a + b`` gives ``W' = s[:, None...] * W`` and ``b' = (b - mean) * s + beta``
    with ``s = scale / sqrt(var + epsilon)`` — so BN costs nothing at runtime and never
    reaches the IR. Works for a 2-D dense ``W`` ``(n_out, n_in)`` and a 4-D conv kernel
    ``(out_ch, in_ch, kh, kw)``; ``s`` is broadcast along the leading (output) axis.
    """
    scale = np.asarray(scale, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    if epsilon <= 0:
        raise ValueError(f"BatchNorm epsilon must be positive, got {epsilon}")

    n_out = int(weight.shape[0])
    if len(scale) != n_out:
        raise ValueError(
            f"BatchNorm scale length {len(scale)} does not match weight output channels {n_out}"
        )
    if not (len(scale) == len(beta) == len(mean) == len(var)):
        raise ValueError(
            f"BatchNorm parameters must have equal length; got scale={len(scale)}, "
            + f"beta={len(beta)}, mean={len(mean)}, var={len(var)}"
        )

    s = scale / np.sqrt(var + epsilon)
    reshape_dims = (n_out,) + (1,) * (weight.ndim - 1)
    folded_weight = s.reshape(reshape_dims) * weight

    b = np.zeros((n_out,), dtype=np.float64) if bias is None else np.asarray(bias, dtype=np.float64)
    if len(b) != n_out:
        raise ValueError(f"Accumulator bias length {len(b)} does not match output channels {n_out}")
    folded_bias = (b - mean) * s + beta
    return folded_weight, folded_bias
