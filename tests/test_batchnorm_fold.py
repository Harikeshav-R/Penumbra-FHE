"""Tests for BatchNorm folding into preceding Conv2d/Linear weights and bias."""

from __future__ import annotations

import numpy as np
import pytest

from penumbra.quantization.batchnorm import fold_batchnorm


def test_fold_batchnorm_linear():
    """Folding BN into Linear weights and bias equals BN(Linear(x)) within 1e-9."""
    rng = np.random.default_rng(42)
    n_out = 8
    n_in = 16
    batch = 5

    W = rng.normal(size=(n_out, n_in))
    b = rng.normal(size=n_out)
    scale = rng.uniform(0.5, 2.0, size=n_out)
    beta = rng.normal(size=n_out)
    mean = rng.normal(size=n_out)
    var = rng.uniform(0.1, 3.0, size=n_out)
    eps = 1e-5

    W_fold, b_fold = fold_batchnorm(W, b, scale=scale, beta=beta, mean=mean, var=var, epsilon=eps)

    x = rng.normal(size=(batch, n_in))

    # Reference forward:
    y_lin = x @ W.T + b
    # BN forward: scale * (y - mean) / sqrt(var + eps) + beta
    y_bn = scale * (y_lin - mean) / np.sqrt(var + eps) + beta

    # Folded forward:
    y_folded = x @ W_fold.T + b_fold

    np.testing.assert_allclose(y_folded, y_bn, rtol=1e-9, atol=1e-9)


def test_fold_batchnorm_conv2d():
    """Folding BN into Conv2d weights and bias equals BN(Conv(x)) within 1e-9."""
    rng = np.random.default_rng(123)
    out_ch = 6
    in_ch = 3
    kh, kw = 3, 3

    W = rng.normal(size=(out_ch, in_ch, kh, kw))
    b = rng.normal(size=out_ch)
    scale = rng.uniform(0.5, 2.0, size=out_ch)
    beta = rng.normal(size=out_ch)
    mean = rng.normal(size=out_ch)
    var = rng.uniform(0.1, 3.0, size=out_ch)
    eps = 1e-5

    W_fold, b_fold = fold_batchnorm(W, b, scale=scale, beta=beta, mean=mean, var=var, epsilon=eps)

    # Compare folded weight and bias directly:
    s = scale / np.sqrt(var + eps)
    expected_W = s[:, None, None, None] * W
    expected_b = (b - mean) * s + beta

    np.testing.assert_allclose(W_fold, expected_W, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(b_fold, expected_b, rtol=1e-9, atol=1e-9)


def test_fold_batchnorm_none_bias():
    """fold_batchnorm handles bias=None correctly."""
    W = np.ones((4, 2))
    scale = np.ones(4)
    beta = np.zeros(4)
    mean = np.zeros(4)
    var = np.ones(4)
    eps = 1e-5

    W_fold, b_fold = fold_batchnorm(
        W, None, scale=scale, beta=beta, mean=mean, var=var, epsilon=eps
    )
    assert W_fold.shape == (4, 2)
    assert b_fold.shape == (4,)
    np.testing.assert_allclose(b_fold, np.zeros(4), atol=1e-9)


def test_fold_batchnorm_mismatches_raise():
    """fold_batchnorm rejects length and epsilon mismatches loudly."""
    W = np.ones((4, 2))
    b = np.zeros(4)
    scale = np.ones(4)
    beta = np.zeros(4)
    mean = np.zeros(4)
    var = np.ones(4)

    # Non-positive epsilon:
    with pytest.raises(ValueError, match="epsilon must be positive"):
        _ = fold_batchnorm(W, b, scale=scale, beta=beta, mean=mean, var=var, epsilon=0.0)

    # Scale length mismatch:
    with pytest.raises(ValueError, match="output channels"):
        _ = fold_batchnorm(W, b, scale=np.ones(5), beta=beta, mean=mean, var=var, epsilon=1e-5)

    # Parameter lengths mismatch:
    with pytest.raises(ValueError, match="equal length"):
        _ = fold_batchnorm(W, b, scale=scale, beta=np.ones(3), mean=mean, var=var, epsilon=1e-5)

    # Bias length mismatch:
    with pytest.raises(ValueError, match="does not match output channels"):
        _ = fold_batchnorm(W, np.zeros(3), scale=scale, beta=beta, mean=mean, var=var, epsilon=1e-5)
