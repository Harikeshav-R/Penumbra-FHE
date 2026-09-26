"""Tests for affine activation lookup table generation (penumbra.quantization.lut)."""

from __future__ import annotations

import pytest

from penumbra.bitwidth import MESSAGE_BITS
from penumbra.quantization.activations import gelu, hardswish, leaky_relu, relu, tanh
from penumbra.quantization.lut import (
    affine_activation_codomain,
    make_activation_lut,
    make_affine_activation_lut,
    validate_lut,
)
from penumbra.quantization.spec import QuantSpec


def test_affine_activation_lut_tanh():
    """Verify affine LUT for tanh with hand-computed values."""
    # levels = 4; domain codes v in {0, 1, 2, 3}.
    # in_zero_point = 1, in_scale = 2.0.
    # v=0: (0-1)*2.0 = -2.0 -> tanh(-2) = -0.9640
    # v=1: (1-1)*2.0 = 0.0  -> tanh(0) = 0.0
    # v=2: (2-1)*2.0 = 2.0  -> tanh(2) = 0.9640
    # v=3: (3-1)*2.0 = 4.0  -> tanh(4) = 0.9993
    out_scale, out_zp = affine_activation_codomain(tanh, in_scale=2.0, in_zero_point=1, act_bits=2)
    lut = make_affine_activation_lut(
        tanh,
        in_scale=2.0,
        in_zero_point=1,
        out_scale=out_scale,
        out_zero_point=out_zp,
        out_bits=2,
    )
    assert len(lut) == 1 << MESSAGE_BITS
    assert all(0 <= e < (1 << MESSAGE_BITS) for e in lut)
    # v=1 corresponds to x=0.0 -> tanh(0)=0.0 -> should map to code out_zp
    assert lut[1] == out_zp
    # v=0 (negative) should be smaller than out_zp
    assert lut[0] < out_zp
    # v=2, 3 (positive) should be larger than out_zp
    assert lut[2] > out_zp
    assert lut[3] >= lut[2]


def test_affine_activation_lut_leaky_relu():
    """Verify affine LUT for leaky ReLU."""
    out_scale, out_zp = affine_activation_codomain(
        leaky_relu, in_scale=1.0, in_zero_point=2, act_bits=2
    )
    lut = make_affine_activation_lut(
        leaky_relu,
        in_scale=1.0,
        in_zero_point=2,
        out_scale=out_scale,
        out_zero_point=out_zp,
        out_bits=2,
    )
    assert len(lut) == 1 << MESSAGE_BITS
    assert all(0 <= e < (1 << MESSAGE_BITS) for e in lut)
    assert lut[2] == out_zp  # x=0 maps to out_zp


def test_affine_activation_lut_hardswish_and_gelu():
    """Verify affine LUT generation for HardSwish and GELU runs and produces valid tables."""
    for fn in (hardswish, gelu):
        out_scale, out_zp = affine_activation_codomain(
            fn, in_scale=1.5, in_zero_point=1, act_bits=2
        )
        lut = make_affine_activation_lut(
            fn,
            in_scale=1.5,
            in_zero_point=1,
            out_scale=out_scale,
            out_zero_point=out_zp,
            out_bits=2,
        )
        assert len(lut) == 1 << MESSAGE_BITS
        assert all(0 <= e < (1 << MESSAGE_BITS) for e in lut)


def test_affine_lut_reproduces_zero_point_free_lut_for_relu():
    """With in_zero_point=0 and out_zero_point=0, affine LUT reproduces make_activation_lut."""
    spec_in = QuantSpec(scale=0.5, bits=2, signed=False)
    spec_out = QuantSpec(scale=0.5, bits=2, signed=False)
    legacy_lut = make_activation_lut(relu, spec_in, spec_out)
    affine_lut = make_affine_activation_lut(
        relu,
        in_scale=0.5,
        in_zero_point=0,
        out_scale=0.5,
        out_zero_point=0,
        out_bits=2,
    )
    assert legacy_lut == affine_lut


def test_validate_lut_catches_invalid_tables():
    """validate_lut rejects tables of wrong length or containing out-of-range codes."""
    with pytest.raises(ValueError, match="entries"):
        validate_lut([0, 1, 2])
    with pytest.raises(ValueError, match="does not fit one shortint block"):
        validate_lut([0, 1, 2, 4])
    with pytest.raises(ValueError, match="negative"):
        validate_lut([0, -1, 2, 3])


def test_affine_activation_codomain_zero_preserving_for_tanh():
    """For symmetric or near-symmetric tanh inputs, 0 maps to out_zp."""
    out_scale, out_zp = affine_activation_codomain(tanh, in_scale=1.0, in_zero_point=1, act_bits=2)
    assert out_zp >= 0
    assert out_scale > 0
    # x=0 is at v=1; fn(0) = 0.0 -> round(0.0/out_scale) + out_zp = out_zp
    code_zero = round(tanh(0.0) / out_scale) + out_zp
    assert code_zero == out_zp
