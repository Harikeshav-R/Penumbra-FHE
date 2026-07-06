"""Tests for integer-domain LUT generation (``penumbra.quantization.lut``, Phase 5).

These pin the quantization service's LUT generators against the backend's single-block LUT
contract (``runtime/src/ops/activation.rs``, ``runtime/src/ops/requant.rs``): the tables must
cover the full ``2**MESSAGE_BITS`` message space with every entry in ``[0, 2**MESSAGE_BITS)``,
and they must be produced in the quantized-integer domain consistent with the chosen scales —
an off-by-scale here silently wrecks accuracy (``PROJECT.md`` §8). None of these need FHE or
the ML stack (NumPy-only, hermetic), so they run instantly.

The headline pin is that :func:`make_activation_lut` reproduces the hand-built clamp LUT in
``examples/mnist/train_quantize_export.py`` (``relu_lut = [min(v, 2) for v in range(4)]``) and
that :func:`identity_clamp_lut` agrees value-for-value with the inline ``compile._clamp_lut``,
so the canonical generator and the existing callers cannot drift.
"""

from __future__ import annotations

import pytest

from penumbra.bitwidth import MESSAGE_BITS
from penumbra.compile import _clamp_lut
from penumbra.quantization import QuantSpec
from penumbra.quantization.lut import (
    identity_clamp_lut,
    lut_output_bits,
    make_activation_lut,
    validate_lut,
)

# The message space the backend LUTs are indexed over: 4 entries under MESSAGE_BITS == 2.
MESSAGE_SPACE = 1 << MESSAGE_BITS


def relu(x: float) -> float:
    return max(x, 0.0)


def clamped_relu_at_2(x: float) -> float:
    """ReLU clamped at 2 — the float activation whose integer table is ``min(v, 2)``."""
    return min(max(x, 0.0), 2.0)


# --- make_activation_lut ----------------------------------------------------------------


def test_clamped_relu_reproduces_example_hand_built_lut():
    """A clamped ReLU on unit scales reproduces the example's ``[min(v, 2) for v in range(4)]``.

    This is the exact table ``examples/mnist/train_quantize_export.py`` builds by hand for the
    standalone Activation/PBS path; the generator must produce it identically so the example
    can later consume the service instead of hand-rolling the LUT.
    """
    in_spec = QuantSpec(scale=1.0, bits=MESSAGE_BITS, signed=False)
    out_spec = QuantSpec(scale=1.0, bits=MESSAGE_BITS, signed=False)

    lut = make_activation_lut(clamped_relu_at_2, in_spec, out_spec)

    expected = [min(v, 2) for v in range(MESSAGE_SPACE)]
    assert lut == expected == [0, 1, 2, 2]


def test_plain_relu_unit_scales_is_identity_over_domain():
    """Plain ReLU on unit-ish scales is the identity over the non-negative message space."""
    in_spec = QuantSpec(scale=1.0, bits=MESSAGE_BITS, signed=False)
    out_spec = QuantSpec(scale=1.0, bits=MESSAGE_BITS, signed=False)

    lut = make_activation_lut(relu, in_spec, out_spec)

    assert lut == [0, 1, 2, 3]


def test_output_scale_rescales_then_clamps():
    """A finer output scale rescales the value up, and the result saturates at the block max.

    With ``in_scale = 1`` and ``out_scale = 0.5``, ``q = round(relu(v) / 0.5) = 2v``: the
    table is ``[0, 2, 4, 6]`` before clamping, then saturated into ``[0, 3]``. This pins the
    dequant -> fn -> requant -> clamp pipeline end to end with the two scales differing.
    """
    in_spec = QuantSpec(scale=1.0, bits=MESSAGE_BITS, signed=False)
    out_spec = QuantSpec(scale=0.5, bits=MESSAGE_BITS, signed=False)

    lut = make_activation_lut(relu, in_spec, out_spec)

    assert lut == [0, 2, 3, 3]


def test_input_scale_dequantizes_before_fn():
    """The input scale dequantizes ``v`` before the activation sees it.

    With ``in_scale = 2`` the dequantized inputs are ``[0, 2, 4, 6]``; a clamp-at-2 ReLU maps
    them to ``[0, 2, 2, 2]`` and (unit output scale) requantizes back unchanged. This isolates
    the *input*-side dequantize from the output-side requantize.
    """
    in_spec = QuantSpec(scale=2.0, bits=MESSAGE_BITS, signed=False)
    out_spec = QuantSpec(scale=1.0, bits=MESSAGE_BITS, signed=False)

    lut = make_activation_lut(clamped_relu_at_2, in_spec, out_spec)

    assert lut == [0, 2, 2, 2]


def test_make_activation_lut_always_validates():
    """The generator returns a full-message-space, in-range table (validate_lut passes)."""
    in_spec = QuantSpec(scale=1.0, bits=MESSAGE_BITS, signed=False)
    out_spec = QuantSpec(scale=1.0, bits=MESSAGE_BITS, signed=False)

    lut = make_activation_lut(relu, in_spec, out_spec)

    assert len(lut) == MESSAGE_SPACE
    validate_lut(lut)  # must not raise


# --- identity_clamp_lut -----------------------------------------------------------------


def test_identity_clamp_lut_known_tables():
    """``identity_clamp_lut`` matches ``min(v, 2**out_bits - 1)`` for the documented cases."""
    assert identity_clamp_lut(2) == [0, 1, 2, 3]
    assert identity_clamp_lut(1) == [0, 1, 1, 1]


@pytest.mark.parametrize("out_bits", [1, 2])
def test_identity_clamp_lut_matches_compile_clamp_lut(out_bits: int):
    """Pin that the canonical generator agrees value-for-value with ``compile._clamp_lut``.

    ``compile.py`` builds this table inline today; this assertion guarantees the two cannot
    drift once the compile pass reuses the service's generator (``AGENTS.md`` §5 spirit — one
    source of truth for a value the runtime checks bit-for-bit).
    """
    assert identity_clamp_lut(out_bits) == _clamp_lut(out_bits)


# --- validate_lut -----------------------------------------------------------------------


def test_validate_lut_rejects_wrong_length():
    with pytest.raises(ValueError, match="message space"):
        validate_lut([0, 1, 2])  # 3 entries, not MESSAGE_SPACE
    with pytest.raises(ValueError, match="message space"):
        validate_lut([0, 1, 2, 3, 0])  # too long


def test_validate_lut_rejects_out_of_range_entry():
    # An entry >= 2**MESSAGE_BITS would silently wrap modulo the message modulus in the PBS.
    with pytest.raises(ValueError, match="does not fit one shortint block"):
        validate_lut([0, 1, 2, MESSAGE_SPACE])  # entry == 4 is out of range


def test_validate_lut_rejects_negative_entry():
    with pytest.raises(ValueError, match="negative"):
        validate_lut([0, -1, 2, 3])


def test_validate_lut_accepts_full_in_range_table():
    validate_lut([0, 1, 2, 3])  # must not raise


# --- lut_output_bits --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lut", "expected"),
    [
        ([0, 0, 0, 0], 1),  # all-zero table still occupies a 1-bit value
        ([0, 1, 1, 1], 1),  # max == 1 -> 1 bit
        ([0, 1, 2, 2], 2),  # max == 2 -> 2 bits
        ([0, 1, 2, 3], 2),  # max == 3 -> 2 bits
    ],
)
def test_lut_output_bits(lut: list[int], expected: int):
    assert lut_output_bits(lut) == expected
