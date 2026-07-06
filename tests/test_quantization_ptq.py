"""Tests for the Phase-5 calibration observers and per-layer PTQ quantizers.

Two things are pinned here:

* **Calibration observers** (:mod:`penumbra.quantization.calibration`) produce sane symmetric
  scales — MinMax reproduces :func:`symmetric_spec` exactly (it is the streaming form of it),
  Percentile clips a heavy tail, and MSE picks a sub-peak clip that lowers round-trip error.
* **Per-layer quantizers** (:mod:`penumbra.quantization.ptq`) reproduce, integer-for-integer, the
  inline quantization math the examples hand-roll (``train_quantize_export.py`` for Linear,
  ``cnn_export.py`` for Conv and the integer-feature head). That equality is the contract: the
  quantized weights/biases must be exactly what the FHE op computes on, so the quantized-cleartext
  oracle the golden test compares against does not drift (``AGENTS.md`` §1.1).

Hermetic and fast: NumPy only, no torch/brevitas/network.
"""

from __future__ import annotations

import numpy as np
import pytest

from penumbra.bitwidth import requant_internal_bits
from penumbra.quantization.calibration import (
    Calibrator,
    MinMaxObserver,
    MSEObserver,
    PercentileObserver,
)
from penumbra.quantization.ptq import (
    choose_requant_params,
    quantize_conv,
    quantize_linear,
    quantize_linear_integer_input,
)
from penumbra.quantization.spec import symmetric_spec, symmetric_spec_per_channel

# --- The Phase-4 fixed conv filters, lifted from cnn_export.py for an exact-match check. -----
CONV_FILTERS = np.array(
    [
        [[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]],  # (out_ch=2, in_ch=1, 3, 3)
        [[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]],
    ],
    dtype=np.float64,
)


# ======================================================================================
# Observers
# ======================================================================================


def test_minmax_observer_matches_symmetric_spec() -> None:
    """MinMax over batches == symmetric_spec over the whole set (it is the streaming form)."""
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 3.0, size=(200, 9))

    obs = MinMaxObserver()
    # Feed in several batches; the running peak must equal the global peak.
    for batch in np.array_split(data, 5):
        obs.update(batch)

    streamed = obs.spec(8, signed=True)
    whole = symmetric_spec(data, 8, signed=True)
    assert streamed.scale == pytest.approx(whole.scale)
    assert streamed.bits == whole.bits
    assert streamed.signed == whole.signed
    assert obs.magnitude() == pytest.approx(float(np.max(np.abs(data))))


def test_minmax_observer_all_zero_yields_unit_scale() -> None:
    """An all-zero (or never-updated) observer yields a unit-scale spec (everything -> 0)."""
    obs = MinMaxObserver()
    obs.update(np.zeros((4, 4)))
    spec = obs.spec(8, signed=True)
    assert spec.scale == 1.0
    # A unit scale collapses small values to 0 (nothing nonzero was observed).
    assert np.array_equal(spec.quantize(np.array([0.0, 0.4, -0.4])), np.array([0, 0, 0]))

    never = MinMaxObserver()
    assert never.magnitude() == 0.0
    assert never.spec(8, signed=True).scale == 1.0


def test_percentile_observer_clips_outliers() -> None:
    """A heavy tail makes Percentile report a far smaller magnitude than MinMax."""
    # 10k samples near 1.0 plus one extreme outlier at 1000.
    data = np.concatenate([np.full(10_000, 1.0), np.array([1000.0])])

    pct = PercentileObserver(99.0, num_bins=4096)
    pct.update(data)
    mm = MinMaxObserver()
    mm.update(data)

    assert mm.magnitude() == pytest.approx(1000.0)
    # The 99th percentile sits in the bulk near 1.0, far below the outlier-driven peak.
    assert pct.magnitude() < 10.0
    assert pct.magnitude() > 0.0


def test_percentile_observer_p100_approaches_peak() -> None:
    """At percentile 100 the estimate is the peak (within one histogram bin, from above)."""
    rng = np.random.default_rng(1)
    data = rng.uniform(-5.0, 5.0, size=50_000)
    peak = float(np.max(np.abs(data)))

    pct = PercentileObserver(100.0, num_bins=8192)
    pct.update(data)
    # Right-edge readout is an upper bound on the true percentile, never under-clipping.
    assert pct.magnitude() >= peak - 1e-9
    assert pct.magnitude() <= peak * (1.0 + 1.0 / 8192) + 1e-9


def test_percentile_observer_streaming_range_growth() -> None:
    """Feeding a later, larger-range batch still tracks the peak (histogram range auto-grows)."""
    pct = PercentileObserver(100.0, num_bins=4096)
    pct.update(np.full(1000, 1.0))
    pct.update(np.array([50.0]))  # overflows the initial [0, 1] histogram range
    assert pct.magnitude() >= 50.0 - 1e-9


def test_percentile_observer_streaming_rebins_on_peak_growth() -> None:
    """A peak-growing later batch must NOT inflate the percentile of the earlier bulk.

    Regression guard for the re-binning bug: with 1000 values near 1.0 then a single 100.0, the
    99th percentile of the *combined* set is still ~1.0 (the 100.0 is the top ~0.1%). Before the
    fix the earlier bulk's counts stayed at their old bin indices and were reinterpreted at the
    new peak (~100x), reporting p99 ~= 96. The estimate must stay a tight upper bound near 1.0,
    matching a one-shot observer over the same data.
    """
    data = np.concatenate([np.full(1000, 1.0), np.array([100.0])])

    streamed = PercentileObserver(99.0, num_bins=2048)
    streamed.update(np.full(1000, 1.0))
    streamed.update(np.array([100.0]))  # grows the peak 1 -> 100

    one_shot = PercentileObserver(99.0, num_bins=2048)
    one_shot.update(data)

    # p99 sits in the bulk near 1.0 (one bin of 100/2048 ~= 0.05 slack), nowhere near 100.
    assert streamed.magnitude() < 2.0
    # Streaming with peak growth agrees with the one-shot histogram (both bin over [0, 100]).
    assert streamed.magnitude() == pytest.approx(one_shot.magnitude())


def test_mse_observer_streaming_rebins_on_peak_growth() -> None:
    """MSEObserver's clip is unaffected by batch order once the peak has grown (re-binning)."""
    rng = np.random.default_rng(3)
    bulk = rng.normal(0.0, 1.0, size=50_000)
    tail = np.array([80.0])  # a lone far outlier that grows the peak
    data = np.concatenate([bulk, tail])

    streamed = MSEObserver(grid_size=200, num_bins=4096)
    streamed.update(bulk)
    streamed.update(tail)  # grows the peak ~5 -> 80
    streamed.spec(4, signed=True)

    one_shot = MSEObserver(grid_size=200, num_bins=4096)
    one_shot.update(data)
    one_shot.spec(4, signed=True)

    # Same histogram after re-binning -> same MSE-optimal clip regardless of streaming order.
    assert streamed.magnitude() == pytest.approx(one_shot.magnitude())


def test_mse_observer_picks_subpeak_clip_on_gaussian() -> None:
    """On a Gaussian, MSE clips below the empirical peak and lowers round-trip MSE vs no-clip."""
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, size=200_000)
    peak = float(np.max(np.abs(data)))

    mse = MSEObserver(grid_size=200, num_bins=4096)
    mse.update(data)
    spec4 = mse.spec(4, signed=True)
    clip4 = mse.magnitude()
    assert 0.0 < clip4 < peak  # the sparse tail is worth clipping at 4 bits

    def round_trip_mse(clip: float, bits: int) -> float:
        sp = symmetric_spec(np.array([-clip, clip]), bits, signed=True)
        q = np.clip(np.round(data / sp.scale), sp.qmin, sp.qmax)
        return float(np.mean((data - q * sp.scale) ** 2))

    # The chosen clip must beat (or tie) clipping at the full peak — that is the whole point.
    assert round_trip_mse(clip4, 4) <= round_trip_mse(peak, 4) + 1e-12
    # The returned spec's scale is consistent with the chosen clip.
    assert spec4.scale == pytest.approx(clip4 / spec4.qmax)


def test_mse_observer_more_bits_clip_closer_to_peak() -> None:
    """More bits -> the MSE-optimal clip moves toward the peak (less aggressive clipping)."""
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, size=200_000)

    mse4 = MSEObserver(grid_size=200, num_bins=4096)
    mse4.update(data)
    mse4.spec(4, signed=True)

    mse8 = MSEObserver(grid_size=200, num_bins=4096)
    mse8.update(data)
    mse8.spec(8, signed=True)

    assert mse8.magnitude() > mse4.magnitude()


def test_mse_observer_all_zero_yields_unit_scale() -> None:
    """All-zero data -> a unit-scale spec, no division blow-up in the grid search."""
    mse = MSEObserver()
    mse.update(np.zeros(100))
    spec = mse.spec(8, signed=True)
    assert spec.scale == 1.0


def test_calibrator_drives_named_observers() -> None:
    """The Calibrator routes per-name batches and reads out per-name specs with overrides."""
    rng = np.random.default_rng(2)
    acts = rng.normal(0.0, 2.0, size=(100, 5))
    weights = rng.normal(0.0, 0.5, size=(100, 5))

    cal = Calibrator({"act": MinMaxObserver(), "w": MinMaxObserver()})
    for ab, wb in zip(np.array_split(acts, 4), np.array_split(weights, 4), strict=True):
        cal.observe({"act": ab, "w": wb})

    specs = cal.specs(8, signed=False, overrides={"w": (4, True)})
    # 'act' uses the defaults (8-bit unsigned); 'w' is overridden to 4-bit signed.
    assert specs["act"].bits == 8 and specs["act"].signed is False
    assert specs["w"].bits == 4 and specs["w"].signed is True
    # Each spec reproduces symmetric_spec over its own stream.
    assert specs["act"].scale == pytest.approx(symmetric_spec(acts, 8, signed=False).scale)
    assert specs["w"].scale == pytest.approx(symmetric_spec(weights, 4, signed=True).scale)


def test_calibrator_unknown_tensor_fails_loudly() -> None:
    """An unknown tensor name raises rather than silently dropping calibration data."""
    cal = Calibrator({"act": MinMaxObserver()})
    with pytest.raises(KeyError, match="no observer for tensor 'bogus'"):
        cal.observe({"bogus": np.ones(3)})


def test_calibrator_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one named observer"):
        Calibrator({})


# ======================================================================================
# Per-layer quantizers — exact reproduction of the inline example math
# ======================================================================================


def test_quantize_linear_per_tensor_matches_inline_math() -> None:
    """quantize_linear reproduces train_quantize_export.py's inline Linear quantization."""
    rng = np.random.default_rng(3)
    w_f = rng.normal(size=(3, 5))
    b_f = rng.normal(size=3)
    in_scale = 0.125
    bits = 4

    # Inline math, lifted verbatim from train_quantize_export.py (generalized to (n_out, n_in)).
    w_spec = symmetric_spec(w_f, bits, signed=True)
    w_q_inline = w_spec.quantize(w_f)
    acc_scale = in_scale * w_spec.scale
    b_q_inline = np.round(b_f / acc_scale).astype(np.int64)

    w_q, b_q, spec = quantize_linear(w_f, b_f, in_scale, bits=bits)
    assert np.array_equal(w_q, w_q_inline)
    assert np.array_equal(b_q, b_q_inline)
    assert w_q.dtype == np.int64 and b_q.dtype == np.int64
    assert w_q.shape == (3, 5) and b_q.shape == (3,)
    assert spec.scale == pytest.approx(w_spec.scale)


def test_quantize_linear_single_row_matches_phase2_example() -> None:
    """The Phase-2 single-logit case: a (1, n_in) weight quantizes identically to the example."""
    rng = np.random.default_rng(4)
    w_f = rng.normal(size=(1, 64))  # one logit, 64 features
    b_f = np.array([0.7])
    in_scale = symmetric_spec(rng.uniform(0, 16, size=(40, 64)), 4, signed=False).scale

    w_spec = symmetric_spec(w_f, 4, signed=True)
    acc_scale = in_scale * w_spec.scale
    expected_b = int(np.round(b_f[0] / acc_scale))

    w_q, b_q, _ = quantize_linear(w_f, b_f, in_scale, bits=4)
    assert np.array_equal(w_q[0], w_spec.quantize(w_f[0]))
    assert int(b_q[0]) == expected_b


def test_quantize_linear_none_bias_is_zero() -> None:
    """A bias-free Linear yields an all-zero int bias of the right length."""
    w_f = np.array([[1.0, -2.0, 0.5], [0.25, 0.75, -1.0]])
    w_q, b_q, _ = quantize_linear(w_f, None, 0.5, bits=4)
    assert np.array_equal(b_q, np.zeros(2, dtype=np.int64))
    assert b_q.dtype == np.int64


def test_quantize_linear_per_channel_per_row_scales_and_bias() -> None:
    """Per-channel gives one scale per output row and scales each bias by its own acc_scale."""
    # Row 0 large magnitude, row 1 tiny — a per-tensor scale would crush row 1.
    w_f = np.array([[10.0, -8.0, 2.0], [0.10, -0.05, 0.08]])
    b_f = np.array([5.0, 0.3])
    in_scale = 0.5
    bits = 4

    w_q, b_q, specs = quantize_linear(w_f, b_f, in_scale, bits=bits, per_channel=True)

    expected_specs = symmetric_spec_per_channel(w_f, bits, signed=True, axis=0)
    assert len(specs) == 2
    assert specs[0].scale != specs[1].scale  # genuinely per-row
    for i in range(2):
        assert specs[i].scale == pytest.approx(expected_specs[i].scale)
        assert np.array_equal(w_q[i], expected_specs[i].quantize(w_f[i]))
        expected_b = int(np.round(b_f[i] / (in_scale * expected_specs[i].scale)))
        assert int(b_q[i]) == expected_b
    assert w_q.shape == (2, 3) and b_q.shape == (2,)


def test_quantize_conv_matches_cnn_export_math() -> None:
    """quantize_conv reproduces cnn_export.py's conv weight quantization + flat layout."""
    bits = 4
    # Inline: one signed per-tensor scale over the whole 4-D kernel, then flatten per out-ch.
    w1_spec = symmetric_spec(CONV_FILTERS, bits, signed=True)
    w1_q = w1_spec.quantize(CONV_FILTERS)  # (2, 1, 3, 3)
    inline_flat = np.stack([w1_q[c].reshape(-1) for c in range(CONV_FILTERS.shape[0])])

    w_q, b_q, spec = quantize_conv(CONV_FILTERS, bits=bits)
    assert np.array_equal(w_q, inline_flat)
    assert w_q.shape == (2, 9)  # [out_ch][in_ch*kh*kw]
    assert b_q is None  # Phase-4 conv has no bias
    assert spec.scale == pytest.approx(w1_spec.scale)


def test_quantize_conv_flatten_is_in_channel_row_col_order() -> None:
    """The flattened row follows the IR's in-channel / kernel-row / kernel-col fastest order."""
    # 1 out-ch, 2 in-ch, 2x2 kernel with distinct values so order is observable.
    w_f = np.arange(8, dtype=np.float64).reshape(1, 2, 2, 2)  # values 0..7
    # Large scale so quantization is ~identity over this tiny range; check the *order* not values.
    w_q, _, _ = quantize_conv(w_f, bits=8)
    # reshape(out_ch, -1) on a contiguous (out,in,kh,kw) array is exactly in/row/col-fastest.
    assert np.array_equal(w_q[0].argsort(), w_f.reshape(1, -1)[0].argsort())


def test_quantize_conv_with_bias() -> None:
    """An optional conv bias is quantized into accumulator units (in_scale * weight_scale)."""
    rng = np.random.default_rng(5)
    w_f = rng.normal(size=(3, 1, 3, 3))
    b_f = rng.normal(size=3)
    in_scale = 0.2
    bits = 4

    w_spec = symmetric_spec(w_f, bits, signed=True)
    expected_b = np.round(b_f / (in_scale * w_spec.scale)).astype(np.int64)

    w_q, b_q, _ = quantize_conv(w_f, bits=bits, in_scale=in_scale, b_f=b_f)
    assert b_q is not None
    assert np.array_equal(b_q, expected_b)
    assert w_q.shape == (3, 9)


def test_quantize_conv_bias_without_in_scale_fails() -> None:
    """A bias with no input scale fails loudly — the accumulator units are undefined."""
    w_f = np.zeros((2, 1, 3, 3))
    with pytest.raises(ValueError, match="needs in_scale"):
        quantize_conv(w_f, bits=4, b_f=np.zeros(2))


def test_quantize_conv_per_channel() -> None:
    """Per-channel conv gives one scale per output channel, applied to the flattened row."""
    rng = np.random.default_rng(6)
    w_f = rng.normal(size=(2, 1, 3, 3))
    w_f[0] *= 100.0  # make channel 0 far larger so per-channel scales clearly differ
    bits = 4

    w_q, b_q, specs = quantize_conv(w_f, bits=bits, per_channel=True)
    expected_specs = symmetric_spec_per_channel(w_f, bits, signed=True, axis=0)
    assert len(specs) == 2
    assert specs[0].scale != specs[1].scale
    flat = w_f.reshape(2, -1)
    for i in range(2):
        assert np.array_equal(w_q[i], expected_specs[i].quantize(flat[i]))
    assert b_q is None


def test_quantize_linear_integer_input_matches_cnn_head_math() -> None:
    """The integer-feature head shares the weight scale for its bias (cnn_export.py head)."""
    rng = np.random.default_rng(7)
    w2_f = rng.normal(size=(10, 6))  # (n_classes, n_features), one row per class
    b2_f = rng.normal(size=10)
    bits = 4

    # Inline math from cnn_export.py: bias divides by the *weight* scale (input scale is 1).
    w2_spec = symmetric_spec(w2_f, bits, signed=True)
    w2_q_inline = w2_spec.quantize(w2_f)
    b2_q_inline = np.round(b2_f / w2_spec.scale).astype(np.int64)

    w_q, b_q, spec = quantize_linear_integer_input(w2_f, b2_f, bits=bits)
    assert np.array_equal(w_q, w2_q_inline)
    assert np.array_equal(b_q, b2_q_inline)
    assert spec.scale == pytest.approx(w2_spec.scale)
    # Equivalent to quantize_linear with in_scale = 1.0.
    w_q2, b_q2, _ = quantize_linear(w2_f, b2_f, 1.0, bits=bits)
    assert np.array_equal(w_q, w_q2) and np.array_equal(b_q, b_q2)


# ======================================================================================
# Requant (mult, shift, round_bias) calibration
# ======================================================================================


def test_choose_requant_params_power_of_two_prefers_pure_shift() -> None:
    """A power-of-two rescale ratio yields the legacy pure shift (mult=1) — no extra width."""
    # acc_scale / out_scale = 1/64 -> shift 6, mult 1, round_bias 32.
    mult, shift, round_bias = choose_requant_params(1.0, 64.0)
    assert mult == 1
    assert shift == 6
    assert round_bias == 1 << (shift - 1)


def test_choose_requant_params_non_power_of_two_uses_multiplier() -> None:
    """A non-power-of-two ratio is approximated more accurately by a fixed-point multiplier."""
    # ratio = 1/3: a pure shift (1/2 or 1/4) is ~33% off; a multiplier does much better.
    acc_scale, out_scale = 1.0, 3.0
    mult, shift, round_bias = choose_requant_params(acc_scale, out_scale, max_mult_bits=5)
    approx = mult / (1 << shift)
    assert abs(approx - 1.0 / 3.0) / (1.0 / 3.0) < 0.05  # within 5% — better than any pure shift
    if shift > 0:
        assert round_bias == 1 << (shift - 1)


def test_choose_requant_params_respects_internal_budget() -> None:
    """An over-wide multiplier for the radix is rejected loudly (internal-peak budget)."""
    # A small radix with a wide input: a large multiplier would overflow the transient peak.
    with pytest.raises(ValueError, match="transient bits|exceeding"):
        choose_requant_params(1.0, 3.0, max_mult_bits=8, input_bits=14, radix_capacity_bits=14)


def test_choose_requant_params_internal_peak_matches_tracker() -> None:
    """The params chosen fit the radix exactly per the bit-width tracker's internal-peak rule."""
    mult, shift, round_bias = choose_requant_params(
        1.0, 3.0, max_mult_bits=4, input_bits=12, radix_capacity_bits=18
    )
    peak = requant_internal_bits(12, mult, round_bias)
    assert peak <= 18
    assert mult <= (1 << 4)


def test_choose_requant_params_rejects_bad_scales() -> None:
    with pytest.raises(ValueError, match="acc_scale"):
        choose_requant_params(0.0, 1.0)
    with pytest.raises(ValueError, match="out_scale"):
        choose_requant_params(1.0, -1.0)


def test_choose_requant_params_rejects_amplifying_ratio() -> None:
    """ratio > 1 (would amplify) is rejected loudly — a Requant only narrows (`AGENTS.md` §1.4).

    Regression guard: the search could otherwise return an amplifying ``(mult>1, shift=0)`` for a
    ratio just above 1, or silently collapse a large ratio to the identity ~1.0. Both are wrong;
    an amplifying rescale is now a loud error, not a silent one.
    """
    # A modest amplification (2.5x) that used to yield (mult=2, shift=0) ~= 2.0.
    with pytest.raises(ValueError, match="only narrows|cannot amplify"):
        choose_requant_params(2.5, 1.0)
    # A large ratio (100x) that used to silently collapse to ~1.0.
    with pytest.raises(ValueError, match="only narrows|cannot amplify"):
        choose_requant_params(100.0, 1.0)


def test_choose_requant_params_never_amplifies_for_narrowing_ratio() -> None:
    """For every narrowing ratio (<= 1) the chosen mult/2**shift never exceeds 1 (narrows-only)."""
    for acc, out in [(1.0, 1.0), (0.99, 1.0), (0.85, 1.0), (0.7, 1.0), (1.0, 64.0), (1.0, 3.0)]:
        mult, shift, _ = choose_requant_params(acc, out, max_mult_bits=6)
        assert mult / (1 << shift) <= 1.0 + 1e-9, (acc, out, mult, shift)


def test_choose_requant_params_ratio_one_is_identity() -> None:
    """ratio == 1 (acc_scale == out_scale) yields the no-op pure shift (mult=1, shift=0)."""
    mult, shift, round_bias = choose_requant_params(1.0, 1.0)
    assert (mult, shift, round_bias) == (1, 0, 0)


# ======================================================================================
# Edge cases
# ======================================================================================


def test_quantize_linear_all_zero_weights() -> None:
    """All-zero weights -> unit weight scale, all-zero ints, no division blow-up."""
    w_f = np.zeros((2, 4))
    b_f = np.zeros(2)
    w_q, b_q, spec = quantize_linear(w_f, b_f, 0.5, bits=4)
    assert spec.scale == 1.0
    assert np.array_equal(w_q, np.zeros((2, 4), dtype=np.int64))
    assert np.array_equal(b_q, np.zeros(2, dtype=np.int64))


def test_quantize_conv_single_channel() -> None:
    """A single-output-channel conv quantizes and flattens correctly."""
    w_f = np.array([[[[1.0, -1.0], [0.5, -0.5]]]])  # (1, 1, 2, 2)
    w_q, b_q, spec = quantize_conv(w_f, bits=4)
    assert w_q.shape == (1, 4)
    assert b_q is None
    assert np.array_equal(w_q[0], spec.quantize(w_f).reshape(-1))


def test_quantize_linear_wrong_rank_fails() -> None:
    """A 1-D weight (not (n_out, n_in)) fails loudly rather than mis-broadcasting."""
    with pytest.raises(ValueError, match="2-D weights"):
        quantize_linear(np.zeros(5), None, 0.5, bits=4)


def test_quantize_conv_wrong_rank_fails() -> None:
    """A non-4-D conv weight fails loudly."""
    with pytest.raises(ValueError, match="4-D weights"):
        quantize_conv(np.zeros((2, 3)), bits=4)


def test_quantize_linear_bias_shape_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="does not match n_out"):
        quantize_linear(np.zeros((3, 4)), np.zeros(2), 0.5, bits=4)
