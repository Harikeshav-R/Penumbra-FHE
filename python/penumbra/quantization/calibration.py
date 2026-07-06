"""Calibration: observe float tensor ranges from data, turn them into :class:`QuantSpec`s.

Symmetric PTQ needs one number per tensor — the clipping *magnitude* ``max|x|`` that maps to
the integer range edge (:func:`penumbra.quantization.spec.symmetric_spec`). For weights that
number is read straight off the tensor, but for **activations** it must be estimated from
representative inputs: the user supplies calibration data, never scales, and the library picks
the range (``PROJECT.md`` §8, §12). That estimation is what this module owns.

The choice of clipping magnitude is the main accuracy lever in PTQ (``PROJECT.md`` §8): too
large and most of the integer range is wasted on rare outliers, crushing precision on the bulk
of the distribution; too small and the bulk is clipped. We offer three observers that trade off
differently, all producing a symmetric (zero-point-free) magnitude consistent with the signed
radix the FHE backend computes on:

    MinMaxObserver       exact peak ``max|x|`` — no clipping, streaming equivalent of
                         :func:`symmetric_spec` over the whole set. Safe default for weights.
    PercentileObserver   a high quantile of ``|x|`` (default 99.99%) — clips the extreme tail,
                         outlier-robust. Better for heavy-tailed activations.
    MSEObserver          the clip that minimizes round-trip quantization MSE at the target bits
                         — searches a grid of candidate maxima and simulates quantize/dequantize.

Bit-width budget link (``PROJECT.md`` §9): the *magnitude* an observer reports does not by
itself set the bit-width — ``bits`` is chosen by the caller and kept small (<=6-8) for
activations. The observer only fixes the float<->int scale; the accumulator-width budget is
enforced separately by :mod:`penumbra.bitwidth`.

NumPy only — calibration is a host-side, cleartext step (it never sees ciphertext), so it has
no crypto dependency and stays hermetic and fast.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from penumbra.quantization.spec import QuantSpec, symmetric_spec


def _rebin_counts(
    counts: np.ndarray, old_peak: float, new_peak: float, num_bins: int
) -> np.ndarray:
    """Re-bin a fixed-range histogram from ``[0, old_peak]`` onto ``[0, new_peak]`` (peak grew).

    The streaming observers keep an ``O(num_bins)`` histogram of ``|x|`` over ``[0, peak]``. When a
    later batch grows ``peak``, the retained counts were binned against the *old* edges; leaving
    them in place silently reinterprets their mass at the new, larger magnitudes — a huge
    over-estimate (a batch of ``~1`` values followed by a single ``100`` reported ``p99 ~= 96``
    instead of ``~1``). Since ``new_peak > old_peak`` the old range is a sub-range of the new one,
    so each old bin's mass moves to a *lower* new-bin index. We map each old bin by its **right
    edge** (``(i+1)/num_bins * old_peak``) into the new histogram: the new bin holding that edge
    has a right edge ``>=`` the old one, so the percentile read-out stays an upper bound and never
    under-clips (the module's documented safe direction).
    """
    new_counts = np.zeros(num_bins, dtype=np.int64)
    if counts.sum() == 0:
        return new_counts
    # Right edge of each old bin, remapped to a new-bin index over [0, new_peak].
    right_edges = np.arange(1, num_bins + 1) / num_bins * old_peak
    new_idx = np.minimum((right_edges / new_peak * num_bins).astype(np.int64), num_bins - 1)
    np.add.at(new_counts, new_idx, counts)
    return new_counts


class Observer(ABC):
    """Abstract range observer: fold in float batches, then emit a :class:`QuantSpec`.

    The contract is two calls: :meth:`update` one or more times with batches of float values
    (any shape — they are flattened), then :meth:`spec` once to read out the chosen symmetric
    scale at a target bit-width. An observer that has seen no nonzero value yields a unit-scale
    spec (everything quantizes to 0), matching :func:`symmetric_spec`'s all-zero guard.
    """

    @abstractmethod
    def update(self, values: np.ndarray) -> None:
        """Fold a batch of float values into the running range estimate."""

    @abstractmethod
    def magnitude(self) -> float:
        """The symmetric clipping magnitude observed so far (``>= 0``)."""

    def spec(self, bits: int, *, signed: bool) -> QuantSpec:
        """Build the symmetric :class:`QuantSpec` for the observed magnitude at ``bits``.

        Delegates to :func:`symmetric_spec` on a tiny two-point array ``[-m, +m]`` so the scale
        math (range edge, all-zero unit-scale guard, degenerate-bit-width error) is defined in
        exactly one place. ``signed`` selects the integer range; activations fed to the FHE
        backend are typically unsigned post-ReLU, weights are signed.
        """
        m = self.magnitude()
        # symmetric_spec keys off max|x|; a symmetric pair reproduces that peak exactly while
        # reusing the spec's edge/guard logic rather than recomputing the scale here.
        return symmetric_spec(np.array([-m, m], dtype=np.float64), bits, signed=signed)


class MinMaxObserver(Observer):
    """Track the exact peak ``max|x|`` across all batches (no clipping).

    This is the streaming equivalent of calling :func:`symmetric_spec` on the whole concatenated
    dataset: the running magnitude is the max of per-batch peaks, which equals the global peak.
    It is the conservative choice — nothing is clipped — and the right default for weights, where
    the tensor is fully known and outliers are real signal, not noise.
    """

    def __init__(self) -> None:
        self._peak = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size:
            self._peak = max(self._peak, float(np.max(np.abs(values))))

    def magnitude(self) -> float:
        return self._peak


class PercentileObserver(Observer):
    """Track a high percentile of ``|x|`` — clips the extreme tail, outlier-robust.

    Activations can be heavy-tailed: a handful of large values would force :class:`MinMaxObserver`
    to spend most of the integer range on outliers, starving the bulk of the distribution of
    precision. Clipping at, say, the 99.99th percentile of ``|x|`` keeps the common case sharp
    at the cost of saturating the rare tail — usually a net accuracy win for activations
    (``PROJECT.md`` §8).

    Memory: an exact percentile needs the samples, so a streaming observer must either retain
    them or histogram them. We use a **fixed-range running histogram** (``num_bins`` bins over
    ``[0, peak]`` of ``|x|``) so memory is O(num_bins) regardless of data volume. The histogram
    range auto-grows: if a batch exceeds the current ``peak`` the retained counts are **re-binned**
    onto the wider range (:func:`_rebin_counts`) *before* the new batch is folded in, so old mass
    keeps its magnitude instead of being silently reinterpreted at the larger scale. The
    percentile is read as the right edge of the bin containing the target rank — an upper bound
    on the true percentile, so it never under-clips (it errs toward keeping range, the safe
    direction). Larger ``num_bins`` tightens the estimate.
    """

    def __init__(self, percentile: float = 99.99, *, num_bins: int = 2048) -> None:
        if not 0.0 < percentile <= 100.0:
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}")
        self.percentile = float(percentile)
        self.num_bins = int(num_bins)
        self._peak = 0.0  # exact running max|x|; also the histogram's upper edge
        self._counts = np.zeros(self.num_bins, dtype=np.int64)
        self._total = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not values.size:
            return
        mags = np.abs(values).ravel()
        batch_peak = float(np.max(mags))
        # Grow the histogram range if this batch overflows it, re-binning the retained counts onto
        # the wider range first so their mass keeps its true magnitude (see _rebin_counts). This is
        # the fix for the streaming bug where old counts left in place get reinterpreted at the new
        # larger peak, inflating the percentile ~100x.
        if batch_peak > self._peak:
            if self._peak > 0.0:
                self._counts = _rebin_counts(self._counts, self._peak, batch_peak, self.num_bins)
            self._peak = batch_peak
        self._counts += np.histogram(mags, bins=self.num_bins, range=(0.0, self._peak))[0].astype(
            np.int64
        )
        self._total += mags.size

    def magnitude(self) -> float:
        if self._total == 0 or self._peak == 0.0:
            return 0.0
        # Rank of the percentile within the sorted magnitudes (ceil so e.g. p100 -> last sample).
        target = int(np.ceil(self.percentile / 100.0 * self._total))
        target = min(max(target, 1), self._total)
        cumulative = np.cumsum(self._counts)
        bin_idx = int(np.searchsorted(cumulative, target, side="left"))
        bin_idx = min(bin_idx, self.num_bins - 1)
        # Right edge of the containing bin: an upper bound on the true percentile magnitude.
        edge = (bin_idx + 1) / self.num_bins * self._peak
        return float(min(edge, self._peak))


class MSEObserver(Observer):
    """Choose the clip that minimizes round-trip quantization MSE at the target bits.

    Both MinMax (no clip) and a fixed percentile are heuristics for the real goal: the clipping
    magnitude that loses the least information once values are quantized to ``bits``. This
    observer optimizes that directly. Because the optimal clip depends on the bit-width (more
    bits -> clip closer to the peak; fewer bits -> clip tighter to protect the bulk), the search
    is deferred to :meth:`spec`, when ``bits``/``signed`` are known.

    Memory/search: like :class:`PercentileObserver` it keeps a fixed-range running histogram of
    ``|x|`` (re-binned on peak growth via :func:`_rebin_counts`, so streaming multiple
    peak-growing batches gives the same clip as one-shot). At
    ``spec`` time it sweeps a **grid of candidate maxima** — ``grid_size`` fractions of the
    observed peak, ``peak * i / grid_size`` for ``i = 1..grid_size`` — simulates symmetric
    quantize->dequantize of the histogram at each candidate, and returns the min-MSE clip. The
    grid is linear in the peak fraction; the default 100 points is plenty for an 8-bit scale and
    keeps the sweep trivially fast.
    """

    def __init__(self, *, grid_size: int = 100, num_bins: int = 2048) -> None:
        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}")
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}")
        self.grid_size = int(grid_size)
        self.num_bins = int(num_bins)
        self._peak = 0.0
        self._counts = np.zeros(self.num_bins, dtype=np.int64)
        self._total = 0
        # Selecting a clip needs a fixed bit-width; it is only known at spec() time, so the
        # search runs there. We keep magnitude() returning the chosen clip for the *last* spec()
        # so the base-class spec() and direct magnitude() callers stay consistent.
        self._last_magnitude = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not values.size:
            return
        mags = np.abs(values).ravel()
        batch_peak = float(np.max(mags))
        # Re-bin retained counts onto the wider range on peak growth (see _rebin_counts), so the
        # histogram the MSE sweep reads keeps each bin's true magnitude across streaming batches.
        if batch_peak > self._peak:
            if self._peak > 0.0:
                self._counts = _rebin_counts(self._counts, self._peak, batch_peak, self.num_bins)
            self._peak = batch_peak
        self._counts += np.histogram(mags, bins=self.num_bins, range=(0.0, self._peak))[0].astype(
            np.int64
        )
        self._total += mags.size

    def magnitude(self) -> float:
        # The MSE-optimal clip is bit-width dependent; absent a spec() call we have no bits to
        # optimize against, so report the last chosen clip (or the peak before any spec()).
        return self._last_magnitude if self._last_magnitude > 0.0 else self._peak

    def spec(self, bits: int, *, signed: bool) -> QuantSpec:
        """Search the candidate-max grid for the min-MSE clip at ``bits``, then build the spec."""
        if self._total == 0 or self._peak == 0.0:
            self._last_magnitude = 0.0
            return symmetric_spec(np.zeros(1, dtype=np.float64), bits, signed=signed)

        # Bin centers carry the histogram mass; MSE is summed over them weighted by counts.
        centers = (np.arange(self.num_bins) + 0.5) / self.num_bins * self._peak
        weights = self._counts.astype(np.float64)

        best_clip = self._peak
        best_mse = np.inf
        for i in range(1, self.grid_size + 1):
            clip = self._peak * i / self.grid_size
            # Simulate the symmetric quantizer at this clip: q = clip/edge is the step; round to
            # the nearest level and clamp into range, then dequantize. This mirrors exactly what
            # QuantSpec.quantize / dequantize would do, on the histogram centers.
            spec = symmetric_spec(np.array([-clip, clip], dtype=np.float64), bits, signed=signed)
            q = np.clip(np.round(centers / spec.scale), spec.qmin, spec.qmax)
            deq = q * spec.scale
            mse = float(np.sum(weights * (centers - deq) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_clip = clip

        self._last_magnitude = best_clip
        clip_pair = np.array([-best_clip, best_clip], dtype=np.float64)
        return symmetric_spec(clip_pair, bits, signed=signed)


class Calibrator:
    """Drive a set of named observers over batches of float tensors, then read out specs.

    A thin convenience over :class:`Observer`: register one observer per tensor name, feed
    per-batch dicts ``{name: values}`` through :meth:`observe`, then call :meth:`specs` to get
    ``{name: QuantSpec}`` at a chosen bit-width. The reusable substance is the observers; this
    just bookkeeps them so a model adapter does not hand-roll the loop.

    Per-tensor ``bits``/``signed`` may be overridden in :meth:`specs` for tensors that need a
    different range (e.g. signed weights vs. unsigned post-ReLU activations).
    """

    def __init__(self, observers: dict[str, Observer]) -> None:
        if not observers:
            raise ValueError("Calibrator needs at least one named observer")
        self.observers = dict(observers)

    def observe(self, batch: dict[str, np.ndarray]) -> None:
        """Fold one batch of named float tensors into their observers.

        Unknown names fail loudly (`AGENTS.md` §1.4) rather than silently dropping data — a typo
        in a tensor name would otherwise leave an observer un-calibrated and mis-scale the layer.
        """
        for name, values in batch.items():
            if name not in self.observers:
                raise KeyError(
                    f"Calibrator has no observer for tensor {name!r}; "
                    f"known: {sorted(self.observers)}"
                )
            self.observers[name].update(values)

    def specs(
        self,
        bits: int,
        *,
        signed: bool,
        overrides: dict[str, tuple[int, bool]] | None = None,
    ) -> dict[str, QuantSpec]:
        """Read out ``{name: QuantSpec}`` at the default ``bits``/``signed``.

        ``overrides`` maps a tensor name to its own ``(bits, signed)`` when it differs from the
        defaults — so one calibration pass can produce both signed-weight and unsigned-activation
        specs without separate calibrators.
        """
        overrides = overrides or {}
        out: dict[str, QuantSpec] = {}
        for name, obs in self.observers.items():
            b, s = overrides.get(name, (bits, signed))
            out[name] = obs.spec(b, signed=s)
        return out
