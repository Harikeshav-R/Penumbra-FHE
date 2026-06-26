"""Quantization service: float graph -> int graph + scales + lookup tables.

Quantization is ~80% of the engineering effort and where ML accuracy lives or dies
(``PROJECT.md`` §8). The library owns it as a **service** so non-crypto users never
compute scales by hand — they supply calibration data, not scales (``PROJECT.md`` §12).

Two paths:
    - **PTQ** (post-training quantization): quantize a trained float model using
      calibration data to choose per-tensor (then optionally per-channel) scales. Start
      here.
    - **QAT** (quantization-aware training): wrap **Brevitas** rather than writing our
      own quantizer (``PROJECT.md`` §8, §15). Needed for harder models.

Bit-width budget link (``PROJECT.md`` §9): keep activation bit-widths small (<=6-8 bits);
LUTs must be generated in the quantized-integer domain consistent with the chosen scales
— an off-by-scale here silently wrecks accuracy.

Verification invariant: the quantized-cleartext output is the oracle — FHE must match it
bit-for-bit (``AGENTS.md`` §1.1). The quantization service must never break this.

## Module map

    spec.py         QuantSpec + symmetric scale selection (per-tensor / per-channel)
    calibration.py  observers + Calibrator: choose ranges from calibration data
    ptq.py          per-layer quantizers + requant (mult, shift) calibration
    lut.py          activation/requant LUT generation in the integer domain
    accuracy.py     float-vs-quantized accuracy + per-layer SQNR sensitivity report

The user-facing entry point is :meth:`penumbra.model.Model.quantize`, which composes these.
"""

from __future__ import annotations

from penumbra.quantization.accuracy import (
    AccuracyReport,
    accuracy_report,
    layer_sqnr_report,
    sqnr_db,
)
from penumbra.quantization.calibration import (
    Calibrator,
    MinMaxObserver,
    MSEObserver,
    Observer,
    PercentileObserver,
)
from penumbra.quantization.lut import (
    identity_clamp_lut,
    lut_output_bits,
    make_activation_lut,
    validate_lut,
)
from penumbra.quantization.ptq import (
    choose_requant_params,
    quantize_conv,
    quantize_linear,
    quantize_linear_integer_input,
)
from penumbra.quantization.spec import (
    QuantSpec,
    linear_logit_int,
    symmetric_spec,
    symmetric_spec_per_channel,
)

__all__ = [
    # Specs + scale selection (spec.py)
    "QuantSpec",
    "symmetric_spec",
    "symmetric_spec_per_channel",
    "linear_logit_int",
    # Calibration observers (calibration.py)
    "Observer",
    "MinMaxObserver",
    "PercentileObserver",
    "MSEObserver",
    "Calibrator",
    # Per-layer PTQ quantizers + requant calibration (ptq.py)
    "quantize_linear",
    "quantize_conv",
    "quantize_linear_integer_input",
    "choose_requant_params",
    # LUT generation (lut.py)
    "make_activation_lut",
    "identity_clamp_lut",
    "validate_lut",
    "lut_output_bits",
    # Accuracy + sensitivity reporting (accuracy.py)
    "AccuracyReport",
    "accuracy_report",
    "sqnr_db",
    "layer_sqnr_report",
]
