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

TODO(phase-5): PTQ, calibration, LUT generation, and the Brevitas-backed QAT path.
"""
