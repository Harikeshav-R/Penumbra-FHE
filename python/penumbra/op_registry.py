"""Supported-op registry: ONNX op -> internal narrow-waist op.

A declarative table is the single source of truth for what Penumbra-FHE accepts. The
ONNX loader validates every node against it and **fails loudly at load time** with an
actionable message when an op is unsupported (``PROJECT.md`` §10, ``AGENTS.md`` §1.4).

An ONNX op is FHE-viable only if it reduces to the TFHE primitives — plaintext-weight
arithmetic, ciphertext adds, or a single-input LUT — within the bit-width budget
(``PROJECT.md`` §9). Document the rationale per op.

The documented supported-op list (``docs/SUPPORTED-OPS.md``) must always match what this
registry actually accepts — this is testable, keep it true (``AGENTS.md`` §5).

Initial target mapping (ROADMAP.md Phase 6):
    Gemm / MatMul        -> Linear
    Conv                 -> Conv2d
    Relu / Sigmoid / ... -> Activation (LUT)
    MaxPool / AveragePool-> Pool
    Add                  -> Add
    Reshape / Flatten    -> shape ops (no-op / layout)

TODO(phase-6): build the declarative registry + validation entry point.
"""
