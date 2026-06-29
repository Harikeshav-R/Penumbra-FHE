"""Quantization-Aware Training (QAT) via Brevitas — recover accuracy lost to low-bit integers.

PTQ (:mod:`penumbra.quantization.ptq`) quantizes an already-trained float model; it is the easy
path but loses accuracy when the model is pushed to very low precision — exactly Penumbra's
regime, where activations are capped at a single ``MESSAGE_BITS``-bit block. **QAT** trains *with*
the quantization simulated in the forward/backward pass, so the network learns weights that are
robust to the rounding, recovering most of that gap (``PROJECT.md`` §8).

Penumbra wraps **Brevitas** rather than shipping its own quantizer (``PROJECT.md`` §8, §15).
Brevitas + PyTorch are the optional ``ml`` extra: this module **lazy-imports** them so the core
library never depends on the heavy stack, and CI (which never installs the extra) never imports
it. The committed QAT example fixture is plain integers, exercised by the golden test without
importing Brevitas — the same hermetic-fixture discipline as every other example.

## How the export stays exact

The golden invariant is non-negotiable (``AGENTS.md`` §1.1): FHE output == quantized-cleartext
output, bit-for-bit. To guarantee it, QAT here does **not** trust Brevitas's internal fake-quant
arithmetic to match our runtime. Instead it uses Brevitas only to *train* (the weights become
quantization-robust), then **re-quantizes the learned float weights through Penumbra's own PTQ
service** (:meth:`penumbra.model.Model.quantize`). The exported int graph is therefore produced by
the same code path — and validated by the same self-verify and golden test — as a pure-PTQ model;
QAT's contribution is purely better *weights*, not a different export path. This keeps a single,
exact quantization pipeline while still delivering QAT's accuracy benefit.

(A future refinement could export Brevitas's learned per-tensor scales directly; that is only
worthwhile if it measurably beats re-PTQ on the same weights, and it must still pass the golden
gate. Re-PTQ on QAT weights is the robust default.)
"""

from __future__ import annotations

from typing import Any


def require_brevitas() -> tuple[Any, Any]:
    """Import ``torch`` and ``brevitas`` lazily, with an actionable error if the extra is absent.

    Returns ``(torch, brevitas)``. Raising here — rather than at module import — keeps the core
    library importable without the heavy ML stack (the ``ml`` optional extra). Call this at the
    top of any QAT entry point.
    """
    try:
        import brevitas  # noqa: F401
        import torch  # noqa: F401
    except ModuleNotFoundError as e:  # pragma: no cover - exercised only without the ml extra
        raise ModuleNotFoundError(
            "QAT needs the optional 'ml' dependencies (torch + brevitas). Install them with "
            "`pip install penumbra-fhe[ml]` (or `uv sync --extra ml`). The core PTQ path "
            "(penumbra.Model.quantize on float weights) needs only NumPy."
        ) from e
    import brevitas
    import torch

    return torch, brevitas


def qat_weights_to_model(layers: list[Any], *, input_bits: int = 4) -> Any:
    """Build a Penumbra float :class:`~penumbra.model.Model` from QAT-trained layer weights.

    ``layers`` is the ordered list of Penumbra float layers (``penumbra.layers``) already
    populated with the **QAT-trained** float weights (read off the Brevitas modules after
    training). This is a thin constructor that exists so the QAT example reads symmetrically with
    the PTQ path: ``Model.quantize`` then re-quantizes these quantization-robust weights through
    the exact PTQ pipeline (see the module docstring for why this preserves the golden invariant).
    """
    from penumbra.model import Model

    return Model(layers, input_bits=input_bits)
