"""Layer-3 adapters: convenience builders that produce IR graphs.

This is the layer that *grows per use case* — and it contains **no cryptography**
(``PROJECT.md`` §4). An adapter's only job is to turn a model (sklearn, torch, a tree
ensemble, ...) into a graph of standard narrow-waist ops. Adding a use case means adding
a graph here, never editing the Rust backend (``AGENTS.md`` §1.2).

Most users will go through ``onnx_loader.load_onnx()`` instead; adapters are optional
sugar for framework-native models that skip the ONNX round trip.
"""

from penumbra.adapters.trees import (
    TreeEnsembleModel,
    from_sklearn,
    from_xgboost,
)

__all__ = [
    "TreeEnsembleModel",
    "from_sklearn",
    "from_xgboost",
]
