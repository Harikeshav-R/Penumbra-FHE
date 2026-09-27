"""Replay committed MNIST fixtures under FHE.

Run: uv run python examples/mnist/run.py [--model KEY] [--samples N] [--backend tfhe|ckks]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # examples/ is not a package; share examples/_replay.py

from _replay import Example, main  # noqa: E402

EXAMPLES = [
    Example("phase2_logreg", HERE / "phase2_fixture.json"),
    Example("phase4_cnn", HERE / "phase4_cnn_fixture.json"),
    Example("phase5_digits", HERE / "phase5_digits_fixture.json"),
    Example("phase5_qat", HERE / "phase5_qat_fixture.json"),
    Example("phase6_onnx", HERE / "phase6_onnx_fixture.json"),
    Example("phase6_sklearn", HERE / "phase6_sklearn_fixture.json"),
    Example("phase8_bn_cnn", HERE / "phase8_bn_cnn_fixture.json"),
    Example("phase8_branch", HERE / "phase8_branch_fixture.json", ckks_max_poly_degree=3),
    Example("phase8_gap_cnn", HERE / "phase8_gap_cnn_fixture.json"),
    Example("phase8_tanh", HERE / "phase8_tanh_fixture.json"),
]

if __name__ == "__main__":
    main(EXAMPLES, default="phase6_onnx", description=__doc__ or "")
