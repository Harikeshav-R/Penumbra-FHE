"""Unified one-command runner for Penumbra-FHE examples.

Usage:
    uv run python examples/run.py <suite> [options]

Available suites:
    mnist    - Digit classification models (logreg, CNNs, QAT, sklearn)
    faces    - Olivetti faces face-recognition CNN
    tabular  - Breast Cancer Wisconsin tabular MLP
    trees    - Tree ensembles (Random Forest, XGBoost)
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from examples._replay import Example, main
except ImportError:
    from _replay import Example, main  # type: ignore[import-not-found]

HERE = Path(__file__).resolve().parent

EXAMPLES_BY_SUITE: dict[str, tuple[list[Example], str, str]] = {
    "mnist": (
        [
            Example("phase2_logreg", HERE / "mnist" / "phase2_fixture.json"),
            Example("phase4_cnn", HERE / "mnist" / "phase4_cnn_fixture.json"),
            Example("phase5_digits", HERE / "mnist" / "phase5_digits_fixture.json"),
            Example("phase5_qat", HERE / "mnist" / "phase5_qat_fixture.json"),
            Example("phase6_onnx", HERE / "mnist" / "phase6_onnx_fixture.json"),
            Example("phase6_sklearn", HERE / "mnist" / "phase6_sklearn_fixture.json"),
            Example("phase8_bn_cnn", HERE / "mnist" / "phase8_bn_cnn_fixture.json"),
            Example(
                "phase8_branch",
                HERE / "mnist" / "phase8_branch_fixture.json",
                ckks_max_poly_degree=3,
            ),
            Example("phase8_gap_cnn", HERE / "mnist" / "phase8_gap_cnn_fixture.json"),
            Example("phase8_tanh", HERE / "mnist" / "phase8_tanh_fixture.json"),
        ],
        "phase6_onnx",
        "Replay committed MNIST fixtures under FHE.",
    ),
    "faces": (
        [
            Example("phase7_faces", HERE / "faces" / "phase7_faces_fixture.json"),
        ],
        "phase7_faces",
        "Replay committed Olivetti faces fixture under FHE.",
    ),
    "tabular": (
        [
            Example(
                "phase11_tabular_mlp",
                HERE / "tabular" / "phase11_tabular_mlp_fixture.json",
            ),
        ],
        "phase11_tabular_mlp",
        "Replay committed tabular MLP fixture under FHE.",
    ),
    "trees": (
        [
            Example("phase8_trees", HERE / "trees" / "phase8_trees_fixture.json"),
            Example("phase8_xgb", HERE / "trees" / "phase8_xgb_fixture.json"),
        ],
        "phase8_trees",
        "Replay committed tree ensemble fixtures under FHE.",
    ),
}


def run_suite(suite: str, argv: list[str] | None = None) -> None:
    if suite not in EXAMPLES_BY_SUITE:
        suites = ", ".join(EXAMPLES_BY_SUITE.keys())
        print(f"Unknown suite '{suite}'. Available suites: {suites}", file=sys.stderr)
        sys.exit(2)
    examples, default, desc = EXAMPLES_BY_SUITE[suite]
    main(examples, default=default, description=desc, argv=argv)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        suites = ", ".join(EXAMPLES_BY_SUITE.keys())
        print(__doc__ or "")
        print(f"Available suites: {suites}\n")
        print("Run `python examples/run.py <suite> --help` for suite-specific options.")
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)

    suite_name = sys.argv[1]
    run_suite(suite_name, sys.argv[2:])
