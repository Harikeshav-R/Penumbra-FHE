"""Tests for the tree-ensemble-to-IR adapter (Layer 3).

Guarded by pytest.importorskip so the hermetic CI job skips cleanly and the ML job/local dev
runs the full suite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from penumbra.reference import evaluate_graph_int


def test_threshold_conversion_math() -> None:
    """Unit-test the threshold integerisation formulas and radix clamping."""
    # Scale s_j = 0.5, input_bits = 4 (domain [0, 15], max clamp 16)
    s_j = 0.5
    input_bits = 4
    max_thresh = 1 << input_bits  # 16

    # 1. scikit-learn: T = floor(t / s_j) + 1
    # For t = 2.0: 2.0 / 0.5 = 4.0 -> floor is 4 -> T = 5.
    # An integer input x_int = 4 (corresponding to 2.0) has x_int >= 5 (False, i.e. <= 2.0).
    # An integer input x_int = 5 (corresponding to 2.5) has x_int >= 5 (True, i.e. > 2.0).
    t = 2.0
    t_sklearn = math.floor(t / s_j) + 1
    assert t_sklearn == 5

    # Boundary: negative threshold clamped to 0
    t_neg = -10.0
    t_clamped_neg = max(0, min(math.floor(t_neg / s_j) + 1, max_thresh))
    assert t_clamped_neg == 0

    # Boundary: large threshold clamped to 2**input_bits
    t_large = 100.0
    t_clamped_large = max(0, min(math.floor(t_large / s_j) + 1, max_thresh))
    assert t_clamped_large == max_thresh

    # 2. XGBoost: T = ceil(c / s_j)
    # For c = 2.1: 2.1 / 0.5 = 4.2 -> ceil is 5 -> T = 5.
    # For c = 2.0: 2.0 / 0.5 = 4.0 -> ceil is 4 -> T = 4.
    c = 2.0
    t_xgb = math.ceil(c / s_j)
    assert t_xgb == 4

    c_frac = 2.1
    assert math.ceil(c_frac / s_j) == 5

    t_xgb_neg = max(0, min(math.ceil(-5.0 / s_j), max_thresh))
    assert t_xgb_neg == 0

    t_xgb_large = max(0, min(math.ceil(50.0 / s_j), max_thresh))
    assert t_xgb_large == max_thresh


def test_sklearn_tree_adapter_accuracy_and_structure() -> None:
    """Verify sklearn RandomForestClassifier lowering matches predict on >= 95% of test samples."""
    _ = pytest.importorskip("sklearn")
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    from penumbra.adapters.trees import from_sklearn

    x, y = load_breast_cancer(return_X_y=True)
    x_tr_raw, x_te_raw, y_tr_raw, _ = train_test_split(x, y, test_size=0.2, random_state=42)
    x_tr = np.asarray(x_tr_raw, dtype=np.float64)
    x_te = np.asarray(x_te_raw, dtype=np.float64)
    y_tr = np.asarray(y_tr_raw, dtype=np.int64)

    clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    clf.fit(x_tr, y_tr)

    model = from_sklearn(clf, x_tr, input_bits=8, leaf_bits=4)
    g = model.graph

    # Verify 4-node lowering sequence
    assert [n.op.op_type for n in g.nodes] == ["Compare", "Linear", "Compare", "Linear"]

    # Verify predictions on test set track float model
    float_preds = clf.predict(x_te)
    x_te_q = model.quantize_input(x_te)
    quant_preds = [
        int(np.argmax(evaluate_graph_int(g, {"x": row})[g.outputs[0]])) for row in x_te_q
    ]

    agreement = np.mean(float_preds == quant_preds)
    assert agreement >= 0.95, f"agreement {agreement:.4f} is below 95% target"


def _run_xgboost_adapter_check() -> None:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    from penumbra.adapters.trees import from_xgboost

    x, y = load_breast_cancer(return_X_y=True)
    x_tr_raw, x_te_raw, y_tr_raw, _ = train_test_split(x, y, test_size=0.2, random_state=42)
    x_tr = np.asarray(x_tr_raw, dtype=np.float64)
    x_te = np.asarray(x_te_raw, dtype=np.float64)
    y_tr = np.asarray(y_tr_raw, dtype=np.int64)

    clf = XGBClassifier(n_estimators=5, max_depth=3, random_state=0)
    clf.fit(x_tr, y_tr)

    model = from_xgboost(clf, x_tr, input_bits=8, leaf_bits=4)
    g = model.graph

    # Verify 4-node lowering sequence
    assert [n.op.op_type for n in g.nodes] == ["Compare", "Linear", "Compare", "Linear"]

    float_preds = clf.predict(x_te)
    x_te_q = model.quantize_input(x_te)
    quant_preds = [
        int(np.argmax(evaluate_graph_int(g, {"x": row})[g.outputs[0]])) for row in x_te_q
    ]

    agreement = np.mean(float_preds == quant_preds)
    assert agreement >= 0.95, f"agreement {agreement:.4f} is below 95% target"


def test_xgboost_tree_adapter_accuracy_and_structure() -> None:
    """Verify XGBoost XGBClassifier lowering matches predict on >= 95% of test samples."""
    _ = pytest.importorskip("xgboost")
    _ = pytest.importorskip("sklearn")
    import subprocess
    import sys

    # Run in an isolated subprocess to prevent OpenMP runtime collision on macOS
    cmd = (
        "from tests.test_tree_adapter import _run_xgboost_adapter_check; "
        "_run_xgboost_adapter_check()"
    )
    res = subprocess.run([sys.executable, "-c", cmd], capture_output=True, text=True)
    assert res.returncode == 0, f"XGBoost adapter check failed:\n{res.stderr}"


def test_degenerate_all_leaf_ensemble_raises_value_error() -> None:
    """A tree with 0 splits (all leaves) must raise a clear ValueError."""
    _ = pytest.importorskip("sklearn")
    from sklearn.tree import DecisionTreeClassifier

    from penumbra.adapters.trees import from_sklearn

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 0])  # single class -> 0 splits
    clf = DecisionTreeClassifier().fit(x, y)

    with pytest.raises(ValueError, match="no splits"):
        _ = from_sklearn(clf, x)
