"""Tree ensemble (decision trees, random forests, XGBoost) -> IR adapter (Layer 3).

Lowers tree ensembles to standard narrow-waist Penumbra IR ops without ciphertext-ciphertext
multiplications, using the 4-stage sum-of-comparisons formulation:
  1. `split_cmp`:  Compare op (threshold split evaluations across all trees)
  2. `leaf_score`: Linear op (path condition accumulator)
  3. `leaf_sel`:   Compare op (one-hot leaf indicators)
  4. `logits`:     Linear op (leaf value dot product to class logits)

Contains NO cryptography (PROJECT.md §4, AGENTS.md §1.2).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from penumbra.bitwidth import minimal_num_blocks
from penumbra.ir import SCHEMA_VERSION, CompareSpec, Graph, LinearSpec, Node
from penumbra.quantization.spec import symmetric_spec, symmetric_spec_per_channel
from penumbra.reference import evaluate_graph_int


@dataclass(frozen=True)
class _Split:
    feature: int
    threshold: float  # go RIGHT iff x[feature] > threshold


@dataclass(frozen=True)
class _ParsedTree:
    splits: list[_Split]  # internal nodes, tree-local index
    leaf_values: list[list[float]]  # per leaf: contribution per class
    leaf_paths: list[list[tuple[int, bool]]]  # per leaf: (tree-local split index, went_right)


def _parse_sklearn_tree(tree_: Any, n_classes: int, weight: float) -> _ParsedTree:
    """Parse a scikit-learn ``Tree`` object via DFS into a neutral ``_ParsedTree``."""
    splits: list[_Split] = []
    leaf_values: list[list[float]] = []
    leaf_paths: list[list[tuple[int, bool]]] = []

    def dfs(node_id: int, current_path: list[tuple[int, bool]]) -> None:
        left = int(tree_.children_left[node_id])
        right = int(tree_.children_right[node_id])
        if left == -1 and right == -1:
            val_raw = np.asarray(tree_.value[node_id][0], dtype=np.float64)
            s = float(val_raw.sum())
            if s > 0:
                normalized = val_raw / s
            else:
                normalized = np.zeros(n_classes, dtype=np.float64)
            leaf_values.append((normalized * weight).tolist())
            leaf_paths.append(list(current_path))
            return

        feat = int(tree_.feature[node_id])
        thresh = float(tree_.threshold[node_id])
        split_idx = len(splits)
        splits.append(_Split(feature=feat, threshold=thresh))

        # scikit-learn rule: left iff x <= threshold (went_right = False),
        # right iff x > threshold (went_right = True)
        dfs(left, current_path + [(split_idx, False)])
        dfs(right, current_path + [(split_idx, True)])

    dfs(0, [])
    return _ParsedTree(splits=splits, leaf_values=leaf_values, leaf_paths=leaf_paths)


def _parse_xgboost_tree(node_json: dict[str, Any], n_classes: int, class_index: int) -> _ParsedTree:
    """Parse one XGBoost JSON tree dump into a neutral ``_ParsedTree``."""
    splits: list[_Split] = []
    leaf_values: list[list[float]] = []
    leaf_paths: list[list[tuple[int, bool]]] = []

    def walk(curr: dict[str, Any], current_path: list[tuple[int, bool]]) -> None:
        if "leaf" in curr:
            vals = [0.0] * n_classes
            vals[class_index] = float(curr["leaf"])
            leaf_values.append(vals)
            leaf_paths.append(list(current_path))
            return

        split_str = curr.get("split")
        if (
            not isinstance(split_str, str)
            or not split_str.startswith("f")
            or not split_str[1:].isdigit()
        ):
            nodeid = curr.get("nodeid", "?")
            raise ValueError(
                f"XGBoost node {nodeid} has split {split_str!r}; expected 'f<index>' format"
            )
        feat = int(split_str[1:])
        cond = float(curr["split_condition"])
        split_idx = len(splits)
        splits.append(_Split(feature=feat, threshold=cond))

        yes_id = curr["yes"]
        no_id = curr["no"]
        children = {c["nodeid"]: c for c in curr.get("children", [])}
        if yes_id not in children or no_id not in children:
            raise ValueError(
                f"XGBoost node {curr.get('nodeid')} missing children for yes={yes_id} / no={no_id}"
            )

        # XGBoost rule: yes iff x < split_condition (went_right = False)
        # no iff x >= split_condition (went_right = True)
        walk(children[yes_id], current_path + [(split_idx, False)])
        walk(children[no_id], current_path + [(split_idx, True)])

    walk(node_json, [])
    return _ParsedTree(splits=splits, leaf_values=leaf_values, leaf_paths=leaf_paths)


def _build_tree_graph(
    parsed_trees: list[_ParsedTree],
    calibration_data: np.ndarray,
    *,
    input_bits: int,
    leaf_bits: int,
    is_xgboost: bool,
    base_margin: float,
    estimator_name: str,
) -> tuple[Graph, list[float], float]:
    """Compile neutral parsed trees into a 4-node Penumbra IR graph."""
    calib = np.asarray(calibration_data, dtype=np.float64)
    if calib.ndim != 2:
        raise ValueError("calibration_data must be 2-D array of shape (N, n_features)")

    # Per-channel input quantization scales
    specs = symmetric_spec_per_channel(calib, input_bits, signed=False, axis=1)
    input_scales = [float(spec.scale) for spec in specs]

    # 1. Flatten all splits across all trees
    global_splits: list[_Split] = []
    tree_split_offsets: list[int] = []
    for tree in parsed_trees:
        tree_split_offsets.append(len(global_splits))
        global_splits.extend(tree.splits)

    if not global_splits:
        msg = (
            f"Ensemble {estimator_name} has no splits (all trees are bare leaves); "
            "cannot build a valid graph"
        )
        raise ValueError(msg)

    # 2. Flatten all leaves across all trees with globally offset paths
    all_leaf_values: list[list[float]] = []
    all_leaf_paths: list[list[tuple[int, bool]]] = []
    for tree_idx, tree in enumerate(parsed_trees):
        offset = tree_split_offsets[tree_idx]
        for lv, path in zip(tree.leaf_values, tree.leaf_paths, strict=True):
            all_leaf_values.append(lv)
            global_path = [(offset + split_idx, went_right) for split_idx, went_right in path]
            all_leaf_paths.append(global_path)

    n_leaves = len(all_leaf_values)
    total_splits = len(global_splits)

    # Stage 1: split_cmp (Compare op)
    cmp_indices: list[int] = []
    cmp_thresholds: list[int] = []
    for split in global_splits:
        j = split.feature
        s_j = input_scales[j]
        if not is_xgboost:
            t_thresh = math.floor(split.threshold / s_j) + 1
        else:
            t_thresh = math.ceil(split.threshold / s_j)
        t_clamped = max(0, min(t_thresh, 1 << input_bits))
        cmp_indices.append(j)
        cmp_thresholds.append(t_clamped)
    split_cmp_node = Node(
        name="split_cmp",
        inputs=["x"],
        outputs=["split_bits"],
        op=CompareSpec(indices=cmp_indices, thresholds=cmp_thresholds),
    )

    # Stage 2: leaf_score (Linear op)
    # score_l = Σ_{g ∈ right} (+1)·b_g + Σ_{g ∈ left} (-1)·b_g + |left|
    w_mat: list[list[int]] = [[0] * total_splits for _ in range(n_leaves)]
    b_vec: list[int] = [0] * n_leaves
    for leaf_idx, path in enumerate(all_leaf_paths):
        left_count = 0
        for g, went_right in path:
            if went_right:
                w_mat[leaf_idx][g] = 1
            else:
                w_mat[leaf_idx][g] = -1
                left_count += 1
        b_vec[leaf_idx] = left_count

    leaf_score_node = Node(
        name="leaf_score",
        inputs=["split_bits"],
        outputs=["leaf_scores"],
        op=LinearSpec(weights=w_mat, bias=b_vec, weight_bits=1),
    )

    # Stage 3: leaf_sel (Compare op)
    # [score_l >= depth_l]
    sel_indices = list(range(n_leaves))
    sel_thresholds = [len(path) for path in all_leaf_paths]

    leaf_sel_node = Node(
        name="leaf_sel",
        inputs=["leaf_scores"],
        outputs=["leaf_onehot"],
        op=CompareSpec(indices=sel_indices, thresholds=sel_thresholds),
    )

    # Stage 4: logits (Linear op)
    # V[c][l] = round(leaf_values[l][c] / leaf_scale)
    n_classes = len(all_leaf_values[0])
    all_lv_arr = np.asarray(all_leaf_values, dtype=np.float64)
    leaf_spec = symmetric_spec(all_lv_arr, leaf_bits, signed=True)
    leaf_scale = float(leaf_spec.scale)

    v_mat: list[list[int]] = [
        [int(round(all_leaf_values[leaf_idx][c] / leaf_scale)) for leaf_idx in range(n_leaves)]
        for c in range(n_classes)
    ]
    if is_xgboost and n_classes == 2:
        bias_c = [0, int(round(base_margin / leaf_scale))]
    else:
        bias_c = [0] * n_classes

    logits_node = Node(
        name="logits",
        inputs=["leaf_onehot"],
        outputs=["logits"],
        op=LinearSpec(weights=v_mat, bias=bias_c, weight_bits=leaf_bits),
    )

    nodes = [split_cmp_node, leaf_score_node, leaf_sel_node, logits_node]

    # Compute minimal num_blocks via bit-width tracker
    provisional = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=2,
        input_bits=input_bits,
        inputs=["x"],
        outputs=["logits"],
        nodes=nodes,
    )
    num_blocks = minimal_num_blocks(provisional)

    graph = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=num_blocks,
        input_bits=input_bits,
        inputs=["x"],
        outputs=["logits"],
        nodes=nodes,
    )

    # In-quantize smoke check on a sample of calibration inputs
    sample = calib[: min(4, len(calib))]
    scales_arr = np.asarray(input_scales, dtype=np.float64)
    max_int = (1 << input_bits) - 1
    for row in sample:
        xq = np.clip(np.round(row / scales_arr), 0, max_int).astype(int).tolist()
        _ = evaluate_graph_int(graph, {"x": xq})

    return graph, input_scales, leaf_scale


@dataclass(frozen=True)
class TreeEnsembleModel:
    """Compiled tree ensemble model ready for cleartext or encrypted inference."""

    graph: Graph
    input_scales: list[float]  # one per feature
    leaf_scale: float  # float logit ≈ int logit * leaf_scale
    n_features: int
    n_classes: int
    input_bits: int

    def quantize_input(self, x: np.ndarray | list[float] | list[list[float]]) -> list[list[int]]:
        """Quantize float input features into integer arrays for the graph."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != self.n_features:
            raise ValueError(f"expected input with {self.n_features} features, got {arr.shape[1]}")
        scales = np.asarray(self.input_scales, dtype=np.float64)
        max_int = (1 << self.input_bits) - 1
        scaled = np.round(arr / scales)
        clipped = np.clip(scaled, 0, max_int).astype(np.int64)
        return clipped.tolist()

    def export(self, path: str | Path) -> None:
        """Serialize the IR graph to JSON."""
        p = Path(path)
        p.write_text(self.graph.to_json())

    def predict_encrypted(
        self,
        x: np.ndarray,
        *,
        return_logits: bool = False,
        keys: Any = None,
        backend: str = "tfhe",
        profile: Any = None,
    ):
        """Run encrypted forward pass on float array ``x``, returning predicted label(s)."""
        from penumbra.client import run_encrypted

        arr = np.asarray(x, dtype=np.float64)
        single = arr.ndim == 1
        int_inputs = self.quantize_input(arr)
        raw_outputs = run_encrypted(
            self.graph, int_inputs, keys=keys, backend=backend, profile=profile
        )
        labels = [int(np.argmax(row)) for row in raw_outputs]
        if single:
            if return_logits:
                return labels[0], raw_outputs[0]
            return labels[0]
        if return_logits:
            return labels, raw_outputs
        return labels


def from_sklearn(
    estimator: Any,
    calibration_data: np.ndarray,
    *,
    input_bits: int = 8,
    leaf_bits: int = 4,
) -> TreeEnsembleModel:
    """Lower a fitted scikit-learn DecisionTreeClassifier or RandomForestClassifier to IR."""
    calib = np.asarray(calibration_data, dtype=np.float64)
    if hasattr(estimator, "estimators_"):
        trees_to_parse = list(estimator.estimators_)
        n_estimators = len(trees_to_parse)
        weight = 1.0 / n_estimators
        n_classes = int(getattr(estimator, "n_classes_", 2))
        n_features = int(getattr(estimator, "n_features_in_", calib.shape[1]))
    elif hasattr(estimator, "tree_"):
        trees_to_parse = [estimator]
        weight = 1.0
        n_classes = int(getattr(estimator, "n_classes_", 2))
        n_features = int(getattr(estimator, "n_features_in_", calib.shape[1]))
    else:
        raise ValueError(
            f"Unsupported sklearn estimator {type(estimator).__name__}; "
            "expected DecisionTreeClassifier or RandomForestClassifier"
        )

    parsed_trees = [_parse_sklearn_tree(est.tree_, n_classes, weight) for est in trees_to_parse]

    graph, input_scales, leaf_scale = _build_tree_graph(
        parsed_trees,
        calib,
        input_bits=input_bits,
        leaf_bits=leaf_bits,
        is_xgboost=False,
        base_margin=0.0,
        estimator_name=type(estimator).__name__,
    )

    return TreeEnsembleModel(
        graph=graph,
        input_scales=input_scales,
        leaf_scale=leaf_scale,
        n_features=n_features,
        n_classes=n_classes,
        input_bits=input_bits,
    )


def from_xgboost(
    model: Any,
    calibration_data: np.ndarray,
    *,
    input_bits: int = 8,
    leaf_bits: int = 4,
) -> TreeEnsembleModel:
    """Lower a fitted XGBoost model (XGBClassifier or Booster) to IR."""
    try:
        import xgboost  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "from_xgboost requires xgboost; install with 'uv sync --extra ml' (AGENTS.md §7)"
        ) from e

    calib = np.asarray(calibration_data, dtype=np.float64)
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    booster.feature_names = None

    cfg = json.loads(booster.save_config())
    learner_param = cfg.get("learner", {}).get("learner_model_param", {})
    num_class = int(learner_param.get("num_class", "0"))
    if num_class <= 1:
        n_classes = 2
        is_binary = True
    else:
        n_classes = num_class
        is_binary = False

    bs_raw = learner_param.get("base_score", "0.5")
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(bs_raw))
    bs_val = float(matches[0]) if matches else 0.5
    if is_binary and 0.0 < bs_val < 1.0:
        base_margin = math.log(bs_val / (1.0 - bs_val))
    else:
        base_margin = 0.0

    dump = booster.get_dump(dump_format="json")
    parsed_trees = []
    for pos, tree_str in enumerate(dump):
        tree_json = json.loads(tree_str)
        c_idx = 1 if is_binary else (pos % n_classes)
        parsed = _parse_xgboost_tree(tree_json, n_classes, c_idx)
        parsed_trees.append(parsed)

    graph, input_scales, leaf_scale = _build_tree_graph(
        parsed_trees,
        calib,
        input_bits=input_bits,
        leaf_bits=leaf_bits,
        is_xgboost=True,
        base_margin=base_margin,
        estimator_name=type(model).__name__,
    )

    return TreeEnsembleModel(
        graph=graph,
        input_scales=input_scales,
        leaf_scale=leaf_scale,
        n_features=calib.shape[1],
        n_classes=n_classes,
        input_bits=input_bits,
    )
