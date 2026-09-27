"""Train, quantize, and export an XGBoost tree ensemble to Penumbra IR.

Phase-8 tabular example: an XGBClassifier trained on the Wisconsin Breast Cancer
dataset (30 features, 2 classes). Lowered via penumbra.adapters.from_xgboost into a 4-node
IR graph:
    1. split_cmp:  Compare (evaluates all tree splits in parallel)
    2. leaf_score: Linear (accumulates path conditions)
    3. leaf_sel:   Compare (selects active leaves)
    4. logits:     Linear (computes class logits)

Exports:
    examples/trees/phase8_xgb_fixture.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import penumbra as fhe
from penumbra.adapters.trees import from_xgboost
from penumbra.ir import CompareSpec
from penumbra.quantization.accuracy import accuracy_report
from penumbra.reference import evaluate_graph_int

THIS_DIR = Path(__file__).parent
FIXTURE_PATH = THIS_DIR / "phase8_xgb_fixture.json"

INPUT_BITS = 8
LEAF_BITS = 4
N_TEST = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--fixture",
        type=Path,
        default=FIXTURE_PATH,
        help="Path to write the fixture JSON (default: phase8_xgb_fixture.json)",
    )
    args = parser.parse_args()

    x, y = load_breast_cancer(return_X_y=True)
    x_tr_raw, x_te_raw, y_tr_raw, y_te_raw = train_test_split(x, y, test_size=0.2, random_state=42)
    x_tr = np.asarray(x_tr_raw, dtype=np.float64)
    x_te = np.asarray(x_te_raw, dtype=np.float64)
    y_tr = np.asarray(y_tr_raw, dtype=np.int64)
    y_te = np.asarray(y_te_raw, dtype=np.int64)

    clf = XGBClassifier(n_estimators=5, max_depth=3, random_state=0)
    clf.fit(x_tr, y_tr)

    tree_model = from_xgboost(clf, x_tr, input_bits=INPUT_BITS, leaf_bits=LEAF_BITS)
    graph = tree_model.graph

    x_te_q = tree_model.quantize_input(x_te)

    out_key = graph.outputs[0]
    logits_q = [evaluate_graph_int(graph, {"x": row})[out_key] for row in x_te_q]
    labels_q = [int(np.argmax(logit)) for logit in logits_q]

    def float_predict(samples: np.ndarray) -> np.ndarray:
        return clf.predict(samples)

    def quant_predict(samples: np.ndarray) -> np.ndarray:
        q_rows = tree_model.quantize_input(samples)
        return np.array(
            [int(np.argmax(evaluate_graph_int(graph, {"x": r})[out_key])) for r in q_rows]
        )

    report = accuracy_report(float_predict, quant_predict, x_te, y_te)

    x_batch_q = x_te_q[:N_TEST]
    labels_batch = labels_q[:N_TEST]
    logits_batch = logits_q[:N_TEST]

    total_cmps = sum(
        len(node.op.thresholds) for node in graph.nodes if isinstance(node.op, CompareSpec)
    )

    fixture = {
        "_comment": (
            "Phase-8 XGBoost tree ensemble fixture: an XGBClassifier "
            "(5 estimators, max_depth=3) trained on Wisconsin Breast Cancer (30 features, "
            "2 classes) lowered via penumbra.adapters.from_xgboost into a 4-node IR graph "
            "(split_cmp -> leaf_score -> leaf_sel -> logits). FHE output must equal these "
            "quantized-cleartext logits/labels bit-for-bit (AGENTS.md §1.1)."
        ),
        "graph": graph.to_dict(),
        "scales": {
            "input": tree_model.input_scales,
            "leaf": tree_model.leaf_scale,
        },
        "bit_plan": {
            "input_bits": INPUT_BITS,
            "leaf_bits": LEAF_BITS,
        },
        "accuracy": {
            "float": report.float_accuracy,
            "quantized": report.quantized_accuracy,
        },
        "test_inputs": x_batch_q,
        "expected_labels": labels_batch,
        "expected_logits": logits_batch,
    }

    args.fixture.write_text(json.dumps(fixture, indent=2) + "\n")

    print(f"wrote {args.fixture}")
    print("  architecture       = XGBClassifier(5 trees, max_depth=3) [BreastCancer 30->2]")
    print(
        f"  num_blocks         = {graph.num_blocks} "
        f"({fhe.radix_capacity_bits(graph.num_blocks)}-bit radix)"
    )
    print(f"  node count         = {len(graph.nodes)}")
    print(f"  total comparisons  = {total_cmps}")
    print(f"  float accuracy     = {report.float_accuracy:.4f}")
    print(f"  quantized accuracy = {report.quantized_accuracy:.4f}  (gap {report.gap:+.4f})")
    print(f"  test batch         = {len(labels_batch)} samples")


if __name__ == "__main__":
    main()
