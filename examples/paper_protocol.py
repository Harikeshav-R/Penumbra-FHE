"""Paper evaluation protocol: dataset reconstruction, frozen quantization, and reference generation.

This module establishes immutable evaluation datasets and references against frozen committed
graphs for the research paper evaluation track (Phase 15).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Avoid OpenMP runtime clashes on macOS between onnxruntime and xgboost in the same process
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from penumbra.ir import Graph  # noqa: E402
from penumbra.reference import evaluate_graph_int  # noqa: E402

CALIBRATION_SEED = 1507
SPOT_CHECK_SEED = 1503
CALIBRATION_LIMIT = 128
TFHE_SPOT_CHECK_COUNT = 30
CKKS_BOUND_MARGIN = 2.0


FIXTURE_PATHS: dict[str, Path] = {
    "phase2_logreg": REPO_ROOT / "examples/mnist/phase2_fixture.json",
    "phase4_cnn": REPO_ROOT / "examples/mnist/phase4_cnn_fixture.json",
    "phase5_digits": REPO_ROOT / "examples/mnist/phase5_digits_fixture.json",
    "phase5_qat": REPO_ROOT / "examples/mnist/phase5_qat_fixture.json",
    "phase6_onnx": REPO_ROOT / "examples/mnist/phase6_onnx_fixture.json",
    "phase6_sklearn": REPO_ROOT / "examples/mnist/phase6_sklearn_fixture.json",
    "phase7_faces": REPO_ROOT / "examples/faces/phase7_faces_fixture.json",
    "phase8_trees": REPO_ROOT / "examples/trees/phase8_trees_fixture.json",
    "phase8_branch": REPO_ROOT / "examples/mnist/phase8_branch_fixture.json",
    "phase8_bn_cnn": REPO_ROOT / "examples/mnist/phase8_bn_cnn_fixture.json",
    "phase8_gap_cnn": REPO_ROOT / "examples/mnist/phase8_gap_cnn_fixture.json",
    "phase8_tanh": REPO_ROOT / "examples/mnist/phase8_tanh_fixture.json",
    "phase8_xgb": REPO_ROOT / "examples/trees/phase8_xgb_fixture.json",
    "phase11_tabular_mlp": REPO_ROOT / "examples/tabular/phase11_tabular_mlp_fixture.json",
}


@dataclass(frozen=True)
class SpotCheckRef:
    split: str  # "test" | "calibration"
    index: int

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "index": self.index}


@dataclass(frozen=True)
class Sample:
    id: str
    inputs: list[int]
    expected_output: list[int]
    expected_label: int
    target: int
    expected_scores: list[int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "inputs": self.inputs,
            "expected_output": self.expected_output,
            "expected_label": self.expected_label,
            "target": self.target,
            "expected_scores": self.expected_scores,
        }


@dataclass(frozen=True)
class FloatAccuracy:
    value: float
    source: str  # "recomputed_frozen_onnx" | "reproduced_generator" | "historical_fixture"
    source_path: str | None
    source_sha256: str | None
    sample_count: int
    recomputed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "sample_count": self.sample_count,
            "recomputed": self.recomputed,
        }


@dataclass(frozen=True)
class PaperData:
    schema_version: int
    dataset: str
    graph_sha256: str
    output_kind: str  # "logits" | "label"
    score_tensor: str
    decision_threshold: int | None
    calibration: list[Sample]
    test: list[Sample]
    tfhe_spot_check: list[SpotCheckRef]
    float_accuracy: FloatAccuracy

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "graph_sha256": self.graph_sha256,
            "output_kind": self.output_kind,
            "score_tensor": self.score_tensor,
            "decision_threshold": self.decision_threshold,
            "calibration": [s.to_dict() for s in self.calibration],
            "test": [s.to_dict() for s in self.test],
            "tfhe_spot_check": [r.to_dict() for r in self.tfhe_spot_check],
            "float_accuracy": self.float_accuracy.to_dict(),
        }


def file_sha256(path: Path) -> str:
    """Compute hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def canonical_graph_hash(graph_dict: dict[str, Any]) -> str:
    """Compute canonical SHA-256 of an IR graph dict."""
    canonical = json.dumps(graph_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sample_rank_key(seed: int, model_key: str, sample_id: str) -> tuple[bytes, str]:
    """Deterministic selection rank: SHA-256 of `<seed>:<model_key>:<sample_id>`."""
    digest = hashlib.sha256(f"{seed}:{model_key}:{sample_id}".encode()).digest()
    return (digest, sample_id)


def select_calibration_samples(
    model_key: str,
    train_samples: list[Any],
    seed: int = CALIBRATION_SEED,
    limit: int = CALIBRATION_LIMIT,
) -> list[Any]:
    """Select the first min(limit, N_train) samples by seeded hash rank."""
    ranked = sorted(
        train_samples,
        key=lambda s: sample_rank_key(seed, model_key, s.id if hasattr(s, "id") else s["id"]),
    )
    return ranked[: min(limit, len(ranked))]


def select_spot_checks(
    model_key: str,
    test_samples: list[Any],
    cal_samples: list[Any],
    is_faces: bool,
) -> list[SpotCheckRef]:
    """Select exactly 30 spot check references.

    For faces: all 20 test samples plus 10 hash-ranked calibration samples.
    For other models: 30 distinct test samples under SPOT_CHECK_SEED.
    """
    if is_faces:
        test_refs = [SpotCheckRef(split="test", index=i) for i in range(len(test_samples))]
        ranked_cal_indices = sorted(
            range(len(cal_samples)),
            key=lambda idx: sample_rank_key(
                SPOT_CHECK_SEED,
                model_key,
                cal_samples[idx].id if hasattr(cal_samples[idx], "id") else cal_samples[idx]["id"],
            ),
        )
        chosen_cal = sorted(ranked_cal_indices[:10])
        cal_refs = [SpotCheckRef(split="calibration", index=idx) for idx in chosen_cal]
        return test_refs + cal_refs

    ranked_test_indices = sorted(
        range(len(test_samples)),
        key=lambda idx: sample_rank_key(
            SPOT_CHECK_SEED,
            model_key,
            test_samples[idx].id if hasattr(test_samples[idx], "id") else test_samples[idx]["id"],
        ),
    )
    chosen_indices = sorted(ranked_test_indices[:TFHE_SPOT_CHECK_COUNT])
    return [SpotCheckRef(split="test", index=idx) for idx in chosen_indices]


def _quantize_flat(x: np.ndarray, scale: float | np.ndarray, input_bits: int) -> list[list[int]]:
    """Quantize flat feature rows: round(x / scale), clip to [0, 2^bits - 1]."""
    x_arr = np.asarray(x)
    q = np.clip(np.round(x_arr / scale), 0, (1 << input_bits) - 1).astype(int)
    return q.tolist()


def generate_paper_data(model_key: str, fixture_path: Path | None = None) -> PaperData:
    """Generate paper evaluation data for a registered model key."""
    if fixture_path is None:
        fixture_path = FIXTURE_PATHS[model_key]

    with open(fixture_path) as f:
        fixture = json.load(f)

    graph_dict = fixture["graph"]
    graph = Graph.from_dict(graph_dict)
    graph_hash = canonical_graph_hash(graph_dict)
    input_bits = graph.input_bits
    input_scale = fixture["scales"]["input"]

    if model_key == "phase2_logreg":
        from examples.mnist.train_quantize_export import (
            N_TRAIN,
            make_dataset,
            train_logreg,
        )

        rng = np.random.default_rng(0)
        x_all, y_all = make_dataset(rng)
        x_tr, y_tr = x_all[:N_TRAIN], y_all[:N_TRAIN]
        x_te, y_te = x_all[N_TRAIN:], y_all[N_TRAIN:]

        w_f, b_f = train_logreg(x_tr, y_tr)
        float_preds = ((x_te @ w_f + b_f) >= 0).astype(int)
        float_acc_val = float(np.mean(float_preds == y_te))

        q_tr = _quantize_flat(x_tr, input_scale, input_bits)
        q_te = _quantize_flat(x_te, input_scale, input_bits)

        # Temporary graph view to get both label and logit score
        eval_graph = Graph(
            schema_version=graph.schema_version,
            num_blocks=graph.num_blocks,
            input_bits=graph.input_bits,
            inputs=graph.inputs,
            outputs=["label", "logit"],
            nodes=graph.nodes,
        )

        def make_samples(
            inputs_q: list[list[int]], targets: np.ndarray, offset: int
        ) -> list[Sample]:
            samples: list[Sample] = []
            for i, (inp, target) in enumerate(zip(inputs_q, targets, strict=True)):
                res = evaluate_graph_int(eval_graph, {"x": inp})
                label = res["label"][0]
                logit = res["logit"]
                samples.append(
                    Sample(
                        id=str(offset + i),
                        inputs=inp,
                        expected_output=[label],
                        expected_label=label,
                        target=int(target),
                        expected_scores=logit,
                    )
                )
            return samples

        train_samples = make_samples(q_tr, y_tr, offset=0)
        test_samples = make_samples(q_te, y_te, offset=N_TRAIN)

        cal_samples = select_calibration_samples(model_key, train_samples)
        spot_checks = select_spot_checks(model_key, test_samples, cal_samples, is_faces=False)

        float_src_path = "examples/mnist/train_quantize_export.py"
        float_accuracy = FloatAccuracy(
            value=float_acc_val,
            source="reproduced_generator",
            source_path=float_src_path,
            source_sha256=file_sha256(REPO_ROOT / float_src_path),
            sample_count=len(test_samples),
            recomputed=True,
        )

        return PaperData(
            schema_version=1,
            dataset="synthetic_two_blob",
            graph_sha256=graph_hash,
            output_kind="label",
            score_tensor="logit",
            decision_threshold=0,
            calibration=cal_samples,
            test=test_samples,
            tfhe_spot_check=spot_checks,
            float_accuracy=float_accuracy,
        )

    if model_key == "phase4_cnn":
        from examples.mnist.cnn_export import (
            ACT_BITS,
            CONV_CH,
            CONV_FILTERS,
            IN_H,
            IN_W,
            KERNEL,
            N_TRAIN,
            avgpool_sum,
            conv2d_valid,
            make_dataset,
            softmax_train,
        )
        from penumbra.quantization import quantize_conv

        rng = np.random.default_rng(0)
        x_all, y_all = make_dataset(rng)
        x_tr, y_tr = x_all[:N_TRAIN], y_all[:N_TRAIN]
        x_te, y_te = x_all[N_TRAIN:], y_all[N_TRAIN:]

        q_tr = _quantize_flat(x_tr.reshape(len(x_tr), -1), input_scale, input_bits)
        q_te = _quantize_flat(x_te.reshape(len(x_te), -1), input_scale, input_bits)

        shift = fixture["scales"]["requant_shift"]
        act_ceiling = (1 << ACT_BITS) - 1
        w1_q_fl, _, _ = quantize_conv(CONV_FILTERS[:, None, :, :], bits=4)
        w1_q = w1_q_fl.reshape(CONV_CH, KERNEL, KERNEL)

        x_tr_q_arr = np.array(q_tr).reshape(len(x_tr), IN_H, IN_W)
        x_te_q_arr = np.array(q_te).reshape(len(x_te), IN_H, IN_W)

        conv_tr = conv2d_valid(x_tr_q_arr.astype(np.float64), w1_q.astype(np.float64)).astype(
            np.int64
        )
        conv_te = conv2d_valid(x_te_q_arr.astype(np.float64), w1_q.astype(np.float64)).astype(
            np.int64
        )

        act_tr = np.clip(np.maximum(conv_tr >> shift, 0), 0, act_ceiling)
        act_te = np.clip(np.maximum(conv_te >> shift, 0), 0, act_ceiling)

        pooled_tr = avgpool_sum(act_tr.astype(np.float64)).reshape(len(x_tr), -1)
        pooled_te = avgpool_sum(act_te.astype(np.float64)).reshape(len(x_te), -1)

        w2_f, b2_f = softmax_train(pooled_tr, y_tr)
        logits_f = pooled_te @ w2_f + b2_f
        labels_f = logits_f.argmax(1)
        float_acc_val = float(np.mean(labels_f == y_te))

        out_name = graph.outputs[0]

        def make_samples(
            inputs_q: list[list[int]], targets: np.ndarray, offset: int
        ) -> list[Sample]:
            samples: list[Sample] = []
            for i, (inp, target) in enumerate(zip(inputs_q, targets, strict=True)):
                res = evaluate_graph_int(graph, {"x": inp})
                logits = res[out_name]
                label = int(np.argmax(logits))
                samples.append(
                    Sample(
                        id=str(offset + i),
                        inputs=inp,
                        expected_output=logits,
                        expected_label=label,
                        target=int(target),
                        expected_scores=None,
                    )
                )
            return samples

        train_samples = make_samples(q_tr, y_tr, offset=0)
        test_samples = make_samples(q_te, y_te, offset=N_TRAIN)

        cal_samples = select_calibration_samples(model_key, train_samples)
        spot_checks = select_spot_checks(model_key, test_samples, cal_samples, is_faces=False)

        float_src_path = "examples/mnist/cnn_export.py"
        float_accuracy = FloatAccuracy(
            value=float_acc_val,
            source="reproduced_generator",
            source_path=float_src_path,
            source_sha256=file_sha256(REPO_ROOT / float_src_path),
            sample_count=len(test_samples),
            recomputed=True,
        )

        return PaperData(
            schema_version=1,
            dataset="synthetic_templates_cnn",
            graph_sha256=graph_hash,
            output_kind="logits",
            score_tensor=out_name,
            decision_threshold=None,
            calibration=cal_samples,
            test=test_samples,
            tfhe_spot_check=spot_checks,
            float_accuracy=float_accuracy,
        )

    if model_key in {
        "phase5_digits",
        "phase5_qat",
        "phase6_onnx",
        "phase6_sklearn",
        "phase8_branch",
        "phase8_bn_cnn",
        "phase8_gap_cnn",
        "phase8_tanh",
    }:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        digits = load_digits()
        dtype = np.float32 if model_key == "phase6_sklearn" else np.float64
        x_all = digits.images.astype(dtype)  # (1797, 8, 8)
        y_all = digits.target.astype(np.int64)
        indices = np.arange(len(x_all))

        x_tr, x_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            x_all, y_all, indices, test_size=0.2, random_state=0, stratify=y_all
        )

        q_tr = _quantize_flat(x_tr.reshape(len(x_tr), -1), input_scale, input_bits)
        q_te = _quantize_flat(x_te.reshape(len(x_te), -1), input_scale, input_bits)

        out_name = graph.outputs[0]

        def make_samples(
            inputs_q: list[list[int]], targets: np.ndarray, orig_indices: np.ndarray
        ) -> list[Sample]:
            samples: list[Sample] = []
            for inp, target, orig_idx in zip(inputs_q, targets, orig_indices, strict=True):
                res = evaluate_graph_int(graph, {"x": inp})
                logits = res[out_name]
                label = int(np.argmax(logits))
                samples.append(
                    Sample(
                        id=str(orig_idx),
                        inputs=inp,
                        expected_output=logits,
                        expected_label=label,
                        target=int(target),
                        expected_scores=None,
                    )
                )
            return samples

        train_samples = make_samples(q_tr, y_tr, idx_tr)
        test_samples = make_samples(q_te, y_te, idx_te)

        cal_samples = select_calibration_samples(model_key, train_samples)
        spot_checks = select_spot_checks(model_key, test_samples, cal_samples, is_faces=False)

        onnx_rel_paths: dict[str, str] = {
            "phase5_digits": "examples/mnist/digit_cnn.onnx",
            "phase6_onnx": "examples/mnist/digit_cnn.onnx",
            "phase6_sklearn": "examples/mnist/digit_linear_sklearn.onnx",
            "phase8_branch": "examples/mnist/digit_branch_mlp.onnx",
            "phase8_bn_cnn": "examples/mnist/digit_bn_cnn.onnx",
            "phase8_gap_cnn": "examples/mnist/digit_gap_cnn.onnx",
            "phase8_tanh": "examples/mnist/digit_tanh_mlp.onnx",
        }

        if model_key == "phase5_qat":
            float_accuracy = FloatAccuracy(
                value=float(fixture["accuracy"]["float"]),
                source="historical_fixture",
                source_path="examples/mnist/phase5_qat_fixture.json",
                source_sha256=file_sha256(REPO_ROOT / "examples/mnist/phase5_qat_fixture.json"),
                sample_count=len(test_samples),
                recomputed=False,
            )
        else:
            import onnxruntime as ort

            onnx_path = onnx_rel_paths[model_key]
            full_onnx = REPO_ROOT / onnx_path
            sess = ort.InferenceSession(str(full_onnx))
            inp_meta = sess.get_inputs()[0]
            shape = inp_meta.shape
            preds: list[int] = []
            for sample in x_te:
                if len(shape) == 4:
                    inp = sample.reshape(1, 1, 8, 8).astype(np.float32)
                elif len(shape) == 2:
                    inp = sample.reshape(1, -1).astype(np.float32)
                else:
                    inp = sample.reshape(shape).astype(np.float32)
                out = sess.run(None, {inp_meta.name: inp})[0]
                preds.append(int(np.argmax(out.flatten())))
            float_acc_val = float(np.mean(np.array(preds) == y_te))

            float_accuracy = FloatAccuracy(
                value=float_acc_val,
                source="recomputed_frozen_onnx",
                source_path=onnx_path,
                source_sha256=file_sha256(full_onnx),
                sample_count=len(test_samples),
                recomputed=True,
            )

        return PaperData(
            schema_version=1,
            dataset="sklearn_digits_8x8",
            graph_sha256=graph_hash,
            output_kind="logits",
            score_tensor=out_name,
            decision_threshold=None,
            calibration=cal_samples,
            test=test_samples,
            tfhe_spot_check=spot_checks,
            float_accuracy=float_accuracy,
        )

    if model_key == "phase7_faces":
        import onnxruntime as ort
        from sklearn.datasets import fetch_olivetti_faces
        from sklearn.model_selection import train_test_split

        from examples.faces.olivetti_export import _downsample

        faces = fetch_olivetti_faces()
        x_all = faces.images.astype(np.float32)
        y_all = faces.target.astype(np.int64)

        enrolled = y_all < 8
        enrolled_indices = np.where(enrolled)[0]
        x_enrolled = _downsample(x_all[enrolled])
        y_enrolled = y_all[enrolled]

        x_tr, x_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            x_enrolled,
            y_enrolled,
            enrolled_indices,
            test_size=0.25,
            random_state=0,
            stratify=y_enrolled,
        )

        q_tr = _quantize_flat(x_tr.reshape(len(x_tr), -1), input_scale, input_bits)
        q_te = _quantize_flat(x_te.reshape(len(x_te), -1), input_scale, input_bits)

        out_name = graph.outputs[0]

        def make_samples(
            inputs_q: list[list[int]], targets: np.ndarray, orig_indices: np.ndarray
        ) -> list[Sample]:
            samples: list[Sample] = []
            for inp, target, orig_idx in zip(inputs_q, targets, orig_indices, strict=True):
                res = evaluate_graph_int(graph, {"x": inp})
                logits = res[out_name]
                label = int(np.argmax(logits))
                samples.append(
                    Sample(
                        id=str(orig_idx),
                        inputs=inp,
                        expected_output=logits,
                        expected_label=label,
                        target=int(target),
                        expected_scores=None,
                    )
                )
            return samples

        train_samples = make_samples(q_tr, y_tr, idx_tr)
        test_samples = make_samples(q_te, y_te, idx_te)

        cal_samples = select_calibration_samples(model_key, train_samples)
        spot_checks = select_spot_checks(model_key, test_samples, cal_samples, is_faces=True)

        onnx_path = "examples/faces/face_cnn.onnx"
        full_onnx = REPO_ROOT / onnx_path
        sess = ort.InferenceSession(str(full_onnx))
        preds = []
        for sample in x_te:
            inp = sample.reshape(1, 1, 16, 16).astype(np.float32)
            out = sess.run(None, {"x": inp})[0]
            preds.append(int(np.argmax(out.flatten())))
        float_acc_val = float(np.mean(np.array(preds) == y_te))

        float_accuracy = FloatAccuracy(
            value=float_acc_val,
            source="recomputed_frozen_onnx",
            source_path=onnx_path,
            source_sha256=file_sha256(full_onnx),
            sample_count=len(test_samples),
            recomputed=True,
        )

        return PaperData(
            schema_version=1,
            dataset="olivetti_faces_16x16_first8",
            graph_sha256=graph_hash,
            output_kind="logits",
            score_tensor=out_name,
            decision_threshold=None,
            calibration=cal_samples,
            test=test_samples,
            tfhe_spot_check=spot_checks,
            float_accuracy=float_accuracy,
        )

    if model_key in {"phase8_trees", "phase8_xgb"}:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        from penumbra.adapters import from_sklearn, from_xgboost

        x_raw, y_raw = load_breast_cancer(return_X_y=True)
        indices = np.arange(len(x_raw))
        x_tr_raw, x_te_raw, y_tr_raw, y_te_raw, idx_tr, idx_te = train_test_split(
            x_raw, y_raw, indices, test_size=0.2, random_state=42
        )
        x_tr = np.asarray(x_tr_raw, dtype=np.float64)
        x_te = np.asarray(x_te_raw, dtype=np.float64)
        y_tr = np.asarray(y_tr_raw, dtype=np.int64)
        y_te = np.asarray(y_te_raw, dtype=np.int64)

        if model_key == "phase8_trees":
            from sklearn.ensemble import RandomForestClassifier

            clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
            clf.fit(x_tr, y_tr)
            tree_model = from_sklearn(clf, x_tr, input_bits=input_bits, leaf_bits=4)
            src_path = "examples/trees/tree_export.py"
        else:
            from xgboost import XGBClassifier

            clf = XGBClassifier(n_estimators=5, max_depth=3, random_state=0, n_jobs=1)
            clf.fit(x_tr, y_tr)
            tree_model = from_xgboost(clf, x_tr, input_bits=input_bits, leaf_bits=4)
            src_path = "examples/trees/xgb_export.py"

        q_tr = tree_model.quantize_input(x_tr)
        q_te = tree_model.quantize_input(x_te)

        out_name = graph.outputs[0]

        def make_samples(
            inputs_q: list[list[int]], targets: np.ndarray, orig_indices: np.ndarray
        ) -> list[Sample]:
            samples: list[Sample] = []
            for inp, target, orig_idx in zip(inputs_q, targets, orig_indices, strict=True):
                res = evaluate_graph_int(graph, {"x": inp})
                logits = res[out_name]
                label = int(np.argmax(logits))
                samples.append(
                    Sample(
                        id=str(orig_idx),
                        inputs=inp,
                        expected_output=logits,
                        expected_label=label,
                        target=int(target),
                        expected_scores=None,
                    )
                )
            return samples

        train_samples = make_samples(q_tr, y_tr, idx_tr)
        test_samples = make_samples(q_te, y_te, idx_te)

        cal_samples = select_calibration_samples(model_key, train_samples)
        spot_checks = select_spot_checks(model_key, test_samples, cal_samples, is_faces=False)

        float_preds = clf.predict(x_te)
        float_acc_val = float(np.mean(float_preds == y_te))

        float_accuracy = FloatAccuracy(
            value=float_acc_val,
            source="reproduced_generator",
            source_path=src_path,
            source_sha256=file_sha256(REPO_ROOT / src_path),
            sample_count=len(test_samples),
            recomputed=True,
        )

        return PaperData(
            schema_version=1,
            dataset="breast_cancer_wisconsin_tabular",
            graph_sha256=graph_hash,
            output_kind="logits",
            score_tensor=out_name,
            decision_threshold=None,
            calibration=cal_samples,
            test=test_samples,
            tfhe_spot_check=spot_checks,
            float_accuracy=float_accuracy,
        )

    if model_key == "phase11_tabular_mlp":
        import onnxruntime as ort
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        x_raw, y_raw = load_breast_cancer(return_X_y=True)
        indices = np.arange(len(x_raw))
        x_tr_raw, x_te_raw, y_tr_raw, y_te_raw, idx_tr, idx_te = train_test_split(
            x_raw, y_raw, indices, test_size=0.2, random_state=42
        )
        x_tr = np.asarray(x_tr_raw, dtype=np.float64)
        x_te = np.asarray(x_te_raw, dtype=np.float64)
        y_tr = np.asarray(y_tr_raw, dtype=np.int64)
        y_te = np.asarray(y_te_raw, dtype=np.int64)

        lo = x_tr.min(0)
        span = x_tr.max(0) - lo
        span[span == 0] = 1.0
        x_tr = (x_tr - lo) / span
        x_te = np.clip((x_te - lo) / span, 0.0, 1.0)

        q_tr = _quantize_flat(x_tr, input_scale, input_bits)
        q_te = _quantize_flat(x_te, input_scale, input_bits)

        out_name = graph.outputs[0]

        def make_samples(
            inputs_q: list[list[int]], targets: np.ndarray, orig_indices: np.ndarray
        ) -> list[Sample]:
            samples: list[Sample] = []
            for inp, target, orig_idx in zip(inputs_q, targets, orig_indices, strict=True):
                res = evaluate_graph_int(graph, {"x": inp})
                logits = res[out_name]
                label = int(np.argmax(logits))
                samples.append(
                    Sample(
                        id=str(orig_idx),
                        inputs=inp,
                        expected_output=logits,
                        expected_label=label,
                        target=int(target),
                        expected_scores=None,
                    )
                )
            return samples

        train_samples = make_samples(q_tr, y_tr, idx_tr)
        test_samples = make_samples(q_te, y_te, idx_te)

        cal_samples = select_calibration_samples(model_key, train_samples)
        spot_checks = select_spot_checks(model_key, test_samples, cal_samples, is_faces=False)

        onnx_path = "examples/tabular/breast_cancer_mlp.onnx"
        full_onnx = REPO_ROOT / onnx_path
        sess = ort.InferenceSession(str(full_onnx))
        preds = []
        for sample in x_te:
            inp = sample.reshape(1, 30).astype(np.float32)
            out = sess.run(None, {"x": inp})[0]
            preds.append(int(np.argmax(out.flatten())))
        float_acc_val = float(np.mean(np.array(preds) == y_te))

        float_accuracy = FloatAccuracy(
            value=float_acc_val,
            source="recomputed_frozen_onnx",
            source_path=onnx_path,
            source_sha256=file_sha256(full_onnx),
            sample_count=len(test_samples),
            recomputed=True,
        )

        return PaperData(
            schema_version=1,
            dataset="breast_cancer_wisconsin_tabular",
            graph_sha256=graph_hash,
            output_kind="logits",
            score_tensor=out_name,
            decision_threshold=None,
            calibration=cal_samples,
            test=test_samples,
            tfhe_spot_check=spot_checks,
            float_accuracy=float_accuracy,
        )

    raise ValueError(f"Unknown model key: {model_key}")


def export_paper_protocol(
    model_keys: list[str] | None = None,
    fixture_paths: dict[str, Path] | None = None,
) -> None:
    """Generate and write `paper` data into fixture JSON files after verifying golden prefixes."""
    if fixture_paths is None:
        fixture_paths = FIXTURE_PATHS
    if model_keys is None:
        model_keys = list(fixture_paths.keys())

    staged_updates: dict[str, tuple[Path, dict[str, Any]]] = {}

    for key in model_keys:
        path = fixture_paths[key]
        with open(path) as f:
            fixture = json.load(f)

        paper_data = generate_paper_data(key, fixture_path=path)

        # Validate golden prefix: all legacy rows must match exactly
        legacy_inputs = fixture["test_inputs"]
        legacy_labels = fixture["expected_labels"]
        legacy_logits = fixture.get("expected_logits")

        n_legacy = len(legacy_inputs)
        if len(paper_data.test) < n_legacy:
            raise ValueError(
                f"Model {key}: test set size {len(paper_data.test)} is smaller "
                f"than legacy test batch {n_legacy}"
            )

        for i in range(n_legacy):
            test_sample = paper_data.test[i]
            if test_sample.inputs != legacy_inputs[i]:
                raise ValueError(
                    f"Model {key}: golden mismatch at sample {i} for inputs: "
                    f"legacy={legacy_inputs[i][:5]}..., paper={test_sample.inputs[:5]}..."
                )
            if test_sample.expected_label != legacy_labels[i]:
                raise ValueError(
                    f"Model {key}: golden mismatch at sample {i} for label: "
                    f"legacy={legacy_labels[i]}, paper={test_sample.expected_label}"
                )
            if legacy_logits is not None and test_sample.expected_output != legacy_logits[i]:
                raise ValueError(
                    f"Model {key}: golden mismatch at sample {i} for logits: "
                    f"legacy={legacy_logits[i]}, paper={test_sample.expected_output}"
                )

        fixture["paper"] = paper_data.to_dict()
        staged_updates[key] = (path, fixture)

    # Only write once ALL selected models have successfully generated and validated
    for key, (path, updated_fixture) in staged_updates.items():
        path.write_text(json.dumps(updated_fixture, indent=2) + "\n")
        print(f"Updated {key} at {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys or 'all' (default: all)",
    )
    args = parser.parse_args()

    if args.models == "all":
        keys = list(FIXTURE_PATHS.keys())
    else:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]

    export_paper_protocol(model_keys=keys)


if __name__ == "__main__":
    main()
