"""Phase-6 second-framework example: a scikit-learn model, exported to ONNX, loaded encrypted.

This is the **"train anywhere" proof** (``ROADMAP.md`` Phase 6 exit criterion: ``load_onnx`` works
for at least two models from two different frameworks). The first example (``onnx_export.py``)
trains a CNN in **PyTorch**; this one trains a linear digit classifier in **scikit-learn** and
exports it with ``skl2onnx``. Both flow through the *same* ``penumbra.load_onnx`` front door and the
*same* Phase-5 ``quantize`` service — no framework-specific code, proving the ONNX waist is the only
integration point (``PROJECT.md`` §10).

The model is a plain multinomial linear classifier (``MLPRegressor`` with no hidden layer, trained
on one-hot targets — i.e. a 64->10 least-squares logit map)::

    x (64 pixels)  ->  Linear(64 -> 10 logits)

Why a *regressor* and not ``LogisticRegression``/``MLPClassifier``: skl2onnx lowers those to the
``ai.onnx.ml`` custom-op domain (``LinearClassifier``, ``ZipMap``, ``ArrayFeatureExtractor``) with a
two-output (label + probability) graph — outside Penumbra's supported ``ai.onnx`` subset. A
no-hidden-layer ``MLPRegressor`` exports as a clean single-output ``ai.onnx`` graph
(``Cast -> MatMul -> Add -> Reshape``) that lowers to exactly one ``Linear``: the leading ``Cast``
(skl2onnx's input dtype-normalizer) folds away, and the client argmaxes the 10 logits
(``PROJECT.md`` §11) — the same contract as the CNN.

A single ``Linear`` has **no Requant and no PBS**, so the encrypted forward is cheap (one
plaintext-weight matmul), which is why the sklearn golden gate runs quickly where the CNN gate is
minutes per sample.

Two committed artifacts are written next to this script:

* ``digit_linear_sklearn.onnx`` — the real skl2onnx-exported ONNX model (the second framework proof).
* ``phase6_sklearn_fixture.json`` — the same fixture shape as the other examples. CI reads only the
  JSON (hermetic); ``tests/test_sklearn_fixture.py`` reloads the committed ``.onnx`` with the core
  ``onnx`` dep (no sklearn) and asserts it lowers to this committed graph/labels, and
  ``runtime/tests/golden_sklearn.rs`` is the FHE bit-for-bit gate over the fixture.

scikit-learn + skl2onnx are the optional ``ml`` extra; regenerate only when the example changes::

    cd python && uv run --extra ml --system-certs python ../examples/mnist/sklearn_export.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

import penumbra as fhe
from penumbra.quantization import accuracy_report
from penumbra.reference import evaluate_graph_int

# --- Configuration -------------------------------------------------------------------------
N_FEATURES = 64  # load_digits is 8x8 = 64 pixels, flattened
N_CLASSES = 10

INPUT_BITS = 4  # inputs quantized into [0, 15] (digits pixels are already [0, 16])
WEIGHT_BITS = 8  # a single well-conditioned logit head quantizes tightly at 8-bit per-tensor
SEED = 0

N_TEST = 2  # committed FHE test batch (kept small: the wide-radix Linear is ~a minute per sample)

ONNX_PATH = Path(__file__).resolve().parent / "digit_linear_sklearn.onnx"
FIXTURE_PATH = Path(__file__).resolve().parent / "phase6_sklearn_fixture.json"


def train() -> tuple[MLPRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train a no-hidden-layer sklearn regressor (== multinomial linear head) on 8x8 digits."""
    digits = load_digits()
    x = digits.data.astype(np.float32)  # (N, 64), pixels in [0, 16]
    y = digits.target.astype(np.int64)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)

    # One-hot targets turn the regressor into a least-squares 64->10 logit map; argmax = class.
    y_onehot = np.eye(N_CLASSES, dtype=np.float32)[y_tr]
    model = MLPRegressor(
        hidden_layer_sizes=(),  # no hidden layer -> a pure linear model (one Gemm/MatMul)
        activation="identity",
        max_iter=3000,
        alpha=1e-2,
        random_state=SEED,
    )
    model.fit(x_tr, y_onehot)
    return model, x_tr, y_tr, x_te, y_te


def export_onnx(model: MLPRegressor) -> None:
    """Export the trained sklearn model to a committed ``.onnx`` (single-output ai.onnx graph)."""
    # An explicit float input type keeps the graph single-input; skl2onnx still prepends a
    # dtype-normalizing Cast(to=FLOAT), which the front door folds away as an identity no-op.
    onnx_model = to_onnx(model, initial_types=[("x", FloatTensorType([None, N_FEATURES]))])
    ONNX_PATH.write_bytes(onnx_model.SerializeToString())


def main() -> None:
    model, x_tr, y_tr, x_te, y_te = train()

    # Export to ONNX, then lower it back through the front door — the point of the example: the
    # Model is produced by load_onnx from a *scikit-learn* export, not hand-built or from torch.
    export_onnx(model)
    fmodel = fhe.load_onnx(str(ONNX_PATH), input_bits=INPUT_BITS)

    cal = x_tr.astype(np.float64)
    graph = fmodel.quantize(cal, n_bits=WEIGHT_BITS, calibration="mse")

    # --- Quantized-integer oracle = what FHE must match. Compute over the test set. --------
    in_scale = fmodel.input_scale
    x_te_q = np.clip(np.round(x_te / in_scale), 0, (1 << INPUT_BITS) - 1).astype(np.int64)

    logits_q = np.array(
        [evaluate_graph_int(graph, {"x": row.tolist()})[graph.outputs[0]] for row in x_te_q]
    )
    labels_q = logits_q.argmax(1)

    # --- Honest accuracy: sklearn float model vs quantized-integer pipeline ---------------
    def float_predict(images: np.ndarray) -> np.ndarray:
        return model.predict(images.astype(np.float32)).argmax(1)

    def quant_predict(images: np.ndarray) -> np.ndarray:
        q = np.clip(np.round(images / in_scale), 0, (1 << INPUT_BITS) - 1).astype(np.int64)
        return np.array(
            [
                int(np.argmax(evaluate_graph_int(graph, {"x": r.tolist()})[graph.outputs[0]]))
                for r in q
            ]
        )

    report = accuracy_report(float_predict, quant_predict, x_te, y_te)

    # --- Committed test batch (quantized inputs) ------------------------------------------
    x_batch_q = x_te_q[:N_TEST]
    labels_batch = labels_q[:N_TEST]

    fixture = {
        "_comment": (
            "Phase-6 SECOND-FRAMEWORK example: a scikit-learn linear digit classifier (MLPRegressor "
            "with no hidden layer, trained on one-hot targets) exported to digit_linear_sklearn.onnx "
            "with skl2onnx, then loaded THROUGH THE ONNX FRONT DOOR (penumbra.load_onnx) and "
            "quantized via penumbra.Model.quantize. The leading Cast(to=FLOAT) skl2onnx emits folds "
            "away; the graph is a single Linear whose 10 logits are the graph output, and the client "
            "argmaxes them. FHE output must equal these quantized-cleartext logits/labels "
            "bit-for-bit. Together with the torch CNN (phase6_onnx_fixture.json) this is the "
            "'train anywhere, two frameworks' proof (ROADMAP Phase 6)."
        ),
        "graph": graph.to_dict(),
        "scales": {"input": in_scale},
        "accuracy": {"float": report.float_accuracy, "quantized": report.quantized_accuracy},
        "test_inputs": x_batch_q.tolist(),
        "expected_labels": labels_batch.tolist(),
        "expected_logits": logits_q[:N_TEST].tolist(),
    }
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")

    print(f"wrote {ONNX_PATH}")
    print(f"wrote {FIXTURE_PATH}")
    print(f"  architecture       = Linear({N_FEATURES}->{N_CLASSES})  [scikit-learn via load_onnx]")
    print(
        f"  num_blocks         = {graph.num_blocks} "
        f"({fhe.radix_capacity_bits(graph.num_blocks)}-bit radix)"
    )
    print(f"  float accuracy     = {report.float_accuracy:.4f}")
    print(f"  quantized accuracy = {report.quantized_accuracy:.4f}  (gap {report.gap:+.4f})")
    print(f"  test batch         = {len(labels_batch)} samples")


if __name__ == "__main__":
    main()
