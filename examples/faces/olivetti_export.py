"""Phase-7 example: closed-set face recognition through the UNCHANGED backend.

This is the abstraction-validation milestone (``ROADMAP.md`` Phase 7, ``PROJECT.md`` §4). It adds
a *completely different* use case — "is this one of N enrolled people?" — and runs it through the
**same** ``load_onnx -> quantize -> IR -> encrypted inference`` pipeline the digit examples use,
with **zero edits to the Rust backend** (``runtime/src/ops/``, ``eval.rs``) and no new IR op. If a
new use case forced a crypto-backend change, the narrow-waist abstraction would have leaked
(``AGENTS.md`` §1.2); the whole point of this example is that it does not.

The task is **closed-set** classification over the first ``N_IDENTITIES`` people of the Olivetti
faces dataset (AT&T "Database of Faces": 40 people, 10 grayscale 64x64 images each). Closed-set
(a fixed-output softmax head) is the FHE-friendly starting point; open-set embedding + distance
matching is a documented stretch goal, not this (``PROJECT.md`` §11).

The model is the same narrow-waist CNN shape as the digit examples, just wired for faces::

    downsample 64x64 -> 16x16   (4x4 block-mean, a NumPy preprocessing step, NOT a graph op)
    Conv2d(1 -> 8, 3x3, stride 4)  ->  ReLU  ->  Linear(128 -> N_IDENTITIES)

exported as ``Conv -> Relu -> Flatten -> Gemm`` (the Flatten folds away; the Gemm's ``transB`` and
bias are resolved by the loader), which ``load_onnx`` lowers to ``[Conv2d, Activation, Linear]`` —
the *exact* float layers the digit CNN lowers to. The 16x16 downsample keeps the bootstrap count
feasible: cost ~= CONV_CH * OUT_H * OUT_W = 8 * 4 * 4 = 128 requant PBS/sample, comparable to the
digit CNN's 108. The N_IDENTITIES logits are the graph output; the client argmaxes them
(``PROJECT.md`` §11).

Two committed artifacts are written next to this script (the same 7-key fixture shape as every
other example — ``_comment``, ``graph``, ``scales``, ``accuracy``, ``test_inputs``,
``expected_labels``, ``expected_logits``):

* ``face_cnn.onnx`` — the real PyTorch-exported model (the loader's framework proof).
* ``phase7_faces_fixture.json`` — CI reads only the JSON (hermetic); ``tests/test_faces_fixture.py``
  reloads the committed ``.onnx`` with the core ``onnx`` dep (no torch) and asserts it lowers to
  this committed graph/labels, and ``runtime/tests/golden_faces.rs`` (``#[ignore]``d) is the FHE
  bit-for-bit gate over the fixture.

Torch + scikit-learn are the optional ``ml`` extra. Unlike the digit examples (sklearn's bundled
``load_digits`` ships with the wheel), Olivetti is **downloaded once** (~4 MB) to
``~/scikit_learn_data`` and cached; regenerate only when the example changes::

    cd python && uv run --extra ml --system-certs python ../examples/faces/olivetti_export.py

CI never runs this (it reads only the committed JSON), so the one-time download never touches CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch import nn

import penumbra as fhe
from penumbra.quantization import accuracy_report
from penumbra.reference import evaluate_graph_int

# --- Configuration (matches the digit examples so the two paths are directly comparable) ------
RAW_H = RAW_W = 64  # Olivetti images are 64x64
DOWNSAMPLE = 4  # 4x4 block-mean -> 16x16 (64 = 16 * 4, a clean factor)
IN_H = IN_W = RAW_H // DOWNSAMPLE  # 16
IN_CH = 1
N_IDENTITIES = 8  # closed set: enroll the first 8 people (80 images, 10 each)
KERNEL = 3
STRIDE = 4  # stride-4 conv shrinks the 16x16 map to 4x4, keeping the bootstrap count feasible
CONV_CH = 8

INPUT_BITS = 4  # inputs quantized into [0, 15]
WEIGHT_BITS = 6  # 6-bit signed weights
ACT_BITS = 2  # post-Requant activations land in a single 2-bit block (the hard backend cap)

N_TEST = 2  # committed FHE test batch — tiny on purpose (each sample is minutes under FHE)
EPOCHS = 300
SEED = 0

ONNX_PATH = Path(__file__).resolve().parent / "face_cnn.onnx"
FIXTURE_PATH = Path(__file__).resolve().parent / "phase7_faces_fixture.json"

OUT_H = (IN_H - KERNEL) // STRIDE + 1  # 4
OUT_W = (IN_W - KERNEL) // STRIDE + 1  # 4
N_FEATURES = CONV_CH * OUT_H * OUT_W  # 8 * 4 * 4 = 128


class FaceCNN(nn.Module):
    """Conv (stride 4) -> ReLU -> linear — the same narrow-waist graph as the digit CNN."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(IN_CH, CONV_CH, KERNEL, stride=STRIDE, bias=False)
        self.fc = nn.Linear(N_FEATURES, N_IDENTITIES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)


def _downsample(images: np.ndarray) -> np.ndarray:
    """Block-mean 64x64 -> 16x16. Preprocessing (NumPy), not a graph op — the model sees 16x16.

    Reshaping ``(N, 64, 64) -> (N, 16, 4, 16, 4)`` and averaging the two size-4 block axes is an
    exact 4x4 average pool. Keeping this out of the ONNX graph means the exported model is the
    plain ``Conv -> Relu -> Flatten -> Gemm`` the front door already lowers — no new op, no
    backend change (the Phase-7 invariant).
    """
    n = len(images)
    blocks = images.reshape(n, IN_H, DOWNSAMPLE, IN_W, DOWNSAMPLE)
    return blocks.mean(axis=(2, 4)).astype(np.float32)


def train() -> tuple[FaceCNN, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the float CNN on the first N_IDENTITIES Olivetti people; return model + splits."""
    torch.manual_seed(SEED)
    faces = fetch_olivetti_faces()  # cached after the first ~4 MB download
    x = faces.images.astype(np.float32)  # (400, 64, 64), pixels in [0, 1]
    y = faces.target.astype(np.int64)  # 0..39, 10 images per person

    enrolled = y < N_IDENTITIES  # the closed set: the first N_IDENTITIES people
    x = _downsample(x[enrolled])  # (80, 16, 16)
    y = y[enrolled]

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, random_state=SEED, stratify=y
    )

    model = FaceCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(x_tr).unsqueeze(1)  # (N, 1, 16, 16)
    yt = torch.from_numpy(y_tr)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model, x_tr, y_tr, x_te, y_te


def export_onnx(model: FaceCNN) -> None:
    """Export the trained CNN to a committed ``.onnx`` (opset 13; a static 1x1x16x16 input).

    Uses the legacy TorchScript exporter (``dynamo=False``): it is dependency-light (the newer
    dynamo path pulls in ``onnxscript``) and emits the plain ``Conv -> Relu -> Flatten -> Gemm``
    graph the front door lowers. The static input shape lets ONNX shape inference fully resolve the
    conv/flatten feature dims the loader reads.
    """
    dummy = torch.zeros(1, IN_CH, IN_H, IN_W)
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["x"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes=None,
        dynamo=False,
    )


def main() -> None:
    model, x_tr, y_tr, x_te, y_te = train()

    # Export to ONNX, then lower it back through the front door — this is the whole point of the
    # example: the Model is produced by load_onnx, not hand-built from torch weights, exactly like
    # the digit CNN. No use-case-specific code touches the backend.
    export_onnx(model)
    fmodel = fhe.load_onnx(str(ONNX_PATH), input_bits=INPUT_BITS)

    # Calibration data: the (downsampled) training images, flattened to the model's input layout.
    cal = x_tr.reshape(len(x_tr), -1).astype(np.float64)
    graph = fmodel.quantize(
        cal, n_bits=WEIGHT_BITS, act_bits=ACT_BITS, per_channel=True, calibration="mse"
    )

    # --- Quantized-integer oracle = what FHE must match. Compute over the test set. --------
    in_scale = fmodel.input_scale
    x_te_flat = x_te.reshape(len(x_te), -1).astype(np.float64)
    x_te_q = np.clip(np.round(x_te_flat / in_scale), 0, (1 << INPUT_BITS) - 1).astype(np.int64)

    logits_q = np.array(
        [evaluate_graph_int(graph, {"x": row.tolist()})[graph.outputs[0]] for row in x_te_q]
    )
    labels_q = logits_q.argmax(1)

    # --- Honest accuracy: float CNN vs quantized-integer pipeline -------------------------
    def float_predict(images: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.from_numpy(images.astype(np.float32)).unsqueeze(1)
            return model(xt).argmax(1).numpy()

    def quant_predict(images: np.ndarray) -> np.ndarray:
        flat = images.reshape(len(images), -1).astype(np.float64)
        q = np.clip(np.round(flat / in_scale), 0, (1 << INPUT_BITS) - 1).astype(np.int64)
        preds = [
            int(np.argmax(evaluate_graph_int(graph, {"x": r.tolist()})[graph.outputs[0]]))
            for r in q
        ]
        return np.array(preds)

    report = accuracy_report(float_predict, quant_predict, x_te, y_te)

    # --- Committed test batch (quantized inputs) ------------------------------------------
    x_batch_q = x_te_q[:N_TEST]
    labels_batch = labels_q[:N_TEST]

    fixture = {
        "_comment": (
            "Phase-7 example (the abstraction-validation milestone): a torch CNN trained on the "
            "first 8 people of the Olivetti faces dataset (closed-set face recognition), exported "
            "to face_cnn.onnx, then loaded THROUGH THE ONNX FRONT DOOR (penumbra.load_onnx) and "
            "quantized via penumbra.Model.quantize. 64x64 images are 4x4 block-mean downsampled to "
            "16x16 as NumPy preprocessing (not a graph op). The serialized IR graph is under "
            "'graph'; the 8 identity logits are the graph output and the client argmaxes them. FHE "
            "output must equal these quantized-cleartext logits/labels bit-for-bit. This runs "
            "through the SAME backend as the digit examples with ZERO Rust edits — proving the "
            "narrow-waist abstraction holds (PROJECT.md section 4, AGENTS.md section 1.2)."
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
    print(
        f"  architecture       = Conv2d(1->{CONV_CH},{KERNEL}x{KERNEL},stride{STRIDE}) "
        f"-> Requant+ReLU -> Linear({N_FEATURES}->{N_IDENTITIES})  [via load_onnx]"
    )
    print(
        f"  num_blocks         = {graph.num_blocks} "
        f"({fhe.radix_capacity_bits(graph.num_blocks)}-bit radix)"
    )
    print(f"  bootstraps/sample  = {CONV_CH * OUT_H * OUT_W}  (one requant PBS per conv activation)")
    print(f"  float accuracy     = {report.float_accuracy:.4f}")
    print(f"  quantized accuracy = {report.quantized_accuracy:.4f}  (gap {report.gap:+.4f})")
    print(f"  test batch         = {len(labels_batch)} samples")


if __name__ == "__main__":
    main()
