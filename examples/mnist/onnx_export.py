"""Phase-6 example: train a torch CNN, export to ONNX, load it through the ONNX front door.

This is the first example that exercises the **ONNX front door** (``penumbra.load_onnx``) end to
end: instead of hand-building a ``penumbra.Model`` from torch weights (as ``real_digits_export.py``
does), it trains the *same* CNN, ``torch.onnx.export``s it to a committed ``digit_cnn.onnx``, and
lets ``load_onnx`` parse + validate + lower that ONNX graph back into a ``Model``. The lowered
model then flows through the **unchanged** Phase-5 ``quantize`` service and the integer oracle —
proving the front door produces a graph identical in behavior to the hand-built path (``ROADMAP.md``
Phase 6: "load_onnx works for a real framework-exported .onnx").

The model is the same BN-free digit CNN as ``real_digits_export.py``::

    Conv2d(1 -> 12, 3x3, stride 2)  ->  ReLU  ->  Linear(108 -> 10)

exported as ``Conv -> Relu -> Flatten -> Gemm`` (the Flatten folds away; the Gemm's ``transB`` and
bias are resolved by the loader). The 10 logits are the graph output; the client argmaxes them
(``PROJECT.md`` §11).

Two committed artifacts are written next to this script:

* ``digit_cnn.onnx`` — the real PyTorch-exported ONNX model (the loader's framework proof).
* ``phase6_onnx_fixture.json`` — the same 7-key fixture shape as the other examples
  (``_comment``, ``graph``, ``scales``, ``accuracy``, ``test_inputs``, ``expected_labels``,
  ``expected_logits``). CI reads only the JSON (hermetic); ``tests/test_onnx_fixture.py`` reloads
  the committed ``.onnx`` with the core ``onnx`` dep (no torch) and asserts it lowers to this
  committed graph/labels, and ``runtime/tests/golden_onnx.rs`` (``#[ignore]``d) is the FHE
  bit-for-bit gate over the fixture.

Torch + scikit-learn are the optional ``ml`` extra; regenerate only when the example changes::

    cd python && uv run --extra ml --system-certs python ../examples/mnist/onnx_export.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

import penumbra as fhe
from penumbra.quantization import accuracy_report
from penumbra.reference import evaluate_graph_int

# --- Configuration (matches real_digits_export.py so the two paths are directly comparable) --
IN_H = IN_W = 8  # load_digits is 8x8
IN_CH = 1
N_CLASSES = 10
KERNEL = 3
STRIDE = 2  # stride-2 conv shrinks the feature map to 3x3, keeping the bootstrap count feasible
CONV_CH = 12

INPUT_BITS = 4  # inputs quantized into [0, 15] (digits pixels are already [0, 16])
WEIGHT_BITS = 6  # 6-bit signed weights
ACT_BITS = 2  # post-Requant activations land in a single 2-bit block (the hard backend cap)

N_TEST = 2  # committed FHE test batch — tiny on purpose (each sample is minutes under FHE)
EPOCHS = 150
SEED = 0

ONNX_PATH = Path(__file__).resolve().parent / "digit_cnn.onnx"
FIXTURE_PATH = Path(__file__).resolve().parent / "phase6_onnx_fixture.json"

OUT_H = (IN_H - KERNEL) // STRIDE + 1  # 3
OUT_W = (IN_W - KERNEL) // STRIDE + 1  # 3
N_FEATURES = CONV_CH * OUT_H * OUT_W  # 12 * 3 * 3 = 108


class DigitCNN(nn.Module):
    """Conv (stride 2) -> ReLU -> linear (the same BN-free graph as real_digits_export.py)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(IN_CH, CONV_CH, KERNEL, stride=STRIDE, bias=False)
        self.fc = nn.Linear(N_FEATURES, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)


def train() -> tuple[DigitCNN, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the float CNN on real 8x8 digits; return the model + train/test splits (float)."""
    torch.manual_seed(SEED)
    digits = load_digits()
    x = digits.images.astype(np.float32)  # (N, 8, 8), pixels in [0, 16]
    y = digits.target.astype(np.int64)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)

    model = DigitCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(x_tr).unsqueeze(1)  # (N, 1, 8, 8)
    yt = torch.from_numpy(y_tr)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model, x_tr, y_tr, x_te, y_te


def export_onnx(model: DigitCNN) -> None:
    """Export the trained CNN to a committed ``.onnx`` (opset 13; a static 1x1x8x8 input).

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
    # example: the Model is produced by load_onnx, not hand-built from torch weights.
    export_onnx(model)
    fmodel = fhe.load_onnx(str(ONNX_PATH), input_bits=INPUT_BITS)

    # Calibration data: the training images, flattened to the model's input layout.
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
            "Phase-6 example: a torch CNN trained on real 8x8 handwritten digits (sklearn "
            "load_digits), exported to digit_cnn.onnx, then loaded THROUGH THE ONNX FRONT DOOR "
            "(penumbra.load_onnx) and quantized via penumbra.Model.quantize. The serialized IR "
            "graph is under 'graph'; the 10 logits are the graph output and the client argmaxes "
            "them. FHE output must equal these quantized-cleartext logits/labels bit-for-bit. This "
            "proves load_onnx lowers a real framework-exported ONNX model to the same behavior as "
            "the hand-built path (see phase5_digits_fixture.json)."
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
        f"-> Requant+ReLU -> Linear({N_FEATURES}->{N_CLASSES})  [via load_onnx]"
    )
    print(
        f"  num_blocks         = {graph.num_blocks} "
        f"({fhe.radix_capacity_bits(graph.num_blocks)}-bit radix)"
    )
    print(f"  float accuracy     = {report.float_accuracy:.4f}")
    print(f"  quantized accuracy = {report.quantized_accuracy:.4f}  (gap {report.gap:+.4f})")
    print(f"  test batch         = {len(labels_batch)} samples")


if __name__ == "__main__":
    main()
