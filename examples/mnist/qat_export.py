"""Phase-5 example: Quantization-Aware Training (Brevitas) on real handwritten digits.

The QAT counterpart of ``real_digits_export.py``. Same dataset (sklearn's 8x8 ``load_digits``) and
same FHE op graph (``Conv2d(stride 2) → Requant+ReLU → Linear``), but the CNN is trained with
**Brevitas** quantizers in the loop, so the network learns weights robust to the low-bit rounding
Penumbra's FHE backend forces. The learned weights are then exported through Penumbra's own PTQ
service (:meth:`penumbra.Model.quantize`), so the int IR — and the golden invariant that guards it
— are produced by the exact same path as a pure-PTQ model (see ``penumbra.quantization.qat``). QAT
contributes better *weights*, not a different export, which is what keeps FHE == cleartext exact.

The point of this example is to demonstrate the **QAT path end to end** through the same exact
int export and golden gate as PTQ. With the head quantized against the post-Requant activation
scale (see ``penumbra.model``), QAT on this task closes the quantization gap essentially
completely — the quantized model matches the float model (and here slightly exceeds it, within
the small test set's noise, the quantization acting as a mild regularizer). The committed
``phase5_qat_fixture.json`` records the honest float/quantized numbers.

Hermetic-fixture discipline (like every example): torch + brevitas are the optional ``ml`` extra,
used only by this generator; CI reads the committed integers and never imports them. Regenerate::

    cd python && uv run --extra ml --system-certs python ../examples/mnist/qat_export.py
"""

from __future__ import annotations

import json
from pathlib import Path

import brevitas.nn as qnn
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

import penumbra as fhe
from penumbra.quantization import accuracy_report
from penumbra.reference import evaluate_graph_int

# --- Configuration (mirrors real_digits_export.py so the two are directly comparable) ----
IN_H = IN_W = 8
IN_CH = 1
N_CLASSES = 10
KERNEL = 3
STRIDE = 2
CONV_CH = 12

INPUT_BITS = 4
WEIGHT_BITS = 6  # match the PTQ example so the two are directly comparable
ACT_BITS = 2

N_TEST = 2
EPOCHS = 200
SEED = 0

FIXTURE_PATH = Path(__file__).resolve().parent / "phase5_qat_fixture.json"

OUT_H = (IN_H - KERNEL) // STRIDE + 1  # 3
OUT_W = (IN_W - KERNEL) // STRIDE + 1  # 3
N_FEATURES = CONV_CH * OUT_H * OUT_W  # 108


class QATDigitCNN(nn.Module):
    """Brevitas QAT version of the digit CNN: quantized conv + ReLU + linear.

    The quantizers simulate low-bit rounding during training so the learned float weights are
    robust to it. We read those float weights back out afterwards and let Penumbra's PTQ service
    do the *exact* int export (the golden-invariant-preserving path).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = qnn.QuantConv2d(
            IN_CH, CONV_CH, KERNEL, stride=STRIDE, bias=False, weight_bit_width=WEIGHT_BITS
        )
        self.relu = qnn.QuantReLU(bit_width=max(ACT_BITS, 2))
        self.fc = qnn.QuantLinear(N_FEATURES, N_CLASSES, bias=True, weight_bit_width=WEIGHT_BITS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)


def train() -> tuple[QATDigitCNN, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(SEED)
    digits = load_digits()
    x = digits.images.astype(np.float32)
    y = digits.target.astype(np.int64)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)

    model = QATDigitCNN()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(x_tr).unsqueeze(1)
    yt = torch.from_numpy(y_tr)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss_fn(model(xt), yt).backward()
        opt.step()
    model.eval()
    return model, x_tr, y_tr, x_te, y_te


def main() -> None:
    model, x_tr, y_tr, x_te, y_te = train()

    # Read the QAT-trained float weights off the Brevitas modules and export through Penumbra's
    # PTQ service — the exact int pipeline (golden-invariant preserving). QAT's gift is that these
    # weights quantize much better than weights trained without quantization in the loop.
    conv_w = model.conv.weight.detach().numpy().astype(np.float64)
    fc_w = model.fc.weight.detach().numpy().astype(np.float64)
    fc_b = model.fc.bias.detach().numpy().astype(np.float64)

    fmodel = fhe.Model(
        [
            fhe.Conv2d(weight=conv_w, in_h=IN_H, in_w=IN_W, in_channels=IN_CH, stride=STRIDE),
            fhe.Activation(lambda v: max(v, 0.0)),
            fhe.Linear(weight=fc_w, bias=fc_b),
        ],
        input_bits=INPUT_BITS,
    )
    cal = x_tr.reshape(len(x_tr), -1).astype(np.float64)
    graph = fmodel.quantize(cal, n_bits=WEIGHT_BITS, act_bits=ACT_BITS, per_channel=True)

    in_scale = fmodel.input_scale
    x_te_q = np.clip(
        np.round(x_te.reshape(len(x_te), -1).astype(np.float64) / in_scale),
        0,
        (1 << INPUT_BITS) - 1,
    ).astype(np.int64)
    logits_q = np.array(
        [evaluate_graph_int(graph, {"x": row.tolist()})[graph.outputs[0]] for row in x_te_q]
    )
    labels_q = logits_q.argmax(1)

    def float_predict(images: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.from_numpy(images.astype(np.float32)).unsqueeze(1)
            return model(xt).argmax(1).numpy()

    def quant_predict(images: np.ndarray) -> np.ndarray:
        q = np.clip(
            np.round(images.reshape(len(images), -1).astype(np.float64) / in_scale),
            0,
            (1 << INPUT_BITS) - 1,
        ).astype(np.int64)
        preds = [
            int(np.argmax(evaluate_graph_int(graph, {"x": r.tolist()})[graph.outputs[0]]))
            for r in q
        ]
        return np.array(preds)

    report = accuracy_report(float_predict, quant_predict, x_te, y_te)

    fixture = {
        "_comment": (
            "Phase-5 QAT example: a Brevitas quantization-aware-trained CNN on real 8x8 "
            "handwritten digits (sklearn load_digits), exported through penumbra.Model.quantize. "
            "Same architecture as phase5_digits_fixture (PTQ); QAT recovers accuracy lost to the "
            "2-bit activation cap. The model is the serialized IR graph under 'graph'; the 10 "
            "logits are the graph output and the client argmaxes them. FHE output must equal "
            "these quantized-cleartext logits/labels bit-for-bit (see qat_export.py)."
        ),
        "graph": graph.to_dict(),
        "scales": {"input": in_scale},
        "accuracy": {"float": report.float_accuracy, "quantized": report.quantized_accuracy},
        "test_inputs": x_te_q[:N_TEST].tolist(),
        "expected_labels": labels_q[:N_TEST].tolist(),
        "expected_logits": logits_q[:N_TEST].tolist(),
    }
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")

    print(f"wrote {FIXTURE_PATH}")
    print(
        f"  architecture       = QAT Conv2d(1->{CONV_CH},{KERNEL}x{KERNEL},stride{STRIDE}) "
        f"-> Requant+ReLU -> Linear({N_FEATURES}->{N_CLASSES})"
    )
    print(f"  num_blocks         = {graph.num_blocks}")
    print(f"  float accuracy     = {report.float_accuracy:.4f}")
    print(f"  quantized accuracy = {report.quantized_accuracy:.4f}  (gap {report.gap:+.4f})")


if __name__ == "__main__":
    main()
