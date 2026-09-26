"""Phase-8 example: train a CNN with BatchNorm, export to ONNX, fold BN, and quantize.

Exercises the BatchNorm folding pipeline:
    x(1x8x8) -> Conv(1->4, 3x3, stride 1, no pad) -> BatchNorm2d(4) -> ReLU
             -> AveragePool(2x2, stride 2) [6x6 -> 3x3]
             -> Flatten(36) -> Linear(36->10)
folded to:
    Conv2d -> Requant(fused ReLU) -> Pool(avg) -> Linear
with BatchNorm folded into the Conv kernel and bias at load time (no BN op in the IR).

Regenerate artifacts:
    uv run --extra ml --system-certs python examples/mnist/bn_cnn_export.py
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

# --- Configuration -------------------------------------------------------------------
IN_CHANNELS = 1
IN_H = 8
IN_W = 8
N_CLASSES = 10

INPUT_BITS = 4
WEIGHT_BITS = (4, 4)  # (conv, fc)
ACT_BITS = 2
MAX_MULT_BITS = 5
N_TEST = 2
EPOCHS = 120
SEED = 0

ONNX_PATH = Path(__file__).resolve().parent / "digit_bn_cnn.onnx"
FIXTURE_PATH = Path(__file__).resolve().parent / "phase8_bn_cnn_fixture.json"


class DigitBNCNN(nn.Module):
    """CNN with Conv, BatchNorm, AveragePool, and Linear."""

    conv: nn.Conv2d
    bn: nn.BatchNorm2d
    relu: nn.ReLU
    pool: nn.AvgPool2d
    flatten: nn.Flatten
    fc: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(IN_CHANNELS, 4, kernel_size=3, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 3 * 3, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


def train() -> tuple[DigitBNCNN, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the float CNN on real 8x8 digits; return model and splits."""
    torch.manual_seed(SEED)
    digits = load_digits()
    images = digits.images
    target = digits.target
    x_img = np.asarray(images, dtype=np.float32).reshape(-1, 1, IN_H, IN_W)
    x_flat = np.asarray(images, dtype=np.float32).reshape(-1, IN_H * IN_W)
    y = np.asarray(target, dtype=np.int64)

    # Use stratified split on images
    _, _, y_tr, y_te, x_tr_img, x_te_img, x_tr_flat, x_te_flat = train_test_split(
        x_img, y, x_img, x_flat, test_size=0.2, random_state=SEED, stratify=y
    )

    model = DigitBNCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(x_tr_img)
    yt = torch.from_numpy(y_tr)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model, x_tr_flat, y_tr, x_te_flat, y_te


def export_onnx(model: DigitBNCNN) -> None:
    """Export the trained CNN to a committed .onnx (opset 13)."""
    dummy = torch.zeros(1, 1, IN_H, IN_W)
    torch.onnx.export(
        model,
        (dummy,),
        str(ONNX_PATH),
        input_names=["x"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes=None,
        dynamo=False,
    )


def main() -> None:
    model, x_tr, y_tr, x_te, y_te = train()
    export_onnx(model)

    cal = x_tr.astype(np.float64)

    # Load through the ONNX front door (folds BatchNorm!)
    fmodel = fhe.load_onnx(str(ONNX_PATH), input_bits=INPUT_BITS)
    graph = fmodel.quantize(
        cal,
        n_bits=list(WEIGHT_BITS),
        act_bits=ACT_BITS,
        per_channel=False,
        max_mult_bits=MAX_MULT_BITS,
        calibration="minmax",
    )

    in_scale = fmodel.input_scale
    assert in_scale is not None
    x_te_flat = x_te.astype(np.float64)
    x_te_q = np.clip(np.round(x_te_flat / in_scale), 0, (1 << INPUT_BITS) - 1).astype(np.int64)

    logits_q = np.array(
        [evaluate_graph_int(graph, {"x": row.tolist()})[graph.outputs[0]] for row in x_te_q]
    )
    labels_q = logits_q.argmax(1)

    def float_predict(images_flat: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            img = images_flat.reshape(-1, 1, IN_H, IN_W).astype(np.float32)
            xt = torch.from_numpy(img)
            return model(xt).argmax(1).numpy()

    def quant_predict(images_flat: np.ndarray) -> np.ndarray:
        flat = images_flat.astype(np.float64)
        q = np.clip(np.round(flat / in_scale), 0, (1 << INPUT_BITS) - 1).astype(np.int64)
        preds = [
            int(np.argmax(evaluate_graph_int(graph, {"x": r.tolist()})[graph.outputs[0]]))
            for r in q
        ]
        return np.array(preds)

    report = accuracy_report(float_predict, quant_predict, x_te, y_te)

    x_batch_q = x_te_q[:N_TEST]
    labels_batch = labels_q[:N_TEST]

    # Assert NO BatchNorm node exists in the emitted IR graph
    assert not any(
        "batchnorm" in n.op.op_type.lower() for n in graph.nodes
    ), "BatchNorm must not appear in the IR graph"
    expected_op_types = ["Conv2d", "Requant", "Pool", "Linear"]
    assert [
        n.op.op_type for n in graph.nodes
    ] == expected_op_types, (
        f"Expected op types {expected_op_types}, got {[n.op.op_type for n in graph.nodes]}"
    )

    fixture = {
        "_comment": (
            "Phase-8 example: a PyTorch CNN with folded BatchNorm and AveragePool, trained on "
            "real 8x8 handwritten digits (sklearn load_digits), exported to digit_bn_cnn.onnx, "
            "loaded through the ONNX front door (penumbra.load_onnx) and quantized via "
            "penumbra.Model.quantize. BatchNorm is folded into Conv2d at load time. "
            "The serialized IR graph is under 'graph'; the 10 logits are the graph output. "
            "FHE output must equal these quantized-cleartext logits/labels bit-for-bit."
        ),
        "graph": graph.to_dict(),
        "scales": {"input": in_scale},
        "bit_plan": {
            "input_bits": INPUT_BITS,
            "weight_bits": list(WEIGHT_BITS),
            "act_bits": ACT_BITS,
            "max_mult_bits": MAX_MULT_BITS,
        },
        "accuracy": {"float": report.float_accuracy, "quantized": report.quantized_accuracy},
        "test_inputs": x_batch_q.tolist(),
        "expected_labels": labels_batch.tolist(),
        "expected_logits": logits_q[:N_TEST].tolist(),
    }
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")

    print(f"wrote {ONNX_PATH}")
    print(f"wrote {FIXTURE_PATH}")
    print(
        f"  architecture       = Conv(1->4, 3x3) [folded BN] -> ReLU -> "
        f"Pool(avg 2x2) -> Linear(36->{N_CLASSES})"
    )
    cap_bits = fhe.radix_capacity_bits(graph.num_blocks)
    print(f"  num_blocks         = {graph.num_blocks} ({cap_bits}-bit radix)")
    print(f"  float accuracy     = {report.float_accuracy:.4f}")
    print(f"  quantized accuracy = {report.quantized_accuracy:.4f}  (gap {report.gap:+.4f})")
    print(f"  test batch         = {len(labels_batch)} samples")


if __name__ == "__main__":
    main()
