"""Phase-8 example: train a PyTorch Tanh MLP, export to ONNX, load and quantize with affine LUT.

Exercises the non-ReLU activation pipeline (IR schema 0.8.0):
    Gemm(64 -> 16) -> Tanh -> Gemm(16 -> 10)
lowered to:
    Linear -> Requant(clamp_lo < 0, zero_point > 0) -> Activation(affine LUT) -> Linear
with the activation's output zero-point folded into the second Linear's bias.

Regenerate artifacts:
    uv run --extra ml --system-certs python examples/mnist/tanh_mlp_export.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

import penumbra as fhe
from penumbra.ir import ActivationSpec, RequantSpec
from penumbra.quantization import accuracy_report, affine_activation_codomain
from penumbra.reference import evaluate_graph_int

# --- Configuration -------------------------------------------------------------------
IN_FEATURES = 64  # 8x8 flattened digits
HIDDEN = 16
N_CLASSES = 10

INPUT_BITS = 4
WEIGHT_BITS = (5, 5)  # (fc1, fc2)
ACT_BITS = 2
MAX_MULT_BITS = 5
N_TEST = 2
EPOCHS = 200
SEED = 0

ONNX_PATH = Path(__file__).resolve().parent / "digit_tanh_mlp.onnx"
FIXTURE_PATH = Path(__file__).resolve().parent / "phase8_tanh_fixture.json"


class DigitTanhMLP(nn.Module):
    """Two-layer MLP with Tanh activation: 64 -> 16 -> Tanh -> 10."""

    fc1: nn.Linear
    tanh: nn.Tanh
    fc2: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.tanh(x)
        return self.fc2(x)


def train() -> tuple[DigitTanhMLP, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the float MLP on real 8x8 digits; return model and splits."""
    torch.manual_seed(SEED)
    digits = load_digits()
    images = digits.images
    target = digits.target
    x = np.asarray(images, dtype=np.float32).reshape(-1, IN_FEATURES)
    y = np.asarray(target, dtype=np.int64)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)

    model = DigitTanhMLP()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=0.05)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(x_tr)
    yt = torch.from_numpy(y_tr)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model, x_tr, y_tr, x_te, y_te


def export_onnx(model: DigitTanhMLP) -> None:
    """Export the trained MLP to a committed .onnx (opset 13)."""
    dummy = torch.zeros(1, IN_FEATURES)
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
    export_onnx(model)

    cal = x_tr.astype(np.float64)

    # Load through the ONNX front door
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

    def float_predict(images: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.from_numpy(images.astype(np.float32))
            return model(xt).argmax(1).numpy()

    def quant_predict(images: np.ndarray) -> np.ndarray:
        flat = images.astype(np.float64)
        q = np.clip(np.round(flat / in_scale), 0, (1 << INPUT_BITS) - 1).astype(np.int64)
        preds = [
            int(np.argmax(evaluate_graph_int(graph, {"x": r.tolist()})[graph.outputs[0]]))
            for r in q
        ]
        return np.array(preds)

    report = accuracy_report(float_predict, quant_predict, x_te, y_te)

    x_batch_q = x_te_q[:N_TEST]
    labels_batch = labels_q[:N_TEST]

    # Find Requant and Activation nodes for diagnostics
    rq_node = next(n for n in graph.nodes if isinstance(n.op, RequantSpec))
    act_node = next(n for n in graph.nodes if isinstance(n.op, ActivationSpec))
    assert isinstance(rq_node.op, RequantSpec)
    assert isinstance(act_node.op, ActivationSpec)

    fixture = {
        "_comment": (
            "Phase-8 example: a PyTorch Tanh MLP trained on real 8x8 handwritten digits "
            "(sklearn load_digits), exported to digit_tanh_mlp.onnx, loaded through the "
            "ONNX front door (penumbra.load_onnx) and quantized via penumbra.Model.quantize. "
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
        f"  architecture       = Linear({IN_FEATURES}->{HIDDEN}) -> Tanh -> "
        f"Linear({HIDDEN}->{N_CLASSES})"
    )
    cap_bits = fhe.radix_capacity_bits(graph.num_blocks)
    print(f"  num_blocks         = {graph.num_blocks} ({cap_bits}-bit radix)")
    print(f"  float accuracy     = {report.float_accuracy:.4f}")
    print(f"  quantized accuracy = {report.quantized_accuracy:.4f}  (gap {report.gap:+.4f})")
    print(f"  test batch         = {len(labels_batch)} samples")
    act_lo = min(float(fmodel.layers[0].forward(cal).min()), 0.0)
    act_hi = max(float(fmodel.layers[0].forward(cal).max()), 0.0)
    act_scale = (act_hi - act_lo) / ((1 << ACT_BITS) - 1)
    out_scale, out_zero_point = affine_activation_codomain(
        math.tanh, in_scale=act_scale, in_zero_point=rq_node.op.zero_point, act_bits=ACT_BITS
    )
    print("  === Affine Activation Domain ===")
    print(f"  act_scale          = {act_scale:.6f}")
    print(f"  clamp_lo           = {rq_node.op.clamp_lo}")
    print(f"  zero_point         = {rq_node.op.zero_point}")
    print(f"  out_scale          = {out_scale:.6f}")
    print(f"  out_zero_point     = {out_zero_point}")
    print(f"  LUT                = {act_node.op.lut}")


if __name__ == "__main__":
    main()
