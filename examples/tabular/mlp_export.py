"""Phase-11 tabular example: train a PyTorch MLP on Breast Cancer, export to ONNX, quantize.

Architecture:
    Gemm(30 -> 8) -> ReLU -> Gemm(8 -> 2)
lowered to:
    Linear -> Requant(clamp_lo=0, fused ReLU) -> Linear

Note on preprocessing:
    Raw Breast Cancer features span widely (~0.05 to ~2500 across 30 dimensions).
    Client-side min-max feature scaling to [0.0, 1.0] using train statistics is performed
    in NumPy prior to integer quantization, not as an in-graph FHE op, because input scaling
    is per-tensor unsigned.

Regenerate artifacts:
    uv run --extra ml --system-certs python examples/tabular/mlp_export.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch import nn

import penumbra as fhe
from penumbra.quantization import accuracy_report
from penumbra.reference import evaluate_graph_int

# --- Configuration -------------------------------------------------------------------
IN_FEATURES = 30
HIDDEN = 8
N_CLASSES = 2

INPUT_BITS = 4
WEIGHT_BITS = (5, 5)  # (fc1, fc2)
ACT_BITS = 2
MAX_MULT_BITS = 5
PER_CHANNEL = False
CALIBRATION = "minmax"
N_TEST = 4
EPOCHS = 300
LR = 1e-2
SEED = 0
SPLIT_SEED = 42

ONNX_PATH = Path(__file__).resolve().parent / "breast_cancer_mlp.onnx"
FIXTURE_PATH = Path(__file__).resolve().parent / "phase11_tabular_mlp_fixture.json"


class BreastCancerMLP(nn.Module):
    """Two-layer MLP with ReLU activation: 30 -> 8 -> ReLU -> 2."""

    fc1: nn.Linear
    relu: nn.ReLU
    fc2: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def train() -> tuple[BreastCancerMLP, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the float MLP on min-max scaled Breast Cancer features; return model and splits."""
    torch.manual_seed(SEED)
    x, y = load_breast_cancer(return_X_y=True)
    x_tr_raw, x_te_raw, y_tr_raw, y_te_raw = train_test_split(
        x, y, test_size=0.2, random_state=SPLIT_SEED
    )
    x_tr = np.asarray(x_tr_raw, dtype=np.float64)
    x_te = np.asarray(x_te_raw, dtype=np.float64)
    y_tr = np.asarray(y_tr_raw, dtype=np.int64)
    y_te = np.asarray(y_te_raw, dtype=np.int64)

    # Min-max scale using train split statistics
    lo = x_tr.min(0)
    span = x_tr.max(0) - lo
    span[span == 0] = 1.0
    x_tr = (x_tr - lo) / span
    x_te = np.clip((x_te - lo) / span, 0.0, 1.0)

    model = BreastCancerMLP()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(x_tr.astype(np.float32))
    yt = torch.from_numpy(y_tr)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model, x_tr, y_tr, x_te, y_te


def export_onnx(model: BreastCancerMLP) -> None:
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
        per_channel=PER_CHANNEL,
        max_mult_bits=MAX_MULT_BITS,
        calibration=CALIBRATION,
    )

    in_scale = fmodel.input_scale
    assert in_scale is not None
    x_te_flat = x_te.astype(np.float64)
    x_te_q = np.clip(np.round(x_te_flat / in_scale), 0, (1 << INPUT_BITS) - 1).astype(np.int64)

    in_name = graph.inputs[0]
    out_name = graph.outputs[0]

    logits_q = np.array(
        [evaluate_graph_int(graph, {in_name: row.tolist()})[out_name] for row in x_te_q]
    )
    labels_q = logits_q.argmax(1)

    def float_predict(features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.from_numpy(features.astype(np.float32))
            return model(xt).argmax(1).numpy()

    def quant_predict(features: np.ndarray) -> np.ndarray:
        flat = features.astype(np.float64)
        q = np.clip(np.round(flat / in_scale), 0, (1 << INPUT_BITS) - 1).astype(np.int64)
        preds = [
            int(np.argmax(evaluate_graph_int(graph, {in_name: r.tolist()})[out_name])) for r in q
        ]
        return np.array(preds)

    report = accuracy_report(float_predict, quant_predict, x_te, y_te)

    x_batch_q = x_te_q[:N_TEST]
    labels_batch = labels_q[:N_TEST]

    fixture = {
        "_comment": (
            "Phase-11 tabular example: a PyTorch MLP (30→8→ReLU→2) trained on min-max-scaled "
            "Wisconsin Breast Cancer features (sklearn load_breast_cancer, same split as "
            "examples/trees/), exported to breast_cancer_mlp.onnx, loaded through "
            "penumbra.load_onnx and quantized via penumbra.Model.quantize. The serialized IR "
            "graph is under 'graph'; the 2 logits are the graph output and the client argmaxes "
            "them. FHE output must equal these quantized-cleartext logits/labels bit-for-bit "
            "under TFHE, and stay within PHASE11_TABULAR_MLP under CKKS."
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
        f"  architecture       = Linear({IN_FEATURES}->{HIDDEN}) -> ReLU -> "
        f"Linear({HIDDEN}->{N_CLASSES})"
    )
    cap_bits = fhe.radix_capacity_bits(graph.num_blocks)
    print(f"  num_blocks         = {graph.num_blocks} ({cap_bits}-bit radix)")
    print(f"  float accuracy     = {report.float_accuracy:.4f}")
    print(f"  quantized accuracy = {report.quantized_accuracy:.4f}  (gap {report.gap:+.4f})")
    print(f"  test batch         = {len(labels_batch)} samples")


if __name__ == "__main__":
    main()
