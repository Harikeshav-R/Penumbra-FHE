"""Phase-8 example: train a branching PyTorch MLP, export to ONNX, load and quantize.

Exercises the branching graph pipeline (IR schema 0.9.0):
    x(64) -> fc1 -> ReLU -> h1(16)
    h1 -> Split([8, 8]) -> s0, s1
    s0 -> fc2a(8->8) -> ReLU -> a(8)
    s1 -> fc2b(8->8) -> ReLU -> b(8)
    c = Concat(a, b) (16)
    r = Add(h1, c)   (16)   # fan-out: h1 feeds Split and Add
    logits = fc3(16->10)

Regenerate artifacts:
    uv run --extra ml --system-certs python examples/mnist/branch_mlp_export.py
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
from penumbra.ir import ConcatSpec, SplitSpec
from penumbra.quantization import accuracy_report
from penumbra.reference import evaluate_graph_int

# --- Configuration -------------------------------------------------------------------
IN_FEATURES = 64  # 8x8 flattened digits
HIDDEN = 16
N_CLASSES = 10

INPUT_BITS = 4
WEIGHT_BITS = (5, 5, 5, 5)  # (fc1, fc2a, fc2b, fc3)
ACT_BITS = 2
MAX_MULT_BITS = 5
N_TEST = 2
EPOCHS = 150
SEED = 0

ONNX_PATH = Path(__file__).resolve().parent / "digit_branch_mlp.onnx"
FIXTURE_PATH = Path(__file__).resolve().parent / "phase8_branch_fixture.json"


class DigitBranchMLP(nn.Module):
    """Branching MLP with Split, Concat, and residual Add."""

    fc1: nn.Linear
    relu1: nn.ReLU
    fc2a: nn.Linear
    relu2a: nn.ReLU
    fc2b: nn.Linear
    relu2b: nn.ReLU
    fc3: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN)
        self.relu1 = nn.ReLU()
        self.fc2a = nn.Linear(8, 8)
        self.relu2a = nn.ReLU()
        self.fc2b = nn.Linear(8, 8)
        self.relu2b = nn.ReLU()
        self.fc3 = nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.relu1(self.fc1(x))
        s0, s1 = torch.split(h1, [8, 8], dim=1)
        a = self.relu2a(self.fc2a(s0))
        b = self.relu2b(self.fc2b(s1))
        c = torch.cat([a, b], dim=1)
        r = h1 + c
        return self.fc3(r)


def train() -> tuple[DigitBranchMLP, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the float MLP on real 8x8 digits; return model and splits."""
    torch.manual_seed(SEED)
    digits = load_digits()
    images = digits.images
    target = digits.target
    x = np.asarray(images, dtype=np.float32).reshape(-1, IN_FEATURES)
    y = np.asarray(target, dtype=np.int64)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)

    model = DigitBranchMLP()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.01)
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


def export_onnx(model: DigitBranchMLP) -> None:
    """Export the trained MLP to a committed .onnx (opset 13)."""
    dummy = torch.zeros(1, IN_FEATURES)
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

    # Load through the ONNX front door
    fmodel = fhe.load_onnx(str(ONNX_PATH), input_bits=INPUT_BITS)
    graph = fmodel.quantize(
        cal,
        n_bits=list(WEIGHT_BITS),
        act_bits=ACT_BITS,
        per_channel=False,
        max_mult_bits=MAX_MULT_BITS,
        calibration="mse",
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

    # Verify Split, Concat, and Add nodes exist in the emitted IR
    assert any(isinstance(n.op, SplitSpec) for n in graph.nodes), "SplitSpec must be present"
    assert any(isinstance(n.op, ConcatSpec) for n in graph.nodes), "ConcatSpec must be present"
    assert any(n.op.op_type == "Add" for n in graph.nodes), "AddSpec must be present"

    fixture = {
        "_comment": (
            "Phase-8 example: a branching PyTorch MLP with Split, Concat, and residual Add, "
            "trained on real 8x8 handwritten digits (sklearn load_digits), exported to "
            "digit_branch_mlp.onnx, loaded through the ONNX front door (penumbra.load_onnx) "
            "and quantized via penumbra.Model.quantize. The serialized IR graph is under 'graph'; "
            "the 10 logits are the graph output. FHE output must equal these quantized-cleartext "
            "logits/labels bit-for-bit."
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
        f"  architecture       = Linear({IN_FEATURES}->{HIDDEN}) -> Split([8,8]) -> "
        f"Linear(8->8) + Linear(8->8) -> Concat(16) -> Add(16) -> Linear(16->{N_CLASSES})"
    )
    cap_bits = fhe.radix_capacity_bits(graph.num_blocks)
    print(f"  num_blocks         = {graph.num_blocks} ({cap_bits}-bit radix)")
    print(f"  float accuracy     = {report.float_accuracy:.4f}")
    print(f"  quantized accuracy = {report.quantized_accuracy:.4f}  (gap {report.gap:+.4f})")
    print(f"  test batch         = {len(labels_batch)} samples")


if __name__ == "__main__":
    main()
