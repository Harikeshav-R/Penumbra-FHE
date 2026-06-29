"""Phase-5 example: train a real torch CNN on real handwritten digits, quantize via the service.

This is the first example that uses a **real dataset** and a **real PyTorch model** end to end,
exercising the Phase-5 quantization service (``penumbra.Model.quantize``) rather than hand-rolled
quantization. It trains a small CNN on scikit-learn's bundled **8x8 handwritten-digit** dataset
(``sklearn.datasets.load_digits`` — the UCI digits set, real pen-written digits, no network),
quantizes it through the library, and writes ``phase5_digits_fixture.json``. The Rust runtime
deserializes the IR graph and the golden test asserts FHE output == quantized-cleartext output,
bit-for-bit (``AGENTS.md`` §1.1).

Honest scope: this is **8x8 digits**, not 28x28 MNIST. The small input is deliberate — each FHE
sample is minutes (the cost is the per-activation bootstraps, ``PROJECT.md`` §5), so a 28x28 conv
model would be impractical for a committed golden test. 8x8 digits are real handwritten data in
the same [0, 16] pixel domain the synthetic examples used, so this is a genuine "train a real
model, run it encrypted" slice at a feasible size (ROADMAP Phase 5/6).

The model is::

    Conv2d(1 -> 12, 3x3, stride 2)  ->  [auto Requant + ReLU]  ->  Linear(108 -> 10)

(the stride-2 conv shrinks 8x8 to a 3x3 feature map, so no separate pooling op is needed). The 10
logits are the graph output; the client decrypts them and argmaxes (``PROJECT.md`` §11).

The fixture is **committed**, so CI never retrains or imports torch — it just reads the integers
(the hermetic-fixture discipline, like the other examples). Torch + scikit-learn are the optional
``ml`` extra; regenerate only when the example changes::

    cd python && uv run --extra ml --system-certs python ../examples/mnist/real_digits_export.py

Accuracy is honest, not headline: ~0.96 float drops to ~0.69 quantized — the cost of capping
activations at a single 2-bit block (``MESSAGE_BITS``). The QAT example (``qat_export.py``)
recovers much of that gap by training with the quantization simulated in the loop.

Quantization is the library's job here: after training the float CNN, ``Model.quantize`` calibrates
on the training set, quantizes weights/bias, fuses the ReLU into the conv's Requant, sizes the
radix, and self-verifies — no manual scale math in this file. The committed ``expected_logits`` /
``expected_labels`` are the quantized-integer oracle's output, which the FHE path must match.
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

# --- Configuration (the only knobs) -----------------------------------------------------
IN_H = IN_W = 8  # load_digits is 8x8
IN_CH = 1
N_CLASSES = 10
KERNEL = 3
STRIDE = 2  # stride-2 conv shrinks the feature map to 3x3, keeping the bootstrap count feasible
# 12 learned conv filters. Activations are capped at a single 2-bit block (MESSAGE_BITS), so each
# feature is brutally low-precision; accuracy comes from having *many* features rather than
# precise ones — the realistic lever within the FHE budget. The cost is ~1 requant bootstrap per
# post-conv activation = CONV_CH * conv_positions = 12 * 3 * 3 = 108 PBS/sample (~3x the Phase-4
# golden). 12ch reaches ~0.85 quantized on real digits; fewer channels drops accuracy sharply
# (8ch -> ~0.64), more channels costs proportionally more FHE time for marginal gain.
CONV_CH = 12

INPUT_BITS = 4  # inputs quantized into [0, 15] (digits pixels are already [0, 16])
WEIGHT_BITS = 4
ACT_BITS = 2  # post-Requant activations land in a single 2-bit block

N_TEST = 2  # committed FHE test batch — tiny on purpose (each sample is minutes in CI)
EPOCHS = 150
SEED = 0

FIXTURE_PATH = Path(__file__).resolve().parent / "phase5_digits_fixture.json"

OUT_H = (IN_H - KERNEL) // STRIDE + 1  # 3
OUT_W = (IN_W - KERNEL) // STRIDE + 1  # 3
N_FEATURES = CONV_CH * OUT_H * OUT_W  # 12 * 3 * 3 = 108 (no pooling: stride-2 already shrinks)


class DigitCNN(nn.Module):
    """Conv (stride 2) -> ReLU -> linear, matching the narrow-waist op vocabulary.

    Deliberately mirrors the FHE op graph: a single strided conv (no bias, like the Phase-4
    example so the accumulator stays narrow), ReLU, then a dense head. The stride-2 conv already
    shrinks the 8x8 input to a 3x3 feature map, so no separate pooling op is needed — this keeps
    the per-activation bootstrap count (and FHE latency) feasible. The quantization service maps
    each module to its IR op.
    """

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


def main() -> None:
    model, x_tr, y_tr, x_te, y_te = train()

    # --- Lift the trained float weights into the library's float Model (Layer 3) -----------
    # The float CNN and the int IR pipeline are the *same* function (strided conv -> ReLU ->
    # dense), so the float-vs-quantized accuracy gap is an honest quantization measurement. The
    # quantization service calibrates the conv accumulator range and chooses the fused-ReLU
    # Requant rescale that maps post-conv activations into the 2-bit domain the head reads.
    conv_w = model.conv.weight.detach().numpy().astype(np.float64)  # (CONV_CH, 1, 3, 3)
    fc_w = model.fc.weight.detach().numpy().astype(np.float64)  # (10, N_FEATURES)
    fc_b = model.fc.bias.detach().numpy().astype(np.float64)  # (10,)

    fmodel = fhe.Model(
        [
            fhe.Conv2d(weight=conv_w, in_h=IN_H, in_w=IN_W, in_channels=IN_CH, stride=STRIDE),
            fhe.Activation(lambda v: max(v, 0.0)),  # ReLU, fused into the conv's Requant
            fhe.Linear(weight=fc_w, bias=fc_b),
        ],
        input_bits=INPUT_BITS,
    )

    # Calibration data: the training images, flattened to the model's input layout.
    cal = x_tr.reshape(len(x_tr), -1).astype(np.float64)
    # per_channel weight scales recover meaningful accuracy at 2-bit activations (ROADMAP P5).
    graph = fmodel.quantize(cal, n_bits=WEIGHT_BITS, act_bits=ACT_BITS, per_channel=True)

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
            "Phase-5 example: a real torch CNN trained on real 8x8 handwritten digits "
            "(sklearn load_digits), quantized through penumbra.Model.quantize. The model is the "
            "serialized IR graph under 'graph'; the 10 logits are the graph output and the client "
            "argmaxes them. FHE output must equal these quantized-cleartext logits/labels "
            "bit-for-bit. 8x8 digits (not 28x28 MNIST) keep FHE latency feasible (see "
            "real_digits_export.py)."
        ),
        "graph": graph.to_dict(),
        "scales": {"input": in_scale},
        "accuracy": {"float": report.float_accuracy, "quantized": report.quantized_accuracy},
        "test_inputs": x_batch_q.tolist(),
        "expected_labels": labels_batch.tolist(),
        "expected_logits": logits_q[:N_TEST].tolist(),
    }
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")

    print(f"wrote {FIXTURE_PATH}")
    print(
        f"  architecture       = Conv2d(1->{CONV_CH},{KERNEL}x{KERNEL},stride{STRIDE}) "
        f"-> Requant+ReLU -> Linear({N_FEATURES}->{N_CLASSES})"
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
