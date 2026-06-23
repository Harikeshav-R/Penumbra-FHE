"""Phase-2 example: train -> quantize -> export a binary classifier fixture.

This is the Layer-3 producer side of the Phase-2 golden slice (ROADMAP Phase 2). It trains
a logistic-regression classifier on a **2-class** problem, quantizes it by hand (Phase 5
automates this), and writes ``phase2_fixture.json`` — pure data (quantized weights, bias,
threshold, an activation LUT, and a batch of quantized test inputs with expected labels).
The Rust runtime hand-assembles the op graph from this fixture and the golden test asserts
FHE output == quantized-cleartext output, bit-for-bit (``AGENTS.md`` §1.1).

The fixture is **committed**, so CI never needs to retrain or hit the network — it just
reads the integers. Regenerate it only when the example changes::

    cd python && uv run python ../examples/mnist/train_quantize_export.py

Dataset note: Phase 2's goal is to prove the pipeline + the golden invariant, not model
accuracy. To keep this generator hermetic and dependency-light (NumPy only — no torch /
sklearn / network), it uses a **deterministic synthetic 8x8 two-blob dataset** and a small
hand-rolled logistic-regression trainer (gradient descent). The op graph
(``Linear -> Argmax``, plus a standalone ``Activation`` LUT) and the integer arithmetic are
identical to what a real MNIST 0-vs-1 model would produce; swapping in a real dataset +
trained model is a drop-in change once the ML stack is installable. This is flagged so the
synthetic stand-in is not mistaken for the eventual MNIST example (ROADMAP Phase 2/4).

Bit-width budget (``PROJECT.md`` §9): 64 features, inputs 4-bit unsigned (max 15), weights
4-bit signed (min -8). A single term reaches ``|15 * -8| = 120`` (~7 bits); summing 64 of
them reaches ~7680 (~13 bits). The quantized bias can independently be larger than the
summed products, so the accumulator width is ``max(sum_bits, bias_bits) + 1`` — here the
bias is ~11 bits, giving ~14 bits total, which still fits comfortably in a 16-bit signed
radix (``num_blocks = 8`` x 2 message bits). The fixture records ``num_blocks`` so the
runtime keygen uses the same budget, and ``Linear::output_bits`` enforces this same rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from penumbra.quantization import linear_logit_int, symmetric_spec

# --- Configuration (the only knobs) -----------------------------------------------------
N_FEATURES = 64  # 8x8, the MNIST-downsample feature count Phase 2 targets
INPUT_BITS = 4  # small non-negative input range
WEIGHT_BITS = 4  # signed weights, range -8..7
NUM_BLOCKS = 8  # 16-bit signed radix under PARAM_MESSAGE_2_CARRY_2 (the budget)
N_TRAIN = 400
N_TEST = 4  # committed test batch — small on purpose: each FHE sample is ~30s in CI
SEED = 0

FIXTURE_PATH = Path(__file__).resolve().parent / "phase2_fixture.json"


def make_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic two-blob dataset in a non-negative pixel-like range [0, 16].

    Class 0 and class 1 differ by a fixed offset on a subset of features, so a linear
    classifier separates them well — enough to exercise the pipeline meaningfully.
    """
    n = N_TRAIN + 256
    y = rng.integers(0, 2, size=n)
    base = rng.uniform(2.0, 6.0, size=(n, N_FEATURES))
    # Push the two classes apart on the first half of the features.
    signal = np.zeros((n, N_FEATURES))
    signal[:, : N_FEATURES // 2] = y[:, None] * 5.0
    x = np.clip(base + signal + rng.normal(0, 1.0, size=(n, N_FEATURES)), 0.0, 16.0)
    return x, y.astype(np.int64)


def train_logreg(x: np.ndarray, y: np.ndarray, *, iters: int = 2000, lr: float = 0.05):
    """Tiny hand-rolled logistic regression (full-batch gradient descent).

    Returns ``(weights, bias)`` for a single logit. Standardizing inputs internally keeps
    the gradient well-conditioned; the learned weights are mapped back to raw-input space
    so the exported model consumes the same pixel domain the test inputs are quantized in.
    """
    mu, sigma = x.mean(0), x.std(0) + 1e-8
    xs = (x - mu) / sigma
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(iters):
        z = xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad = p - y
        w -= lr * (xs.T @ grad) / len(y)
        b -= lr * float(grad.mean())
    # Map weights/bias from standardized space back to raw-input space:
    # z = ((x - mu)/sigma) . w + b = x . (w/sigma) + (b - sum(mu*w/sigma)).
    w_raw = w / sigma
    b_raw = b - float(np.sum(mu * w_raw))
    return w_raw, b_raw


def main() -> None:
    rng = np.random.default_rng(SEED)
    x, y = make_dataset(rng)
    x_tr, y_tr = x[:N_TRAIN], y[:N_TRAIN]
    x_te, y_te = x[N_TRAIN:], y[N_TRAIN:]

    w_f, b_f = train_logreg(x_tr, y_tr)

    # --- Quantize (symmetric per-tensor PTQ) --------------------------------------------
    # Inputs are non-negative; calibrate the scale on the training data. Weights are
    # signed. Both quantize into the same integer domain the FHE Linear op computes in.
    x_spec = symmetric_spec(x_tr, INPUT_BITS, signed=False)
    w_spec = symmetric_spec(w_f, WEIGHT_BITS, signed=True)

    w_q = w_spec.quantize(w_f)  # (64,) signed ints
    # The accumulator lives in units of (input_scale * weight_scale); the bias must too.
    acc_scale = x_spec.scale * w_spec.scale
    bias_q = int(np.round(b_f / acc_scale))

    x_te_q = x_spec.quantize(x_te)  # (n_te, 64) ints

    # Decision: float logit (w_f . x + b_f) >= 0  <=>  int logit (w_q . x_q + bias_q) >= 0.
    # So the integer threshold is simply 0.
    threshold = 0

    # --- Cleartext quantized oracle (this is what FHE must match bit-for-bit) -----------
    logits_q = linear_logit_int(x_te_q, w_q, bias_q)  # (n_te,)
    labels_q = (logits_q >= threshold).astype(np.int64)

    # Trim to a fixed-size committed batch.
    x_batch = x_te_q[:N_TEST]
    labels_batch = labels_q[:N_TEST]

    # Report the quantization gap honestly (float vs quantized-cleartext accuracy).
    def float_predict(xb: np.ndarray) -> np.ndarray:
        return ((xb @ w_f + b_f) >= 0).astype(np.int64)

    acc_float = float(np.mean(float_predict(x_te) == y_te))
    acc_quant = float(np.mean(labels_q == y_te))

    # --- A narrow-domain activation LUT, exercised standalone by the golden test --------
    # ReLU-like clamp over the 2-bit message space (0..3): proves the Activation(LUT)/PBS
    # path bit-exactly (the binary decision itself uses the threshold, not this LUT).
    message_space = 1 << 2  # PARAM_MESSAGE_2_CARRY_2 message modulus
    relu_lut = [min(v, 2) for v in range(message_space)]  # clamp at 2

    fixture = {
        "_comment": (
            "Phase-2 golden-test fixture. Pure data; the Rust runtime assembles the op "
            "graph. FHE output must equal these quantized-cleartext labels bit-for-bit. "
            "Synthetic dataset stand-in (see train_quantize_export.py)."
        ),
        "num_blocks": NUM_BLOCKS,
        "input_bits": INPUT_BITS,
        "weight_bits": WEIGHT_BITS,
        "scales": {"input": x_spec.scale, "weight": w_spec.scale, "acc": acc_scale},
        "accuracy": {"float": acc_float, "quantized": acc_quant},
        "linear": {"weights": [w_q.tolist()], "bias": [bias_q]},
        "argmax": {"threshold": threshold},
        "activation": {
            "lut": relu_lut,
            "output_bits": 2,
            "test_inputs": list(range(message_space)),
            "expected": [relu_lut[v] for v in range(message_space)],
        },
        "test_inputs": x_batch.tolist(),
        "expected_labels": labels_batch.tolist(),
    }

    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH}")
    print(f"  float accuracy     = {acc_float:.4f}")
    print(f"  quantized accuracy = {acc_quant:.4f}")
    print(
        f"  test batch         = {len(labels_batch)} samples, num_blocks={NUM_BLOCKS}"
    )


if __name__ == "__main__":
    main()
