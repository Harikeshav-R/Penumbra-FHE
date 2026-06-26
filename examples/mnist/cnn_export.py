"""Phase-4 example: train -> quantize -> export a small 10-class CNN fixture.

The Layer-3 producer side of the Phase-4 golden slice (ROADMAP Phase 4). It builds a tiny
convolutional classifier for a **10-class** problem, quantizes it by hand (Phase 5 automates
this), runs the **automatic Requant insertion** compile pass (:func:`penumbra.insert_requants`),
and writes ``phase4_cnn_fixture.json``. The Rust runtime deserializes the IR graph under the
``"graph"`` key and walks it; the golden test asserts FHE output == quantized-cleartext output,
bit-for-bit (``AGENTS.md`` §1.1).

The model is::

    Conv2d(1 -> C, 3x3, stride 1)  ->  [auto Requant + ReLU]  ->  Pool(avg 2x2)  ->  Linear(-> 10)

and the **10 logits are the graph output**: the client decrypts them and takes the argmax
(``PROJECT.md`` §11). No wide-domain in-FHE argmax is needed this phase — the 2-class ``Argmax``
op is unchanged, and the privacy story holds (the server only ever sees ciphertext; the client
would learn the label anyway).

The fixture is **committed**, so CI never retrains or hits the network — it just reads the
integers. Regenerate it only when the example changes::

    cd python && uv run python ../examples/mnist/cnn_export.py

Dataset note (same discipline as the Phase-2 example): to stay hermetic and dependency-light
(NumPy only — no torch / sklearn / network), this uses a **deterministic synthetic 10-class**
dataset of per-class template images plus noise, fixed 3x3 edge-detector conv filters, and a
hand-rolled softmax-regression head trained on the pooled features. The op graph and the
integer arithmetic are identical to what a real MNIST CNN would produce; swapping in a real
dataset + trained CNN is a drop-in change once the ML stack is wired up (ROADMAP Phase 5/6).

Quantization is integer-consistent by construction: inputs, conv filters, and head weights are
symmetric-quantized; the Requant ``shift`` is chosen by **calibration** (the observed conv-
accumulator magnitude on training data) so post-ReLU activations land in the single-block
``[0, 2^out_bits-1]`` domain. The quantized-integer pipeline is the oracle the FHE path must
match bit-for-bit; argmax of the quantized logits is what the fixture commits as the label.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from penumbra import insert_requants, propagate_bit_widths, radix_capacity_bits
from penumbra.bitwidth import MESSAGE_BITS
from penumbra.ir import SCHEMA_VERSION, Conv2dSpec, Graph, LinearSpec, Node, PoolSpec
from penumbra.quantization import (
    quantize_conv,
    quantize_linear_integer_input,
    symmetric_spec,
)

# --- Configuration (the only knobs) -----------------------------------------------------
IN_H = IN_W = 6  # 6x6 single-channel input (small, to keep FHE latency feasible)
IN_CH = 1
N_CLASSES = 10
KERNEL = 3  # 3x3 conv, stride 1, no padding -> 4x4 feature map
CONV_CH = 2  # two fixed edge-detector filters
POOL = 2  # 2x2 average pool, stride 2 -> 2x2 per channel

INPUT_BITS = 4  # inputs in [0, 15]
WEIGHT_BITS = 4  # signed conv/linear weights in [-8, 7]
ACT_BITS = MESSAGE_BITS  # post-Requant activations land in a single 2-bit block [0, 3]

N_TRAIN = 600
N_TEST = 2  # committed FHE test batch — small on purpose (each sample is minutes in CI)
SEED = 0

FIXTURE_PATH = Path(__file__).resolve().parent / "phase4_cnn_fixture.json"

# Two fixed 3x3 edge detectors (vertical and horizontal gradients) — generic features that
# do not need training; only the linear head is trained.
CONV_FILTERS = np.array(
    [
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],  # vertical edge
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],  # horizontal edge
    ],
    dtype=np.float64,
)

OUT_H = (IN_H - KERNEL) + 1
OUT_W = (IN_W - KERNEL) + 1
POOL_H = OUT_H // POOL
POOL_W = OUT_W // POOL
N_FEATURES = CONV_CH * POOL_H * POOL_W  # flattened pooled-feature length feeding the head


def make_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 10-class dataset: per-class template images plus low noise, in [0, 16].

    Each class has a fixed random template; samples are the template plus small Gaussian
    noise, clipped to the pixel range. With low noise the classes are well separated, so the
    conv+pool+linear pipeline (and its quantized integer form) classify them meaningfully —
    enough to exercise the slice and report an honest quantization gap.
    """
    templates = rng.uniform(0.0, 16.0, size=(N_CLASSES, IN_H, IN_W))
    n = N_TRAIN + 256
    y = rng.integers(0, N_CLASSES, size=n)
    x = np.clip(templates[y] + rng.normal(0.0, 1.0, size=(n, IN_H, IN_W)), 0.0, 16.0)
    return x, y.astype(np.int64)


def conv2d_valid(images: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Float 'valid' 2-D convolution of (N, H, W) images by (C, kh, kw) filters.

    Returns (N, C, OUT_H, OUT_W). Plain reference (correlation, the conv convention used by
    the runtime op): no flipping, matching how the quantized op walks the kernel taps.
    """
    n = images.shape[0]
    out = np.zeros((n, filters.shape[0], OUT_H, OUT_W))
    for c in range(filters.shape[0]):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                patch = images[:, oy : oy + KERNEL, ox : ox + KERNEL]
                out[:, c, oy, ox] = np.einsum("nij,ij->n", patch, filters[c])
    return out


def avgpool_sum(feature: np.ndarray) -> np.ndarray:
    """Sum-pool (N, C, OUT_H, OUT_W) over POOLxPOOL windows -> (N, C, POOL_H, POOL_W).

    The op emits the window **sum** (the /k averaging is folded into the next layer's scale),
    so the reference sums too — keeping cleartext and FHE identical.
    """
    n, c = feature.shape[:2]
    out = np.zeros((n, c, POOL_H, POOL_W))
    for py in range(POOL_H):
        for px in range(POOL_W):
            window = feature[:, :, py * POOL : py * POOL + POOL, px * POOL : px * POOL + POOL]
            out[:, :, py, px] = window.sum(axis=(2, 3))
    return out


def softmax_train(feats: np.ndarray, y: np.ndarray, *, iters: int = 4000, lr: float = 0.1):
    """Hand-rolled multiclass softmax-regression head (full-batch gradient descent).

    Trained on the (float) pooled features so the learned head is a real classifier; it is
    quantized afterwards. Standardizing features internally keeps the gradient well-conditioned;
    the learned weights are mapped back to the raw-feature space the integer pipeline uses.
    """
    mu, sigma = feats.mean(0), feats.std(0) + 1e-8
    fs = (feats - mu) / sigma
    n, d = fs.shape
    w = np.zeros((d, N_CLASSES))
    b = np.zeros(N_CLASSES)
    onehot = np.eye(N_CLASSES)[y]
    for _ in range(iters):
        logits = fs @ w + b
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        grad = p - onehot
        w -= lr * (fs.T @ grad) / n
        b -= lr * grad.mean(0)
    # Map standardized-space weights back to raw-feature space:
    # logits = ((f - mu)/sigma) . w + b = f . (w/sigma) + (b - sum(mu*w/sigma)).
    w_raw = w / sigma[:, None]
    b_raw = b - mu @ w_raw
    return w_raw, b_raw


def main() -> None:
    rng = np.random.default_rng(SEED)
    x, y = make_dataset(rng)
    x_tr, y_tr = x[:N_TRAIN], y[:N_TRAIN]
    x_te, y_te = x[N_TRAIN:], y[N_TRAIN:]

    # --- Quantize inputs and the fixed conv filters via the quantization service --------
    # The conv quantizer takes (out_ch, in_ch, kh, kw) float kernels and returns the IR's flat
    # [out_ch][in_ch*kh*kw] int layout; reshaped back to (CONV_CH, 3, 3) for the float-conv
    # reference. (Reproduces the former inline symmetric PTQ; pinned by the ptq tests.)
    x_spec = symmetric_spec(x_tr, INPUT_BITS, signed=False)
    w1_q_flat, _, w1_spec = quantize_conv(CONV_FILTERS[:, None, :, :], bits=WEIGHT_BITS)
    w1_q = w1_q_flat.reshape(CONV_CH, KERNEL, KERNEL)  # (CONV_CH, 3, 3) signed ints

    x_tr_q = x_spec.quantize(x_tr)  # (N_TRAIN, H, W) ints
    x_te_q = x_spec.quantize(x_te)

    # --- Integer conv accumulator (no bias on the conv) ---------------------------------
    conv_tr = conv2d_valid(x_tr_q.astype(np.float64), w1_q.astype(np.float64)).astype(np.int64)
    conv_te = conv2d_valid(x_te_q.astype(np.float64), w1_q.astype(np.float64)).astype(np.int64)

    # --- Calibrate the Requant shift: pick shift so the largest post-ReLU conv value over
    # the training set maps to the top of the [0, 2^ACT_BITS-1] activation domain. This is
    # exactly the rescale the quantization service would choose; insert_requants uses it. ---
    relu_max = int(max(1, conv_tr.clip(min=0).max()))
    act_ceiling = (1 << ACT_BITS) - 1  # 3
    # shift s such that relu_max >> s <= act_ceiling, i.e. s = ceil(log2((relu_max+1)/(ceil+1))).
    shift = 0
    while (relu_max >> shift) > act_ceiling:
        shift += 1

    def requant(acc: np.ndarray) -> np.ndarray:
        # Mirror the runtime op exactly: clamp(max(acc >> shift, 0), 0, 2^ACT_BITS - 1).
        return np.clip(np.maximum(acc >> shift, 0), 0, act_ceiling)

    act_tr = requant(conv_tr)
    act_te = requant(conv_te)

    # --- Pool (sum) and flatten to the head's feature vector ----------------------------
    pooled_tr = avgpool_sum(act_tr.astype(np.float64)).reshape(len(x_tr), -1)
    pooled_te = avgpool_sum(act_te.astype(np.float64)).reshape(len(x_te), -1)

    # --- Train the float head on the integer pooled features, then quantize it ----------
    # The head consumes the *integer* pooled features directly (effective input scale 1), so the
    # integer-input quantizer shares the weight scale for the bias — argmax-preserving. One row
    # per class: (N_CLASSES, N_FEATURES). (Reproduces the former inline math; ptq tests pin it.)
    w2_f, b2_f = softmax_train(pooled_tr, y_tr)
    w2_q, b2_q, w2_spec = quantize_linear_integer_input(w2_f.T, b2_f, bits=WEIGHT_BITS)

    # --- Quantized-integer pipeline = the oracle. Compute labels on the test set. -------
    pooled_te_i = pooled_te.astype(np.int64)
    logits_q = pooled_te_i @ w2_q.T + b2_q  # (n_te, N_CLASSES)
    labels_q = logits_q.argmax(1).astype(np.int64)

    # Honest accuracy reporting: float pipeline vs quantized-integer pipeline.
    logits_f = pooled_te @ w2_f + b2_f
    labels_f = logits_f.argmax(1)
    acc_float = float(np.mean(labels_f == y_te))
    acc_quant = float(np.mean(labels_q == y_te))

    # --- Build the IR graph (no Requant yet) and run the automatic insertion pass -------
    # Conv weights flatten to [out_ch][in_ch*kh*kw] = [CONV_CH][9]; the runtime walks taps in
    # in-channel / kernel-row / kernel-col order, which for a single in-channel is row-major.
    conv_weights = [w1_q[c].reshape(-1).tolist() for c in range(CONV_CH)]
    raw = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=1,  # placeholder; set below once widths are known
        input_bits=INPUT_BITS,
        inputs=["x"],
        outputs=["logits"],
        nodes=[
            Node(
                name="conv",
                inputs=["x"],
                outputs=["conv_acc"],
                op=Conv2dSpec(
                    weights=conv_weights,
                    bias=[0] * CONV_CH,
                    weight_bits=WEIGHT_BITS,
                    in_h=IN_H,
                    in_w=IN_W,
                    in_channels=IN_CH,
                    kernel_h=KERNEL,
                    kernel_w=KERNEL,
                    stride=1,
                    padding=0,
                ),
            ),
            Node(
                name="pool",
                inputs=["conv_acc"],
                outputs=["pooled"],
                op=PoolSpec(
                    mode="avg",
                    in_h=OUT_H,
                    in_w=OUT_W,
                    channels=CONV_CH,
                    pool_h=POOL,
                    pool_w=POOL,
                    stride=POOL,
                ),
            ),
            Node(
                name="head",
                inputs=["pooled"],
                outputs=["logits"],
                op=LinearSpec(weights=w2_q.tolist(), bias=b2_q.tolist(), weight_bits=WEIGHT_BITS),
            ),
        ],
    )

    # Size num_blocks to the widest tensor of the *requant-inserted* graph, then re-run the
    # pass against that real budget. We first insert with a generous radix to learn the widths.
    probe = insert_requants(replace(raw, num_blocks=64), shifts={"conv": shift})
    max_bits = max(propagate_bit_widths(probe).values())
    num_blocks = (max_bits + MESSAGE_BITS - 1) // MESSAGE_BITS
    graph = insert_requants(replace(raw, num_blocks=num_blocks), shifts={"conv": shift})
    assert max(propagate_bit_widths(graph).values()) <= radix_capacity_bits(num_blocks)

    # --- Committed test batch (quantized inputs flattened channel-major, row-major) -----
    x_batch_q = x_te_q[:N_TEST].reshape(N_TEST, -1)  # (N_TEST, IN_CH*IN_H*IN_W)
    labels_batch = labels_q[:N_TEST]

    fixture = {
        "_comment": (
            "Phase-4 golden-test fixture: a small 10-class CNN "
            "(Conv2d -> auto Requant+ReLU -> avg Pool -> Linear). The model is the serialized "
            "IR graph under 'graph' (the Rust runtime deserializes and walks it). The 10 logits "
            "are the graph output; the client argmaxes them. FHE output must equal these "
            "quantized-cleartext logits/labels bit-for-bit. Synthetic dataset stand-in "
            "(see cnn_export.py)."
        ),
        "graph": graph.to_dict(),
        "scales": {
            "input": x_spec.scale,
            "conv_weight": w1_spec.scale,
            "head_weight": w2_spec.scale,
            "requant_shift": shift,
        },
        "accuracy": {"float": acc_float, "quantized": acc_quant},
        "test_inputs": x_batch_q.tolist(),
        "expected_labels": labels_batch.tolist(),
        "expected_logits": logits_q[:N_TEST].tolist(),
    }

    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH}")
    print(
        f"  architecture       = Conv2d(1->{CONV_CH},{KERNEL}x{KERNEL}) -> Requant(shift={shift})"
        f" -> avgPool{POOL}x{POOL} -> Linear({N_FEATURES}->{N_CLASSES})"
    )
    print(f"  num_blocks         = {num_blocks} ({radix_capacity_bits(num_blocks)}-bit radix)")
    print(f"  float accuracy     = {acc_float:.4f}")
    print(f"  quantized accuracy = {acc_quant:.4f}")
    print(f"  test batch         = {len(labels_batch)} samples")


if __name__ == "__main__":
    main()
