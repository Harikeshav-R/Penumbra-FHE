"""Phase-16 example: train a torch CNN on raw 28x28 MNIST, quantize and export probe fixture.

This script implements Task 4 of Phase 16:
1. Downloads official MNIST IDX gzip resources from https://ossci-datasets.s3.amazonaws.com/mnist/
   and verifies published MD5 checksums pinned from torchvision:
   https://raw.githubusercontent.com/pytorch/vision/v0.23.0/torchvision/datasets/mnist.py
   into ignored data/. Corrupted local data raises an error naming the file and URL.
2. Trains the approved probe architecture:
   Conv2d(1, 4, kernel_size=3, stride=4, bias=False) -> ReLU -> Flatten -> Linear(196, 10).
3. Preserves trained weights under ignored target/phase16/mnist28-trained.pt immediately
   after five epochs. If checkpoint exists, supports resuming export/quantization
   without retraining.
   - seed: 0
   - 60,000 train / 10,000 test official split
   - pixels normalized by 255.0
   - Adam learning rate 1e-3, batch size 128, 5 epochs
   - Quantization: input_bits=3, weight_bits=(5, 6), act_bits=2, max_mult_bits=1,
     per_channel=True, calibration='mse'
   - 128 calibration examples from training split with seed 1507
5. Exports ONNX opset 13 using legacy TorchScript exporter (dynamo=False).
6. Lowers through penumbra.load_onnx and quantizes via Model.quantize.
7. Computes float and quantized test accuracy and generates phase16_mnist28_fixture.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import penumbra as fhe
from penumbra.reference import evaluate_graph_int

# --- Configuration & Constants -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "target" / "phase16" / "mnist28-trained.pt"
FIXTURE_PATH = Path(__file__).resolve().parent / "phase16_mnist28_fixture.json"

MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
PRIMARY_CHECKSUM_SOURCE = (
    "https://raw.githubusercontent.com/pytorch/vision/v0.23.0/torchvision/datasets/mnist.py"
)

MNIST_FILES: dict[str, str] = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Published MD5 checksums pinned from torchvision.datasets.MNIST resources:
# https://raw.githubusercontent.com/pytorch/vision/v0.23.0/torchvision/datasets/mnist.py
MNIST_MD5: dict[str, str] = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

SEED = 0
CALIBRATION_SEED = 1507
CALIBRATION_LIMIT = 128

IN_H = 28
IN_W = 28
IN_CH = 1
KERNEL = 3
STRIDE = 4
CONV_CH = 4
N_CLASSES = 10

OUT_H = (IN_H - KERNEL) // STRIDE + 1  # 7
OUT_W = (IN_W - KERNEL) // STRIDE + 1  # 7
N_FEATURES = CONV_CH * OUT_H * OUT_W  # 4 * 7 * 7 = 196

INPUT_BITS = 3
WEIGHT_BITS = (5, 6)
ACT_BITS = 2
MAX_MULT_BITS = 1

EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


class AccuracyDict(TypedDict):
    float: float
    quantized: float


class AccuracyMetricCountsDict(TypedDict):
    correct: int
    total: int


class AccuracyCountsDict(TypedDict):
    float: AccuracyMetricCountsDict
    quantized: AccuracyMetricCountsDict


class DatasetDict(TypedDict):
    name: str
    source_url: str
    primary_checksum_source: str
    hash_algorithm: str
    checksums: dict[str, str]


class BitPlanDict(TypedDict):
    input_bits: int
    weight_bits: list[int]
    act_bits: int
    max_mult_bits: int


class TrainingProtocolDict(TypedDict):
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    loss: str
    calibration_seed: int
    calibration_samples: int
    train_samples: int
    test_samples: int


class FixtureDict(TypedDict):
    _comment: str
    graph: dict[str, object]
    scales: dict[str, float]
    bit_plan: BitPlanDict
    accuracy: AccuracyDict
    accuracy_counts: AccuracyCountsDict
    sample_ids: list[str]
    calibration_sample_ids: list[int]
    test_inputs: list[list[int]]
    expected_labels: list[int]
    expected_logits: list[list[int]]
    dataset: DatasetDict
    training_protocol: TrainingProtocolDict


class MnistData(NamedTuple):
    x_train: np.ndarray  # (60000, 1, 28, 28) float32 in [0, 1]
    y_train: np.ndarray  # (60000,) int64
    x_test: np.ndarray  # (10000, 1, 28, 28) float32 in [0, 1]
    y_test: np.ndarray  # (10000,) int64


class Mnist28CNN(nn.Module):
    """Conv2d(1, 4, 3, stride=4, bias=False) -> ReLU -> Flatten -> Linear(196, 10)."""

    conv: nn.Conv2d
    fc: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(IN_CH, CONV_CH, KERNEL, stride=STRIDE, bias=False)
        self.fc = nn.Linear(N_FEATURES, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        flat = h.flatten(1)
        out: torch.Tensor = self.fc(flat)
        return out


def compute_file_md5(path: Path) -> str:
    """Compute hex MD5 of a local file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify_resource(filename: str, target_dir: Path) -> Path:
    """Download a file from MNIST_BASE_URL if missing, verifying published MD5.

    Does NOT silently unlink or re-download corrupted local files; propagates corruption
    with full path and URL context.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / filename
    expected_md5 = MNIST_MD5[filename]
    url = f"{MNIST_BASE_URL}{filename}"

    if dest.exists():
        actual_md5 = compute_file_md5(dest)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"Local file {dest} is corrupted (from {url}): "
                f"expected MD5 {expected_md5}, got {actual_md5}. "
                "Refusing to silently overwrite."
            )
        return dest

    print(f"Downloading {url} -> {dest}...")
    try:
        _ = urllib.request.urlretrieve(url, str(dest))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        raise RuntimeError(f"Failed to download {filename} from {url}: {exc}") from exc

    actual_md5 = compute_file_md5(dest)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"Checksum verification failed after download for {dest} from {url}: "
            f"expected MD5 {expected_md5}, got {actual_md5}"
        )
    return dest


def parse_idx_images(path: Path, expected_count: int) -> np.ndarray:
    """Parse gzip compressed IDX3 image file into (N, 1, 28, 28) float32 array in [0, 1]."""
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError(f"Truncated header in {path}")
        magic, num_images, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {path}, expected 2051")
        if num_images != expected_count:
            raise ValueError(
                f"Unexpected image count {num_images} in {path}, expected {expected_count}"
            )
        if rows != IN_H or cols != IN_W:
            raise ValueError(f"Unexpected image dimensions {rows}x{cols} in {path}, expected 28x28")
        data = f.read()
        expected_bytes = num_images * rows * cols
        if len(data) != expected_bytes:
            raise ValueError(
                f"Data byte length mismatch in {path}: expected {expected_bytes}, got {len(data)}"
            )
        raw = np.frombuffer(data, dtype=np.uint8)
        images = (raw.astype(np.float32) / 255.0).reshape(num_images, 1, IN_H, IN_W)
        return images


def parse_idx_labels(path: Path, expected_count: int) -> np.ndarray:
    """Parse gzip compressed IDX1 label file into (N,) int64 array."""
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Truncated header in {path}")
        magic, num_labels = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {path}, expected 2049")
        if num_labels != expected_count:
            raise ValueError(
                f"Unexpected label count {num_labels} in {path}, expected {expected_count}"
            )
        data = f.read()
        if len(data) != num_labels:
            raise ValueError(
                f"Data byte length mismatch in {path}: expected {num_labels}, got {len(data)}"
            )
        raw = np.frombuffer(data, dtype=np.uint8)
        return raw.astype(np.int64)


def load_mnist_dataset(data_dir: Path = DATA_DIR) -> MnistData:
    """Download (if needed) and parse official MNIST dataset files."""
    tr_img_path = download_and_verify_resource(MNIST_FILES["train_images"], data_dir)
    tr_lbl_path = download_and_verify_resource(MNIST_FILES["train_labels"], data_dir)
    te_img_path = download_and_verify_resource(MNIST_FILES["test_images"], data_dir)
    te_lbl_path = download_and_verify_resource(MNIST_FILES["test_labels"], data_dir)

    x_train = parse_idx_images(tr_img_path, 60000)
    y_train = parse_idx_labels(tr_lbl_path, 60000)
    x_test = parse_idx_images(te_img_path, 10000)
    y_test = parse_idx_labels(te_lbl_path, 10000)

    return MnistData(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def select_calibration_indices(
    n_total: int,
    seed: int = CALIBRATION_SEED,
    limit: int = CALIBRATION_LIMIT,
) -> list[int]:
    """Deterministically select calibration indices using SHA-256 rank matching paper protocol."""
    ranked = sorted(
        range(n_total),
        key=lambda i: (
            hashlib.sha256(f"{seed}:mnist28:{i}".encode()).digest(),
            str(i),
        ),
    )
    return ranked[: min(limit, n_total)]


def train_model(data: MnistData, checkpoint_path: Path | None = None) -> Mnist28CNN:
    """Train the CNN model following approved experiment constants and save checkpoint."""
    _ = torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = Mnist28CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.from_numpy(data.x_train), torch.from_numpy(data.y_train))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    _ = model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            _ = optimizer.step()
            total_loss += float(loss.item())
        mean_loss = total_loss / len(loader)
        print(f"  Epoch {epoch + 1}/{EPOCHS} complete - mean loss: {mean_loss:.4f}")

    _ = model.eval()

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Preserved trained weights to {checkpoint_path}")

    return model


def export_and_quantize(
    model: Mnist28CNN,
    data: MnistData,
    out_fixture: Path = FIXTURE_PATH,
) -> FixtureDict:
    """Export to ONNX opset 13, lower through load_onnx, quantize, and write fixture JSON."""
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_onnx:
        onnx_file = tmp_onnx.name
        dummy_input = torch.zeros(1, IN_CH, IN_H, IN_W)
        _ = torch.onnx.export(
            model,
            (dummy_input,),
            onnx_file,
            input_names=["x"],
            output_names=["logits"],
            opset_version=13,
            dynamic_axes=None,
            dynamo=False,
        )

        fmodel = fhe.load_onnx(onnx_file, input_bits=INPUT_BITS)

    # Calibration set: 128 samples selected via CALIBRATION_SEED (1507)
    cal_indices = select_calibration_indices(
        len(data.x_train),
        seed=CALIBRATION_SEED,
        limit=CALIBRATION_LIMIT,
    )
    x_train_flat = data.x_train.reshape(len(data.x_train), -1).astype(np.float64)
    cal_data = x_train_flat[cal_indices]

    graph = fmodel.quantize(
        cal_data,
        n_bits=list(WEIGHT_BITS),
        act_bits=ACT_BITS,
        per_channel=True,
        max_mult_bits=MAX_MULT_BITS,
        calibration="mse",
    )

    assert fmodel.input_scale is not None, "fmodel.input_scale must be initialized by quantize"
    in_scale = float(fmodel.input_scale)
    x_test_flat = data.x_test.reshape(len(data.x_test), -1).astype(np.float64)
    x_test_q = np.clip(np.round(x_test_flat / in_scale), 0, (1 << INPUT_BITS) - 1).astype(np.int64)

    # Evaluate float test accuracy
    _ = model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(data.x_test)
        float_preds_list: list[int] = []
        for i in range(0, len(test_tensor), 500):
            batch = test_tensor[i : i + 500]
            preds = model(batch).argmax(dim=1).cpu().numpy().tolist()
            float_preds_list.extend(preds)
    float_correct = int(np.sum(np.array(float_preds_list) == data.y_test))
    float_total = len(data.y_test)
    float_acc = float(float_correct / float_total)

    # Evaluate quantized test accuracy on full test split
    print("Evaluating quantized test accuracy with evaluate_graph_int...")
    quant_preds_list: list[int] = []
    quant_logits_list: list[list[int]] = []
    in_name = graph.inputs[0]
    out_name = graph.outputs[0]

    for i, row in enumerate(x_test_q):
        out = evaluate_graph_int(graph, {in_name: row.tolist()})
        logits = out[out_name]
        pred_label = int(np.argmax(logits))
        quant_preds_list.append(pred_label)
        if i < 2:
            quant_logits_list.append(logits)

    quant_correct = int(np.sum(np.array(quant_preds_list) == data.y_test))
    quant_total = len(data.y_test)
    quant_acc = float(quant_correct / quant_total)
    print(
        f"Float accuracy: {float_acc:.4f} ({float_correct}/{float_total}), "
        f"Quantized accuracy: {quant_acc:.4f} ({quant_correct}/{quant_total})"
    )

    # Build fixture dict
    sample_ids = ["test_0", "test_1"]
    test_inputs = [row.tolist() for row in x_test_q[:2]]
    expected_labels = quant_preds_list[:2]
    expected_logits = quant_logits_list[:2]

    fixture_content: FixtureDict = {
        "_comment": (
            "Phase-16 scale probe fixture: PyTorch CNN on raw 28x28 MNIST (784 encrypted pixels), "
            "exported to ONNX opset 13, lowered via penumbra.load_onnx, and quantized via "
            "penumbra.Model.quantize. The model is Conv2d(1, 4, 3, stride=4) -> ReLU -> "
            "Linear(196, 10). The 10 logits are graph outputs. Decrypted FHE logits must match "
            "expected_logits bit-for-bit."
        ),
        "graph": graph.to_dict(),
        "scales": {"input": in_scale},
        "bit_plan": {
            "input_bits": INPUT_BITS,
            "weight_bits": list(WEIGHT_BITS),
            "act_bits": ACT_BITS,
            "max_mult_bits": MAX_MULT_BITS,
        },
        "accuracy": {
            "float": float_acc,
            "quantized": quant_acc,
        },
        "accuracy_counts": {
            "float": {
                "correct": float_correct,
                "total": float_total,
            },
            "quantized": {
                "correct": quant_correct,
                "total": quant_total,
            },
        },
        "sample_ids": sample_ids,
        "calibration_sample_ids": [int(i) for i in cal_indices],
        "test_inputs": test_inputs,
        "expected_labels": expected_labels,
        "expected_logits": expected_logits,
        "dataset": {
            "name": "MNIST",
            "source_url": MNIST_BASE_URL,
            "primary_checksum_source": PRIMARY_CHECKSUM_SOURCE,
            "hash_algorithm": "MD5",
            "checksums": MNIST_MD5,
        },
        "training_protocol": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "calibration_seed": CALIBRATION_SEED,
            "calibration_samples": CALIBRATION_LIMIT,
            "train_samples": len(data.x_train),
            "test_samples": len(data.x_test),
        },
    }

    out_fixture.parent.mkdir(parents=True, exist_ok=True)
    out_fixture.write_text(json.dumps(fixture_content, indent=2) + "\n")
    print(f"Wrote fixture to {out_fixture}")
    return fixture_content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to store/read MNIST gz files",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to save or load trained weights checkpoint (.pt)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training and resume export/quantization from existing checkpoint",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=FIXTURE_PATH,
        help="Output path for fixture JSON",
    )
    args = parser.parse_args()

    data_dir_arg: Path = args.data_dir
    checkpoint_arg: Path = args.checkpoint
    out_fixture_arg: Path = args.out
    export_only_arg: bool = args.export_only

    print(f"Loading official MNIST from {data_dir_arg}...")
    data = load_mnist_dataset(data_dir_arg)

    model = Mnist28CNN()
    if export_only_arg or checkpoint_arg.exists():
        if not checkpoint_arg.exists():
            raise FileNotFoundError(
                f"--export-only requested but checkpoint not found at {checkpoint_arg}"
            )
        print(f"Loading trained weights from checkpoint {checkpoint_arg} (train-once resume)...")
        state = torch.load(checkpoint_arg, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        _ = model.eval()
    else:
        print(
            f"Training probe CNN (epochs={EPOCHS}, lr={LEARNING_RATE}, "
            f"batch_size={BATCH_SIZE})..."
        )
        model = train_model(data, checkpoint_path=checkpoint_arg)

    print("Exporting ONNX opset 13 and quantizing via Penumbra...")
    _ = export_and_quantize(model, data, out_fixture_arg)


if __name__ == "__main__":
    main()
