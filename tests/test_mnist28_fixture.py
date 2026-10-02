"""Phase-16 28x28 MNIST scale probe fixture guard (ROADMAP Phase 16).

This module guards the Phase 16 raw 28x28 MNIST probe fixture:
1. Fixture contract regression:
   - Exactly 784 integer inputs per sample (1x1x28x28 raw encrypted pixels).
   - Graph Conv2d in_h=28, in_w=28, stride=4, producing a 4x7x7 feature map.
   - 10 reference logits per sample.
   - evaluate_graph_int reproduces stored logits and first-maximum labels bit-for-bit.
   - Default tests inspect the committed fixture without importing torch or using the network.
2. Guarded synthetic lowering test:
   - Exercises PyTorch ONNX export (opset 13) and Penumbra load_onnx/quantize pipeline on
     synthetic data, completely independent of downloading official MNIST data or training.
   - Skipped when torch is not available.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import pytest

from penumbra.bitwidth import check_bit_width_budget
from penumbra.ir import Conv2dSpec, Graph
from penumbra.reference import evaluate_graph_int

if TYPE_CHECKING:
    import torch

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "mnist" / "phase16_mnist28_fixture.json"
)


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


def _load_fixture() -> FixtureDict:
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(f"Fixture not found at {FIXTURE_PATH}")
    raw = json.loads(FIXTURE_PATH.read_text())
    assert isinstance(raw, dict), "Fixture root must be a JSON object"
    content = cast(FixtureDict, cast(object, raw))
    return content


def test_mnist28_fixture_contract() -> None:
    """Committed fixture strictly conforms to the approved 28x28 probe specifications."""
    fx = _load_fixture()

    assert "graph" in fx, "fixture must contain 'graph'"
    graph_dict = fx["graph"]
    assert isinstance(graph_dict, dict)
    g = Graph.from_dict(graph_dict)  # type: ignore[arg-type]
    assert Graph.from_json(g.to_json()) == g, "Graph IR serialization must round-trip exactly"
    check_bit_width_budget(g)

    # Validate Conv2d dimensions
    conv_nodes = [n for n in g.nodes if isinstance(n.op, Conv2dSpec)]
    assert len(conv_nodes) >= 1, "Graph must contain at least one Conv2d node"
    conv_op = conv_nodes[0].op
    assert isinstance(conv_op, Conv2dSpec)
    assert conv_op.in_h == 28, f"Expected in_h=28, got {conv_op.in_h}"
    assert conv_op.in_w == 28, f"Expected in_w=28, got {conv_op.in_w}"
    assert conv_op.stride == 4, f"Expected stride=4, got {conv_op.stride}"
    assert conv_op.kernel_h == 3, f"Expected kernel_h=3, got {conv_op.kernel_h}"
    assert conv_op.kernel_w == 3, f"Expected kernel_w=3, got {conv_op.kernel_w}"
    assert conv_op.in_channels == 1, f"Expected in_channels=1, got {conv_op.in_channels}"

    # Validate sample IDs
    assert fx["sample_ids"] == ["test_0", "test_1"], f"Unexpected sample_ids: {fx['sample_ids']}"
    assert (
        len(fx["calibration_sample_ids"]) == 128
    ), f"Expected 128 calibration IDs, got {len(fx['calibration_sample_ids'])}"

    # Validate test inputs and reference outputs
    test_inputs = fx["test_inputs"]
    expected_logits = fx["expected_logits"]
    expected_labels = fx["expected_labels"]

    assert len(test_inputs) >= 2, f"Expected >= 2 test samples, got {len(test_inputs)}"
    assert len(test_inputs) == len(expected_logits) == len(expected_labels)

    for i, (inp, exp_logit, exp_label) in enumerate(
        zip(test_inputs, expected_logits, expected_labels, strict=True)
    ):
        assert len(inp) == 784, f"Sample {i}: expected 784 integer inputs, got {len(inp)}"
        assert len(exp_logit) == 10, f"Sample {i}: expected 10 logits, got {len(exp_logit)}"

        # Evaluate cleartext integer reference
        out = evaluate_graph_int(g, {g.inputs[0]: inp})
        actual_logits: list[int] = out[g.outputs[0]]
        assert actual_logits == exp_logit, f"Sample {i}: logits drifted from evaluate_graph_int"

        # First maximum argmax tie-breaker matching NumPy / integer reference
        first_max_label = int(np.argmax(actual_logits))
        assert first_max_label == exp_label, f"Sample {i}: label drifted from first-argmax"

    # Validate accuracy metadata (numeric floats conforming to existing fixture convention)
    acc = fx["accuracy"]
    assert 0.0 <= acc["quantized"] <= 1.0
    assert 0.0 <= acc["float"] <= 1.0

    # Validate detailed accuracy counts
    counts = fx["accuracy_counts"]
    assert counts["float"]["total"] == 10000
    assert counts["quantized"]["total"] == 10000
    assert counts["float"]["correct"] == 9047
    assert counts["quantized"]["correct"] == 8709
    assert abs(acc["float"] - counts["float"]["correct"] / counts["float"]["total"]) < 1e-4
    assert (
        abs(acc["quantized"] - counts["quantized"]["correct"] / counts["quantized"]["total"]) < 1e-4
    )
    # Validate protocol metadata
    assert fx["dataset"]["name"] == "MNIST"
    assert fx["dataset"]["hash_algorithm"] == "MD5"
    assert fx["dataset"]["primary_checksum_source"].startswith("https://")
    assert "checksums" in fx["dataset"], "Fixture must record official MNIST checksums"
    assert "training_protocol" in fx, "Fixture must document training protocol"


def test_mnist28_synthetic_lowering_and_quantization() -> None:
    """Guarded test verifying PyTorch ONNX export and Penumbra load_onnx/quantize pipeline."""
    torch_mod = pytest.importorskip("torch")
    import penumbra as fhe

    class Mnist28CNN(torch_mod.nn.Module):
        conv: torch.nn.Conv2d
        fc: torch.nn.Linear

        def __init__(self) -> None:
            super().__init__()
            self.conv = torch_mod.nn.Conv2d(1, 4, 3, stride=4, bias=False)
            self.fc = torch_mod.nn.Linear(196, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h: torch.Tensor = torch_mod.relu(self.conv(x))
            out: torch.Tensor = self.fc(h.flatten(1))
            return out

    _ = torch_mod.manual_seed(42)
    model = Mnist28CNN()
    _ = model.eval()

    dummy = torch_mod.zeros(1, 1, 28, 28)
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        _ = torch_mod.onnx.export(
            model,
            (dummy,),
            tmp.name,
            input_names=["x"],
            output_names=["logits"],
            opset_version=13,
            dynamic_axes=None,
            dynamo=False,
        )

        fmodel = fhe.load_onnx(tmp.name, input_bits=3)
        assert fmodel is not None

        # Synthetic calibration set: 8 samples of 1x28x28 normalized pixels in [0, 1]
        np.random.seed(42)
        cal = np.random.uniform(0.0, 1.0, size=(8, 784)).astype(np.float64)

        graph = fmodel.quantize(
            cal,
            n_bits=[5, 6],
            act_bits=2,
            per_channel=True,
            max_mult_bits=1,
            calibration="mse",
        )
        assert graph is not None

        # Check graph structure
        conv_nodes = [n for n in graph.nodes if isinstance(n.op, Conv2dSpec)]
        assert len(conv_nodes) == 1
        conv_op = conv_nodes[0].op
        assert isinstance(conv_op, Conv2dSpec)
        assert conv_op.in_h == 28
        assert conv_op.in_w == 28
        assert conv_op.stride == 4

        # Run integer evaluation on one synthetic sample
        assert fmodel.input_scale is not None
        s_in = float(fmodel.input_scale)
        q_arr = np.clip(np.round(cal[0] / s_in), 0, (1 << 3) - 1).astype(np.int64)
        q_sample: list[int] = q_arr.tolist()
        assert len(q_sample) == 784
        out = evaluate_graph_int(graph, {graph.inputs[0]: q_sample})
        logits: list[int] = out[graph.outputs[0]]
        assert len(logits) == 10
