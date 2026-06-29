"""Phase-5 QAT fixture guard + optional Brevitas training smoke test (``AGENTS.md`` §1.1, §5).

Two tiers, matching the hermetic-fixture discipline:

* **Always (CI):** guard the committed ``phase5_qat_fixture.json`` for self-consistency with the
  integer reference oracle — no FHE, no torch/brevitas. This is the half CI runs every time.
* **Only with the ml extra (local):** a `pytest.importorskip("brevitas")` smoke test that the QAT
  training + export path runs end to end and produces a graph the oracle can evaluate. Skipped in
  CI (which never installs the extra), exercised by a developer who ran `uv sync --extra ml`.

The bit-for-bit FHE gate for the QAT model lives in Rust (``runtime/tests/golden_qat.rs``,
``#[ignore]``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from penumbra.bitwidth import check_bit_width_budget
from penumbra.compile import insert_requants
from penumbra.ir import Graph
from penumbra.reference import evaluate_graph_int

FIXTURE = Path(__file__).resolve().parent.parent / "examples" / "mnist" / "phase5_qat_fixture.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_qat_graph_round_trips_and_fits_budget():
    g = Graph.from_dict(_fixture()["graph"])
    assert Graph.from_json(g.to_json()) == g
    check_bit_width_budget(g)


def test_qat_graph_is_conv_requant_linear_and_idempotent():
    g = Graph.from_dict(_fixture()["graph"])
    assert [n.op.op_type for n in g.nodes] == ["Conv2d", "Requant", "Linear"]
    assert insert_requants(g) == g, "insert_requants must be idempotent on the committed graph"


def test_qat_fixture_logits_and_labels_match_oracle():
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    for i, (x, expected) in enumerate(zip(fx["test_inputs"], fx["expected_logits"], strict=True)):
        logits = evaluate_graph_int(g, {"x": x})[g.outputs[0]]
        assert logits == expected, f"sample {i}: logits drifted from the oracle"
        assert int(np.argmax(logits)) == fx["expected_labels"][i], f"sample {i}: label drifted"


def test_qat_training_path_runs_end_to_end():
    """Optional: the Brevitas QAT train+export path runs and yields an oracle-evaluable graph.

    Skipped unless the optional ``ml`` extra (torch + brevitas + scikit-learn) is installed, so CI
    stays hermetic. This is a smoke test of the *pipeline*, not an accuracy assertion — it trains
    for only a few epochs to stay fast.
    """
    pytest.importorskip("brevitas")
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")

    import brevitas.nn as qnn
    import torch
    from sklearn.datasets import load_digits

    import penumbra as fhe
    from penumbra.reference import evaluate_graph_int as _eval

    torch.manual_seed(0)
    digits = load_digits()
    x = digits.images.astype(np.float32)
    y = digits.target.astype(np.int64)

    # A tiny QAT model, trained for a handful of epochs — enough to exercise the path.
    conv = qnn.QuantConv2d(1, 4, 3, stride=2, bias=False, weight_bit_width=4)
    relu = qnn.QuantReLU(bit_width=2)
    fc = qnn.QuantLinear(4 * 3 * 3, 10, bias=True, weight_bit_width=4)
    params = list(conv.parameters()) + list(relu.parameters()) + list(fc.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    xt = torch.from_numpy(x).unsqueeze(1)
    yt = torch.from_numpy(y)
    for _ in range(5):
        opt.zero_grad()
        h = relu(conv(xt)).flatten(1)
        torch.nn.functional.cross_entropy(fc(h), yt).backward()
        opt.step()

    model = fhe.Model(
        [
            fhe.Conv2d(
                weight=conv.weight.detach().numpy().astype(np.float64),
                in_h=8,
                in_w=8,
                in_channels=1,
                stride=2,
            ),
            fhe.Activation(lambda v: max(v, 0.0)),
            fhe.Linear(
                weight=fc.weight.detach().numpy().astype(np.float64),
                bias=fc.bias.detach().numpy().astype(np.float64),
            ),
        ],
        input_bits=4,
    )
    cal = x.reshape(len(x), -1).astype(np.float64)
    graph = model.quantize(cal, n_bits=4, act_bits=2, per_channel=True)  # self-verifies internally

    # The produced graph evaluates through the oracle to 10 logits.
    xq = np.clip(np.round(cal[0] / model.input_scale), 0, 15).astype(np.int64).tolist()
    out = _eval(graph, {"x": xq})
    assert len(out[graph.outputs[0]]) == 10
