"""Tests for the Phase-9 subprocess bridge: ``Model.predict_encrypted`` + ``penumbra.client``.

The bridge quantizes a float input, hands the exported IR + integer batch to the Rust
runtime's ``predict`` binary (keygen -> encrypt -> evaluate -> decrypt under FHE), and decodes
the decrypted outputs client-side (argmax / label, ``PROJECT.md`` §11).

Two tiers, mirroring the project's hermetic-fixture discipline:

* **Fast unit tests (run in CI, no FHE, no cargo):** pin the *client-side* logic — input
  quantization, output decoding for both terminal-op shapes, and the loud-failure contract.
  Where the runtime would be invoked, a **fake** ``run_encrypted`` (the pure-Python integer
  oracle :func:`penumbra.reference.evaluate_graph_int`) is injected, so the decode/quantize
  path is exercised without a Rust toolchain.
* **Opt-in end-to-end golden (skipped unless ``PENUMBRA_E2E=1``):** the real bridge gate —
  runs actual FHE via ``cargo`` and asserts the decrypted prediction equals the
  quantized-cleartext oracle bit-for-bit (``AGENTS.md`` §1.1). Off by default because FHE is
  minutes-per-sample and needs cargo (CI stays hermetic, like the ``#[ignore]``d Rust goldens).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from penumbra import Linear, Model
from penumbra.ir import SCHEMA_VERSION, ArgmaxSpec, Graph, LinearSpec, Node
from penumbra.quantization.spec import QuantSpec
from penumbra.reference import evaluate_graph_int


def _fake_runtime(graph: Graph):
    """A drop-in ``run_encrypted`` that evaluates the graph with the integer oracle.

    TFHE is exact, so the real runtime returns exactly what ``evaluate_graph_int`` computes
    (``AGENTS.md`` §1.1). Substituting it lets the fast tests exercise the full quantize ->
    (encrypt/eval/decrypt) -> decode path deterministically, with no cargo and no FHE.
    """

    def _run(g: Graph, int_inputs: list[list[int]], *, keys=None) -> list[list[int]]:
        assert g is graph
        return [evaluate_graph_int(g, {"x": row})[g.outputs[0]] for row in int_inputs]

    return _run


# --- fast client-side unit tests (CI) --------------------------------------------------------


def test_input_quantization_matches_quantspec(monkeypatch):
    """predict_encrypted quantizes the input exactly as QuantSpec.quantize would."""
    rng = np.random.default_rng(0)
    model = Model([Linear(weight=rng.normal(size=(3, 8)), bias=rng.normal(size=3))], input_bits=4)
    model.quantize(rng.uniform(0.0, 16.0, size=(64, 8)), n_bits=4)

    seen: list[list[int]] = []

    def _capture(graph, int_inputs, *, keys=None):
        seen.extend(int_inputs)
        # Return a well-formed output so decode succeeds (values irrelevant here).
        return [[0, 0, 0] for _ in int_inputs]

    monkeypatch.setattr("penumbra.model.run_encrypted", _capture)

    x = rng.uniform(0.0, 16.0, size=(2, 8))
    model.predict_encrypted(x)

    in_spec = QuantSpec(scale=model.input_scale, bits=model.input_bits, signed=False)
    expected = [in_spec.quantize(row).tolist() for row in x]
    assert seen == expected


def test_decode_wide_logits_argmaxes(monkeypatch):
    """A wide multi-logit head is argmaxed client-side; single input returns a scalar label."""
    rng = np.random.default_rng(1)
    model = Model([Linear(weight=rng.normal(size=(4, 6)), bias=rng.normal(size=4))], input_bits=4)
    model.quantize(rng.uniform(0.0, 16.0, size=(32, 6)), n_bits=4)

    # class 2 is the argmax of this logit row
    monkeypatch.setattr("penumbra.model.run_encrypted", lambda g, xs, **kw: [[3, 1, 9, 4]])
    label = model.predict_encrypted(rng.uniform(0.0, 16.0, size=6))
    assert label == 2
    assert isinstance(label, int)


def test_decode_batch_returns_list_and_logits(monkeypatch):
    """A batch returns a list of labels; return_logits surfaces the raw output rows."""
    rng = np.random.default_rng(2)
    model = Model([Linear(weight=rng.normal(size=(3, 5)), bias=rng.normal(size=3))], input_bits=4)
    model.quantize(rng.uniform(0.0, 16.0, size=(32, 5)), n_bits=4)

    rows = [[5, 2, 1], [0, 7, 3]]
    monkeypatch.setattr("penumbra.model.run_encrypted", lambda g, xs, **kw: rows)
    labels, logits = model.predict_encrypted(
        rng.uniform(0.0, 16.0, size=(2, 5)), return_logits=True
    )
    assert labels == [0, 1]
    assert logits == rows


def test_decode_terminal_argmax_returns_label_bit():
    """A terminal Argmax head emits the label bit directly — no client-side argmax."""
    # Hand-build a Linear -> Argmax graph and decode a runtime output of [[1]] / [[0]].
    graph = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=8,
        input_bits=4,
        inputs=["x"],
        outputs=["label"],
        nodes=[
            Node("fc", ["x"], ["logit"], LinearSpec(weights=[[1, 1]], bias=[0], weight_bits=4)),
            Node("head", ["logit"], ["label"], ArgmaxSpec(threshold=5)),
        ],
    )
    model = Model([Linear(weight=np.ones((1, 2)), bias=np.zeros(1))], input_bits=4)
    model.graph = graph  # bypass quantize(): we only exercise _decode's terminal-op branch
    assert model._decode([[1]], single=True, return_logits=False) == 1
    assert model._decode([[0]], single=True, return_logits=False) == 0
    # Contrast: a wide head would argmax; here the raw bit is returned unchanged even though
    # np.argmax([1]) / np.argmax([0]) would both be 0.
    assert model._decode([[1]], single=False, return_logits=False) == [1]


def test_predict_before_quantize_raises():
    model = Model([Linear(weight=np.ones((2, 3)), bias=np.zeros(2))], input_bits=4)
    with pytest.raises(RuntimeError, match="quantize"):
        model.predict_encrypted(np.zeros(3))


def test_run_encrypted_missing_runtime_dir_raises(monkeypatch):
    """A bad PENUMBRA_RUNTIME_DIR fails loudly and actionably (AGENTS.md §1.4)."""
    from penumbra.client import run_encrypted

    monkeypatch.setenv("PENUMBRA_RUNTIME_DIR", "/nonexistent/penumbra/runtime")
    graph = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=8,
        input_bits=4,
        inputs=["x"],
        outputs=["y"],
        nodes=[Node("fc", ["x"], ["y"], LinearSpec(weights=[[1]], bias=[0], weight_bits=4))],
    )
    with pytest.raises(FileNotFoundError, match="runtime crate not found"):
        run_encrypted(graph, [[1]])


def test_run_encrypted_empty_batch_short_circuits():
    """An empty batch needs no toolchain and returns an empty list."""
    from penumbra.client import run_encrypted

    graph = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=8,
        input_bits=4,
        inputs=["x"],
        outputs=["y"],
        nodes=[Node("fc", ["x"], ["y"], LinearSpec(weights=[[1]], bias=[0], weight_bits=4))],
    )
    assert run_encrypted(graph, []) == []


def test_predict_matches_oracle_with_fake_runtime(monkeypatch):
    """End-to-end client logic against the integer oracle (no FHE): the decoded label equals
    argmax of the cleartext logits for every sample."""
    rng = np.random.default_rng(3)
    model = Model([Linear(weight=rng.normal(size=(5, 8)), bias=rng.normal(size=5))], input_bits=4)
    cal = rng.uniform(0.0, 16.0, size=(64, 8))
    model.quantize(cal, n_bits=4)

    monkeypatch.setattr("penumbra.model.run_encrypted", _fake_runtime(model.graph))

    labels, logits = model.predict_encrypted(cal[:5], return_logits=True)
    in_spec = QuantSpec(scale=model.input_scale, bits=model.input_bits, signed=False)
    for i, row in enumerate(cal[:5]):
        xq = in_spec.quantize(row).tolist()
        ref = evaluate_graph_int(model.graph, {"x": xq})[model.graph.outputs[0]]
        assert logits[i] == ref
        assert labels[i] == int(np.argmax(ref))


# --- opt-in end-to-end golden (real FHE via cargo) -------------------------------------------


@pytest.mark.skipif(
    os.environ.get("PENUMBRA_E2E") != "1",
    reason="real-FHE bridge gate; set PENUMBRA_E2E=1 (needs cargo, minutes/sample)",
)
def test_predict_encrypted_matches_cleartext_bit_for_bit():
    """The bridge's golden gate: the encrypted prediction equals the quantized-cleartext oracle
    bit-for-bit (AGENTS.md §1.1). Kept to a single small Linear so it runs in seconds."""
    rng = np.random.default_rng(4)
    model = Model([Linear(weight=rng.normal(size=(3, 8)), bias=rng.normal(size=3))], input_bits=4)
    cal = rng.uniform(0.0, 16.0, size=(64, 8))
    model.quantize(cal, n_bits=4)

    samples = cal[:3]
    labels, logits = model.predict_encrypted(samples, return_logits=True)

    in_spec = QuantSpec(scale=model.input_scale, bits=model.input_bits, signed=False)
    for i, row in enumerate(samples):
        xq = in_spec.quantize(row).tolist()
        ref = evaluate_graph_int(model.graph, {"x": xq})[model.graph.outputs[0]]
        assert logits[i] == ref, f"FHE logits {logits[i]} != cleartext {ref} at sample {i}"
        assert labels[i] == int(np.argmax(ref))
