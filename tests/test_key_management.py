"""Tests for the Phase-9 key management + client/server split (``penumbra.client.KeySet``).

The bridge persists a client/server key pair so a client can generate it once and **reuse** it
across inferences, and runs the roles as separate processes when a ``KeySet`` is passed
(``PROJECT.md`` §11). Two tiers, mirroring ``tests/test_predict_bridge.py``:

* **Fast unit tests (CI, no FHE, no cargo):** ``KeySet`` save/load persistence, the loud
  key/model ``num_blocks`` mismatch, and that ``keys=None`` keeps the all-in-one path — with a
  fake runtime injected where FHE would run, so no Rust toolchain is needed.
* **Opt-in end-to-end (``PENUMBRA_E2E=1``):** real FHE via cargo — generate a ``KeySet`` and
  reuse it across two ``predict_encrypted`` calls, asserting both equal the quantized-cleartext
  oracle bit-for-bit (``AGENTS.md`` §1.1). Proves key reuse and the split round trip.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from penumbra import KeySet, Linear, Model
from penumbra.client import _CLIENT_KEY, _KEY_META, _SERVER_KEY
from penumbra.quantization.spec import QuantSpec
from penumbra.reference import evaluate_graph_int


def _stub_keyset(directory, num_blocks: int) -> KeySet:
    """Materialize a KeySet on disk without cargo — placeholder key bytes + real metadata.

    The fast tests exercise the *bookkeeping* around keys (persistence, num_blocks matching),
    not the crypto, so opaque placeholder bytes stand in for the bincode key blobs.
    """
    import json
    from pathlib import Path

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / _CLIENT_KEY).write_bytes(b"stub-client-key")
    (d / _SERVER_KEY).write_bytes(b"stub-server-key")
    (d / _KEY_META).write_text(json.dumps({"num_blocks": int(num_blocks)}))
    return KeySet(directory=d, num_blocks=int(num_blocks))


# --- fast unit tests (CI) --------------------------------------------------------------------


def test_keyset_save_and_load_roundtrip(tmp_path):
    """A KeySet copies to a new directory and loads back with its num_blocks intact."""
    ks = _stub_keyset(tmp_path / "orig", num_blocks=8)
    saved = ks.save(tmp_path / "durable")
    assert saved.client_key.is_file() and saved.server_key.is_file()

    loaded = KeySet.load(tmp_path / "durable")
    assert loaded.num_blocks == 8
    assert loaded.client_key.read_bytes() == b"stub-client-key"
    assert loaded.server_key.read_bytes() == b"stub-server-key"


def test_keyset_load_missing_fails_loudly(tmp_path):
    """Loading a directory without key files raises an actionable error (AGENTS.md §1.4)."""
    with pytest.raises(FileNotFoundError, match="no Penumbra KeySet"):
        KeySet.load(tmp_path / "empty")


def test_run_encrypted_mismatched_keys_raises(tmp_path):
    """A KeySet whose num_blocks != the model's fails loudly, before any FHE (AGENTS.md §1.4).

    The num_blocks guard is the first thing the split path checks, so this raises without a Rust
    toolchain (no cargo call is reached) even with placeholder key bytes.
    """
    rng = np.random.default_rng(0)
    model = Model([Linear(weight=rng.normal(size=(3, 8)), bias=rng.normal(size=3))], input_bits=4)
    model.quantize(rng.uniform(0.0, 16.0, size=(64, 8)), n_bits=4)
    assert model.graph is not None

    from penumbra.client import run_encrypted

    wrong = _stub_keyset(tmp_path / "wrong", num_blocks=model.graph.num_blocks + 1)
    with pytest.raises(ValueError, match="key/model mismatch"):
        run_encrypted(model.graph, [[1] * 8], keys=wrong)


def test_predict_without_keys_uses_all_in_one(monkeypatch):
    """keys=None keeps the single-process path: run_encrypted is called with keys=None."""
    rng = np.random.default_rng(1)
    model = Model([Linear(weight=rng.normal(size=(4, 6)), bias=rng.normal(size=4))], input_bits=4)
    cal = rng.uniform(0.0, 16.0, size=(32, 6))
    model.quantize(cal, n_bits=4)
    assert model.graph is not None

    seen = {}

    def _fake(graph, int_inputs, *, keys=None):
        seen["keys"] = keys
        return [evaluate_graph_int(graph, {"x": row})[graph.outputs[0]] for row in int_inputs]

    monkeypatch.setattr("penumbra.model.run_encrypted", _fake)

    label = model.predict_encrypted(cal[0])
    assert seen["keys"] is None
    xq = QuantSpec(scale=model.input_scale, bits=model.input_bits, signed=False).quantize(cal[0])
    ref = evaluate_graph_int(model.graph, {"x": xq.tolist()})[model.graph.outputs[0]]
    assert label == int(np.argmax(ref))


def test_predict_forwards_keys_to_runtime(tmp_path, monkeypatch):
    """A matching KeySet is forwarded through predict_encrypted to run_encrypted (split path)."""
    rng = np.random.default_rng(2)
    model = Model([Linear(weight=rng.normal(size=(3, 5)), bias=rng.normal(size=3))], input_bits=4)
    model.quantize(rng.uniform(0.0, 16.0, size=(32, 5)), n_bits=4)
    assert model.graph is not None
    ks = _stub_keyset(tmp_path / "ks", num_blocks=model.graph.num_blocks)

    seen = {}

    def _fake(graph, int_inputs, *, keys=None):
        seen["keys"] = keys
        return [[0, 0, 0] for _ in int_inputs]

    monkeypatch.setattr("penumbra.model.run_encrypted", _fake)
    model.predict_encrypted(rng.uniform(0.0, 16.0, size=5), keys=ks)
    assert seen["keys"] is ks


# --- opt-in end-to-end (real FHE via cargo) --------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("PENUMBRA_E2E") != "1",
    reason="real-FHE key-reuse gate; set PENUMBRA_E2E=1 (needs cargo, seconds/sample)",
)
def test_key_reuse_across_calls_matches_oracle(tmp_path):
    """Generate a KeySet once, reuse it across two split-path calls; both match the oracle
    bit-for-bit (AGENTS.md §1.1). Proves key reuse + the client/server split end to end."""
    rng = np.random.default_rng(3)
    model = Model([Linear(weight=rng.normal(size=(3, 8)), bias=rng.normal(size=3))], input_bits=4)
    cal = rng.uniform(0.0, 16.0, size=(64, 8))
    model.quantize(cal, n_bits=4)
    assert model.graph is not None

    keys = KeySet.generate(model.graph.num_blocks, directory=tmp_path / "keys")
    in_spec = QuantSpec(scale=model.input_scale, bits=model.input_bits, signed=False)

    for call in range(2):  # reuse the SAME keys across calls
        samples = cal[call * 3 : call * 3 + 3]
        labels, logits = model.predict_encrypted(samples, return_logits=True, keys=keys)
        for i, row in enumerate(samples):
            xq = in_spec.quantize(row).tolist()
            ref = evaluate_graph_int(model.graph, {"x": xq})[model.graph.outputs[0]]
            assert logits[i] == ref, f"call {call} sample {i}: {logits[i]} != {ref}"
            assert labels[i] == int(np.argmax(ref))
