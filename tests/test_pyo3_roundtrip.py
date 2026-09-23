"""End-to-end in-process PyO3 round trip tests (Phase 9).

Proves that:
1. All-in-one in-process inference matches the quantized-cleartext oracle bit-for-bit.
2. The client/server split (KeySet generate/save/load -> run_encrypted) matches bit-for-bit.
3. The "gaussian" crypto profile evaluates bit-for-bit equal to cleartext (TFHE is exact).
"""

from __future__ import annotations

import numpy as np

from penumbra import CryptoProfile, KeySet, Linear, Model
from penumbra.quantization.spec import QuantSpec
from penumbra.reference import evaluate_graph_int


def _make_test_model():
    rng = np.random.default_rng(42)
    model = Model(
        [Linear(weight=rng.normal(size=(3, 8)), bias=rng.normal(size=3))],
        input_bits=4,
    )
    cal = rng.uniform(0.0, 16.0, size=(64, 8))
    model.quantize(cal, n_bits=4)
    samples = rng.uniform(0.0, 16.0, size=(3, 8))
    return model, samples


def test_all_in_one_predict_matches_oracle_exactly():
    model, samples = _make_test_model()
    assert model.graph is not None

    labels, logits = model.predict_encrypted(samples, return_logits=True)

    in_spec = QuantSpec(scale=model.input_scale, bits=model.input_bits, signed=False)
    for i, row in enumerate(samples):
        xq = in_spec.quantize(row).tolist()
        ref = evaluate_graph_int(model.graph, {"x": xq})[model.graph.outputs[0]]
        assert (
            logits[i] == ref
        ), f"sample {i}: encrypted logits {logits[i]} != cleartext reference {ref}"
        assert labels[i] == int(np.argmax(ref))


def test_split_predict_with_saved_keyset_matches_oracle_exactly(tmp_path):
    model, samples = _make_test_model()
    assert model.graph is not None

    # Generate, save, and reload KeySet
    ks = KeySet.generate(model.graph.num_blocks, directory=tmp_path / "orig")
    ks.save(tmp_path / "saved")
    loaded = KeySet.load(tmp_path / "saved")

    labels, logits = model.predict_encrypted(samples, return_logits=True, keys=loaded)

    in_spec = QuantSpec(scale=model.input_scale, bits=model.input_bits, signed=False)
    for i, row in enumerate(samples):
        xq = in_spec.quantize(row).tolist()
        ref = evaluate_graph_int(model.graph, {"x": xq})[model.graph.outputs[0]]
        assert (
            logits[i] == ref
        ), f"sample {i}: encrypted logits {logits[i]} != cleartext reference {ref}"
        assert labels[i] == int(np.argmax(ref))


def test_gaussian_profile_predict_matches_oracle_exactly():
    model, samples = _make_test_model()
    assert model.graph is not None

    profile = CryptoProfile.tfhe("gaussian")
    labels, logits = model.predict_encrypted(samples, return_logits=True, profile=profile)

    in_spec = QuantSpec(scale=model.input_scale, bits=model.input_bits, signed=False)
    for i, row in enumerate(samples):
        xq = in_spec.quantize(row).tolist()
        ref = evaluate_graph_int(model.graph, {"x": xq})[model.graph.outputs[0]]
        assert logits[i] == ref, f"sample {i} under gaussian profile: {logits[i]} != {ref}"
        assert labels[i] == int(np.argmax(ref))
