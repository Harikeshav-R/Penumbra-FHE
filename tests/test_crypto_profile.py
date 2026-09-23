"""Tests for crypto-parameter profile APIs and error handling (Phase 9).

Verifies that all known error paths fail loudly with actionable messages:
1. Unknown TFHE profile names
2. Invalid CKKS polynomial degrees (< 1)
3. Unknown backend identifiers
4. Missing CKKS feature build guidance
5. Profile/backend crossing
6. Passing both keys and profile
"""

from __future__ import annotations

import pytest

from penumbra import CryptoProfile, KeySet, available_backends
from penumbra.client import run_encrypted
from penumbra.ir import SCHEMA_VERSION, Graph, LinearSpec, Node


def _dummy_graph() -> Graph:
    return Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=4,
        input_bits=4,
        inputs=["x"],
        outputs=["y"],
        nodes=[
            Node(
                "fc",
                ["x"],
                ["y"],
                LinearSpec(weights=[[1, 0], [0, 1]], bias=[0, 0], weight_bits=4),
            )
        ],
    )


def test_invalid_tfhe_profile_raises():
    with pytest.raises(ValueError, match="available profiles: default, gaussian"):
        CryptoProfile.tfhe("nope")


def test_invalid_ckks_degree_raises():
    with pytest.raises(ValueError, match="max_poly_degree must be >= 1"):
        CryptoProfile.ckks(0)


def test_unknown_backend_raises():
    graph = _dummy_graph()
    with pytest.raises(ValueError, match="unknown backend 'nope'"):
        run_encrypted(graph, [[1, 2]], backend="nope")


def test_ckks_unavailable_message_or_valid():
    graph = _dummy_graph()
    backends = available_backends()
    if "ckks" not in backends:
        with pytest.raises(ValueError, match="nightly Rust toolchain"):
            run_encrypted(graph, [[1, 2]], backend="ckks")
    else:
        pytest.skip("CKKS is compiled into this build")


def test_profile_backend_crossing_raises():
    graph = _dummy_graph()
    profile = CryptoProfile.ckks(15)
    with pytest.raises(ValueError, match="profile/backend mismatch"):
        run_encrypted(graph, [[1, 2]], backend="tfhe", profile=profile)


def test_keys_and_profile_together_raises(tmp_path):
    graph = _dummy_graph()
    ks = KeySet.generate(graph.num_blocks, directory=tmp_path / "keys")
    profile = CryptoProfile.tfhe("default")
    with pytest.raises(ValueError, match="cannot specify both 'keys' and 'profile'"):
        run_encrypted(graph, [[1, 2]], keys=ks, profile=profile)
