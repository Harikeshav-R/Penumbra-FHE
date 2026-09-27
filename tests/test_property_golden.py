"""In-process property test of the golden invariant under TFHE (ROADMAP Phase 11)."""

from __future__ import annotations

import os

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from penumbra.client import run_encrypted
from penumbra.testing import FAMILIES, random_model


@settings(
    max_examples=int(os.environ.get("PENUMBRA_PROPERTY_EXAMPLES", "6")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)
@given(seed=st.integers(0, 2**31 - 1), family=st.sampled_from(FAMILIES))
def test_random_model_tfhe_matches_reference_bit_for_bit(seed: int, family: str) -> None:
    """Random small models under TFHE match the quantized-cleartext oracle bit-for-bit."""
    m = random_model(seed, family)
    got = run_encrypted(m.graph, m.test_inputs, backend="tfhe")
    assert (
        got == m.expected_outputs()
    ), f"GOLDEN VIOLATION on {m.name}: {got} != {m.expected_outputs()}"
