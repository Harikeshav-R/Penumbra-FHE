"""Smoke tests for Penumbra."""

import penumbra_fhe


def test_import_success() -> None:
    """Ensure the package can be imported."""
    assert penumbra_fhe is not None
