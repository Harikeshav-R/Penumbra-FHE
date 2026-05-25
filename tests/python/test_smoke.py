"""Smoke tests for Penumbra."""

import penumbra_fhe


def test_import_success() -> None:
    """Ensure the package can be imported."""
    assert penumbra_fhe is not None


def test_version() -> None:
    """Ensure the version function works and returns a string."""
    v = penumbra_fhe.version()
    assert isinstance(v, str) and bool(v)
