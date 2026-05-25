"""
Smoke tests for Penumbra.
"""

import pytest

def test_import_success():
    """Ensure the package can be imported."""
    import penumbra_fhe
    assert penumbra_fhe is not None
