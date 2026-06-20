"""Placeholder test so `pytest` is green on a clean checkout (ROADMAP.md Phase 0).

Replaced by real tests as the front end is built out. The headline test to come is the
**golden exactness** test (``AGENTS.md`` §1.1, ``tests/test_quantized_vs_fhe.py``):
FHE output == quantized-cleartext output, bit-for-bit.
"""

import penumbra


def test_package_imports_and_has_version():
    assert isinstance(penumbra.__version__, str)
    assert penumbra.__version__
