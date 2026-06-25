"""Cross-language bit-width conformance — the Python half (``AGENTS.md`` §1.3, §5).

The bit-width growth rules live in two places — Python (:mod:`penumbra.bitwidth`, which the
compile pass uses to place ``Requant`` nodes) and Rust (``Op::output_bits`` in
``runtime/src/ops``, which gates evaluation). If they disagree, a model could pass the Python
budget check and then overflow the radix under FHE (a §1.1/§1.3 violation), or vice versa.

This test pins them together: a committed table of ``(op, input_bits) -> expected`` cases
(``tests/fixtures/bitwidth_cases.json``) that **both** languages reproduce. This half asserts
the Python tracker matches every ``expected``; ``runtime/tests/bitwidth_conformance.rs`` does
the same for Rust. The shared committed table is the meeting point (the two CI jobs never call
each other), exactly like the IR conformance fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

from penumbra.bitwidth import output_bits
from penumbra.ir import OpSpec

CASES = Path(__file__).resolve().parent / "fixtures" / "bitwidth_cases.json"


def _load_cases() -> list[dict]:
    return json.loads(CASES.read_text())["cases"]


def test_python_output_bits_matches_committed_table():
    """Every committed case's ``expected`` is reproduced by the Python tracker."""
    cases = _load_cases()
    assert cases, "the bit-width conformance table must not be empty"
    for case in cases:
        op = OpSpec.from_dict(case["op"])
        got = output_bits(op, list(case["input_bits"]))
        assert got == case["expected"], (
            f"bit-width drift in case {case['name']!r}: Python output_bits gave {got}, "
            f"committed table expects {case['expected']} — Python and Rust rules disagree"
        )


def test_table_covers_every_op_type():
    """Guard against silently dropping an op from the conformance table as ops are added."""
    covered = {case["op"]["op_type"] for case in _load_cases()}
    expected = {"Linear", "Conv2d", "Pool", "Requant", "Add", "Activation", "Argmax"}
    assert expected <= covered, f"bit-width table is missing op types: {expected - covered}"
