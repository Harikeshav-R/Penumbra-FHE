"""Drift guard for the model zoo documentation (``docs/MODEL-ZOO.md``).

Ensures that every committed example fixture in ``examples/``:
1. Is documented in ``docs/MODEL-ZOO.md``.
2. Reports IR ops and float/quantized accuracy matching the committed JSON.
3. Points to generator scripts, TFHE golden tests, and CKKS tests/bounds that actually exist.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = ROOT / "docs" / "MODEL-ZOO.md"
BOUNDS_PATH = ROOT / "crates" / "penumbra-ckks" / "src" / "bounds.rs"
MODELS_PATH = ROOT / "crates" / "penumbra-bench" / "src" / "models.rs"


def _parse_zoo_table() -> list[list[str]]:
    """Parse rows from the Markdown table under '## Validated models'."""
    text = DOC_PATH.read_text(encoding="utf-8")
    heading = "## Validated models"
    if heading not in text:
        raise ValueError(f"Heading {heading!r} not found in {DOC_PATH}")
    after = text.split(heading, 1)[1]

    lines = after.splitlines()
    in_table = False
    rows: list[list[str]] = []

    for line in lines:
        stripped = line.strip()
        if not in_table:
            if stripped.startswith("|") and "Fixture" in stripped:
                in_table = True
            continue
        # In table
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        # Skip separator
        if re.fullmatch(r"\|[\s:|-]+\|", stripped):
            continue
        cols = [c.strip() for c in stripped.strip("|").split("|")]
        rows.append(cols)

    return rows


def test_zoo_lists_every_committed_fixture():
    """Every committed *_fixture.json in examples/ is listed in the zoo table."""
    rows = _parse_zoo_table()
    fixtures_in_zoo: set[str] = set()
    for row in rows:
        m = re.search(r"`([^`]+_fixture\.json)`", row[0])
        assert m, f"first column must contain backticked fixture path: {row[0]!r}"
        fixtures_in_zoo.add(m.group(1))

    committed = {p.relative_to(ROOT).as_posix() for p in ROOT.glob("examples/**/*_fixture.json")}
    assert fixtures_in_zoo == committed, (
        f"Zoo fixtures mismatch.\nIn zoo only: {fixtures_in_zoo - committed}\n"
        f"Committed only: {committed - fixtures_in_zoo}"
    )


def test_zoo_ops_and_accuracy_match_fixtures():
    """IR ops and accuracy in the zoo table match each fixture JSON."""
    rows = _parse_zoo_table()
    for row in rows:
        m = re.search(r"`([^`]+_fixture\.json)`", row[0])
        assert m
        rel_path = m.group(1)
        fx_data: dict[str, Any] = json.loads((ROOT / rel_path).read_text(encoding="utf-8"))

        op_names = [n["op"]["op_type"] for n in fx_data["graph"]["nodes"]]
        expected_ops = "`" + " → ".join(op_names) + "`"
        assert row[4] == expected_ops, f"row {rel_path}: ops {row[4]!r} != {expected_ops!r}"

        expected_float = f"{fx_data['accuracy']['float']:.3f}"
        expected_quant = f"{fx_data['accuracy']['quantized']:.3f}"
        assert (
            row[5] == expected_float
        ), f"row {rel_path}: float acc {row[5]!r} != {expected_float!r}"
        assert (
            row[6] == expected_quant
        ), f"row {rel_path}: quant acc {row[6]!r} != {expected_quant!r}"


def test_zoo_referenced_paths_and_bounds_exist():
    """All scripts, tests, bounds, and bench keys referenced in the zoo exist."""
    rows = _parse_zoo_table()
    bounds_text = BOUNDS_PATH.read_text(encoding="utf-8")
    models_text = MODELS_PATH.read_text(encoding="utf-8")

    for row in rows:
        # Col 1: generator script
        for path_token in re.findall(r"`([^`]+)`", row[1]):
            if "/" in path_token:
                assert (ROOT / path_token).exists(), f"generator script {path_token} does not exist"

        # Col 7: TFHE golden test
        for path_token in re.findall(r"`([^`]+)`", row[7]):
            if "/" in path_token:
                assert (ROOT / path_token).exists(), f"TFHE golden test {path_token} does not exist"

        # Col 8: CKKS test and bound
        for path_token in re.findall(r"`([^`]+)`", row[8]):
            if "/" in path_token:
                assert (ROOT / path_token).exists(), f"CKKS test {path_token} does not exist"
            elif re.fullmatch(r"PHASE\w+", path_token):
                decl = f"pub const {path_token}"
                assert decl in bounds_text, f"CKKS bound {path_token} not found in bounds.rs"

        # Col 9: Bench key
        bench_key = row[9].strip()
        if bench_key not in ("—", "-"):
            key_token = re.search(r"`?([a-zA-Z0-9_]+)`?", bench_key)
            assert key_token, f"invalid bench key in col 9: {bench_key}"
            k = key_token.group(1)
            target = f'key: "{k}"'
            assert target in models_text, f"bench key {k!r} not found in models.rs"
