"""Doc<->registry lockstep for the ONNX front door (``AGENTS.md`` §5), Phase 6.

The documented supported-op list must always match what the validator actually accepts. This
parses the **ONNX op -> internal op** mapping table out of ``docs/SUPPORTED-OPS.md`` (the table
under the "ONNX front door (Phase 6)" section) and asserts its ONNX op set equals
``op_registry.supported_onnx_ops()`` exactly. If someone adds a registry entry without documenting
it (or vice versa), this fails — keeping the doc honest and testable (the ROADMAP Phase 6 exit
criterion "the documented supported-op list matches what the validator actually accepts").
"""

from __future__ import annotations

import re
from pathlib import Path

from penumbra import op_registry

DOC = Path(__file__).resolve().parent.parent / "docs" / "SUPPORTED-OPS.md"
SECTION = "## ONNX front door (Phase 6)"


def _parse_onnx_mapping_ops() -> list[str]:
    """Return the ONNX op names (first column) from the front-door mapping table in the doc.

    The table is the first Markdown table after the ``## ONNX front door (Phase 6)`` heading; its
    header row is ``| ONNX op | Internal op | Attribute constraints |``. Op names are wrapped in
    backticks in the first column.
    """
    text = DOC.read_text()
    start = text.index(SECTION)
    end = text.find("\n## ", start + len(SECTION))  # next top-level section
    section = text[start : end if end != -1 else len(text)]

    lines = section.splitlines()
    # Find the mapping table's header, then consume contiguous table rows after the separator.
    ops: list[str] = []
    in_table = False
    seen_separator = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| ONNX op "):
            in_table = True
            seen_separator = False
            continue
        if in_table:
            if re.fullmatch(r"\|[\s:|-]+\|", stripped):  # the |---|---|---| separator row
                seen_separator = True
                continue
            if not stripped.startswith("|"):
                break  # table ended
            if not seen_separator:
                continue
            first_col = stripped.split("|")[1].strip()
            m = re.match(r"`([^`]+)`", first_col)
            assert m, f"table row's first column is not a `code` op name: {first_col!r}"
            ops.append(m.group(1))
    return ops


def test_doc_onnx_mapping_matches_registry_exactly():
    """The doc's ONNX->internal table lists exactly the registry's supported ONNX ops."""
    doc_ops = _parse_onnx_mapping_ops()
    registry_ops = op_registry.supported_onnx_ops()
    assert doc_ops, "no ONNX mapping table found in docs/SUPPORTED-OPS.md"
    assert sorted(doc_ops) == registry_ops, (
        "docs/SUPPORTED-OPS.md ONNX mapping drifted from op_registry: "
        f"doc-only={sorted(set(doc_ops) - set(registry_ops))}, "
        f"registry-only={sorted(set(registry_ops) - set(doc_ops))}"
    )
    # No duplicate rows in the doc table.
    assert len(doc_ops) == len(set(doc_ops)), f"duplicate op rows in the doc table: {doc_ops}"


def test_doc_declares_supported_opset_range():
    """The doc states the same supported opset range the registry pins."""
    text = DOC.read_text()
    assert (
        f"{op_registry.SUPPORTED_OPSET_MIN}–{op_registry.SUPPORTED_OPSET_MAX}" in text
        or f"{op_registry.SUPPORTED_OPSET_MIN}-{op_registry.SUPPORTED_OPSET_MAX}" in text
    ), "docs/SUPPORTED-OPS.md must state the supported opset range from op_registry"
