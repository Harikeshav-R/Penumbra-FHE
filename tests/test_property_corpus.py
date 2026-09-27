"""Drift guard for the committed property corpus (ROADMAP Phase 11)."""

from __future__ import annotations

import json
from pathlib import Path

from penumbra.ir import Graph
from penumbra.reference import evaluate_graph_int
from penumbra.testing import build_corpus

CORPUS_PATH = Path(__file__).resolve().parent / "fixtures" / "property_corpus.json"


def test_committed_corpus_matches_generator() -> None:
    """The committed property corpus matches build_corpus() exactly."""
    committed = json.loads(CORPUS_PATH.read_text())
    fresh = build_corpus()
    assert committed == fresh, (
        "property_corpus.json has drifted from penumbra.testing:build_corpus(); "
        "re-run: uv run python -m penumbra.testing --emit tests/fixtures/property_corpus.json"
    )


def test_corpus_expected_outputs_match_reference_oracle() -> None:
    """Every model's expected_outputs matches evaluate_graph_int per sample."""
    committed = json.loads(CORPUS_PATH.read_text())
    assert committed.get("schema") == "penumbra-property-corpus/1"

    for entry in committed["models"]:
        name = entry["name"]
        graph = Graph.from_dict(entry["graph"])
        test_inputs = entry["test_inputs"]
        expected_outputs = entry["expected_outputs"]

        assert len(test_inputs) == len(expected_outputs), f"{name}: inputs/outputs length mismatch"
        out_name = graph.outputs[0]

        for s, (inp, want) in enumerate(zip(test_inputs, expected_outputs, strict=True)):
            got = evaluate_graph_int(graph, {"x": inp})[out_name]
            assert got == want, f"{name} sample {s}: oracle recomputation {got} != corpus {want}"
