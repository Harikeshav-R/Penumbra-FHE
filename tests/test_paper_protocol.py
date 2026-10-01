"""Behavior and schema tests for the Phase 15 paper benchmark protocol."""

import json

import pytest
from examples.paper_protocol import (
    CALIBRATION_LIMIT,
    CALIBRATION_SEED,
    CKKS_BOUND_MARGIN,
    FIXTURE_PATHS,
    SPOT_CHECK_SEED,
    TFHE_SPOT_CHECK_COUNT,
    canonical_graph_hash,
    export_paper_protocol,
    generate_paper_data,
    sample_rank_key,
    select_calibration_samples,
    select_spot_checks,
)


def test_constants():
    assert CALIBRATION_SEED == 1507
    assert SPOT_CHECK_SEED == 1503
    assert CALIBRATION_LIMIT == 128
    assert TFHE_SPOT_CHECK_COUNT == 30
    assert CKKS_BOUND_MARGIN == 2.0


def test_canonical_graph_hash():
    graph_dict = {
        "schema_version": "0.10.0",
        "num_blocks": 2,
        "input_bits": 2,
        "inputs": ["x"],
        "outputs": ["y"],
        "nodes": [],
    }
    h1 = canonical_graph_hash(graph_dict)
    # Check deterministic serialization: key ordering shouldn't change hash
    graph_dict_reversed = {
        "nodes": [],
        "outputs": ["y"],
        "inputs": ["x"],
        "input_bits": 2,
        "num_blocks": 2,
        "schema_version": "0.10.0",
    }
    h2 = canonical_graph_hash(graph_dict_reversed)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex string


def test_sample_selection_rank():
    # Rank must be deterministic and byte-order based
    key1 = sample_rank_key(1507, "model_a", "sample_1")
    key2 = sample_rank_key(1507, "model_a", "sample_2")
    key1_again = sample_rank_key(1507, "model_a", "sample_1")
    assert key1 == key1_again
    assert key1 != key2


def test_calibration_selection_disjoint_from_test():
    train_ids = [f"tr_{i}" for i in range(200)]
    test_ids = [f"te_{i}" for i in range(50)]

    cal_samples = select_calibration_samples(
        model_key="test_model",
        train_samples=[{"id": sid} for sid in train_ids],
        seed=CALIBRATION_SEED,
        limit=CALIBRATION_LIMIT,
    )
    assert len(cal_samples) == CALIBRATION_LIMIT
    cal_ids = {s["id"] for s in cal_samples}
    # Calibration must be drawn strictly from train
    assert cal_ids.issubset(set(train_ids))
    # Calibration and test must be completely disjoint
    assert cal_ids.isdisjoint(set(test_ids))


def test_spot_check_selection_non_faces():
    test_samples = [{"id": f"te_{i}"} for i in range(100)]
    spot_checks = select_spot_checks(
        model_key="phase2_logreg",
        test_samples=test_samples,
        cal_samples=[{"id": f"tr_{i}"} for i in range(128)],
        is_faces=False,
    )
    assert len(spot_checks) == 30
    assert all(ref.split == "test" for ref in spot_checks)
    # Check all indices are within bounds and distinct
    indices = [ref.index for ref in spot_checks]
    assert len(set(indices)) == 30
    assert all(0 <= idx < 100 for idx in indices)


def test_spot_check_selection_faces():
    test_samples = [{"id": f"te_{i}"} for i in range(20)]
    cal_samples = [{"id": f"tr_{i}"} for i in range(60)]
    spot_checks = select_spot_checks(
        model_key="phase7_faces",
        test_samples=test_samples,
        cal_samples=cal_samples,
        is_faces=True,
    )
    assert len(spot_checks) == 30
    test_refs = [ref for ref in spot_checks if ref.split == "test"]
    cal_refs = [ref for ref in spot_checks if ref.split == "calibration"]
    assert len(test_refs) == 20
    assert len(cal_refs) == 10
    # All 20 test samples included
    assert {ref.index for ref in test_refs} == set(range(20))
    # 10 distinct calibration indices
    cal_indices = {ref.index for ref in cal_refs}
    assert len(cal_indices) == 10
    assert all(0 <= idx < 60 for idx in cal_indices)


def test_qat_float_evidence_marked_historical():
    """Verify that committed QAT fixture preserves historical float accuracy
    without recomputation."""
    with open(FIXTURE_PATHS["phase5_qat"]) as f:
        fixture = json.load(f)
    assert "paper" in fixture
    fa_committed = fixture["paper"]["float_accuracy"]
    assert fa_committed["source"] == "historical_fixture"
    assert fa_committed["recomputed"] is False
    assert abs(fa_committed["value"] - 0.9333333333333333) < 1e-12
    assert fa_committed["source_path"] is not None
    assert fa_committed["sample_count"] == 360


def test_qat_float_evidence_generation():
    """Verify that live paper data generator marks QAT float accuracy as historical."""
    pytest.importorskip("sklearn")
    data = generate_paper_data("phase5_qat")
    fa = data.float_accuracy
    assert fa.source == "historical_fixture"
    assert fa.recomputed is False
    assert abs(fa.value - 0.9333333333333333) < 1e-12
    assert fa.source_path is not None
    assert fa.sample_count == 360


def test_frozen_input_scale_and_golden_match():
    data = generate_paper_data("phase2_logreg")
    with open("examples/mnist/phase2_fixture.json") as f:
        fix = json.load(f)
    # Check leading rows match legacy fixture
    for i in range(len(fix["test_inputs"])):
        assert data.test[i].inputs == fix["test_inputs"][i]
        assert data.test[i].expected_label == fix["expected_labels"][i]


def test_mismatched_golden_prefix_rejected(tmp_path):
    # Corrupt a fixture copy and ensure export raises without writing
    bad_fixture_path = tmp_path / "bad_fixture.json"
    with open("examples/mnist/phase2_fixture.json") as f:
        fix = json.load(f)
    fix["test_inputs"][0][0] ^= 1  # corrupt one input pixel
    bad_fixture_path.write_text(json.dumps(fix))

    with pytest.raises(ValueError, match="golden mismatch"):
        export_paper_protocol(
            model_keys=["phase2_logreg"], fixture_paths={"phase2_logreg": bad_fixture_path}
        )
