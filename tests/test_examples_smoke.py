"""Smoke tests for examples and tutorials (ROADMAP Phase 11).

Default CI runs fast smoke tests (<= 60 s each):
  - Client/server demo (examples/client_server/demo.py)
  - MNIST logistic regression replay (examples/mnist/run.py --model phase2_logreg --samples 1)
  - Tabular tree ensemble replay (examples/trees/run.py --samples 1)
  - Tabular MLP replay (examples/tabular/run.py --samples 1)
  - Rejection hint test when CKKS is unavailable (examples/trees/run.py --backend ckks --samples 1)

Full end-to-end runs require PENUMBRA_E2E=1 because of long TFHE bootstrap durations:
  - MNIST CNN replay (examples/mnist/run.py --samples 1, default phase6_onnx, ~218 s)
  - Olivetti faces CNN replay (examples/faces/run.py --samples 1, ~371 s)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from penumbra import available_backends

REPO = Path(__file__).resolve().parent.parent

E2E = os.environ.get("PENUMBRA_E2E") == "1"
slow = pytest.mark.skipif(not E2E, reason="slow FHE example; set PENUMBRA_E2E=1")


def _run_script(rel_path: str, *args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(REPO / rel_path), *args]
    return subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_client_server_demo_runs_and_matches_oracle():
    """Smoke-test the client/server split demo (examples/client_server/demo.py).

    The demo exercises the full untrusted-server lifecycle:
      client: export model -> keygen -> encrypt input
      server: load model -> evaluate under FHE (public key only)
      client: decrypt output -> assert exact bit-for-bit match with oracle
    """
    proc = _run_script("examples/client_server/demo.py")
    assert proc.returncode == 0, (
        f"client_server/demo.py failed with code {proc.returncode}\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )


def test_mnist_logreg_replay_smoke():
    """Fast smoke test for MNIST runner with logistic regression (phase2_logreg)."""
    proc = _run_script("examples/mnist/run.py", "--model", "phase2_logreg", "--samples", "1")
    assert proc.returncode == 0, (
        f"mnist/run.py --model phase2_logreg failed with code {proc.returncode}\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )


def test_trees_replay_smoke():
    """Fast smoke test for tree runner with default model (phase8_trees)."""
    proc = _run_script("examples/trees/run.py", "--samples", "1")
    assert proc.returncode == 0, (
        f"trees/run.py failed with code {proc.returncode}\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )


def test_tabular_mlp_replay_smoke():
    """Smoke test for tabular MLP runner (phase11_tabular_mlp)."""
    proc = _run_script("examples/tabular/run.py", "--samples", "1")
    assert proc.returncode == 0, (
        f"tabular/run.py failed with code {proc.returncode}\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )


def test_unified_runner_smoke():
    """Smoke test for the unified runner (examples/run.py)."""
    proc = _run_script("examples/run.py", "trees", "--samples", "1")
    assert proc.returncode == 0, (
        f"examples/run.py trees failed with code {proc.returncode}\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )


@pytest.mark.skipif("ckks" in available_backends(), reason="CKKS is compiled in")
def test_ckks_unavailable_hint_smoke():
    """Runner prints actionable build hint when CKKS backend is requested without support."""
    proc = _run_script("examples/trees/run.py", "--backend", "ckks", "--samples", "1")
    assert proc.returncode != 0, f"expected non-zero exit, got {proc.returncode}"
    assert "--features ckks" in proc.stderr, f"expected '--features ckks' in stderr: {proc.stderr}"


@slow
def test_mnist_default_replay_slow():
    """Slow full E2E test for MNIST runner with default CNN (phase6_onnx)."""
    proc = _run_script("examples/mnist/run.py", "--samples", "1")
    assert proc.returncode == 0, (
        f"mnist/run.py failed with code {proc.returncode}\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )


@slow
def test_faces_replay_slow():
    """Slow full E2E test for Olivetti faces runner (phase7_faces)."""
    proc = _run_script("examples/faces/run.py", "--samples", "1")
    assert proc.returncode == 0, (
        f"faces/run.py failed with code {proc.returncode}\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )
