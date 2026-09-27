"""Smoke tests for examples and tutorials (ROADMAP Phase 11)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_client_server_demo_runs_and_matches_oracle():
    """Smoke-test the client/server split demo (examples/client_server/demo.py).

    The demo exercises the full untrusted-server lifecycle:
      client: export model -> keygen -> encrypt input
      server: load model -> evaluate under FHE (public key only)
      client: decrypt output -> assert exact bit-for-bit match with oracle
    """
    cmd = [sys.executable, str(REPO / "examples" / "client_server" / "demo.py")]
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"client_server/demo.py failed with code {proc.returncode}\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )
