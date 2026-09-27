"""Replay committed Olivetti faces fixture under FHE.

Run: uv run python examples/faces/run.py [--model KEY] [--samples N] [--backend tfhe|ckks]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # examples/ is not a package; share examples/_replay.py

from _replay import Example, main  # noqa: E402

EXAMPLES = [
    Example("phase7_faces", HERE / "phase7_faces_fixture.json"),
]

if __name__ == "__main__":
    main(EXAMPLES, default="phase7_faces", description=__doc__ or "")
