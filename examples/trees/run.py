"""Replay committed tree ensemble fixtures under FHE.

Run: uv run python examples/trees/run.py [--model KEY] [--samples N] [--backend tfhe|ckks]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # examples/ is not a package; share examples/_replay.py

from _replay import Example, main  # noqa: E402

EXAMPLES = [
    Example("phase8_trees", HERE / "phase8_trees_fixture.json"),
    Example("phase8_xgb", HERE / "phase8_xgb_fixture.json"),
]

if __name__ == "__main__":
    main(EXAMPLES, default="phase8_trees", description=__doc__ or "")
