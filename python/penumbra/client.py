"""Bridge to the Rust runtime + the client-side inference round trip.

Drives the runtime to do keygen -> encrypt -> evaluate -> decrypt and return a
prediction, backing the one-call convenience API ``model.predict_encrypted(x)``
(``PROJECT.md`` §12).

Bridge strategy (``PROJECT.md`` §15):
    - Phase 1-8: IR file + subprocess — Python writes ``model.fhe``, the Rust runtime
      reads it. Simplest; start here. **This module.**
    - Phase 9: PyO3 in-process bindings — better ergonomics, no file/subprocess round
      trip. Be deliberate about what crosses the boundary (IR + ciphertext handles, not
      giant copies). A later refinement.

Privacy model (``PROJECT.md`` §11): the client holds the secret key (encrypt/decrypt);
the server holds the public server key + plaintext weights and only ever touches
ciphertext. In this subprocess bridge both roles run in the same local process for
convenience — the *runtime* still only ever sees the quantized integer input (never a
float or a scale) and the encrypted graph, so the trust boundary the design describes is
faithfully modelled (a real deployment swaps the ``cargo run`` for an RPC to a remote
server, unchanged on the Python side).

This is a **subprocess** bridge, not PyO3: it shells out to the runtime's ``predict``
binary (``runtime/src/bin/predict.rs``) via ``cargo run --release --bin predict``, passing
the exported IR graph as a temp file and the quantized integer input batch on stdin, and
parses the decrypted graph outputs back. No new dependency — ``subprocess``/``json``/
``tempfile`` are stdlib. FHE is minutes-per-sample and needs a Rust toolchain, so this is a
developer/research path, not something CI exercises (the fast tests inject a fake runtime).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from penumbra.ir import Graph

# The runtime crate lives at ``<repo>/runtime`` relative to this file
# (``python/penumbra/client.py`` -> parents[2] is the repo root). Overridable so a wheel
# install or an out-of-tree checkout can point at the crate explicitly.
_ENV_RUNTIME_DIR = "PENUMBRA_RUNTIME_DIR"


def _runtime_dir() -> Path:
    """Locate the Rust runtime crate directory, honoring ``PENUMBRA_RUNTIME_DIR``.

    Fails loudly with an actionable message if the directory (or its ``Cargo.toml``) is
    missing — a silent wrong path would surface as a confusing cargo error later
    (``AGENTS.md`` §1.4).
    """
    override = os.environ.get(_ENV_RUNTIME_DIR)
    if override:
        d = Path(override)
    else:
        d = Path(__file__).resolve().parents[2] / "runtime"
    if not (d / "Cargo.toml").is_file():
        raise FileNotFoundError(
            f"Penumbra runtime crate not found at {d} (no Cargo.toml). Set the "
            f"{_ENV_RUNTIME_DIR} environment variable to the runtime/ directory of a "
            "Penumbra-FHE checkout, or run from the repository."
        )
    return d


def run_encrypted(graph: Graph, int_inputs: list[list[int]]) -> list[list[int]]:
    """Run encrypted inference on ``graph`` for a batch of quantized integer input rows.

    Serializes ``graph`` to a temporary ``.fhe`` file, invokes the runtime's ``predict``
    binary (``cargo run --release --bin predict -- <file>``) with ``int_inputs`` piped on
    stdin as JSON, and returns the decrypted graph-output tensor for each row (in order).
    Keygen happens **once** inside the binary and is reused across the whole batch.

    ``int_inputs`` are already in the graph's input domain — the caller (the client) owns
    quantization; the runtime never sees floats or scales (``PROJECT.md`` §11). The returned
    rows are raw decrypted values (wide logits, or a single 2-class ``Argmax`` label bit);
    the caller interprets them (argmax / label).

    Raises with an actionable message if the toolchain is missing or the binary fails
    (``AGENTS.md`` §1.4).
    """
    if not int_inputs:
        return []
    if shutil.which("cargo") is None:
        raise RuntimeError(
            "the encrypted round trip needs a Rust toolchain: `cargo` was not found on PATH. "
            "Install Rust (https://rustup.rs) to run predict_encrypted; the quantize/export "
            "path works without it."
        )
    runtime_dir = _runtime_dir()

    # Write the exported IR graph to a temp file the binary reads (the same JSON `export`
    # writes). Delete it after the run.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fhe", delete=False, encoding="utf-8") as fh:
        fh.write(graph.to_json())
        model_path = fh.name

    try:
        proc = subprocess.run(
            ["cargo", "run", "--release", "--quiet", "--bin", "predict", "--", model_path],
            cwd=str(runtime_dir),
            input=json.dumps(int_inputs),
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(model_path)

    if proc.returncode != 0:
        raise RuntimeError(
            "the runtime `predict` binary failed "
            f"(exit {proc.returncode}).\n--- stderr ---\n{proc.stderr.strip()}"
        )

    try:
        payload = json.loads(proc.stdout)
        outputs = payload["outputs"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"could not parse runtime output as {{'outputs': [[int, ...], ...]}}: {e}\n"
            f"--- stdout ---\n{proc.stdout.strip()}"
        ) from e
    return [[int(v) for v in row] for row in outputs]
