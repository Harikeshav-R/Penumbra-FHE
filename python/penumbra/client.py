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
ciphertext. Two round-trip paths live here:

    - **All-in-one** (``run_encrypted(graph, batch)`` with no keys): one ``predict`` binary
      call does keygen -> encrypt -> evaluate -> decrypt in a single process. Convenient; the
      keys are ephemeral. This is what ``predict_encrypted`` uses by default.
    - **Client/server split** (``run_encrypted(graph, batch, keys=ks)``): the roles run as
      *separate* processes over files — the client ``encrypt``s and ``decrypt``s with its
      secret key; the server ``serve``s with **only** the public server key + graph +
      ciphertext, never able to decrypt. This is the faithful trust boundary (`PROJECT.md`
      §11); a real deployment swaps the local ``serve`` process for an RPC, unchanged here.

Keys are persisted with :class:`KeySet` so a client can generate a pair once and **reuse**
it across inferences (keygen is the expensive per-``num_blocks`` cost). A key is tied to a
model's radix width (``num_blocks``); using it with a differently-sized model fails loudly.

This is a **subprocess** bridge, not PyO3: it shells out to the runtime's binaries via
``cargo run --release --bin <name>``. No new dependency — ``subprocess``/``json``/
``tempfile`` are stdlib. FHE is minutes-per-sample and needs a Rust toolchain, so this is a
developer/research path, not something CI exercises (the fast tests inject a fake runtime).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from penumbra.ir import Graph

# The runtime crate lives at ``<repo>/runtime`` relative to this file
# (``python/penumbra/client.py`` -> parents[2] is the repo root). Overridable so a wheel
# install or an out-of-tree checkout can point at the crate explicitly.
_ENV_RUNTIME_DIR = "PENUMBRA_RUNTIME_DIR"

# Filenames used inside a KeySet directory.
_CLIENT_KEY = "client.key"
_SERVER_KEY = "server.key"
_KEY_META = "meta.json"  # records num_blocks (the on-disk keys are opaque bincode to Python)


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


def _require_cargo() -> None:
    """Fail loudly if the Rust toolchain is unavailable (``AGENTS.md`` §1.4)."""
    if shutil.which("cargo") is None:
        raise RuntimeError(
            "the encrypted round trip needs a Rust toolchain: `cargo` was not found on PATH. "
            "Install Rust (https://rustup.rs) to run predict_encrypted; the quantize/export "
            "path works without it."
        )


def _run_bin(runtime_dir: Path, bin_name: str, args: list[str], *, stdin: str | None = None) -> str:
    """Run a runtime binary via ``cargo run --release --bin <bin_name> -- <args>``.

    Returns its stdout. Raises an actionable :class:`RuntimeError` with the binary's stderr on a
    non-zero exit (``AGENTS.md`` §1.4). ``stdin`` is piped in when given.
    """
    proc = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--bin", bin_name, "--", *args],
        cwd=str(runtime_dir),
        input=stdin,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"the runtime `{bin_name}` binary failed "
            f"(exit {proc.returncode}).\n--- stderr ---\n{proc.stderr.strip()}"
        )
    return proc.stdout


@dataclass(frozen=True)
class KeySet:
    """A persisted client/server key pair, tied to a model's radix width (``num_blocks``).

    The client generates one pair with :meth:`generate` and **reuses** it across inferences
    (keygen is the expensive per-``num_blocks`` cost, `ROADMAP.md` Phase 9). :attr:`directory`
    holds ``client.key`` (secret — encrypt/decrypt) and ``server.key`` (public — evaluate), plus
    a small ``meta.json`` recording ``num_blocks`` (the bincode key files are opaque to Python).

    A key pair is only usable with a model whose graph declares the same ``num_blocks``; passing
    it to a differently-sized model raises (the "key mismatch" failure mode, `AGENTS.md` §1.4).
    """

    directory: Path
    num_blocks: int

    @property
    def client_key(self) -> Path:
        return self.directory / _CLIENT_KEY

    @property
    def server_key(self) -> Path:
        return self.directory / _SERVER_KEY

    @staticmethod
    def generate(num_blocks: int, directory: str | os.PathLike | None = None) -> KeySet:
        """Generate a fresh key pair for ``num_blocks``, writing it into ``directory``.

        If ``directory`` is omitted, a persistent temporary directory is created (the caller owns
        cleanup — call :meth:`save` to relocate the keys somewhere durable, or delete the dir).
        Shells out to the runtime's ``keygen`` binary; needs a Rust toolchain.
        """
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        _require_cargo()
        runtime_dir = _runtime_dir()
        out_dir = (
            Path(directory)
            if directory is not None
            else Path(tempfile.mkdtemp(prefix="penumbra_keys_"))
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        ck, sk = out_dir / _CLIENT_KEY, out_dir / _SERVER_KEY
        _run_bin(runtime_dir, "keygen", [str(num_blocks), str(ck), str(sk)])
        (out_dir / _KEY_META).write_text(json.dumps({"num_blocks": int(num_blocks)}))
        return KeySet(directory=out_dir, num_blocks=int(num_blocks))

    def save(self, directory: str | os.PathLike) -> KeySet:
        """Copy this key pair into ``directory`` (durable persistence) and return the new handle."""
        dst = Path(directory)
        dst.mkdir(parents=True, exist_ok=True)
        for name in (_CLIENT_KEY, _SERVER_KEY, _KEY_META):
            src = self.directory / name
            if not src.is_file():
                raise FileNotFoundError(f"cannot save KeySet: {src} is missing")
            shutil.copy2(src, dst / name)
        return KeySet(directory=dst, num_blocks=self.num_blocks)

    @staticmethod
    def load(directory: str | os.PathLike) -> KeySet:
        """Load a previously generated/saved key pair from ``directory``.

        Fails loudly if a key file or the ``num_blocks`` metadata is missing (``AGENTS.md`` §1.4).
        """
        d = Path(directory)
        for name in (_CLIENT_KEY, _SERVER_KEY, _KEY_META):
            if not (d / name).is_file():
                raise FileNotFoundError(
                    f"no Penumbra KeySet at {d}: missing {name}. Generate one with "
                    "KeySet.generate(num_blocks) or point at a saved key directory."
                )
        meta = json.loads((d / _KEY_META).read_text())
        return KeySet(directory=d, num_blocks=int(meta["num_blocks"]))


def run_encrypted(
    graph: Graph, int_inputs: list[list[int]], *, keys: KeySet | None = None
) -> list[list[int]]:
    """Run encrypted inference on ``graph`` for a batch of quantized integer input rows.

    ``int_inputs`` are already in the graph's input domain — the caller (the client) owns
    quantization; the runtime never sees floats or scales (``PROJECT.md`` §11). Returns the
    decrypted graph-output tensor for each row (in order); the caller interprets them (argmax /
    label).

    Two paths (``PROJECT.md`` §11):

    * ``keys is None`` — the **all-in-one** ``predict`` binary: keygen -> encrypt -> evaluate ->
      decrypt in one process, keys ephemeral. Keygen runs once and is reused across the batch.
    * ``keys`` given — the **client/server split**: the client ``encrypt``s with its secret key,
      the server ``serve``s with **only** the public server key + graph, and the client
      ``decrypt``s. Reuses the persisted ``keys`` (no keygen). Fails loudly if the keys' radix
      width doesn't match the model (``AGENTS.md`` §1.4).

    Raises with an actionable message if the toolchain is missing or a binary fails
    (``AGENTS.md`` §1.4).
    """
    if not int_inputs:
        return []
    # Validate the key/model radix width up front (before any toolchain work), so the "key
    # mismatch" failure is reported clearly even without cargo present (``AGENTS.md`` §1.4).
    if keys is not None and keys.num_blocks != graph.num_blocks:
        raise ValueError(
            f"key/model mismatch: these keys were generated for num_blocks={keys.num_blocks}, "
            f"but the model needs num_blocks={graph.num_blocks}. Generate a KeySet for this "
            "model (keys are tied to a model's radix width)."
        )
    _require_cargo()
    runtime_dir = _runtime_dir()

    if keys is not None:
        return _run_split(runtime_dir, graph, int_inputs, keys)
    return _run_all_in_one(runtime_dir, graph, int_inputs)


def _run_all_in_one(
    runtime_dir: Path, graph: Graph, int_inputs: list[list[int]]
) -> list[list[int]]:
    """The single-process ``predict`` path (ephemeral keys, keygen reused across the batch)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fhe", delete=False, encoding="utf-8") as fh:
        fh.write(graph.to_json())
        model_path = fh.name
    try:
        stdout = _run_bin(runtime_dir, "predict", [model_path], stdin=json.dumps(int_inputs))
    finally:
        os.unlink(model_path)
    return _parse_outputs(stdout)


def _run_split(
    runtime_dir: Path, graph: Graph, int_inputs: list[list[int]], keys: KeySet
) -> list[list[int]]:
    """The client/server split: encrypt (client) -> serve (server) -> decrypt (client).

    Assumes the key/model ``num_blocks`` have already been validated by :func:`run_encrypted`.
    """
    with tempfile.TemporaryDirectory(prefix="penumbra_run_") as work:
        workdir = Path(work)
        model_path = workdir / "model.fhe"
        in_cts = workdir / "in.cts"
        out_cts = workdir / "out.cts"
        model_path.write_text(graph.to_json())

        # Client: encrypt the inputs with the secret key.
        _run_bin(
            runtime_dir,
            "encrypt",
            [str(keys.client_key), str(in_cts)],
            stdin=json.dumps(int_inputs),
        )
        # Server: evaluate under FHE with ONLY the public server key + graph + ciphertext.
        _run_bin(
            runtime_dir, "serve", [str(model_path), str(keys.server_key), str(in_cts), str(out_cts)]
        )
        # Client: decrypt the result with the secret key.
        stdout = _run_bin(runtime_dir, "decrypt", [str(keys.client_key), str(out_cts)])
    return _parse_outputs(stdout)


def _parse_outputs(stdout: str) -> list[list[int]]:
    """Parse a runtime binary's ``{"outputs": [[int, ...], ...]}`` stdout, failing loudly."""
    try:
        payload = json.loads(stdout)
        outputs = payload["outputs"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"could not parse runtime output as {{'outputs': [[int, ...], ...]}}: {e}\n"
            f"--- stdout ---\n{stdout.strip()}"
        ) from e
    return [[int(v) for v in row] for row in outputs]
