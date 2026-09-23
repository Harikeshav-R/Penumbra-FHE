"""In-process PyO3 bridge to the Penumbra-FHE runtime.

Drives the runtime to do keygen -> encrypt -> evaluate -> decrypt and return a
prediction, backing the one-call convenience API ``model.predict_encrypted(x)``
(``PROJECT.md`` §12).

All crypto runs in-process via compiled PyO3 bindings (``penumbra._penumbra``),
releasing the Python GIL during heavy FHE execution.

Privacy model (``PROJECT.md`` §11):
    - The **client** holds the secret key (encrypt/decrypt) and never sends it to the server.
    - The **server** holds the public server key + graph + weights and evaluates over ciphertext,
      never able to decrypt.

Two execution paths:
    - **All-in-one** (``run_encrypted(graph, batch)`` with no keys): generates ephemeral keys
      and runs keygen -> encrypt -> evaluate -> decrypt in-process. This is what
      ``predict_encrypted`` uses by default.
    - **Client/server split** (``run_encrypted(graph, batch, keys=ks)``): reuses a persisted
      :class:`KeySet` across inferences (keygen is the expensive one-time cost). The client
      encrypts and decrypts with ``client.key``; the server evaluates using ``server.key``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from penumbra import _penumbra
from penumbra._penumbra import CryptoProfile
from penumbra.ir import Graph

DEFAULT_BACKEND = "tfhe"

_CLIENT_KEY = "client.key"
_SERVER_KEY = "server.key"


def available_backends() -> list[str]:
    """Backend names compiled into this build (CKKS is absent from published wheels)."""
    return list(_penumbra.available_backends())


@dataclass(frozen=True)
class KeySet:
    """A persisted client/server key pair, tied to a backend and parameter profile.

    The client generates one pair with :meth:`generate` and **reuses** it across inferences
    (keygen is the expensive one-time cost, `ROADMAP.md` Phase 9). :attr:`directory`
    holds ``client.key`` (secret — encrypt/decrypt) and ``server.key`` (public — evaluate).

    All key metadata (backend, radix width, and crypto profile) is stored within the binary
    envelopes of the key files and recovered via :func:`penumbra._penumbra.key_info`.
    """

    directory: Path
    backend: str
    num_blocks: int | None  # None under CKKS (it has no radix width)
    profile: str

    @property
    def client_key(self) -> Path:
        return self.directory / _CLIENT_KEY

    @property
    def server_key(self) -> Path:
        return self.directory / _SERVER_KEY

    @staticmethod
    def generate(
        num_blocks: int = 1,
        directory: str | os.PathLike | None = None,
        *,
        backend: str = DEFAULT_BACKEND,
        profile: CryptoProfile | None = None,
    ) -> KeySet:
        """Generate a fresh key pair, writing it into ``directory``.

        If ``directory`` is omitted, a persistent temporary directory is created (the caller owns
        cleanup — call :meth:`save` to relocate the keys somewhere durable, or delete the dir).
        """
        if backend == "tfhe" and num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        if profile is not None and profile.backend != backend:
            raise ValueError(
                f"profile/backend mismatch: profile is for backend '{profile.backend}', "
                f"but this run uses '{backend}'"
            )
        out_dir = (
            Path(directory)
            if directory is not None
            else Path(tempfile.mkdtemp(prefix="penumbra_keys_"))
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        ck_bytes, sk_bytes = _penumbra.keygen(backend, num_blocks, profile)
        (out_dir / _CLIENT_KEY).write_bytes(ck_bytes)
        (out_dir / _SERVER_KEY).write_bytes(sk_bytes)

        info = _penumbra.key_info(ck_bytes)
        return KeySet(
            directory=out_dir,
            backend=info.backend,
            num_blocks=info.num_blocks,
            profile=info.profile,
        )

    def save(self, directory: str | os.PathLike) -> KeySet:
        """Copy this key pair into ``directory`` (durable persistence) and return the new handle."""
        dst = Path(directory)
        dst.mkdir(parents=True, exist_ok=True)
        for name in (_CLIENT_KEY, _SERVER_KEY):
            src = self.directory / name
            if not src.is_file():
                raise FileNotFoundError(f"cannot save KeySet: {src} is missing")
            shutil.copy2(src, dst / name)
        return KeySet(
            directory=dst,
            backend=self.backend,
            num_blocks=self.num_blocks,
            profile=self.profile,
        )

    @staticmethod
    def load(directory: str | os.PathLike) -> KeySet:
        """Load a previously generated/saved key pair from ``directory``.

        Fails loudly if a key file is missing (``AGENTS.md`` §1.4).
        """
        d = Path(directory)
        for name in (_CLIENT_KEY, _SERVER_KEY):
            if not (d / name).is_file():
                raise FileNotFoundError(
                    f"no Penumbra KeySet at {d}: missing {name}. Generate one with "
                    "KeySet.generate(num_blocks) or point at a saved key directory."
                )
        ck_bytes = (d / _CLIENT_KEY).read_bytes()
        info = _penumbra.key_info(ck_bytes)
        return KeySet(
            directory=d,
            backend=info.backend,
            num_blocks=info.num_blocks,
            profile=info.profile,
        )


def run_encrypted(
    graph: Graph,
    int_inputs: list[list[int]],
    *,
    keys: KeySet | None = None,
    backend: str = DEFAULT_BACKEND,
    profile: CryptoProfile | None = None,
) -> list[list[int]]:
    """Run encrypted inference on ``graph`` for a batch of quantized integer input rows.

    ``int_inputs`` are already in the graph's input domain — the caller (the client) owns
    quantization; the runtime never sees floats or scales (``PROJECT.md`` §11). Returns the
    decrypted graph-output tensor for each row (in order); the caller interprets them (argmax /
    label).

    Two paths (``PROJECT.md`` §11):
        * ``keys is None`` — the **all-in-one** in-process execution: keygen -> encrypt ->
          evaluate -> decrypt, keys ephemeral. Keygen runs once and is reused across the batch.
        * ``keys`` given — the **client/server split**: the client ``encrypt``s with its secret
          key, the server ``evaluate``s with **only** the public server key + graph, and the
          client ``decrypt``s. Reuses the persisted ``keys`` (no keygen). Fails loudly if the
          keys' radix width or backend doesn't match the model/run (``AGENTS.md`` §1.4).
    """
    if not int_inputs:
        return []
    if keys is not None and profile is not None:
        raise ValueError(
            "cannot specify both 'keys' and 'profile': keys already carry their parameter profile; "
            "regenerate keys under the desired profile instead."
        )
    if profile is not None and profile.backend != backend:
        raise ValueError(
            f"profile/backend mismatch: profile is for backend '{profile.backend}', "
            f"but this run uses '{backend}'"
        )
    if keys is not None and keys.backend != backend:
        raise ValueError(
            f"key/backend mismatch: these keys are for backend '{keys.backend}', "
            f"this run uses '{backend}'. "
            "Key material is not portable across backends (see docs/BACKENDS.md)."
        )
    if keys is not None and keys.num_blocks is not None and keys.num_blocks != graph.num_blocks:
        raise ValueError(
            f"key/model mismatch: these keys were generated for num_blocks={keys.num_blocks}, "
            f"but the model needs num_blocks={graph.num_blocks}. Generate a KeySet for this "
            "model (keys are tied to a model's radix width)."
        )

    if keys is None:
        return _penumbra.predict(backend, graph.to_json(), int_inputs, profile)

    ck_bytes = keys.client_key.read_bytes()
    sk_bytes = keys.server_key.read_bytes()
    in_cts = _penumbra.encrypt(backend, ck_bytes, int_inputs)
    out_cts = _penumbra.evaluate(backend, graph.to_json(), sk_bytes, in_cts)
    return _penumbra.decrypt(backend, ck_bytes, out_cts)
