"""Standalone server process for the client/server split demo.

The server process holds ONLY the public server key, the model IR graph, and the input
ciphertext. It structurally cannot decrypt (it never receives the client secret key).
"""

from __future__ import annotations

import sys
from pathlib import Path

from penumbra import _penumbra


def main() -> None:
    if len(sys.argv) != 5:
        print(
            "usage: python server.py <model.fhe> <server.key> <in.cts> <out.cts>",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = Path(sys.argv[1])
    server_key_path = Path(sys.argv[2])
    in_cts_path = Path(sys.argv[3])
    out_cts_path = Path(sys.argv[4])

    graph_json = model_path.read_text(encoding="utf-8")
    server_key_bytes = server_key_path.read_bytes()
    in_cts_bytes = in_cts_path.read_bytes()

    # Determine backend from server key envelope
    key_info = _penumbra.key_info(server_key_bytes)
    backend = key_info.backend

    # Evaluate encrypted batch under FHE using ONLY the public server key
    out_cts_bytes = _penumbra.evaluate(backend, graph_json, server_key_bytes, in_cts_bytes)

    out_cts_path.write_bytes(out_cts_bytes)


if __name__ == "__main__":
    main()
