"""Client/server split demo — the Phase-9 privacy story, end to end (``PROJECT.md`` §11).

Runs a real encrypted inference across a **process boundary**: the client generates keys and
encrypts, a *separate* server process evaluates the model under FHE holding **only** the public
server key + graph (it structurally cannot decrypt), and the client decrypts the result. This
is the faithful trust boundary — the server "sees" only ciphertext and the model, never the
plaintext input/output or the secret key.

Under the hood, the client and server communicate via serialized files (model graph, public key,
and ciphertext batches). The server process (``examples/client_server/server.py``) is spawned as an
isolated Python process that never receives or accesses ``client.key``.

Kept tiny (a single ``Linear``, small radix) so the whole thing runs in seconds.
No network, no ML stack, no committed artifacts.

Run:  uv run python examples/client_server/demo.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from penumbra import KeySet, Linear, Model, _penumbra
from penumbra.quantization.spec import QuantSpec
from penumbra.reference import evaluate_graph_int


def main() -> None:
    # A trivial 3-class linear classifier over 8 features — the smallest thing that exercises a
    # wide multi-logit head (the client argmaxes the decrypted logits, ``PROJECT.md`` §11).
    rng = np.random.default_rng(0)
    model = Model([Linear(weight=rng.normal(size=(3, 8)), bias=rng.normal(size=3))], input_bits=4)
    calibration = rng.uniform(0.0, 16.0, size=(64, 8))
    model.quantize(calibration, n_bits=4)
    assert model.graph is not None
    print(f"model quantized: {len(model.graph.nodes)} node(s), num_blocks={model.graph.num_blocks}")

    # --- Client: the key ceremony. Generate a key pair ONCE, tied to this model's radix width,
    # and persist it so it can be reused across inferences (keygen is the expensive step). ---
    with tempfile.TemporaryDirectory(prefix="penumbra_demo_keys_") as key_dir:
        print("client: generating a key pair (this holds the SECRET key) ...")
        keys = KeySet.generate(model.graph.num_blocks, directory=key_dir)
        keys = keys.save(Path(key_dir) / "saved")  # prove save/load persistence
        keys = KeySet.load(keys.directory)
        print(f"client: keys for num_blocks={keys.num_blocks} at {keys.directory}")
        print(
            "  client.key = SECRET (encrypt/decrypt, never leaves the client);\n"
            "  server.key = PUBLIC (evaluate only, cannot decrypt) — the only key the server gets."
        )

        samples = calibration[:3]

        # --- Client: quantize and encrypt the input batch using client.key ---
        in_spec = QuantSpec(scale=model.input_scale, bits=model.input_bits, signed=False)
        int_inputs = [in_spec.quantize(row).tolist() for row in samples]

        with tempfile.TemporaryDirectory(prefix="penumbra_demo_work_") as work_dir:
            work = Path(work_dir)
            model_file = work / "model.fhe"
            in_cts_file = work / "in.cts"
            out_cts_file = work / "out.cts"

            model_file.write_text(model.graph.to_json(), encoding="utf-8")
            ck_bytes = keys.client_key.read_bytes()
            in_cts_bytes = _penumbra.encrypt(keys.backend, ck_bytes, int_inputs)
            in_cts_file.write_bytes(in_cts_bytes)

            # --- Server: evaluate in an isolated Python process with ONLY public server.key ---
            server_script = Path(__file__).parent / "server.py"
            print(f"client: launching separate server process ({server_script.name}) ...")
            subprocess.run(
                [
                    sys.executable,
                    str(server_script),
                    str(model_file),
                    str(keys.server_key),
                    str(in_cts_file),
                    str(out_cts_file),
                ],
                check=True,
            )

            # --- Client: decrypt the server response with client secret key ---
            out_cts_bytes = out_cts_file.read_bytes()
            logits = _penumbra.decrypt(keys.backend, ck_bytes, out_cts_bytes)
            labels = [int(np.argmax(row)) for row in logits]

        # --- Verify the golden invariant: the encrypted result equals the quantized-cleartext
        # oracle bit-for-bit (``AGENTS.md`` §1.1). TFHE is exact — any mismatch would be a bug. ---
        ok = True
        for i, row in enumerate(samples):
            xq = in_spec.quantize(row).tolist()
            ref = evaluate_graph_int(model.graph, {"x": xq})[model.graph.outputs[0]]
            match = logits[i] == ref and labels[i] == int(np.argmax(ref))
            ok = ok and match
            print(
                f"  sample {i}: label={labels[i]} logits={logits[i]} "
                f"(cleartext {ref}) {'OK' if match else 'MISMATCH'}"
            )

        if not ok:
            raise SystemExit("GOLDEN VIOLATION: encrypted output != quantized-cleartext oracle")
        print(
            "\nAll samples matched the quantized-cleartext oracle bit-for-bit. The server process "
            "evaluated the model on ciphertext with only the public key — it never saw the input, "
            "the output, or the secret key."
        )


if __name__ == "__main__":
    main()
