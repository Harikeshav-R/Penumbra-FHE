# Client/server split — the privacy story, end to end

A runnable demo of Penumbra-FHE's deployment model (`PROJECT.md` §11, `ROADMAP.md` Phase 9):
encrypted inference across a **real process boundary**, where the server evaluates the model
on ciphertext and **never sees the plaintext input/output or the secret key**.

This example contains **no cryptography** — the crypto lives entirely in the `runtime/` crate.
The demo just orchestrates the roles.

## What it shows

```
┌── CLIENT ─────────────┐                    ┌──────── SERVER ─────────┐
│ KeySet.generate()      │  server.key +      │ serve: evaluate_graph   │
│   client.key  (SECRET) │  encrypted input   │  under FHE, holding     │
│   server.key  (PUBLIC) │ ─────────────────▶ │  ONLY server.key + graph│
│ encrypt(input) ────────┘                    │  (cannot decrypt)       │
│ decrypt(result) ◀───────── encrypted output ─┘                         │
└────────────────────────┘                    └─────────────────────────┘
```

Passing a `KeySet` to `model.predict_encrypted(x, keys=ks)` selects the **split** path: the
Python bridge drives the runtime's `encrypt` (client) → `serve` (server) → `decrypt` (client)
binaries as *separate processes* over files. The `serve` process is handed only the public
`server.key`, the graph, and the encrypted inputs — it is structurally unable to decrypt. The
keys are generated once and **reused** across inferences (keygen is the expensive per-model
step). A real deployment swaps the local `serve` process for an RPC to a remote server, with
the Python side unchanged.

The demo verifies the **golden invariant** (`AGENTS.md` §1.1) on the `tfhe` backend: the
decrypted output equals the quantized-cleartext oracle
(`penumbra.reference.evaluate_graph_int`) bit-for-bit. TFHE is exact, so any mismatch would be
a bug, never crypto noise. The same split runs under any backend — the reference is the same,
and only the comparator changes ([`docs/BACKENDS.md`](../../docs/BACKENDS.md)). Note that a
`KeySet` is **backend-specific**: key material is not portable between schemes.

## Running

Needs a Rust toolchain (`cargo`) on `PATH`; no network, no ML stack, no committed artifacts.
It uses a tiny single-`Linear` model so the whole round trip runs in seconds.

```bash
cd python && uv run python ../examples/client_server/demo.py
```

Expected: each sample prints its label + logits and `OK`, then a summary confirming the server
only ever handled ciphertext + the public key.

## The all-in-one alternative

Without a `KeySet`, `model.predict_encrypted(x)` uses the convenient single-process `predict`
binary (keygen → encrypt → evaluate → decrypt in one process, keys ephemeral) — see
[`docs/DEVELOPMENT.md`](../../docs/DEVELOPMENT.md). The split above is the faithful
privacy demonstration; the all-in-one is the quick path for local experimentation.
