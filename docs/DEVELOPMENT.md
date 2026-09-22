# Development Guide

How to build, test, and work on Penumbra-FHE. Read [`PROJECT.md`](../PROJECT.md) and
[`ROADMAP.md`](../ROADMAP.md) for the architecture and plan, and [`AGENTS.md`](../AGENTS.md)
for the working rules (they apply to humans too).

## Toolchain

| Tool | Version | Notes |
|---|---|---|
| Rust | stable ≥ 1.83 | `tfhe-rs` needs a recent stable. Install via [rustup](https://rustup.rs). |
| Python | 3.10–3.12 | Pinned in `python/pyproject.toml` (`>=3.10,<3.13`). 3.13+ not yet supported by the ML stack. |
| [uv](https://docs.astral.sh/uv/) | latest | **The project standard** for Python env/deps — not poetry/pip/conda. |

> ⚠️ The CKKS backend may require a **different toolchain**: `poulpy` pins a nightly upstream
> and depends on `libm`'s `unstable-float`. Whether the workspace needs nightly is a blocking
> question for the Phase-12.0 spike, and the answer gets recorded in
> [`docs/NOTES-ckks.md`](./NOTES-ckks.md). Whatever it turns out to be, the **TFHE backend
> stays on stable** so an upstream toolchain change cannot break the reference backend's gate.
>
> `poulpy`'s CPU backend is also architecture-specific — `poulpy-cpu-arm` (NEON) on Apple
> Silicon, `poulpy-cpu-avx` (AVX2/FMA) on x86-64. Benchmarks must pin one.

## Layout

```
python/      # Python front end: ONNX loader, quantization, IR emitter (Layer 3)
runtime/     # Rust runtime: ops, IR deserialization, eval loop (Layers 1–2)
examples/    # use cases (mnist, faces) — graphs only, NO crypto
tests/       # cross-cutting + golden exactness tests
docs/        # this guide and the spec docs
```

After the Phase-12.1 workspace refactor, `runtime/` becomes a Cargo workspace under `crates/`
(`PROJECT.md` §13):

```
crates/penumbra-core/    # Layer 2: IR, eval loop, bit-width, the `Backend` trait — NO crypto
crates/penumbra-tfhe/    # Layer 1: the tfhe-rs backend (the reference)
crates/penumbra-ckks/    # Layer 1: the poulpy-ckks backend
crates/penumbra-bench/   # the shared comparison harness
```

## Building & testing

### Rust runtime

```bash
cd runtime
cargo build                # debug build (fine for correctness)
cargo test --release       # run tests — ALWAYS use --release for FHE
```

Each backend can be built, tested, and benchmarked:

```bash
# TFHE backend (stable toolchain)
cargo test --release -p penumbra-tfhe
cargo run -p penumbra-bench --release --bin penumbra-bench-report -- --models phase2_logreg
cargo bench -p penumbra-bench

# CKKS backend (nightly toolchain, requires --features ckks)
cargo +nightly test -p penumbra-ckks --features ckks --release
cargo +nightly test -p penumbra-bench --features ckks --release
cargo +nightly run -p penumbra-bench --features ckks --release --bin penumbra-bench-report -- --models phase2_logreg
PENUMBRA_BENCH_MODELS=phase2_logreg cargo +nightly bench -p penumbra-bench --features ckks
```

> ⚠️ **Build in `--release` for anything that runs FHE.** Debug builds are *extremely* slow
> (orders of magnitude) — true of `poulpy` as much as of `tfhe-rs`. The first compile is slow
> regardless; both libraries pull large dependency trees. The `hello_fhe` test proves the
> toolchain works (encrypt → plaintext-weight arithmetic → LUT-via-PBS → decrypt).

### Python front end

```bash
cd python
uv sync --all-extras       # create .venv and install deps (incl. torch/brevitas)
uv run pytest              # run the Python test suite
```

Use `uv sync` (without `--all-extras`) for the lightweight core (onnx + numpy only),
without the heavy `torch`/`brevitas` ML extra.

> **Behind a corporate proxy / TLS-intercepting firewall?** If `uv` fails to download a
> Python interpreter or packages with `invalid peer certificate: UnknownIssuer`, add
> `--native-tls` (use the OS trust store, which has your corporate CA) — e.g.
> `uv python install 3.12 --native-tls` and `uv sync --dev --native-tls`. You can make this
> the default by setting `UV_NATIVE_TLS=1` in your environment.

## Running encrypted inference from Python

`model.predict_encrypted(x)` runs the real encrypted forward pass (ROADMAP Phase 9). It
quantizes `x`, hands the exported IR + the quantized batch to the Rust runtime, and returns the
client-side prediction:

```python
import penumbra as fhe

model = fhe.load_onnx("model.onnx")
model.quantize(calibration_data, n_bits=6)
pred = model.predict_encrypted(x)             # single sample -> int label; batch -> list[int]
labels, logits = model.predict_encrypted(X, return_logits=True)   # also get the raw logits
```

This is the **subprocess bridge**: Python shells out to the runtime's `predict` binary via
`cargo run --release --bin predict`, so it needs a **Rust toolchain** on `PATH` and a checkout
of this repo (or set `PENUMBRA_RUNTIME_DIR` to the `runtime/` crate). Keygen runs once per call
and is reused across the batch; FHE is seconds-to-minutes per sample. The server side only ever
sees the quantized integer input and the graph — never a float or a scale (`PROJECT.md` §11).
In-process PyO3 bindings and wheels are the remaining Phase-9 work.

> **Choosing a backend.** `predict_encrypted` runs the `tfhe` backend by default; once the
> CKKS backend lands it is selected by name (`backend="ckks"`), and the exported IR file is
> unchanged either way — that is the point (`docs/IR-SPEC.md`, Backend neutrality). Keys and
> ciphertext are **not** portable across backends: a `KeySet` generated for one backend is
> rejected by the other with an actionable message, not a deserialization panic.
>
> Note that the six binary names (`keygen`, `encrypt`, `serve`, `decrypt`, `predict`,
> `inspect`) are a **public contract with the Python bridge** — `client.py` invokes them by
> name. A workspace refactor must keep them resolvable.

You can also drive the binary directly for debugging (a JSON batch of quantized int rows on
stdin, decrypted outputs on stdout):

```bash
cd runtime
echo '[[10,14,10, ...]]' | cargo run --release --bin predict -- ../examples/mnist/phase2_fixture.json
```

The bridge's golden gate — under the `tfhe` backend, decrypted output equals the
quantized-cleartext oracle bit-for-bit (`AGENTS.md` §1.1) — is the opt-in test
`tests/test_predict_bridge.py`, run with real FHE via:

```bash
cd python && PENUMBRA_E2E=1 uv run pytest ../tests/test_predict_bridge.py
```

It is skipped by default (needs `cargo`, minutes/sample) so CI stays hermetic — the fast tests
in that file inject a cleartext-oracle fake for the runtime and run everywhere.

### Reusing keys + the client/server split

By default `predict_encrypted` runs the whole round trip (keygen → encrypt → evaluate →
decrypt) in one `predict` process with **ephemeral** keys. To **reuse** a key pair across calls
— keygen is the expensive per-model step — generate a `KeySet` once and pass it in:

```python
from penumbra import KeySet

keys = KeySet.generate(model.graph.num_blocks)   # client-side key ceremony (once)
keys = keys.save("my_keys/")                       # persist for later runs; KeySet.load("my_keys/")
pred = model.predict_encrypted(x, keys=keys)      # reuse across as many calls as you like
```

Passing `keys=` also switches to the **client/server split**: the client `encrypt`s with its
secret key, a *separate* server process `serve`s holding **only** the public `server.key` +
graph (it cannot decrypt), and the client `decrypt`s — the faithful privacy boundary
(`PROJECT.md` §11). A key pair is tied to a model's radix width (`num_blocks`); using it with a
differently-sized model fails loudly. `*.key` files are git-ignored — never commit key material.

The runnable, self-contained demo is [`examples/client_server/`](../examples/client_server/):

```bash
cd python && uv run python ../examples/client_server/demo.py    # tiny model, ~seconds
```

You can also drive the split binaries by hand (each is a thin CLI over the runtime's public
API): `keygen <num_blocks> <client.key> <server.key>`, `encrypt <client.key> <out.cts>` (JSON
batch on stdin), `serve <model> <server.key> <in.cts> <out.cts>`, `decrypt <client.key>
<cts>` (prints `{"outputs": …}`).

### Failure modes (all caught loudly, `AGENTS.md` §1.4)

The encrypted path fails at the earliest point with an actionable message, never silently:

| Failure | Where | Message gist |
|---|---|---|
| Unsupported ONNX op | `load_onnx` (load time) | `operator X (node '…') not supported` (all at once) |
| Over-budget bit-width | `check_graph_bit_width_budget` (before keygen) | names the offending node + required-vs-available bits |
| Model not quantized | `predict_encrypted` | `call quantize() before predict_encrypted()` |
| No Rust toolchain | `run_encrypted` | `cargo was not found on PATH …` |
| Missing runtime crate | `run_encrypted` | `runtime crate not found … set PENUMBRA_RUNTIME_DIR` |
| Key/model `num_blocks` mismatch | `run_encrypted` / `serve` | `key/model mismatch: keys for num_blocks=A, model needs B` |
| Missing/corrupt key or ciphertext file | key/ciphertext load | `cannot read/deserialize … (is it a Penumbra … file?)` |
| Runtime binary non-zero exit | `run_encrypted` | surfaces the binary's stderr |
| Op unsupported **on the selected backend** | graph load (before keygen) | names the op, the node, and the backend — never a silent approximation (`AGENTS.md` §1.4) |
| Key or ciphertext from a **different backend** | key/ciphertext load | `these keys are for backend X, this run uses Y` — not a deserialization panic (Phase 12.1) |
| Over-budget multiplicative depth (CKKS) | the depth/scale check (before keygen) | names the offending node + required-vs-available levels |

## Linting & formatting

These run in CI and are enforced on PRs. Run them locally before pushing — **warnings are
treated as errors** (`AGENTS.md` §6).

```bash
# Rust
cd runtime
cargo fmt --all                       # format
cargo fmt --all -- --check            # check only (what CI runs)
cargo clippy --all-targets -- -D warnings

# Python
cd python
uv run ruff check .
uv run black --check .
```

## The golden invariant (read this)

> Encrypted output must match the quantized-cleartext output: **bit-for-bit under TFHE,
> within the declared error bound under CKKS.**

The reference never changes — `python/penumbra/reference.py`. Only the comparator is
per-backend, which is what keeps the backends comparable.

TFHE is exact, so if FHE ≠ cleartext it is a quantization or implementation bug, **never
crypto noise** — debug the cleartext quantized path first. CKKS is approximate, so its gate is
a committed per-model bound with the measured error always reported; exceeding it is still a
bug first (scale, level, or polynomial degree). This test is wired into CI from Phase 2 onward
and must never regress. See [`AGENTS.md`](../AGENTS.md) §1 and
[`docs/BACKENDS.md`](./BACKENDS.md).

## Adding an op (the canonical path)

1. Registry entry — map the ONNX op → internal op (`python/penumbra/op_registry.py`).
2. Implementation **in every backend** — or a loud, load-time rejection on backends that
   cannot realize it.
3. Bit-width growth rule — how the op grows the bit-width budget (`PROJECT.md` §9), plus its
   consequence for each backend's resource budget.
4. Golden test — assert the invariant at each backend's comparator.
5. Docs — update `docs/SUPPORTED-OPS.md`, including the per-backend support matrix.

Adding a whole **backend** is a different path — see
[`docs/BACKENDS.md`](./BACKENDS.md#adding-a-backend-the-canonical-path).

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full workflow.
