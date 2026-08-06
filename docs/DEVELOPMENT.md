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

## Layout

```
python/      # Python front end: ONNX loader, quantization, IR emitter (Layer 3)
runtime/     # Rust TFHE backend: ops, IR deserialization, eval loop (Layers 1–2)
examples/    # use cases (mnist, faces) — graphs only, NO crypto
tests/       # cross-cutting + golden exactness tests
docs/        # this guide and the spec docs
```

## Building & testing

### Rust runtime

```bash
cd runtime
cargo build                # debug build (fine for correctness)
cargo test --release       # run tests — ALWAYS use --release for FHE
```

> ⚠️ **Build in `--release` for anything that runs FHE.** Debug builds of `tfhe-rs` are
> *extremely* slow (orders of magnitude). The first compile is slow regardless — `tfhe`
> pulls a large dependency tree. The `hello_fhe` test proves the toolchain works
> (encrypt → plaintext-weight arithmetic → LUT-via-PBS → decrypt).

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

`model.predict_encrypted(x)` runs the real encrypted forward pass (ROADMAP Phase 9, first
slice). It quantizes `x`, hands the exported IR + the quantized batch to the Rust runtime, and
returns the client-side prediction:

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

You can also drive the binary directly for debugging (a JSON batch of quantized int rows on
stdin, decrypted outputs on stdout):

```bash
cd runtime
echo '[[10,14,10, ...]]' | cargo run --release --bin predict -- ../examples/mnist/phase2_fixture.json
```

The bridge's golden gate — decrypted output equals the quantized-cleartext oracle bit-for-bit
(`AGENTS.md` §1.1) — is the opt-in test `tests/test_predict_bridge.py`, run with real FHE via:

```bash
cd python && PENUMBRA_E2E=1 uv run pytest ../tests/test_predict_bridge.py
```

It is skipped by default (needs `cargo`, minutes/sample) so CI stays hermetic — the fast tests
in that file inject a cleartext-oracle fake for the runtime and run everywhere.

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

> FHE output must equal the quantized-cleartext output, **bit-for-bit**.

TFHE is exact. If FHE ≠ cleartext, it is a quantization or implementation bug, **never
crypto noise** — debug the cleartext quantized path first. This test is wired into CI from
Phase 2 onward and must never regress. See [`AGENTS.md`](../AGENTS.md) §1.

## Adding an op (the canonical path)

1. Registry entry — map the ONNX op → internal op (`python/penumbra/op_registry.py`).
2. Rust implementation — against `tfhe-rs` primitives (`runtime/src/ops/`).
3. Bit-width growth rule — how the op grows the bit-width budget (`PROJECT.md` §9).
4. Golden test — assert FHE == quantized-cleartext.
5. Docs — update `docs/SUPPORTED-OPS.md`.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full workflow.
