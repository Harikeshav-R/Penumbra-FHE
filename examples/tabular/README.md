# Tabular MLP — Wisconsin Breast Cancer

Tabular classification on the Wisconsin Breast Cancer dataset using a PyTorch neural network: **train → quantize → export IR → encrypted inference**.

- **Architecture:** `Linear(30 → 8) → ReLU → Linear(8 → 2)`
- **IR Graph:** `Linear → Requant(clamp_lo=0, fused ReLU) → Linear`
- **Dataset:** Wisconsin Breast Cancer (`sklearn.datasets.load_breast_cancer`, 30 features, 2 classes)
- **Radix:** 10 blocks (20-bit signed capacity)
- **Float Accuracy:** 0.965
- **Quantized Accuracy:** 0.956 (gap +0.009)

For tree-ensemble models on the exact same dataset and test split, see [`examples/trees/`](../trees/).

## Preprocessing note

Raw Breast Cancer features vary by orders of magnitude (~0.05 to ~2500 across 30 dimensions).
Min-max feature scaling to `[0.0, 1.0]` using training set statistics is performed client-side
in NumPy before integer quantization, not as an in-graph FHE operation, because the input scale
is per-tensor unsigned. The train/test split (`test_size=0.2, random_state=42`) is identical to
the split in `examples/trees/`, allowing direct comparison between tree ensembles and neural networks.

## Run it (one command)

```bash
uv run python examples/tabular/run.py          # [--model KEY] [--samples N] [--backend tfhe|ckks]
```

Prerequisite is `uv sync` only: it builds the PyO3 extension from source and requires no network
access and no `ml` extra. The command replays the committed fixture's pre-quantized `test_inputs`
in-process through keygen → encrypt → evaluate → decrypt via `penumbra.client.run_encrypted`.
Under TFHE (default), the decrypted outputs are checked bit-for-bit against the quantized-cleartext
reference `evaluate_graph_int`, exiting with code 1 on any mismatch (`AGENTS.md` §1.1). Under CKKS,
evaluation is report-only (max |err| and label agreement) and requires a nightly source build
(`RUSTUP_TOOLCHAIN=nightly uv run --with "maturin>=1.9,<2.0" maturin develop --release --features ckks`)
executed with `uv run --no-sync python ...`; the formal CKKS gate is the Rust golden against
declared bound `PHASE11_TABULAR_MLP = 29.0` in `crates/penumbra-ckks/src/bounds.rs`.

**Latency:** ~30.8 s/sample (measured by `run.py`, TFHE `classic` profile, Apple M3 Pro).
*(There is no CKKS latency reported here; comparative cross-backend performance benchmarks come
exclusively from `penumbra-bench` — see `docs/BENCHMARKS.md`.)*

## Regenerating

```bash
uv run --extra ml --system-certs python examples/tabular/mlp_export.py
```

## Tests

```bash
# Fast Python fixture drift guard:
uv run pytest tests/test_tabular_mlp_fixture.py

# TFHE golden exactness gate (asserts bit-for-bit exactness against integer reference):
cargo test -p penumbra-fhe-runtime --release --test golden_tabular_mlp -- --ignored --nocapture

# CKKS golden error-bound gate (asserts error <= PHASE11_TABULAR_MLP):
cargo +nightly test -p penumbra-ckks --features ckks --release --test ckks_golden_tabular_mlp -- --nocapture
```
