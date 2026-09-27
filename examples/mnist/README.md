# MNIST — the reference example

The first end-to-end use case: **train → quantize → export IR → encrypted inference**.

- **Phase 2:** encrypted logistic regression / 1-layer net on MNIST 0-vs-1, proving the
  narrow waist (`Linear → Activation → Argmax`) and establishing the golden exactness test.
- **Phase 4:** a small CNN on 10-class MNIST, proving multi-layer eval + automatic
  bit-width management (`Conv2d`, `Pool`, `Requant`).

This example contains **no cryptography** — only a model graph and quantized weights
(`PROJECT.md` §4). The crypto lives entirely in the backend crates, and these fixtures are
backend-agnostic: the same committed IR graph is what *every* backend evaluates
([`docs/BACKENDS.md`](../../docs/BACKENDS.md)).

## Run it (one command)

```bash
uv run python examples/mnist/run.py            # [--model KEY] [--samples N] [--backend tfhe|ckks]
```

Prerequisite is `uv sync` only: it builds the PyO3 extension from source and requires no network
access and no `ml` extra. The command replays the committed fixture's pre-quantized `test_inputs`
in-process through keygen → encrypt → evaluate → decrypt via `penumbra.client.run_encrypted`.
Under TFHE (default), the decrypted outputs are checked bit-for-bit against the quantized-cleartext
reference `evaluate_graph_int`, exiting with code 1 on any mismatch (`AGENTS.md` §1.1). Under CKKS,
evaluation is report-only (max |err| and label agreement) and requires a nightly source build
(`RUSTUP_TOOLCHAIN=nightly uv run --with "maturin>=1.9,<2.0" maturin develop --release --features ckks`)
executed with `uv run --no-sync python ...`; the formal CKKS gate is the Rust golden against
declared bounds in `crates/penumbra-ckks/src/bounds.rs`.
Expected wall-clock latency ranges from ~0.52 s/sample for `phase2_logreg` to ~218 s/sample for
the default CNN (`phase6_onnx`), measured on Apple M3 Pro (`docs/BENCHMARKS.md`).

## Models

| `--model` Key | Fixture | Generator | IR Graph | TFHE s/sample |
|---|---|---|---|---:|
| `phase2_logreg` | `phase2_fixture.json` | `train_quantize_export.py` | `Linear → Argmax` | 0.52 s |
| `phase4_cnn` | `phase4_cnn_fixture.json` | `cnn_export.py` | `Conv2d → Requant → Pool → Linear` | 27.1 s |
| `phase5_digits` | `phase5_digits_fixture.json` | `real_digits_export.py` | `Conv2d → Requant → Linear` | 222 s |
| `phase5_qat` | `phase5_qat_fixture.json` | `qat_export.py` | `Conv2d → Requant → Linear` | 179 s |
| `phase6_onnx` (default) | `phase6_onnx_fixture.json` | `onnx_export.py` | `Conv2d → Requant → Linear` | 218 s |
| `phase6_sklearn` | `phase6_sklearn_fixture.json` | `sklearn_export.py` | `Linear` | 40.6 s |
| `phase8_bn_cnn` | `phase8_bn_cnn_fixture.json` | `bn_cnn_export.py` | `Conv2d → Requant → Pool → Linear` | — |
| `phase8_branch` | `phase8_branch_fixture.json` | `branch_mlp_export.py` | `Linear → Requant → Split → Linear → Requant → Linear → Requant → Concat → Add → Linear` | — |
| `phase8_gap_cnn` | `phase8_gap_cnn_fixture.json` | `gap_cnn_export.py` | `Conv2d → Requant → Pool → Pool → Linear` | — |
| `phase8_tanh` | `phase8_tanh_fixture.json` | `tanh_mlp_export.py` | `Linear → Requant → Activation → Linear` | — |

*Latency numbers are eval-total wall clock per sample from `docs/BENCHMARKS.md` Table A (Apple M3 Pro, commit `9b38c1b`, 2026-09-24). Models without published sweep timings show `—`.*

> **Dataset note.** To stay hermetic and dependency-light, the Phase-2 and Phase-4 generators
> use deterministic **synthetic** datasets rather than downloading full MNIST pixels. The op graphs
> and integer arithmetic are identical to real MNIST pipelines; real datasets (such as scikit-learn
> 8×8 digits) are evaluated starting with Phase 5.

## Regenerating

Fixtures are committed and do not need to be regenerated for regular runs. To regenerate them:

```bash
# Phase 2 (NumPy only, no extra dependencies):
uv run python examples/mnist/train_quantize_export.py

# Phases 4-8 (requires ml extra: torch, torchvision, skl2onnx, brevitas):
uv run --extra ml --system-certs python examples/mnist/cnn_export.py
uv run --extra ml --system-certs python examples/mnist/real_digits_export.py
uv run --extra ml --system-certs python examples/mnist/qat_export.py
uv run --extra ml --system-certs python examples/mnist/onnx_export.py
uv run --extra ml --system-certs python examples/mnist/sklearn_export.py
uv run --extra ml --system-certs python examples/mnist/bn_cnn_export.py
uv run --extra ml --system-certs python examples/mnist/branch_mlp_export.py
uv run --extra ml --system-certs python examples/mnist/gap_cnn_export.py
uv run --extra ml --system-certs python examples/mnist/tanh_mlp_export.py
```

## Tests

TFHE golden tests assert bit-for-bit exactness against the quantized-cleartext oracle:

```bash
# Fast golden tests:
cargo test -p penumbra-fhe-runtime --release --test golden_logreg
cargo test -p penumbra-fhe-runtime --release --test golden_cnn

# Slower golden tests (marked #[ignore]):
cargo test -p penumbra-fhe-runtime --release --test golden_digits -- --ignored
cargo test -p penumbra-fhe-runtime --release --test golden_qat -- --ignored
cargo test -p penumbra-fhe-runtime --release --test golden_onnx -- --ignored
cargo test -p penumbra-fhe-runtime --release --test golden_sklearn -- --ignored
cargo test -p penumbra-fhe-runtime --release --test golden_bn_cnn -- --ignored
cargo test -p penumbra-fhe-runtime --release --test golden_branch_mlp -- --ignored
cargo test -p penumbra-fhe-runtime --release --test golden_gap_cnn -- --ignored
cargo test -p penumbra-fhe-runtime --release --test golden_tanh_mlp -- --ignored
```
