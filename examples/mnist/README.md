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

## Phase 2 (current)

`train_quantize_export.py` trains a binary logistic-regression classifier, quantizes it by
hand (symmetric PTQ), and writes **`phase2_fixture.json`** — pure data: quantized weights,
bias, threshold, a narrow activation LUT, and a batch of quantized test inputs with their
expected (quantized-cleartext) labels. The Rust runtime hand-assembles the op graph
(`Linear → Argmax`, plus a standalone `Activation` LUT) from this fixture; the real
serializable IR arrives in Phase 3.

The committed fixture is the input to the **golden test**
(`runtime/tests/golden_logreg.rs`): under the TFHE backend, FHE output must equal these
quantized-cleartext labels bit-for-bit (`AGENTS.md` §1.1). The same fixture is the CKKS
backend's input too — same graph, same reference, a tolerance comparator instead of equality.

```bash
# Regenerate the fixture (only when the example changes; NumPy-only, no network):
uv run python examples/mnist/train_quantize_export.py

# Run the golden test (the gate). Release is mandatory — debug FHE is far too slow:
cargo test --workspace --release
```

> **Dataset note.** To stay hermetic and dependency-light, the Phase-2 generator uses a
> deterministic **synthetic** 8×8 two-class dataset rather than real MNIST pixels. The op
> graph and integer arithmetic are identical to a real MNIST 0-vs-1 model; swapping in a
> trained MNIST model is a drop-in change. Real MNIST + a small CNN comes with Phase 4.

## Phases 4–6 (also current)

The example set grew well past Phase 2; every fixture below is committed, benchmarked
([`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md)), and guarded by a Rust golden test plus a
fast Python self-consistency test.

| Fixture | Generator | Graph | What it proves |
|---|---|---|---|
| `phase4_cnn_fixture.json` | `cnn_export.py` | `Conv2d → Requant → Pool → Linear` | multi-layer eval + automatic bit-width management |
| `phase5_digits_fixture.json` | `real_digits_export.py` | `Conv2d → Requant → Linear` | a **real** dataset and trained model, through the PTQ service |
| `phase5_qat_fixture.json` | `qat_export.py` | `Conv2d → Requant → Linear` | Brevitas QAT, exported through the same int path |
| `phase6_onnx_fixture.json` | `onnx_export.py` | `Conv2d → Requant → Linear` | the ONNX front door, from a PyTorch export |
| `phase6_sklearn_fixture.json` | `sklearn_export.py` | `Linear` | a second framework (`skl2onnx`) through the same waist |

The Phase-5/6 generators need the optional `ml` extra (torch + sklearn + brevitas); see
`docs/BENCHMARKS.md` for the exact commands. Their FHE golden tests are `#[ignore]`d because
they run minutes per sample.
