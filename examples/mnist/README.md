# MNIST — the reference example

The first end-to-end use case: **train → quantize → export IR → encrypted inference**.

- **Phase 2:** encrypted logistic regression / 1-layer net on MNIST 0-vs-1, proving the
  narrow waist (`Linear → Activation → Argmax`) and establishing the golden exactness test.
- **Phase 4:** a small CNN on 10-class MNIST, proving multi-layer eval + automatic
  bit-width management (`Conv2d`, `Pool`, `Requant`).

This example contains **no cryptography** — only a model graph and quantized weights
(`PROJECT.md` §4). The crypto lives entirely in the `runtime/` crate.

## Phase 2 (current)

`train_quantize_export.py` trains a binary logistic-regression classifier, quantizes it by
hand (symmetric PTQ), and writes **`phase2_fixture.json`** — pure data: quantized weights,
bias, threshold, a narrow activation LUT, and a batch of quantized test inputs with their
expected (quantized-cleartext) labels. The Rust runtime hand-assembles the op graph
(`Linear → Argmax`, plus a standalone `Activation` LUT) from this fixture; the real
serializable IR arrives in Phase 3.

The committed fixture is the input to the **golden exactness test**
(`runtime/tests/golden_logreg.rs`): FHE output must equal these quantized-cleartext labels
bit-for-bit (`AGENTS.md` §1.1).

```bash
# Regenerate the fixture (only when the example changes; NumPy-only, no network):
cd python && uv run python ../examples/mnist/train_quantize_export.py

# Run the golden test (the gate). Release is mandatory — debug FHE is far too slow:
cd runtime && cargo test --release
```

> **Dataset note.** To stay hermetic and dependency-light, the Phase-2 generator uses a
> deterministic **synthetic** 8×8 two-class dataset rather than real MNIST pixels. The op
> graph and integer arithmetic are identical to a real MNIST 0-vs-1 model; swapping in a
> trained MNIST model is a drop-in change. Real MNIST + a small CNN comes with Phase 4.

## Phase 4 (planned)

A small CNN on 10-class MNIST, proving multi-layer eval + automatic bit-width management
(`Conv2d`, `Pool`, `Requant`).
