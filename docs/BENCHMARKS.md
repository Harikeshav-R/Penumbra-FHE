# Benchmarks

Accuracy and latency for the committed example models. Numbers are honest and reproducible
from the committed fixtures — **not** marketing figures. Latency is "seconds-to-minutes per
inference, research/prototype territory" (`PROJECT.md` §16).

**Everything below is the `tfhe` backend**, the reference implementation. CKKS numbers arrive
with Phase 12 and get their own columns; see [Cross-backend comparison](#cross-backend-comparison).

> ⚠️ Always benchmark in `--release`. Debug FHE is orders of magnitude slower and the numbers
> are meaningless — true for `poulpy` as much as for `tfhe-rs` (`docs/DEVELOPMENT.md`).

## Methodology

- **Accuracy** is reported by each example's generator (`float` = the float pipeline,
  `quantized` = the quantized-integer pipeline). The **quantization gap** = float − quantized
  is the accuracy lost to low-bit integers. Under TFHE the golden tests guarantee the FHE
  accuracy equals the quantized accuracy *exactly* (bit-for-bit), so there is no separate "FHE
  accuracy" column. Under CKKS there **will** be one — an approximate scheme adds its own
  error term on top of the quantization gap.
- **Latency** is wall-clock for the encrypted forward pass of **one** sample, from the golden
  tests (`cargo test --release`), on the development machine. It is indicative, not a
  controlled benchmark; absolute numbers vary by CPU. The committed test batches are kept tiny
  (`N_TEST`) precisely because each FHE sample is expensive.
- **Crypto profile:** the default `PARAM_MESSAGE_2_CARRY_2_KS_PBS` (`MESSAGE_BITS = 2`), no
  parameter tuning (that is Phase 10). `num_blocks` is sized by the library to the model's
  widest accumulator.
- **Cost proxy:** bootstraps per sample — `runtime ≈ number of bootstraps` (`PROJECT.md` §5).
  This is a **TFHE** proxy; CKKS's is multiplicative depth, rotations, and rescales.

> ⚠️ **These numbers are not yet comparison-grade.** They are hand-recorded wall clock from
> golden-test output on one developer machine. Phase 12.3 replaces that with a shared
> `criterion` harness measuring both backends through the same code path, on a pinned machine
> and a pinned `poulpy` HAL backend. Until then, do not compare a number here against a CKKS
> number produced any other way ([`docs/COMPARISON.md`](./COMPARISON.md)).

## Models

### Phase-2 — binary logistic regression (`examples/mnist/phase2_fixture.json`)

`Linear(64→1) → Argmax` (2-class), synthetic 8×8 two-blob data.

| Metric | Value |
|---|---|
| Float accuracy | 1.00 (synthetic, linearly separable) |
| Quantized accuracy | 1.00 |
| Quantization gap | 0.00 |
| Radix | 8 blocks (16-bit signed) |
| Latency / sample (encrypted) | ~30 s |
| Bootstraps / sample | comparison only (the `Linear` is PBS-free) |

### Phase-4 — small CNN, 10-class (`examples/mnist/phase4_cnn_fixture.json`)

`Conv2d(1→2, 3×3) → Requant+ReLU (auto-inserted) → Pool(avg 2×2) → Linear(8→10 logits)`,
synthetic 6×6 ten-class template data; the client decrypts the 10 logits and argmaxes.

| Metric | Value |
|---|---|
| Float accuracy | 0.98 |
| Quantized accuracy | 0.96 |
| Quantization gap | ~0.02 |
| Radix | 7 blocks (14-bit signed) |
| Latency / sample (encrypted) | ~3–4 min |
| Dominant cost | the `Requant` bootstraps (one PBS per post-conv activation); `Conv2d`/`Pool`/`Linear` are PBS-free |

The CNN's cost is dominated by the per-activation `Requant` bootstraps — exactly the
"runtime ≈ number of bootstraps" lever (`PROJECT.md` §5). The PBS-free `Conv2d`/`Pool`/`Linear`
do many cheap scalar-mul/add operations on the multi-block radix, which is why a wider radix
(more blocks) also costs more. Both are Phase-10 optimization targets (parallelize per-element
work with `rayon`; shrink `num_blocks`/precision to the minimum each layer needs).

### Phase-5 — real handwritten digits, PTQ (`examples/mnist/phase5_digits_fixture.json`)

The first example on a **real dataset** and a **real trained PyTorch model**: scikit-learn's
8×8 `load_digits` (real pen-written digits), quantized through the library service
(`Model.quantize`). `Conv2d(1→12, 3×3, stride 2) → Requant+ReLU → Linear(108→10 logits)`.

| Metric | Value |
|---|---|
| Float accuracy | ~0.96 |
| Quantized accuracy | ~0.93 |
| Quantization gap | ~0.03 |
| Weight / activation bits | 6-bit weights, 2-bit activations |
| Calibration | MSE (clip minimizing round-trip error), per-channel weights |
| Radix | 11 blocks (22-bit signed) |
| Bootstraps / sample | ~108 (one `Requant` PBS per post-conv activation, 12 ch × 3×3) |
| Latency / sample (encrypted) | minutes (the golden test is `#[ignore]`d; see below) |

The remaining ~0.03 gap is the cost of capping activations at a single 2-bit block
(`MESSAGE_BITS`) — the hard TFHE-backend limit. Three service levers close most of the naive gap:
6-bit **per-channel** weights, **MSE** activation calibration (the clip minimizing round-trip
quantization error, not the raw peak), and — the big one — quantizing the head against the
**post-Requant activation scale** (not the wide pre-Requant accumulator scale; getting that wrong
mis-scales the head bias by the requant ratio and was worth ~0.22 accuracy alone). The FHE golden
test (`golden_digits.rs`) is `#[ignore]`d because at ~108 bootstraps/sample it is minutes per
sample; the fast Python guard (`tests/test_real_digits_fixture.py`) checks fixture
self-consistency on every CI run.

### Phase-5 — real handwritten digits, QAT (`examples/mnist/phase5_qat_fixture.json`)

The same architecture and dataset, but trained with **Brevitas quantization-aware training**, then
exported through the same PTQ service (so the int graph and the golden gate are identical).

| Metric | Value |
|---|---|
| Float accuracy | ~0.94 |
| Quantized accuracy | ~0.94 |
| Quantization gap | ~0.00 |
| Weight / activation bits | 6-bit weights, 2-bit activations |
| Calibration | MSE, per-channel weights |
| Radix | 11 blocks (22-bit signed) |
| Latency / sample (encrypted) | minutes (`golden_qat.rs` is `#[ignore]`d) |

With the head correctly quantized against the post-Requant scale, QAT closes the gap essentially
completely on this task — the quantized model matches (and here slightly exceeds, within
small-test-set noise on ~360 samples) the float model, the quantization acting as a mild
regularizer. The example proves the QAT path runs end to end through the exact int export and the
golden invariant.

### Phase-7 — closed-set faces, Olivetti (`examples/faces/phase7_faces_fixture.json`)

The **second use case** and the abstraction-validation milestone: closed-set face recognition
over the first 8 people of the Olivetti dataset (AT&T "Database of Faces"), run through the
**unchanged** backend. 64×64 images are 4×4 block-mean downsampled to 16×16 (NumPy preprocessing,
not a graph op), then `Conv2d(1→8, 3×3, stride 4) → Requant+ReLU → Linear(128→8 logits)` — the
*same* IR op vocabulary the digit CNN lowers to, via the same `load_onnx → quantize` path. Adding
it touched **no `runtime/src/ops/` and no `eval.rs`** — the narrow waist held (`PROJECT.md` §4).

| Metric | Value |
|---|---|
| Float accuracy | 0.95 |
| Quantized accuracy | 0.90 |
| Quantization gap | 0.05 |
| Weight / activation bits | 6-bit weights, 2-bit activations |
| Calibration | MSE (clip minimizing round-trip error), per-channel weights |
| Radix | 11 blocks (22-bit signed) |
| Bootstraps / sample | 128 (one `Requant` PBS per post-conv activation, 8 ch × 4×4) |
| Latency / sample (encrypted) | minutes (`golden_faces.rs` is `#[ignore]`d; see below) |

The ~0.05 gap is the cost of an 8-way decision from tiny 16×16 inputs with activations capped at a
single 2-bit block (`MESSAGE_BITS`, the hard TFHE-backend limit). The value of this example is **not**
its accuracy — it is that a completely different task (faces, not digits) ran encrypted end to end
with zero crypto-backend edits, exactly like `load_onnx` promised. The FHE golden test
(`golden_faces.rs`) is `#[ignore]`d because at 128 bootstraps/sample it is minutes per sample; the
fast Python guard (`tests/test_faces_fixture.py`) checks fixture self-consistency on every CI run.

## Cross-backend comparison

*Pending Phase 12.4 (`ROADMAP.md`).* When the CKKS backend lands, each model's table above
gains a backend dimension — latency, accuracy against the shared quantized-cleartext
reference, ciphertext and key sizes, and each scheme's own cost proxy — all produced by the
shared `penumbra-bench` harness on a single pinned machine and HAL backend.

This document will own the **numbers**; [`docs/COMPARISON.md`](./COMPARISON.md) owns the
**argument** — the hypothesis under test, what is held constant, and the threats to validity
that bound what the numbers mean. Do not restate results in both places; cite across.

Three things must be stated wherever a cross-backend number appears:

1. **The slot-packing decision.** If the CKKS backend runs one value per ciphertext, its
   latency reflects our implementation, not the scheme ([`docs/BACKENDS.md`](./BACKENDS.md)).
2. **That the graph is quantized for TFHE.** The 2-bit activation cap is a PBS constraint;
   imposing it on CKKS is what makes the comparison fair *and* what handicaps CKKS on accuracy
   (`docs/QUANTIZATION.md`).
3. **The maturity asymmetry.** `tfhe-rs` is a mature production library; `poulpy-ckks` is at
   0.8.x, and its Penumbra backend is new.

## Reproducing

```bash
# Regenerate the synthetic fixtures (NumPy only; prints accuracy):
cd python
uv run python ../examples/mnist/train_quantize_export.py   # Phase-2 logreg
uv run python ../examples/mnist/cnn_export.py               # Phase-4 CNN

# Regenerate the real-data fixtures (needs the optional `ml` extra: torch + sklearn + brevitas):
uv run --extra ml --system-certs python ../examples/mnist/real_digits_export.py  # Phase-5 PTQ
uv run --extra ml --system-certs python ../examples/mnist/qat_export.py          # Phase-5 QAT
uv run --extra ml --system-certs python ../examples/faces/olivetti_export.py     # Phase-7 faces (one-time ~4 MB download)

# Time the encrypted forward pass (release; the golden tests carry the timing):
cd ../runtime
cargo test --release --test golden_logreg -- --nocapture                  # ~30 s/sample
cargo test --release --test golden_cnn    -- --nocapture                  # ~3-4 min/sample
cargo test --release --test golden_digits -- --ignored --nocapture        # minutes/sample (real digits)
cargo test --release --test golden_qat    -- --ignored --nocapture        # minutes/sample (QAT)
cargo test --release --test golden_faces  -- --ignored --nocapture        # minutes/sample (faces)

# Inspect a model's per-tensor bit-widths without running FHE:
cargo run --release --bin inspect ../examples/mnist/phase5_digits_fixture.json
```
