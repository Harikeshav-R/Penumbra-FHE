# Benchmarks

Accuracy and latency for the committed example models. Numbers are honest and reproducible
from the committed fixtures — **not** marketing figures. Latency is "seconds-to-minutes per
inference, research/prototype territory" (`PROJECT.md` §16), dominated by programmable
bootstraps (`runtime ≈ number of bootstraps`, `PROJECT.md` §5).

> ⚠️ Always benchmark in `--release`. Debug `tfhe-rs` is orders of magnitude slower and the
> numbers are meaningless (`docs/DEVELOPMENT.md`).

## Methodology

- **Accuracy** is reported by each example's generator (`float` = the float pipeline,
  `quantized` = the quantized-integer pipeline). The **quantization gap** = float − quantized
  is the accuracy lost to low-bit integers; the golden tests guarantee the FHE accuracy equals
  the quantized accuracy *exactly* (bit-for-bit), so there is no separate "FHE accuracy" column.
- **Latency** is wall-clock for the encrypted forward pass of **one** sample, from the golden
  tests (`cargo test --release`), on the development machine. It is indicative, not a
  controlled benchmark; absolute numbers vary by CPU. The committed test batches are kept tiny
  (`N_TEST`) precisely because each FHE sample is expensive.
- **Crypto profile:** the default `PARAM_MESSAGE_2_CARRY_2_KS_PBS` (`MESSAGE_BITS = 2`), no
  parameter tuning (that is Phase 10). `num_blocks` is sized by the library to the model's
  widest accumulator.

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
| Quantized accuracy | ~0.69 |
| Quantization gap | ~0.28 |
| Radix | 10 blocks (20-bit signed) |
| Bootstraps / sample | ~108 (one `Requant` PBS per post-conv activation, 12 ch × 3×3) |
| Latency / sample (encrypted) | minutes (the golden test is `#[ignore]`d; see below) |

The large quantization gap is **honest, not a bug**: capping activations at a single 2-bit block
(`MESSAGE_BITS`) is aggressive, and on real 10-class digits it costs real accuracy. Accuracy comes
from having *many* low-precision features (12 channels) rather than precise ones — the realistic
lever within the FHE budget. The FHE golden test (`golden_digits.rs`) is `#[ignore]`d because at
~108 bootstraps/sample it is minutes per sample; the fast Python guard
(`tests/test_real_digits_fixture.py`) checks fixture self-consistency on every CI run.

### Phase-5 — real handwritten digits, QAT (`examples/mnist/phase5_qat_fixture.json`)

The same architecture and dataset, but trained with **Brevitas quantization-aware training**, then
exported through the same PTQ service (so the int graph and the golden gate are identical).

| Metric | Value |
|---|---|
| Float accuracy | ~0.95 |
| Quantized accuracy | ~0.72 |
| Quantization gap | ~0.23 |
| Radix | 10 blocks (20-bit signed) |
| Latency / sample (encrypted) | minutes (`golden_qat.rs` is `#[ignore]`d) |

QAT gives a **modest** bump over PTQ here (~0.69 → ~0.72): the dominant loss is the hard
single-block activation cap, which the requant calibration handles independently of how the
weights were trained, so re-PTQ on QAT weights only partly closes the gap. The example's value is
proving the QAT path runs end to end through the exact int export and golden invariant — not a
dramatic accuracy recovery.

## Reproducing

```bash
# Regenerate the synthetic fixtures (NumPy only; prints accuracy):
cd python
uv run python ../examples/mnist/train_quantize_export.py   # Phase-2 logreg
uv run python ../examples/mnist/cnn_export.py               # Phase-4 CNN

# Regenerate the real-data fixtures (needs the optional `ml` extra: torch + sklearn + brevitas):
uv run --extra ml --system-certs python ../examples/mnist/real_digits_export.py  # Phase-5 PTQ
uv run --extra ml --system-certs python ../examples/mnist/qat_export.py          # Phase-5 QAT

# Time the encrypted forward pass (release; the golden tests carry the timing):
cd ../runtime
cargo test --release --test golden_logreg -- --nocapture                  # ~30 s/sample
cargo test --release --test golden_cnn    -- --nocapture                  # ~3-4 min/sample
cargo test --release --test golden_digits -- --ignored --nocapture        # minutes/sample (real digits)
cargo test --release --test golden_qat    -- --ignored --nocapture        # minutes/sample (QAT)

# Inspect a model's per-tensor bit-widths without running FHE:
cargo run --release --bin inspect ../examples/mnist/phase5_digits_fixture.json
```
