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

## Reproducing

```bash
# Regenerate the fixtures (NumPy only; prints accuracy):
cd python
uv run python ../examples/mnist/train_quantize_export.py   # Phase-2 logreg
uv run python ../examples/mnist/cnn_export.py               # Phase-4 CNN

# Time the encrypted forward pass (release; the golden tests carry the timing):
cd ../runtime
cargo test --release --test golden_logreg -- --nocapture    # ~30 s/sample
cargo test --release --test golden_cnn    -- --nocapture     # ~3-4 min/sample

# Inspect a model's per-tensor bit-widths without running FHE:
cargo run --release --bin inspect ../examples/mnist/phase4_cnn_fixture.json
```
