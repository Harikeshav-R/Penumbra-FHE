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
- **Harness metrics:** `Eval total` = `GraphProfile::total`, which includes each node's `Backend::build_op`;
  `of which op-build` = the plaintext-weight prep inside it (BSGS diagonal encoding under
  CKKS, weight vectors under TFHE); keygen/encrypt/decrypt are measured outside the walker.
  Both backends were measured from one nightly-built binary on one machine.

### Shared harness (`penumbra-bench`)

Phase 12.3 instruments the Layer-2 graph walker (`penumbra_core::eval::evaluate_graph_profiled`)
and introduces a shared comparison harness driven through `penumbra-bench`:

- **Criterion benchmarks:** `cargo bench -p penumbra-bench` (or `PENUMBRA_BENCH_MODELS=all cargo +nightly bench -p penumbra-bench --features ckks`)
  runs Criterion benchmarks parameterized over backend x model.
- **Report CLI:** `penumbra-bench-report` generates markdown or JSON tables recording:
  - Wall-clock latency per sample (keygen, encrypt, eval, decrypt)
  - Per-op-type time breakdown
  - Key and ciphertext wire sizes
  - Scheme cost proxies (TFHE: bootstraps, cmp_pbs_ops; CKKS: rotations, rescales, depth_levels, poly_evals)
  - Accuracy / error against the shared oracle

Commands:
```bash
# Markdown report for all models on default (TFHE) backend:
cargo run -p penumbra-bench --release --bin penumbra-bench-report

# Compare TFHE and CKKS on Phase-2 logreg:
cargo +nightly run -p penumbra-bench --features ckks --release --bin penumbra-bench-report -- --models phase2_logreg

# JSON output for machine consumption:
cargo run -p penumbra-bench --release --bin penumbra-bench-report -- --models phase2_logreg --format json
```
> ℹ️ The [Cross-backend comparison](#cross-backend-comparison) section below now carries
> shared-harness numbers measured from a single binary. The per-model `Latency / sample (encrypted)`
> rows in the sections immediately below remain hand-recorded TFHE golden-test wall clock
> and are indicative only — cite the cross-backend tables for anything comparative.

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
| Quantized accuracy | 0.9417 |
| Quantization gap | ~0.02 |
| Weight / activation bits | 6-bit weights, 2-bit activations |
| Calibration | MSE (clip minimizing round-trip error), per-channel weights |
| Radix | 11 blocks (22-bit signed) |
| Bootstraps / sample | ~108 (one `Requant` PBS per post-conv activation, 12 ch × 3×3) |
| Latency / sample (encrypted) | minutes (the golden test is `#[ignore]`d; see below) |

The remaining ~0.02 gap is the cost of capping activations at a single 2-bit block
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

**Measured on Apple M3 Pro, macOS 25.6.0, `rustc 1.100.0-nightly (bba531001 2026-09-20)`, HAL backend `FFT64Neon` (`poulpy-ckks 0.8.3`), commit `dc20d05ee332e284045ea971bb8272c5d5ddf522`, 2026-09-22, `--samples 2`. Raw artifact: [`docs/results/phase12-4-comparison.json`](./results/phase12-4-comparison.json).**

This document owns the **numbers**; [`docs/COMPARISON.md`](./COMPARISON.md) owns the
**argument** — the hypothesis under test, what is held constant, and the threats to validity
that bound what the numbers mean. Do not restate results in both places; cite across.

Three things must be stated wherever a cross-backend number appears:

1. **The slot-packing decision.** The CKKS backend evaluates under **Option B (one full tensor per ciphertext, BSGS diagonal transforms over `lt_slots = 256`)** ([`docs/BACKENDS.md`](./BACKENDS.md), `crates/penumbra-ckks/src/params.rs`). Plaintext-weight linear ops (`Linear`, `Conv2d`, `Pool(avg)`) are diagonal-multiplexed SIMD transforms rather than scalar-ciphertext arrays.
2. **That the graph is quantized for TFHE.** The 2-bit activation cap is a PBS constraint;
   imposing it on CKKS is what makes the comparison fair *and* what handicaps CKKS on accuracy
   (`docs/QUANTIZATION.md`).
3. **The maturity asymmetry.** `tfhe-rs` is a mature production library; `poulpy-ckks` is at
   0.8.x, and its Penumbra backend is new.

### Table A — Latency (Wall-Clock per Sample)

| Model | Backend | Keygen (s) | Encrypt (s) | Eval total (s) | of which op-build (s) | Decrypt (s) | TFHE / CKKS eval |
|---|---|---:|---:|---:|---:|---:|---:|
| phase2_logreg | tfhe | 0.513 | 0.022 | 11.846 | 0.000 | 0.000 | 26.7x |
| phase2_logreg | ckks | 1.907 | 0.004 | 0.444 | 0.001 | 0.001 | — |
| phase4_cnn | tfhe | 0.489 | 0.011 | 69.987 | 0.000 | 0.000 | 133.8x |
| phase4_cnn | ckks | 1.900 | 0.004 | 0.523 | 0.001 | 0.001 | — |
| phase5_digits | tfhe | 0.488 | 0.030 | 679.860 | 0.000 | 0.000 | 348.2x |
| phase5_digits | ckks | 1.952 | 0.004 | 1.953 | 0.014 | 0.000 | — |
| phase5_qat | tfhe | 0.487 | 0.030 | 687.919 | 0.000 | 0.000 | 374.4x |
| phase5_qat | ckks | 1.904 | 0.004 | 1.837 | 0.012 | 0.000 | — |
| phase6_onnx | tfhe | 0.492 | 0.030 | 701.855 | 0.000 | 0.000 | 359.9x |
| phase6_onnx | ckks | 1.898 | 0.004 | 1.950 | 0.014 | 0.000 | — |
| phase6_sklearn | tfhe | 0.490 | 0.028 | 159.067 | 0.000 | 0.000 | 363.3x |
| phase6_sklearn | ckks | 1.899 | 0.004 | 0.438 | 0.000 | 0.001 | — |
| phase7_faces | tfhe | 0.499 | 0.121 | 730.216 | 0.000 | 0.000 | 359.1x |
| phase7_faces | ckks | 1.895 | 0.004 | 2.033 | 0.008 | 0.000 | — |

*Variance check (`phase2_logreg`, Criterion 10 samples):* `tfhe` median 11.122 s (95% CI [11.018 s, 11.254 s]); `ckks` median 412.89 ms (95% CI [406.09 ms, 423.43 ms]).

### Table B — Per-Op-Type Eval Breakdown (Mean Seconds per Sample)

Breakdown for `phase2_logreg`, `phase5_digits`, and `phase7_faces` (see [`docs/results/phase12-4-comparison.json`](./results/phase12-4-comparison.json) for the full 7-model op breakdown):

| Model | Backend | Op Type | Calls | Build (s) | Eval (s) |
|---|---|---|---:|---:|---:|
| phase2_logreg | tfhe | Argmax | 1 | 0.0000 | 0.0160 |
| phase2_logreg | tfhe | Linear | 1 | 0.0000 | 11.8300 |
| phase2_logreg | ckks | Argmax | 1 | 0.0012 | 0.1118 |
| phase2_logreg | ckks | Linear | 1 | 0.0000 | 0.3309 |
| phase5_digits | tfhe | Conv2d | 1 | 0.0000 | 338.2062 |
| phase5_digits | tfhe | Linear | 1 | 0.0000 | 287.0420 |
| phase5_digits | tfhe | Requant | 1 | 0.0000 | 54.6112 |
| phase5_digits | ckks | Conv2d | 1 | 0.0001 | 0.6956 |
| phase5_digits | ckks | Linear | 1 | 0.0000 | 0.1755 |
| phase5_digits | ckks | Requant | 1 | 0.0140 | 1.0674 |
| phase7_faces | tfhe | Conv2d | 1 | 0.0000 | 402.4965 |
| phase7_faces | tfhe | Linear | 1 | 0.0000 | 264.1226 |
| phase7_faces | tfhe | Requant | 1 | 0.0000 | 63.5967 |
| phase7_faces | ckks | Conv2d | 1 | 0.0001 | 1.1641 |
| phase7_faces | ckks | Linear | 1 | 0.0000 | 0.1592 |
| phase7_faces | ckks | Requant | 1 | 0.0082 | 0.7017 |

### Table C — Sizes & Scheme Cost Proxies

| Model | Backend | Input CT | Output CT | Client Key | Server Key | Cost Proxy Counters |
|---|---|---:|---:|---:|---:|---|
| phase2_logreg | tfhe | 8.04 MB | 128.7 KB | 23.4 KB | 114.84 MB | cmp_pbs_ops: 1, ct_add: 64, scalar_add: 1, scalar_mul: 64 |
| phase2_logreg | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 5, poly_evals: 1, rescales: 5, rotations: 18 |
| phase4_cnn | tfhe | 3.96 MB | 1.10 MB | 23.4 KB | 114.84 MB | bootstraps: 32, cmp_pbs_ops: 96, ct_add: 296, scalar_add: 42, scalar_mul: 272 |
| phase4_cnn | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 7, poly_evals: 1, rescales: 7, rotations: 45 |
| phase5_digits | tfhe | 11.05 MB | 1.73 MB | 23.4 KB | 114.84 MB | bootstraps: 108, cmp_pbs_ops: 324, ct_add: 2016, scalar_add: 226, scalar_mul: 2124 |
| phase5_digits | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 7, poly_evals: 9, rescales: 7, rotations: 46 |
| phase5_qat | tfhe | 11.05 MB | 1.73 MB | 23.4 KB | 114.84 MB | bootstraps: 108, cmp_pbs_ops: 324, ct_add: 2016, scalar_add: 226, scalar_mul: 2124 |
| phase5_qat | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 7, poly_evals: 8, rescales: 7, rotations: 46 |
| phase6_onnx | tfhe | 11.05 MB | 1.73 MB | 23.4 KB | 114.84 MB | bootstraps: 108, cmp_pbs_ops: 324, ct_add: 2016, scalar_add: 226, scalar_mul: 2124 |
| phase6_onnx | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 7, poly_evals: 9, rescales: 7, rotations: 46 |
| phase6_sklearn | tfhe | 10.05 MB | 1.57 MB | 23.4 KB | 114.84 MB | ct_add: 640, scalar_add: 10, scalar_mul: 640 |
| phase6_sklearn | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 1, rescales: 1, rotations: 19 |
| phase7_faces | tfhe | 44.22 MB | 1.38 MB | 23.4 KB | 114.84 MB | bootstraps: 128, cmp_pbs_ops: 384, ct_add: 2176, scalar_add: 264, scalar_mul: 2304 |
| phase7_faces | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 7, poly_evals: 6, rescales: 7, rotations: 53 |

### Table D — Accuracy and Error

| Model | Float | Quantized (shared ref) | TFHE | CKKS max \|err\| | CKKS mean \|err\| | Declared bound | CKKS labels |
|---|---:|---:|---|---:|---:|---:|---|
| phase2_logreg | 1.0000 | 1.0000 | = quantized, exactly (err = 0.0) | n/a | n/a | 0.5 | 2/2 |
| phase4_cnn | 0.9805 | 0.9570 | = quantized, exactly (err = 0.0) | 3.000 | 0.850 | 10.0 | 2/2 |
| phase5_digits | 0.9639 | 0.9417 | = quantized, exactly (err = 0.0) | 35.000 | 10.950 | 60.0 | 2/2 |
| phase5_qat | 0.9361 | 0.9389 | = quantized, exactly (err = 0.0) | 28.000 | 13.100 | 50.0 | 2/2 |
| phase6_onnx | 0.9639 | 0.9417 | = quantized, exactly (err = 0.0) | 35.000 | 10.950 | 60.0 | 2/2 |
| phase6_sklearn | 0.8944 | 0.8806 | = quantized, exactly (err = 0.0) | 0.000 | 0.000 | 0.001 | 2/2 |
| phase7_faces | 0.9500 | 0.9000 | = quantized, exactly (err = 0.0) | 74.000 | 31.312 | 120.0 | 2/2 |

> ℹ️ **Bound violation resolution (`phase7_faces`):** The previous bound violation on `phase7_faces` Sample 1 (max error 192.0 > 150.0) was diagnosed as a `fit_requant` target-function bias: fitting the continuous pre-floor value introduced a systematic ~+0.5 LSB offset relative to the integer reference across the activation map, which `linear2`'s L1 weight norm amplified into ~200 integer units. Fitting the midpoint of each floor step eliminates the bias; all seven models now pass their bounds, with bounds re-declared at ~1.5x the multi-sample measured error, and golden tests now cover every fixture sample. The CKKS numbers in Tables A–D reflect the re-measured arm; TFHE numbers are carried over unchanged from the initial sweep (recorded in `meta.ckks_rerun` in [`docs/results/phase12-4-comparison.json`](./results/phase12-4-comparison.json)).

## Reproducing

```bash
# Regenerate the synthetic fixtures (NumPy only; prints accuracy):
uv run python examples/mnist/train_quantize_export.py   # Phase-2 logreg
uv run python examples/mnist/cnn_export.py               # Phase-4 CNN
# Regenerate the real-data fixtures (needs the optional `ml` extra: torch + sklearn + brevitas):
uv run --extra ml --system-certs python examples/mnist/real_digits_export.py  # Phase-5 PTQ
uv run --extra ml --system-certs python examples/mnist/qat_export.py          # Phase-5 QAT
uv run --extra ml --system-certs python examples/faces/olivetti_export.py     # Phase-7 faces (one-time ~4 MB download)

# Time the encrypted forward pass (release; the golden tests carry the timing):
cargo test --workspace --release --test golden_logreg -- --nocapture                  # ~30 s/sample
cargo test --release --test golden_cnn    -- --nocapture                  # ~3-4 min/sample
cargo test --release --test golden_digits -- --ignored --nocapture        # minutes/sample (real digits)
cargo test --release --test golden_qat    -- --ignored --nocapture        # minutes/sample (QAT)
cargo test --release --test golden_faces  -- --ignored --nocapture        # minutes/sample (faces)

# Inspect a model's per-tensor bit-widths without running FHE:
cargo run --release --bin inspect ../examples/mnist/phase5_digits_fixture.json

# Full cross-backend comparison sweep across all 7 fixtures (machine otherwise idle):
cargo +nightly build -p penumbra-bench --features ckks --release --bin penumbra-bench-report
mkdir -p target/bench-results
for m in phase2_logreg phase4_cnn phase5_digits phase5_qat phase6_onnx phase6_sklearn phase7_faces; do
  ./target/release/penumbra-bench-report \
    --models "$m" --backends tfhe,ckks --samples 2 \
    --format json --out "target/bench-results/$m.json"
done

# Run Criterion variance check:
PENUMBRA_BENCH_MODELS=phase2_logreg cargo +nightly bench -p penumbra-bench --features ckks
```
