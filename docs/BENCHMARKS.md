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
- **Crypto profile:** the tuned default `PARAM_MESSAGE_2_CARRY_2_KS_PBS` (`classic` profile,
  `MESSAGE_BITS = 2`, p-fail = 2^-129.581, algorithmic cost ~ 113). The parameter sweep evaluated
  discrete-Gaussian noise and multi-bit PBS parameter sets (grouping factors 2, 3, 4 with
  deterministic execution); `classic` was confirmed as the tuned default because outer rayon
  parallelism already saturates available CPU cores, where classic's lower serial cost outperforms
  multi-bit blind rotation while preserving 128-bit security and the smallest server key (114.84 MB).
  `num_blocks` is sized by the library to the model's widest accumulator.
- **Cost proxy:** bootstraps per sample — `runtime ≈ number of bootstraps` (`PROJECT.md` §5).
  This is a **TFHE** proxy; CKKS's is multiplicative depth, rotations, and rescales.
- **Regression gate:** `penumbra-bench-report --baseline <PATH>` gates machine-independent metrics:
  deterministic cost proxies (`cost_proxy`, `measured_totals` / PBS count), wire sizes (client/server
  keys, input/output ciphertexts), radix block count (`num_blocks`), and label correctness. Machine
  wall-clock timing is reported but intentionally not gated to avoid flake on variable CI runners.
  Regenerate the baseline with `penumbra-bench-report --write-baseline <PATH>` when an architectural
  or parameter change intentionally alters these properties.
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
| Radix | 6 blocks (12-bit signed, minimized Phase 10) |
| Latency / sample (encrypted) | ~0.52 s (was ~1.99 s pre-minimization, ~30 s baseline) |
| Bootstraps / sample | comparison plus carry-propagation PBS in radix arithmetic (136 PBS in `Linear`, 1 in `Argmax`, 137 total) |
### Phase-4 — small CNN, 10-class (`examples/mnist/phase4_cnn_fixture.json`)

`Conv2d(1→2, 3×3) → Requant+ReLU (auto-inserted) → Pool(avg 2×2) → Linear(8→10 logits)`,
synthetic 6×6 ten-class template data; the client decrypts the 10 logits and argmaxes.

| Metric | Value |
|---|---|
| Float accuracy | 0.98 |
| Quantized accuracy | 0.96 |
| Quantization gap | ~0.02 |
| Radix | 7 blocks (14-bit signed) |
| Latency / sample (encrypted) | ~27.1 s (was ~3–4 min baseline) |
| Dominant cost | radix MAC carry-propagation bootstraps in `Conv2d` and `Linear`; `Requant` is ~9 % of runtime |

As measured ground truth via `tfhe`'s `pbs-stats` counter reveals, the CNN's cost is
dominated by the internal carry-propagation bootstraps issued during radix addition and
scalar multiplication in `Conv2d` and `Linear`, while `Requant` accounts for only ~9 % of
runtime (1024 PBS out of 9104 total). The MAC loop's carry propagation makes multi-block radix
operations bootstrap-bearing (~2 PBS per block per add). Phase 10 optimizes this by grouping
inputs by weight value and evaluating multi-term sums via `sum_ciphertexts_parallelized`, cutting
carry propagation and latency by ~50 %. Evaluating per-element work in parallel across cores via
`rayon` (Phase 10) further cuts CNN evaluation latency from 38.912 s to 26.899 s (a 1.45x speedup),
with `Conv2d`, `Requant`, and `Pool` outputs processed concurrently.

### Phase-5 — real handwritten digits, PTQ (`examples/mnist/phase5_digits_fixture.json`)

The first example on a **real dataset** and a **real trained PyTorch model**: scikit-learn's
8×8 `load_digits` (real pen-written digits), quantized through the library service
(`Model.quantize`). `Conv2d(1→12, 3×3, stride 2) → Requant+ReLU → Linear(108→10 logits)`.

| Metric | Value |
|---|---|
| Float accuracy | ~0.96 (0.9639) |
| Quantized accuracy | 0.9167 (test batch; ~0.95 train split) |
| Quantization gap | ~0.047 |
| Weight / activation bits | (5, 6)-bit weights (conv, head; minimized Phase 10), 2-bit activations |
| Calibration | MSE (clip minimizing round-trip error), per-channel weights, `max_mult_bits = 1` |
| Radix | 9 blocks (18-bit signed; down from 11 blocks / 22-bit) |
| Bootstraps / sample | 75,153 PBS total (108 `Requant` PBS, 12 ch × 3×3; down from 109,045 PBS) |
| Latency / sample (encrypted) | ~222 s (was ~315 s pre-minimization, ~680 s baseline) |

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
| Float accuracy | ~0.93 (0.9333) |
| Quantized accuracy | 0.9361 |
| Quantization gap | ~0.00 (-0.0028) |
| Weight / activation bits | (5, 4)-bit weights (conv, head; minimized Phase 10), 2-bit activations |
| Calibration | MSE, per-channel weights, `max_mult_bits = 1` |
| Radix | 8 blocks (16-bit signed; down from 11 blocks / 22-bit) |
| Bootstraps / sample | 56,470 PBS total (108 `Requant` PBS; down from 119,096 PBS) |
| Latency / sample (encrypted) | ~179 s (was ~337 s pre-minimization, ~688 s baseline) |

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
| Bootstraps / sample | 128,571 PBS total (128 `Requant` PBS, 8 ch × 4×4) |
| Latency / sample (encrypted) | ~371 s (was ~730 s baseline) |

The ~0.05 gap is the cost of an 8-way decision from tiny 16×16 inputs with activations capped at a
single 2-bit block (`MESSAGE_BITS`, the hard TFHE-backend limit). The value of this example is **not**
its accuracy — it is that a completely different task (faces, not digits) ran encrypted end to end
with zero crypto-backend edits, exactly like `load_onnx` promised. The FHE golden test
(`golden_faces.rs`) is `#[ignore]`d because at 128 bootstraps/sample it is minutes per sample; the
fast Python guard (`tests/test_faces_fixture.py`) checks fixture self-consistency on every CI run.

## Cross-backend comparison

**Measured on Apple M3 Pro, macOS 25.6.0, `rustc 1.100.0-nightly (bba531001 2026-09-20)`, HAL backend `FFT64Neon` (`poulpy-ckks 0.8.3`), commit `9b38c1b`, 2026-09-24, `--samples 2`. Raw artifact: [`docs/results/phase10-final-sweep.json`](./results/phase10-final-sweep.json).**

> ℹ️ **Phase 10 final sweep:** Both TFHE and CKKS numbers in Tables A, B, C, and D are freshly measured from one nightly-built binary path on the pinned development machine across all seven bit-width-minimized fixtures.
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

| Model | Backend | Profile | Keygen (s) | Encrypt (s) | Eval total (s) | of which op-build (s) | Decrypt (s) | TFHE / CKKS eval |
|---|---|---|---:|---:|---:|---:|---:|---:|
| phase2_logreg | tfhe | classic | 0.500 | 0.017 | 0.521 | 0.000 | 0.000 | 0.8x |
| phase2_logreg | ckks | - | 2.153 | 0.007 | 0.647 | 0.001 | 0.001 | — |
| phase4_cnn | tfhe | classic | 0.524 | 0.011 | 27.149 | 0.000 | 0.000 | 41.1x |
| phase4_cnn | ckks | - | 2.184 | 0.007 | 0.660 | 0.001 | 0.001 | — |
| phase5_digits | tfhe | classic | 0.500 | 0.025 | 222.328 | 0.000 | 0.000 | 162.6x |
| phase5_digits | ckks | - | 2.176 | 0.005 | 1.367 | 0.013 | 0.001 | — |
| phase5_qat | tfhe | classic | 0.496 | 0.022 | 178.933 | 0.000 | 0.000 | 152.9x |
| phase5_qat | ckks | - | 2.301 | 0.009 | 1.170 | 0.011 | 0.001 | — |
| phase6_onnx | tfhe | classic | 0.501 | 0.025 | 217.966 | 0.000 | 0.000 | 162.1x |
| phase6_onnx | ckks | - | 2.095 | 0.007 | 1.345 | 0.013 | 0.001 | — |
| phase6_sklearn | tfhe | classic | 0.492 | 0.028 | 40.561 | 0.000 | 0.000 | 79.8x |
| phase6_sklearn | ckks | - | 1.915 | 0.008 | 0.508 | 0.000 | 0.001 | — |
| phase7_faces | tfhe | classic | 0.520 | 0.126 | 370.552 | 0.000 | 0.000 | 161.1x |
| phase7_faces | ckks | - | 2.458 | 0.007 | 2.300 | 0.008 | 0.000 | — |

*Variance check (`phase2_logreg`, Criterion 10 samples):* `tfhe` mean 488.82 ms (95% CI [470.38 ms, 510.38 ms], −74.0% change); `ckks` mean 350.44 ms (95% CI [347.15 ms, 354.35 ms], −15.1% change).

### Table B — Per-Op-Type Eval Breakdown (Mean Seconds per Sample)

Breakdown for `phase2_logreg`, `phase5_digits`, and `phase7_faces` (see [`docs/results/phase10-parallel-tuned-sweep.json`](./results/phase10-parallel-tuned-sweep.json) for the full 7-model op breakdown):

| Model | Backend | Op Type | Calls | Build (s) | Eval (s) | PBS (measured) |
|---|---|---|---:|---:|---:|---:|
| phase2_logreg | tfhe | Argmax | 1 | 0.0000 | 0.0160 | 1 |
| phase2_logreg | tfhe | Linear | 1 | 0.0000 | 0.5053 | 136 |
| phase2_logreg | ckks | Argmax | 1 | 0.0013 | 0.1420 | - |
| phase2_logreg | ckks | Linear | 1 | 0.0001 | 0.5033 | - |
| phase5_digits | tfhe | Conv2d | 1 | 0.0000 | 149.2259 | 48888 |
| phase5_digits | tfhe | Linear | 1 | 0.0000 | 51.4580 | 19137 |
| phase5_digits | tfhe | Requant | 1 | 0.0000 | 21.6435 | 7128 |
| phase5_digits | ckks | Conv2d | 1 | 0.0001 | 0.8807 | - |
| phase5_digits | ckks | Linear | 1 | 0.0000 | 0.2326 | - |
| phase5_digits | ckks | Requant | 1 | 0.0030 | 0.2505 | - |
| phase7_faces | tfhe | Conv2d | 1 | 0.0000 | 252.4591 | 84880 |
| phase7_faces | tfhe | Linear | 1 | 0.0000 | 68.8302 | 26971 |
| phase7_faces | tfhe | Requant | 1 | 0.0000 | 49.2624 | 16720 |
| phase7_faces | ckks | Conv2d | 1 | 0.0001 | 1.4143 | - |
| phase7_faces | ckks | Linear | 1 | 0.0000 | 0.1582 | - |
| phase7_faces | ckks | Requant | 1 | 0.0084 | 0.7193 | - |

### Table C — Sizes & Scheme Cost Proxies

| Model | Backend | Input CT | Output CT | Client Key | Server Key | Cost Proxy Counters |
|---|---|---:|---:|---:|---:|---|
| phase2_logreg | tfhe | 6.03 MB | 96.5 KB | 23.4 KB | 114.84 MB | cmp_pbs_ops: 1, ct_add: 31, scalar_add: 1, scalar_mul: 1, measured pbs: 137 |
| phase2_logreg | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 5, poly_evals: 1, rescales: 5, rotations: 16 |
| phase4_cnn | tfhe | 3.96 MB | 1.10 MB | 23.4 KB | 114.84 MB | bootstraps: 32, cmp_pbs_ops: 96, ct_add: 243, scalar_add: 42, scalar_mul: 114, measured pbs: 9104 |
| phase4_cnn | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 7, poly_evals: 1, rescales: 7, rotations: 45 |
| phase5_digits | tfhe | 9.04 MB | 1.41 MB | 23.4 KB | 114.84 MB | bootstraps: 108, cmp_pbs_ops: 324, ct_add: 1791, scalar_add: 226, scalar_mul: 1064, measured pbs: 75153 |
| phase5_digits | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 7, poly_evals: 2, rescales: 7, rotations: 46 |
| phase5_qat | tfhe | 8.04 MB | 1.26 MB | 23.4 KB | 114.84 MB | bootstraps: 108, cmp_pbs_ops: 324, ct_add: 1652, scalar_add: 226, scalar_mul: 966, measured pbs: 56470 |
| phase5_qat | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 7, poly_evals: 1, rescales: 7, rotations: 46 |
| phase6_onnx | tfhe | 9.04 MB | 1.41 MB | 23.4 KB | 114.84 MB | bootstraps: 108, cmp_pbs_ops: 324, ct_add: 1791, scalar_add: 226, scalar_mul: 1064, measured pbs: 75153 |
| phase6_onnx | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 7, poly_evals: 2, rescales: 7, rotations: 46 |
| phase6_sklearn | tfhe | 10.05 MB | 1.57 MB | 23.4 KB | 114.84 MB | ct_add: 457, scalar_add: 10, scalar_mul: 140, measured pbs: 13376 |
| phase6_sklearn | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 1, rescales: 1, rotations: 19 |
| phase7_faces | tfhe | 44.22 MB | 1.38 MB | 23.4 KB | 114.84 MB | bootstraps: 128, cmp_pbs_ops: 384, ct_add: 1962, scalar_add: 264, scalar_mul: 1500, measured pbs: 128571 |
| phase7_faces | ckks | 4.75 MB | 4.75 MB | 128.1 KB | 1782.50 MB | depth_levels: 7, poly_evals: 6, rescales: 7, rotations: 53 |

### Table D — Accuracy and Error

| Model | Float | Quantized (shared ref) | TFHE | CKKS max \|err\| | CKKS mean \|err\| | Declared bound | CKKS labels |
|---|---:|---:|---|---:|---:|---:|---|
| phase2_logreg | 1.0000 | 1.0000 | = quantized, exactly (err = 0.0) | n/a | n/a | 0.75 | 2/2 |
| phase4_cnn | 0.9805 | 0.9570 | = quantized, exactly (err = 0.0) | 4.000 | 1.600 | 6.0 | 2/2 |
| phase5_digits | 0.9639 | 0.9167 | = quantized, exactly (err = 0.0) | 38.000 | 12.700 | 60.0 | 2/2 |
| phase5_qat | 0.9333 | 0.9361 | = quantized, exactly (err = 0.0) | 10.000 | 4.800 | 15.0 | 2/2 |
| phase6_onnx | 0.9639 | 0.9167 | = quantized, exactly (err = 0.0) | 38.000 | 12.700 | 60.0 | 2/2 |
| phase6_sklearn | 0.8944 | 0.8806 | = quantized, exactly (err = 0.0) | 0.000155 | 0.000042 | 0.0005 | 2/2 |
| phase7_faces | 0.9500 | 0.9000 | = quantized, exactly (err = 0.0) | 74.000 | 31.312 | 120.0 | 2/2 |

### Phase 10 — Bootstrap reduction

Phase 10 implemented ground-truth PBS counting (reading `tfhe`'s `pbs-stats` internal atomic counter) and targeted the true dominant cost identified in profiling: the carry-propagation bootstraps issued by radix additions and scalar multiplications in the MAC loop.

#### 1. Same-Commit Before / After (Fast Pair)

Measured on Apple M3 Pro, commit `78f5db7` (optimized) vs commit `24acfdc` (instrumented baseline), identical binaries, `--samples 2`:

| Model | Baseline Eval (s) | Optimized Eval (s) | Speedup | Baseline PBS | Optimized PBS | PBS Reduction |
|---|---:|---:|---:|---:|---:|---:|
| phase2_logreg | 11.731 | 4.238 | 2.77x | 3326 | 1269 | -61.8% |
| phase4_cnn | 71.076 | 38.912 | 1.83x | 17811 | 10328 | -42.0% |

#### 2. Full-Sweep Comparison (Phase 10 vs Phase 12.4 Baseline Arm)

> ⚠️ **Threat to validity:** The baseline arm was measured at commit `dc20d05` (`rustc 1.100.0-nightly`), whereas Phase 10 was measured at commit `78f5db7` (`rustc 1.98.1`). The rigorous same-commit comparison is the fast pair above; the table below illustrates the system-wide impact across all 7 fixtures on the pinned Apple M3 Pro machine:

| Model | Phase 12.4 TFHE (s) | Phase 10 TFHE (s) | Speedup | Latency Reduction | Phase 10 Measured PBS |
|---|---:|---:|---:|---:|---:|
| phase2_logreg | 11.846 | 4.238 | 2.80x | -64.2% | 1269 |
| phase4_cnn | 69.987 | 38.912 | 1.80x | -44.4% | 10328 |
| phase5_digits | 679.860 | 358.398 | 1.90x | -47.3% | 109045 |
| phase5_qat | 687.919 | 395.328 | 1.74x | -42.5% | 119096 |
| phase6_onnx | 701.855 | 365.622 | 1.92x | -47.9% | 109045 |
| phase6_sklearn | 159.067 | 43.253 | 3.68x | -72.8% | 13376 |
| phase7_faces | 730.216 | 437.005 | 1.67x | -40.2% | 128571 |

### Phase 10 — Parallelism and parameter tuning

Following bootstrap reduction, Phase 10 tasks 3 and 4 parallelized per-element operations across CPU cores with `rayon` and evaluated `tfhe-rs` parameter profiles to establish a tuned default.

#### 1. Rayon Parallelism Before / After (Classic Profile)

Measured on Apple M3 Pro (11 cores), commit `6605986` (parallel) vs `78f5db7` (pre-parallel baseline), `--samples 2`:

| Model | Baseline Eval (s) | Parallel Eval (s) | Speedup | Latency Reduction | Baseline PBS | Parallel PBS | PBS Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| phase2_logreg | 4.238 | 1.992 | 2.13x | -53.0% | 555 | 555 | 0 |
| phase4_cnn | 38.912 | 26.899 | 1.45x | -30.9% | 9104 | 9104 | 0 |

As required by the golden invariant, ground-truth measured PBS counts are identical before and after the parallel refactor (555 on `phase2_logreg`, 9104 on `phase4_cnn`). Parallelism changes only *when* operations run across threads, not *what* operations are performed.

#### 2. Crypto Parameter Profile Sweep

Measured across all five 128-bit secure parameter profiles at `MESSAGE_BITS = 2` (`samples 2`, Apple M3 Pro, release build):

| Profile | Noise / Grouping | p-fail | Eval logreg (s) | Eval CNN (s) | Geomean speedup vs classic | Server key | Measured PBS |
|---|---|---|---:|---:|---:|---:|---|
| classic | TUniform / 1-bit PBS | 2^-129.581 | 2.010 | 27.030 | 1.000x | 114.84 MB | 555 (logreg), 9104 (CNN) |
| gaussian | Gaussian / 1-bit PBS | 2^-128.000 | 2.027 | 27.890 | 0.980x | 121.89 MB | 555 (logreg), 9104 (CNN) |
| multibit2 | TUniform / Group 2 | 2^-140.341 | 3.866 | 60.317 | 0.483x | 373.15 MB | 555 (logreg), 9104 (CNN) |
| multibit3 | TUniform / Group 3 | 2^-128.235 | 2.817 | 47.122 | 0.640x | 392.31 MB | 555 (logreg), 9104 (CNN) |
| multibit4 | TUniform / Group 4 | 2^-134.345 | 2.052 | 32.209 | 0.907x | 302.07 MB | 555 (logreg), 9104 (CNN) |

**Decision rule and outcome:** Multi-bit PBS profiles trade a higher serial algorithmic cost (group-2 ~188, group-3 ~143, group-4 ~100 vs classic ~113) for internal multi-threaded blind rotation. Because outer `rayon` parallelization over independent outputs already saturates available CPU cores, multi-bit PBS experiences thread contention with the outer pool, leading to lower net throughput and larger server keys (302–392 MB vs 114.84 MB). Under the decision rule requiring $S(p) \ge 1.15$, no multi-bit candidate qualifies. The `classic` parameter set (`PARAM_MESSAGE_2_CARRY_2_KS_PBS`) is confirmed as the tuned default profile.

### Phase 10 — Bit-width minimization

To minimize radix block capacity (`num_blocks`), Phase 10 implemented deterministic coordinate descent (`penumbra.quantization.minimize`) searching over three precision knobs:
1. `max_mult_bits`: caps the `Requant` fixed-point multiplier and internal transient multiply peak;
2. `input_bits`: graph-level input integer precision;
3. `weight_bits`: per-accumulator-layer weight bit widths.

Selection methodology: coordinate descent with a maximum accuracy drop tolerance of 1.0 absolute percentage point (`accuracy_tolerance = 0.01`). To prevent test set leakage, candidate bit plans were **selected on the calibration/training split** and evaluated on the held-out test split. If a search yielded no plan improving `num_blocks` within the tolerance floor, the baseline plan was retained unchanged.

#### Before / After Bit-Width Minimization (Apple M3 Pro, Classic Profile, `--samples 2`)

| Model | Plan `(in, w, mult)` [before → after] | Blocks [before → after] | PBS [before → after] | Eval (s) [before → after] | Speedup | Quant acc [before → after] |
|---|---|---:|---:|---:|---:|---:|
| phase2_logreg | `(4, [4], 5)` → `(2, [2], 5)` | 8 → 6 | 555 → 137 | 1.992 → 0.521 | 3.82x | 1.0000 → 1.0000 |
| phase4_cnn | `(4, [4], 5)` → `(4, [4], 5)` | 7 → 7 | 9104 → 9104 | 26.899 → 27.149 | 0.99x | 0.9570 → 0.9570 |
| phase5_digits | `(4, [6, 6], 5)` → `(3, [5, 6], 1)` | 11 → 9 | 109045 → 75153 | 314.796 → 222.328 | 1.42x | 0.9000 → 0.9167 |
| phase5_qat | `(4, [6, 6], 5)` → `(3, [5, 4], 1)` | 11 → 8 | 119096 → 56470 | 336.504 → 178.933 | 1.88x | 0.9417 → 0.9361 |
| phase6_onnx | `(4, [6, 6], 5)` → `(3, [5, 6], 1)` | 11 → 9 | 109045 → 75153 | 309.756 → 217.966 | 1.42x | 0.9000 → 0.9167 |
| phase6_sklearn | `(4, [8], 5)` → `(4, [8], 5)` | 10 → 10 | 13376 → 13376 | 40.859 → 40.561 | 1.01x | 0.8806 → 0.8806 |
| phase7_faces | `(4, [6, 6], 5)` → `(4, [6, 6], 5)` | 11 → 11 | 128571 → 128571 | 359.971 → 370.552 | 0.97x | 0.9000 → 0.9000 |

Across all seven committed models, bit-width minimization achieves a **1.46x geometric-mean speedup** on TFHE evaluation latency. For models with multi-channel convolutional layers (`phase5_digits`, `phase5_qat`, `phase6_onnx`), capping `max_mult_bits = 1` compressed the transient multiply peak from 21 bits down to 18 or 16 bits, shedding 2 to 3 radix blocks (saving ~92 s to ~158 s per inference) with negligible impact on accuracy. On `phase2_logreg`, dropping input and weight bits to 2 reduced radix blocks from 8 to 6, yielding a **3.82x speedup** (0.521 s vs 1.992 s) and dropping PBS from 555 to 137.

### Phase 10 — IR load cost and the binary-format decision

`ROADMAP.md` Phase 10 conditions introducing a binary IR format on profiling evidence that JSON deserialization represents meaningful overhead. Instrumented measurements captured in `penumbra-bench` across all seven models:

| Model | Backend | IR bytes | IR load (ms) | Eval total (s) | IR load as % of eval |
|---|---|---:|---:|---:|---:|
| phase2_logreg | tfhe | 5.0 KB | 0.44 | 0.5213 | 0.0838% |
| phase2_logreg | ckks | 5.0 KB | 0.44 | 0.6467 | 0.0675% |
| phase4_cnn | tfhe | 5.6 KB | 0.45 | 27.1493 | 0.0017% |
| phase4_cnn | ckks | 5.6 KB | 0.45 | 0.6595 | 0.0688% |
| phase5_digits | tfhe | 25.6 KB | 0.39 | 222.3275 | 0.0002% |
| phase5_digits | ckks | 25.6 KB | 0.39 | 1.3670 | 0.0288% |
| phase5_qat | tfhe | 25.4 KB | 0.38 | 178.9332 | 0.0002% |
| phase5_qat | ckks | 25.4 KB | 0.38 | 1.1697 | 0.0325% |
| phase6_onnx | tfhe | 25.8 KB | 0.40 | 217.9661 | 0.0002% |
| phase6_onnx | ckks | 25.8 KB | 0.40 | 1.3448 | 0.0297% |
| phase6_sklearn | tfhe | 14.2 KB | 0.41 | 40.5605 | 0.0010% |
| phase6_sklearn | ckks | 14.2 KB | 0.41 | 0.5084 | 0.0806% |
| phase7_faces | tfhe | 27.6 KB | 0.58 | 370.5519 | 0.0002% |
| phase7_faces | ckks | 27.6 KB | 0.58 | 2.3004 | 0.0251% |

**Decision:** Across all models and both backends, IR load time is under **0.6 ms** (0.38 ms – 0.58 ms) and accounts for at most **0.084%** of evaluation time (and less than 0.002% on multi-layer models). Wire sizes are compact (5.0 KB – 27.6 KB). Because IR loading is orders of magnitude below the noise floor of encrypted evaluation, introducing a binary IR format is unnecessary. Penumbra retains its backend-neutral, human-readable JSON IR format (`SCHEMA_VERSION = 0.6.0`, `AGENTS.md` §5).

## Reproducing

```bash
# Run bit-width minimization search on training split (does not write fixtures):
uv run python examples/mnist/train_quantize_export.py --minimize
uv run python examples/mnist/cnn_export.py --minimize
uv run --extra ml --system-certs python examples/mnist/real_digits_export.py --minimize
uv run --extra ml --system-certs python examples/mnist/qat_export.py --minimize
uv run --extra ml --system-certs python examples/mnist/onnx_export.py --minimize
uv run --extra ml --system-certs python examples/mnist/sklearn_export.py --minimize
uv run --extra ml --system-certs python examples/faces/olivetti_export.py --minimize

# Regenerate the synthetic fixtures from committed constants (NumPy only; prints accuracy):
uv run python examples/mnist/train_quantize_export.py   # Phase-2 logreg
uv run python examples/mnist/cnn_export.py               # Phase-4 CNN
# Regenerate the real-data fixtures (needs the optional `ml` extra: torch + sklearn + brevitas):
uv run --extra ml --system-certs python examples/mnist/real_digits_export.py  # Phase-5 PTQ
uv run --extra ml --system-certs python examples/mnist/qat_export.py          # Phase-5 QAT
uv run --extra ml --system-certs python examples/faces/olivetti_export.py     # Phase-7 faces (one-time ~4 MB download)

# Check / regenerate the committed regression baseline:
cargo run -p penumbra-bench --release --bin penumbra-bench-report -- \
  --models phase2_logreg,phase4_cnn,phase6_sklearn --backends tfhe --samples 1 \
  --baseline crates/penumbra-bench/baselines/tfhe-classic.json
cargo run -p penumbra-bench --release --bin penumbra-bench-report -- \
  --models phase2_logreg,phase4_cnn,phase6_sklearn --backends tfhe --samples 1 \
  --write-baseline crates/penumbra-bench/baselines/tfhe-classic.json

# Calibrate and verify CKKS error bounds:
cargo +nightly run -p penumbra-ckks --features ckks --release --example calibrate
cargo +nightly test -p penumbra-ckks --features ckks --release

# Time the encrypted forward pass (release; the golden tests carry the timing):
cargo test --workspace --release --test golden_logreg -- --nocapture                  # ~0.5 s/sample
cargo test --workspace --release --test golden_cnn    -- --nocapture                  # ~27 s/sample
cargo test --workspace --release --test golden_digits -- --ignored --nocapture        # minutes/sample (real digits)
cargo test --workspace --release --test golden_qat    -- --ignored --nocapture        # minutes/sample (QAT)
cargo test --workspace --release --test golden_faces  -- --ignored --nocapture        # minutes/sample (faces)

# Full cross-backend comparison sweep across all 7 fixtures (machine otherwise idle):
cargo +nightly build -p penumbra-bench --features ckks --release --bin penumbra-bench-report
mkdir -p target/bench-results
for m in phase2_logreg phase4_cnn phase5_digits phase5_qat phase6_onnx phase6_sklearn phase7_faces; do
  ./target/release/penumbra-bench-report \
    --models "$m" --backends tfhe,ckks --samples 2 \
    --format json --out "target/bench-results/$m.json"
done
# Result committed to docs/results/phase10-final-sweep.json

# Run Criterion variance check:
PENUMBRA_BENCH_MODELS=phase2_logreg cargo +nightly bench -p penumbra-bench --features ckks
```
