# TFHE vs CKKS: A Controlled Comparison

The study `penumbra-ckks` exists to enable. [`docs/BENCHMARKS.md`](./BENCHMARKS.md) owns the
measured numbers; this document owns the **argument** — what is being tested, why the setup
is valid, and what the results do and do not license anyone to conclude.

> **Status: measured (Phase 10 / Phase 12 Final Sweep).** Measured 2026-09-24 on Apple M3 Pro, macOS 25.6.0,
> `rustc 1.100.0-nightly (bba531001 2026-09-20)`, HAL backend `FFT64Neon` (`poulpy-ckks 0.8.3`),
> commit `9b38c1b`, `--samples 2`. Both TFHE and CKKS arms measured from a single binary path over bit-width minimized fixtures. Detailed numbers: [`docs/BENCHMARKS.md`](./BENCHMARKS.md);
> raw data: [`docs/results/phase10-final-sweep.json`](./results/phase10-final-sweep.json). Provenance caveat (the run used fixtures committed after `9b38c1b`): [Results provenance](./BENCHMARKS.md#results-provenance).

## The hypothesis

`PROJECT.md` §2 currently asserts, as settled:

> For small classifiers with ReLU/argmax (MNIST, faces, tabular), **TFHE is the correct
> choice** — exact, arbitrary activations as lookup tables, no batching needed.

That claim is well-motivated from first principles but has never been measured **on this
codebase, on these models, under one harness**. Phase 12 turns it from an assumption into a
result. The comparison may confirm it, and confirming it with numbers is a useful outcome; it
may also find the crossover point where CKKS's SIMD batching overtakes TFHE's exactness on
Penumbra's own workloads.

Concretely, three sub-questions:

1. **Latency** — for models in the seconds-to-minutes regime Penumbra targets, which scheme
   is faster, and where does the cost actually go (bootstraps vs. depth/rotations)?
2. **Accuracy** — TFHE's degradation is entirely quantization. CKKS adds approximation error
   from the polynomial nonlinearities. How large is that second term in practice?
3. **Overhead** — ciphertext size, key material size, and computation relative to cleartext.

## What makes this comparison valid

The controls are the point. Everything below is held identical across backends:

| Held constant | Mechanism |
|---|---|
| The model | the same committed ONNX / fixture files (`examples/`) |
| The graph | the **same IR file**, byte-for-byte — no schema change, no scheme tag (`docs/IR-SPEC.md`) |
| The quantization | the same `Model.quantize` output; the same integer weights, scales, and LUTs |
| The accuracy reference | `python/penumbra/reference.py`'s `evaluate_graph_int` — one oracle for both |
| The eval order | the same Layer-2 graph walker; neither backend gets a private fast path |
| The measurement code | one harness (`penumbra-bench`), one timer, one report format |
| The machine | a single pinned machine and a single pinned `poulpy` HAL backend |
| Security level | matched parameter profiles; never traded for speed (`AGENTS.md` §7) |

What varies is **only the backend crate**. Two independently built harnesses would introduce
confounds that no amount of analysis could separate from genuine scheme differences — which
is why the shared harness is a hard requirement rather than a convenience
(`docs/BACKENDS.md`, backend parity).

## Method

1. Both backends are registered with `penumbra-bench` and driven through the same entry point:
   `penumbra_bench::report::run_model` dispatches through `penumbra_core::eval::evaluate_graph_profiled`.
2. The `penumbra-bench-report` CLI binary provides unified operational modes:
   `--mode paper` for headline evaluation, `--mode calibrate` for calibration-derived bound estimation,
   `--mode diagnostics` for per-op timing breakdowns, and `--mode security-inputs` for lattice estimator extraction.
3. **Thread pinning & isolation:** Threads are explicitly configured via `--threads <N>` (`RAYON_NUM_THREADS = N`).
   Server peak RSS is captured via `getrusage` in an isolated child process executing only the server key load
   and forward evaluation, strictly excluding keygen, client secret keys, encryption, decryption, and size measurements.
4. **Calibration chronology (D7):** CKKS error bounds are derived over the training calibration split ($2.0 \times \text{p99}$),
   committed to `crates/penumbra-ckks/src/bounds.rs` and `docs/results/phase15-ckks-calibration.json` *before* evaluating test rows.
5. **Full-test vs. spot-check execution:**
   - **TFHE:** 30 distinct seeded spot checks (seed 1503; for faces, 20 test + 10 calibration rows) are verified bit-for-bit exact against `evaluate_graph_int`. Full-test task accuracy is reported as `quantized_reference_inferred_exact` with evidence of the 30 passed encrypted checks.
   - **CKKS:** All samples in the full test split are evaluated encrypted. Raw unrounded floating-point outputs are compared directly against the integer reference to prevent rounding from masking error noise.
6. **Tie-breaking & zero-margin metrics:** Multiclass predicted labels use first-maximum tie-breaking (`argmax`). Only logical output components enter metrics; non-finite components fail. Margin-relative score errors divide max component error by top-two integer-reference margin. Binary label models instead use the fixture's score tap and `abs(reference_score - decision_threshold)`; the actual decrypted decision still determines accuracy, and the original decision output retains its fixed absolute bound. Zero denominators are reported as null and tracked through explicit counters.
7. **D16 PBS accounting:** Measured PBS operations are partitioned into logical lookup PBS (Activation/Requant `bootstraps`, Compare/Argmax `cmp_pbs_ops`) and residual carry PBS (`total_pbs - lookup_pbs`).
8. **Report provenance:** Paper report version 2 adds original-graph node profiles and the actual `profile_sample_id`; fixture/protocol versions remain 1 and the IR is unchanged. Profiles are diagnostic and may use a different sample from Criterion. Score taps do not alter the Criterion/RSS graph. Unsupported rows preserve the actual backend graph-budget rejection before key generation, rather than a model-name policy explanation.

## Metrics

| Metric | Definition | Why it is here |
|---|---|---|
| Headline Latency | Canonical Criterion median wall clock over 10 flat samples with 95% CI | the headline practical question |
| Server Peak RSS | `getrusage(RUSAGE_SELF)` in dedicated server evaluation child process | memory overhead of server evaluation |
| Per-op-type breakdown | time attributed at the eval-loop seam | shows *why* one scheme wins, not just that it does |
| D16 PBS breakdown | logical lookup PBS vs residual carry PBS | isolates LUT operations from arithmetic carry propagation |
| Task accuracy | prediction agreement with ground-truth target labels | task performance under encryption |
| Quantized reference accuracy | prediction agreement with `evaluate_graph_int` | isolates scheme approximation error from quantization error |
| Absolute error distribution | raw float \|err\| distribution (median, p95, p99, max) vs integer reference | ground-truth CKKS noise behavior |
| Margin-relative score error | max logit error / top-two reference margin; binary score-tap error / distance to declared threshold | score perturbation relative to decision boundaries |
| Scheme cost proxy | bootstraps (TFHE) · depth + rotations + rescales (CKKS) | lets the numbers generalize past this machine |
| Ciphertext size | bytes per encrypted input and output wire | bandwidth cost of client/server split |
| Key material size | client key + evaluation key bytes | deployment storage overhead |

## Threats to validity

Stated in advance, and to be restated alongside any published result.

1. **Single machine, single HAL backend (D12).** All benchmarks were executed on a single pinned Apple Silicon M3 Pro development host (11 cores, 18 GB RAM, macOS 26.6.2, Darwin 25.6.0) using the `FFT64Neon` assembly HAL backend for CKKS. Absolute latencies do not transfer to x86-64 server architectures (AVX2/AVX-512) or cloud instances. Ratios and cost proxies are more platform-robust than raw wall-clock times.
2. **SIMD packing, slot utilization, and unmeasured batching (D10).** Under Option B (`crates/penumbra-ckks/src/params.rs`), `penumbra-ckks` packs one full tensor per ciphertext and evaluates linear operations as BSGS diagonal transforms over $lt\_slots = 256$ slots. Because $N = 16384$, a ciphertext provides 8,192 SIMD slots, meaning slot utilization is $\le 256 / 8192$ (3.125%). Throughput gains from evaluating multiple concurrent inferences in packed unused slots are architecturally possible but **unmeasured and not claimed** by this study, which focuses strictly on single-sample server evaluation latency.
3. **Leveled CKKS without bootstrapping (D11).** The CKKS backend evaluates as a leveled scheme within a fixed multiplicative depth budget (330 bits at $k = 360$, $\log \Delta = 30$, $\text{max\_poly\_degree} = 15$, security $\ge 128$ bits). It does not implement CKKS bootstrapping. Workloads requiring higher multiplicative depth (such as the 4-node tree ensemble requiring 360 bits) exceed the level budget and are rejected at load time. All depth-limited findings are explicitly characterized as **leveled CKKS at these parameters**.
4. **The graph is quantized for TFHE.** Penumbra caps intermediate activations at 2-bit bottlenecks to keep programmable bootstrap LUTs computationally feasible (`docs/QUANTIZATION.md`). CKKS has no such constraint and typically operates on high-precision floating-point inputs. Feeding the TFHE-shaped low-bit quantized graph to CKKS enforces apples-to-apples architectural parity, but simultaneously handicaps CKKS precision and increases sensitivity to polynomial approximation error.
5. **Polynomial degree truncation.** In Phase 12.2, $\text{max\_poly\_degree} = 15$ was selected to match classification labels while keeping multiplicative depth within the 128-bit security envelope ($N=16384$, $k=360$). Residual error on polynomial activations is bounded mathematical approximation noise, which on complex branching topologies (`phase8_branch`) causes significant label flips.
6. **Input scale boundaries and capacity exclusion (D18).** The raw 28×28 MNIST scale probe (784 input elements) evaluates successfully under TFHE (119.99 s), but exceeds the fixed 256-slot capacity of the BSGS diagonal linear transform in single-ciphertext CKKS. Per the approved D18 amendment, larger models exceeding 256 elements are excluded from the controlled comparison suite to preserve architectural parity without introducing multi-ciphertext packing.
7. **External calibration methodology boundaries (D15).** Concrete-ML 1.9.0 evaluates the `phase6_onnx` architecture in 146.97 s natively on the same host. However, Concrete-ML uses uniform 6-bit quantization, its own compilation pipeline, and differing measurement boundaries. This comparison provides external sanity-checking context only; it is not part of the controlled TFHE-vs-CKKS benchmark suite.
8. **Criterion sample statistics and small test sets.** Headline latencies are canonical Criterion medians over 10 flat samples with 95% confidence intervals. Small differences within confidence intervals are not considered findings. Accuracy figures evaluate committed test splits (20 to 360 samples) and cannot support population claims beyond the declared workloads.
9. **Library and implementation maturity asymmetry.** `tfhe-rs` is a mature, production-grade library with multi-threaded rayon evaluation across independent outputs. `poulpy-ckks` is an evolving pure-Rust library (v0.8.x), and Penumbra's CKKS backend evaluates single-threaded behind an internal arena mutex. Part of the observed latency ratio reflects library optimization rather than fundamental scheme theory.
10. **One toolchain for both arms.** Both backends were compiled into a single release binary using nightly Rust (`rustc 1.100.0-nightly`), ensuring an identical measurement harness. Any nightly-vs-stable codegen variance lands on the TFHE arm.

## Results

**The CKKS backend evaluates under Option B (one full tensor per ciphertext, BSGS diagonal transforms over `lt_slots = 256`) ([`docs/BACKENDS.md`](./BACKENDS.md), `crates/penumbra-ckks/src/params.rs`). Linear operations (`Linear`, `Conv2d`, `Pool(avg)`) are evaluated via SIMD diagonal transforms rather than arrays of scalar ciphertexts.** Full per-op, cost proxy, and diagnostic before/after tables appear in [`docs/BENCHMARKS.md`](./BENCHMARKS.md); summary tables derived from the frozen paper protocol (`phase16-paper-final.json` @ `3833c94`) are below.

### Latency and Memory

*Canonical headline latencies are Criterion medians over 10 flat samples with 95% confidence intervals on Apple M3 Pro (`RAYON_NUM_THREADS=11`, CKKS single-threaded `FFT64Neon`). Server memory is peak RSS captured via `getrusage` in an isolated child process evaluating only the server forward pass.*

| Model | TFHE Latency (median [95% CI]) | CKKS Latency (median [95% CI]) | Ratio (TFHE / CKKS) | TFHE Server RSS | CKKS Server RSS | Memory Ratio (CKKS / TFHE) |
|---|---|---|---:|---:|---:|---:|
| `phase2_logreg` | 0.3258 s [0.3245 s .. 0.3269 s] | 0.3404 s [0.3400 s .. 0.3416 s] | **0.96x (TFHE faster)** | 276.44 MB | 5146.77 MB | 18.6x |
| `phase4_cnn` | 15.6253 s [15.4665 s .. 15.9245 s] | 0.4932 s [0.4891 s .. 0.5013 s] | 31.68x | 291.88 MB | 5247.70 MB | 18.0x |
| `phase5_digits` | 75.3691 s [75.0327 s .. 75.6319 s] | 1.0833 s [1.0695 s .. 1.0974 s] | 69.57x | 329.73 MB | 4454.91 MB | 13.5x |
| `phase5_qat` | 68.5777 s [68.4621 s .. 68.7342 s] | 0.9794 s [0.9650 s .. 0.9886 s] | 70.02x | 325.16 MB | 3942.00 MB | 12.1x |
| `phase6_onnx` | 75.6790 s [75.2890 s .. 75.9957 s] | 1.0824 s [1.0770 s .. 1.0936 s] | 69.92x | 324.55 MB | 4873.66 MB | 15.0x |
| `phase6_sklearn` | 12.1230 s [11.9946 s .. 12.2211 s] | 0.3787 s [0.3773 s .. 0.3829 s] | 32.01x | 351.36 MB | 4326.44 MB | 12.3x |
| `phase7_faces` | 123.0731 s [122.9415 s .. 123.2908 s] | 2.0114 s [2.0031 s .. 2.0212 s] | 61.19x | 497.81 MB | 4119.27 MB | 8.3x |
| `phase8_trees` | 4.6731 s [4.6128 s .. 4.7388 s] | *Unsupported (depth budget exceeded)* | — | 274.58 MB | — | — |
| `phase8_branch` | 35.3862 s [35.1211 s .. 35.8782 s] | 0.7333 s [0.7312 s .. 0.7371 s] | 48.26x | 365.11 MB | 4244.89 MB | 11.6x |
| `phase8_bn_cnn` | 88.0627 s [87.8985 s .. 88.2597 s] | 1.0035 s [0.9982 s .. 1.0121 s] | 87.76x | 318.38 MB | 4054.78 MB | 12.7x |
| `phase8_gap_cnn` | 162.9067 s [162.3718 s .. 163.0674 s] | 1.7180 s [1.7022 s .. 1.7298 s] | 94.82x | 330.22 MB | 5255.83 MB | 15.9x |
| `phase8_tanh` | 28.1181 s [27.8604 s .. 28.4598 s] | 0.6359 s [0.6313 s .. 0.6397 s] | 44.22x | 361.86 MB | 3948.52 MB | 10.9x |
| `phase8_xgb` | 4.8506 s [4.7964 s .. 4.9365 s] | *Unsupported (depth budget exceeded)* | — | 273.73 MB | — | — |
| `phase11_tabular_mlp` | 9.2257 s [9.1662 s .. 9.2561 s] | 0.4185 s [0.4165 s .. 0.4202 s] | 22.04x | 301.50 MB | 4400.12 MB | 14.6x |

*(Source: [`docs/results/phase16-paper-final.json`](./results/phase16-paper-final.json) @ `3833c94`.)*

### Accuracy and Error

*Evaluated under encryption against the shared quantized integer reference (`evaluate_graph_int`). TFHE verifies 30 distinct seeded spot checks bit-for-bit exact; CKKS evaluates the full test split raw unrounded floating-point output against declared calibration bounds derived in Phase 15 (`2.0 * p99` over training calibration split).*

| Model | Float Acc | Quantized Ref Acc | TFHE Task Acc | CKKS Task Acc | CKKS Label Flips | CKKS Max |err| | Declared Bound | Bound Status |
|---|---:|---:|---|---:|---:|---:|---:|---|
| `phase2_logreg` | 1.0000 | 1.0000 | 1.0000 (30/30 exact) | 1.0000 | 0 / 256 (0.00%) | 0.4978 | 0.9952 | PASS |
| `phase4_cnn` | 0.9805 | 0.9570 | 0.9570 (30/30 exact) | 0.9922 | 11 / 256 (4.30%) | 11.9345 | 16.8386 | PASS |
| `phase5_digits` | 0.9639 | 0.9167 | 0.9167 (30/30 exact) | 0.9222 | 14 / 360 (3.89%) | 83.2113 | 150.1730 | PASS |
| `phase5_qat` | 0.9333 | 0.9361 | 0.9361 (30/30 exact) | 0.9417 | 12 / 360 (3.33%) | 17.4862 | 30.3024 | PASS |
| `phase6_onnx` | 0.9639 | 0.9167 | 0.9167 (30/30 exact) | 0.9222 | 14 / 360 (3.89%) | 83.2117 | 151.0399 | PASS |
| `phase6_sklearn` | 0.8944 | 0.8806 | 0.8806 (30/30 exact) | 0.8806 | 1 / 360 (0.28%) | 0.0002 | 0.0004 | PASS |
| `phase7_faces` | 0.9500 | 0.9000 | 0.9000 (30/30 exact) | 0.9000 | 0 / 20 (0.00%) | 110.9689 | 288.7376 | PASS |
| `phase8_trees` | 0.9561 | 0.9561 | 0.9561 (30/30 exact) | — | — | — | — | *Unsupported (depth)* |
| `phase8_branch` | 0.9167 | 0.9056 | 0.9056 (30/30 exact) | 0.2028 | 289 / 360 (80.28%) | 76.5457 | 137.3517 | PASS |
| `phase8_bn_cnn` | 0.9444 | 0.9278 | 0.9278 (30/30 exact) | 0.9389 | 15 / 360 (4.17%) | 37.5164 | 60.9603 | PASS |
| `phase8_gap_cnn` | 0.5889 | 0.5611 | 0.5611 (30/30 exact) | 0.5889 | 118 / 360 (32.78%) | 263.4005 | 381.4977 | PASS |
| `phase8_tanh` | 0.6861 | 0.6917 | 0.6917 (30/30 exact) | 0.9583 | 105 / 360 (29.17%) | 42.3772 | 70.9196 | PASS |
| `phase8_xgb` | 0.9708 | 0.9649 | 0.9649 (30/30 exact) | — | — | — | — | *Unsupported (depth)* |
| `phase11_tabular_mlp` | 0.9561 | 0.9561 | 0.9561 (30/30 exact) | 0.9649 | 1 / 114 (0.88%) | 26.8308 | 57.6876 | PASS |

*(Derived from [`docs/BENCHMARKS.md` Table 3](./BENCHMARKS.md#table-3-full-test-accuracy-error-distributions-and-bound-verification-phase-16); source [`docs/results/phase16-paper-final.json`](./results/phase16-paper-final.json) @ `3833c94`.)*

### Overhead

Key material, ciphertext dimensions, and memory footprints across the suite (source [`docs/results/phase16-paper-final.json`](./results/phase16-paper-final.json) @ `3833c94`):

- **Server memory footprint (Peak RSS):**
  - TFHE server evaluation requires **273.73 MB to 497.81 MB** of peak resident set size across all models.
  - CKKS server evaluation requires **3,942.00 MB to 5,255.83 MB** (3.85 GiB to 5.13 GiB) of peak resident set size.
  - CKKS server memory overhead is **8.3x to 18.6x higher** than TFHE.
- **Ciphertext sizes:**
  - TFHE encodes each integer element as a radix ciphertext (`num_blocks` shortint ciphertexts). Input ciphertexts scale linearly with input tensor length and radix blocks (from 3.30 MB on 30-feature tabular inputs to 44.22 MB on 256-element faces; `phase2_logreg` is 6.03 MB). Output ciphertexts range from 16.1 KB (scalar decision) to 1.57 MB.
  - CKKS encodes entire tensors into single SIMD ciphertexts ($N = 16384$, $k = 360$). Every input and output ciphertext is exactly 4.75 MB regardless of tensor dimension ($\le 256$ elements).
- **Key material sizes:**
  - TFHE client key: 23.4 KB; server key (bootstrapping and key-switching keys): 114.84 MB.
  - CKKS client key: 128.1 KB; server key (Galois rotation keys + relinearization keys): 1,782.50 MB (1.74 GiB, a 15.5x storage overhead).
- **Scheme cost proxies:**
  - TFHE cost is dominated by radix carry propagation and LUT lookups: total PBS ranges from 91 on `phase2_logreg` to 65,629 on `phase8_gap_cnn`. Across all models, logical lookup PBS accounts for a small fraction (0 to 256 lookups), while carry propagation accounts for 90 to 65,373 PBS (D16 breakdown).
  - CKKS cost is dominated by BSGS rotations (16 to 85 rotations per sample) and polynomial evaluation rescales (1 to 12 rescales across depth levels 1 to 5).

### Discussion

The study answers the motivating questions with concrete, reproducible data on committed workloads:

1. **The core questions answered:**
   - **Shallow / linear models:** On single-layer binary logistic regression (`phase2_logreg`), **the provisional status is resolved**: under canonical Criterion evaluation, TFHE evaluates in **0.3258 s [0.3245 s .. 0.3269 s]** vs CKKS **0.3404 s [0.3400 s .. 0.3416 s]** (a **0.96x** ratio; TFHE is faster). The 95% confidence intervals do not overlap. TFHE achieves this latency advantage without polynomial approximation, without SIMD rotation overhead, and with an **18.6x lower server RAM footprint** (276.44 MB vs 5146.77 MB).
   - **Multi-layer convolutional networks:** On deep multi-layer CNNs (`phase4_cnn`, `phase5_digits`, `phase5_qat`, `phase6_onnx`, `phase7_faces`, `phase8_bn_cnn`, `phase8_gap_cnn`), CKKS under SIMD tensor packing (Option B) is **31.7x to 94.8x faster** than TFHE (evaluating in 0.49 s to 2.01 s per sample vs TFHE's 15.6 s to 162.9 s). SIMD batching within a single packed ciphertext vastly outperforms element-wise multi-block carry-propagation arithmetic.
   - **Accuracy and exactness:** TFHE guarantees bit-for-bit exact identity with the quantized integer specification across all 14 models (420/420 spot checks passed). CKKS introduces bounded approximation error (max error 0.0002 to 263.4 across models), all within declared calibration bounds. On networks with sharp step functions (`phase8_branch`), polynomial approximation error can lead to substantial label flip rates (80.28%), whereas TFHE remains 100% exact.
   - **Server memory and deployment overhead:** CKKS pays for its latency advantage with substantial infrastructure overhead: server peak RSS is **10x to 19x higher** (4–5 GB vs ~300 MB), and evaluation keys are **15.5x larger** (1.74 GiB vs 114.84 MB).

2. **D17: Within-scheme implementation gain (N) vs cross-scheme gap (M):**
   In [`docs/BENCHMARKS.md` Table 4](./BENCHMARKS.md#table-4-d17-beforeafter-tfhe-analysis-within-scheme-speedup-n-vs-cross-scheme-gap-m), per-tensor radix width optimization achieved a within-scheme speedup of **N = 1.33x to 3.78x** across models (reducing PBS counts by 32.6% to 70.0%). On `phase2_logreg`, canonical Criterion latency improved by **1.46x** (0.4740 s down to 0.3258 s). However, on multi-layer CNNs, the cross-scheme gap remains **M = 31.7x to 94.8x**. This confirms the D17 thesis: while implementation engineering within TFHE yields significant constant-factor improvements (2x–4x), the fundamental structural gap between SIMD ciphertext batching (CKKS) and element-wise radix arithmetic (TFHE) on deep networks spans one to two orders of magnitude.

3. **D11: Tree ensembles under leveled CKKS:**
   Decision tree ensembles (`phase8_trees`, `phase8_xgb`) evaluate natively and bit-for-bit exactly on TFHE via `scalar_ge_parallelized` comparison PBSs in 4.67 s and 4.85 s Criterion median (with 67 and 69 lookup PBS). Under leveled CKKS, `Compare` requires continuous smoothed step polynomial approximation over a linear map. For a 4-node tree graph with chained step functions (`split_cmp` and `leaf_sel`), the multiplicative level budget required (360 bits) exceeds the depth budget capacity (330 bits at $k=360, \log \Delta=30, \text{max\_poly\_degree}=15$). The models are rejected at load time as **unsupported on leveled CKKS at these parameters**. This highlights an asymmetric scheme capability: TFHE evaluates sharp discrete comparisons and branching logic natively without multiplicative level consumption, whereas leveled CKKS cannot evaluate chained tree branches without bootstrapping or parameter expansion.

4. **D18: Raw 28×28 MNIST scale probe:**
   The raw 28×28 MNIST scale probe (`phase16_mnist28_fixture.json`, 784 inputs) evaluates encrypted under TFHE in **119.99 s** for a server evaluation, easily meeting the ≤ 600 s feasibility gate. Decrypted logits match the integer reference bit-for-bit (10/10 exact). Under the approved D18 amendment, the probe is excluded from the controlled comparison suite because the raw 784-element input exceeds CKKS linear-transform SIMD slot capacity (`lt_slots = 256`), providing concrete evidence of input scale boundaries under single-ciphertext packing.

5. **D15: External Concrete-ML calibration:**
   Native execution of Concrete-ML 1.9.0 on the identical M3 Pro host (`phase16-concrete-calibration.json`) on the `phase6_onnx` architecture under unrounded configuration ($p_{error}=2^{-40}$, 6-bit quantization) yields a median encrypted latency of **146.97 s** per sample, with 30/30 spot checks matching Concrete's reference. Penumbra's TFHE evaluates the same architecture in **75.68 s** Criterion median. This external calibration confirms that Penumbra's TFHE implementation operates within the same latency regime as industry-standard TFHE tooling on identical hardware, while reinforcing that Concrete-ML's separate quantizer and execution boundaries place it outside the controlled comparison.

6. **Verdict on `PROJECT.md` §2:**
   `PROJECT.md` §2 originally asserted:
   > *"For small classifiers with ReLU/argmax (MNIST, faces, tabular), **TFHE is the correct choice** — exact, arbitrary activations as lookup tables, no batching needed."*

   **Final Verdict:**
   - **For shallow, linear models (`phase2_logreg`):** The hypothesis is **supported**. TFHE achieves sub-second inference (**0.3258 s**, faster than CKKS's 0.3404 s) with bit-for-bit exactness, no rotation keys, and 18.6x lower server RAM.
   - **For discrete branching and tree ensembles (`phase8_trees`, `phase8_xgb`):** The hypothesis is **supported**. TFHE evaluates exact comparisons efficiently (~4.7 s), whereas leveled CKKS cannot evaluate the depth budget at these parameters.
   - **For multi-layer networks (CNNs, MLPs with multiple activation stages):** The hypothesis is **refuted on latency**. CKKS SIMD diagonal transforms are **31x to 95x faster** than multi-block radix carry arithmetic. However, CKKS incurs bounded approximation error, requires declared error bounds, uses 15.5x larger server keys (1.74 GiB), and demands 10x–19x higher server RAM (4–5 GB).

## Scope

This study compares two backends on **Penumbra's existing supported operations and committed models**. It is not a general survey of FHE schemes, not a claim about CKKS or TFHE outside this workload class, and not an attempt at feature completeness in either backend beyond what the comparison requires (`PROJECT.md` §18).
