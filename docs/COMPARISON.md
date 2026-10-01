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
4. **Calibration chronology (D10):** CKKS error bounds are derived over the training calibration split ($2.0 \times \text{p99}$),
   committed to `crates/penumbra-ckks/src/bounds.rs` and `docs/results/phase15-ckks-calibration.json` *before* evaluating test rows.
5. **Full-test vs. spot-check execution:**
   - **TFHE:** 30 distinct seeded spot checks (seed 1503; for faces, 20 test + 10 calibration rows) are verified bit-for-bit exact against `evaluate_graph_int`. Full-test task accuracy is reported as `quantized_reference_inferred_exact` with evidence of the 30 passed encrypted checks.
   - **CKKS:** All samples in the full test split are evaluated encrypted. Raw unrounded floating-point outputs are compared directly against the integer reference to prevent rounding from masking error noise.
6. **Tie-breaking & zero-margin metrics:** Multi-class predicted labels use first-maximum tie-breaking (`argmax`). Margin-relative score errors divide max component error by top-two reference margin. Where reference top-two margins are zero (ties), relative error is reported as null and tracked via explicit zero-margin counters.
7. **D16 PBS accounting:** Measured PBS operations are partitioned into logical lookup PBS (Activation/Requant `bootstraps`, Compare/Argmax `cmp_pbs_ops`) and residual carry PBS (`total_pbs - lookup_pbs`).

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
| Margin-relative score error | max score error divided by top-two reference logit margin | score perturbation relative to decision boundaries |
| Scheme cost proxy | bootstraps (TFHE) · depth + rotations + rescales (CKKS) | lets the numbers generalize past this machine |
| Ciphertext size | bytes per encrypted input and output wire | bandwidth cost of client/server split |
| Key material size | client key + evaluation key bytes | deployment storage overhead |

## Threats to validity

Stated in advance, and to be restated alongside any published result.

1. **SIMD packing (resolved in Phase 12.2).** `penumbra-ckks` packs one full tensor per
   ciphertext (Option B) and evaluates plaintext-weight linear ops (`Linear`, `Conv2d`,
   `Pool(avg)`) as BSGS diagonal transforms over `lt_slots = 256` slots. This leverages
   CKKS SIMD batching as designed without leaking packing concerns into Layer 2.
2. **The graph is quantized for TFHE.** Penumbra caps activations at a single 2-bit block
   because a programmable bootstrap is only feasible over a narrow value
   (`docs/QUANTIZATION.md`). CKKS has no such constraint and would ordinarily run at much
   higher precision. Feeding it the TFHE-shaped graph is what makes the comparison
   apples-to-apples, and it simultaneously handicaps CKKS on accuracy. In Phase 10, bit-width
   minimization further compressed radix capacities (down to 6–9 blocks via 2/3-bit inputs,
   per-layer weights, and capped multipliers), tightening this threat: CKKS evaluates graphs
   quantized even more aggressively for radix integer arithmetic.
3. **Polynomial degree is a free parameter.** In Phase 12.2, `max_poly_degree = 15` was
   calibrated as the default profile knob (`depth = 4` under BSGS `MinDepth`). It provides
   sufficient precision to match classification labels across all committed models while
   keeping depth within the 128-bit classical security envelope ($N=16384$, $k=360$).
4. **Library maturity is asymmetric.** `tfhe-rs` is a mature, heavily optimized production
   library at 1.6+. `poulpy-ckks` is at 0.8.x and self-describes its API as subject to change.
   Any latency difference partly reflects engineering investment, not scheme fundamentals.
5. **Implementation effort is asymmetric.** The TFHE backend is the product of the entire
   project to date; the CKKS backend is new. An unoptimized backend losing on latency is weak
   evidence about the scheme.
6. **Single machine, single HAL backend.** Absolute numbers do not transfer. Ratios are more
   robust than absolutes, and the cost proxies more robust still.
7. **Small models, small test batches.** Penumbra's committed batches are deliberately tiny
   (`N_TEST`) because each FHE sample is expensive. Accuracy differences within
   small-test-set noise must not be reported as findings.
8. **Op build is charged per sample.** The shared walker times `build_op` inside each
   inference; under CKKS that is BSGS diagonal encoding a real deployment would precompute
   once per model. Reported as a separate `of which op-build` column so it can be discounted.
9. **One toolchain for both arms.** Both backends were measured from a single nightly-built
   binary so that the measurement path is identical; TFHE normally ships on stable. Any
   stable-vs-nightly codegen difference lands on the TFHE arm.

## Results

**The CKKS backend evaluates under Option B (one full tensor per ciphertext, BSGS diagonal transforms over `lt_slots = 256`) ([`docs/BACKENDS.md`](./BACKENDS.md), `crates/penumbra-ckks/src/params.rs`). Linear operations (`Linear`, `Conv2d`, `Pool(avg)`) are evaluated via SIMD diagonal transforms rather than arrays of scalar ciphertexts.** Full per-op and size tables appear in [`docs/BENCHMARKS.md`](./BENCHMARKS.md); summary tables are below.

### Latency

| Model | TFHE / sample | CKKS / sample | Ratio |
|---|---:|---:|---:|
| Phase-2 logreg | 0.52 s | 0.65 s | 0.8x |
| Phase-4 CNN | 27.15 s | 0.66 s | 41.2x |
| Phase-5 digits (PTQ) | 222.33 s | 1.37 s | 162.6x |
| Phase-5 digits (QAT) | 178.93 s | 1.17 s | 153.0x |
| Phase-6 ONNX | 217.97 s | 1.35 s | 162.1x |
| Phase-6 sklearn | 40.56 s | 0.51 s | 79.8x |
| Phase-7 faces | 370.55 s | 2.30 s | 161.1x |

*(Source: [`docs/results/phase10-final-sweep.json`](./results/phase10-final-sweep.json) @ `9b38c1b`, via [`docs/BENCHMARKS.md` Table A](./BENCHMARKS.md#table-a-latency-wall-clock-per-sample). Means over N = 2 samples in `--release`, Apple M3 Pro, FFT64Neon HAL; each mean includes the first, cold sample. Ratios derive from unrounded JSON means (e.g. 41.2x, 153.0x); dividing the 2-decimal rounded table values yields 41.1x and 152.9x due to intermediate rounding. The logreg ordering is contradicted by Criterion — see [Logreg timing reconciliation](./BENCHMARKS.md#logreg-timing-reconciliation).)*

### Accuracy

| Model | Float | Quantized (shared reference) | TFHE | CKKS max \|err\| | Declared bound | CKKS labels |
|---|---:|---:|---|---:|---:|---|
| Phase-2 logreg | 1.0000 | 1.0000 | *= quantized, exactly* | n/a | 0.75 | 2/2 |
| Phase-4 CNN | 0.9805 | 0.9570 | *= quantized, exactly* | 4.000 | 6.0 | 2/2 |
| Phase-5 digits (PTQ) | 0.9639 | 0.9167 | *= quantized, exactly* | 38.000 | 60.0 | 2/2 |
| Phase-5 digits (QAT) | 0.9333 | 0.9361 | *= quantized, exactly* | 10.000 | 15.0 | 2/2 |
| Phase-6 ONNX | 0.9639 | 0.9167 | *= quantized, exactly* | 38.000 | 60.0 | 2/2 |
| Phase-6 sklearn | 0.8944 | 0.8806 | *= quantized, exactly* | 0.000 † | 0.0005 | 2/2 |
| Phase-7 faces | 0.9500 | 0.9000 | *= quantized, exactly* | 74.000 | 120.0 | 2/2 |

*(Derived from [`docs/BENCHMARKS.md` Table D](./BENCHMARKS.md#table-d-accuracy-and-error); declared bounds [`crates/penumbra-ckks/src/bounds.rs`](https://github.com/Harikeshav-R/Penumbra-FHE/blob/4e1a320/crates/penumbra-ckks/src/bounds.rs) @ `4e1a320`).*

> **Sample size.** CKKS max |err| and the CKKS labels column come from N = 2 test samples per model (`--samples 2`); Float and Quantized are the fixture generators' test-batch accuracies. Two samples cannot support an accuracy claim; these figures are superseded by the Phase 15–16 protocol (`docs/PAPER.md` D3, D7).

> † **Historical, not citable.** phase6_sklearn CKKS max |err| 0.000155 — introduced in 8537d14; phase10-final-sweep.json records 0.000 for both samples.

### Overhead

Summary of key and ciphertext dimensions across the suite (see [`docs/BENCHMARKS.md` Table C](./BENCHMARKS.md#table-c-sizes-and-scheme-cost-proxies) for the full per-model table). Sizes use the binary units `penumbra-bench` prints (1 KB = 1,024 B, 1 MB = 2^20 B); source [`docs/results/phase10-final-sweep.json`](./results/phase10-final-sweep.json) @ `9b38c1b`:

- **Ciphertext sizes:**
  - TFHE encodes each integer element as a radix ciphertext (`num_blocks` shortint ciphertexts). Input ciphertexts scale linearly with input tensor length and radix blocks (from 3.96 MB on 36-element inputs to 44.22 MB on 256-element inputs; `phase2_logreg` is 6.03 MB with 6 blocks). Output ciphertexts range from 96.5 KB (scalar) to 1.57 MB.
  - CKKS encodes entire tensors into single SIMD ciphertexts ($N = 16384$, $k = 360$). Every input and output ciphertext is exactly 4.75 MB regardless of tensor dimension ($\le 256$ elements).
- **Key material sizes:**
  - TFHE client key: 23.4 KB; server key (bootstrapping and key-switching keys): 114.84 MB.
  - CKKS client key: 128.1 KB; server key (Galois rotation keys + relinearization keys): 1,782.50 MB (1.74 GiB).
- **Scheme cost proxies:**
  - TFHE cost is dominated by radix arithmetic and PBS: scalar additions, scalar multiplications (up to 1,500/sample), and bootstraps (137 PBS on `phase2_logreg`, up to 128,571 PBS on `phase7_faces`).
  - CKKS cost is dominated by rotations (16–53 rotations/sample) and polynomial evaluation rescales (up to 6 polynomial evaluations across 7 depth levels).

### Discussion

The study answers the three motivating sub-questions (§1) with concrete data on committed workloads:

1. **The three sub-questions answered:**
   - **Latency:** On multi-layer convolutional models (`phase4_cnn`, `phase5_digits`, `phase5_qat`, `phase6_onnx`, `phase7_faces`), CKKS is **41.2x to 162.6x faster** than TFHE (evaluating in 0.66 s to 2.30 s per sample, whereas TFHE takes 27.1 s to 370.6 s per sample). On `phase6_sklearn`, TFHE evaluates in 40.56 s vs. CKKS's 0.508 s (79.8x ratio). On single-layer binary logistic regression (`phase2_logreg`), Phase 10 bit-width minimization (radix reduced to 6 blocks, 137 PBS) flips the ordering: TFHE evaluates in **0.521 s** vs. CKKS's **0.647 s** (a 0.8x ratio, **provisional**: Criterion measures the opposite ordering, TFHE 488.82 ms vs CKKS 350.44 ms (`phase10-criterion-logreg.json @ 9b38c1b`); see [Logreg timing reconciliation](./BENCHMARKS.md#logreg-timing-reconciliation)).
   - **Accuracy:** TFHE achieves bit-exact identity with the quantized integer reference (`max |err| = 0.0` everywhere, labels match 2/2 on all models). CKKS adds approximation error: on linear/argmax models (`phase2_logreg`, `phase6_sklearn`), error is zero †; on multi-layer polynomial activations, error is 4.0 on `phase4_cnn`, 10.0 on QAT digits, 38.0 on PTQ digits/ONNX, and 74.0 on `phase7_faces`. Across all 7 models, CKKS labels matched ground truth 14/14 times (100.0% sample label agreement).
   - **Overhead:** The latency advantage of CKKS on multi-layer networks is paid for in server key storage: CKKS requires **1,782.50 MB** (1.74 GiB) of server keys (Galois automorphism and relin keys) compared to TFHE's **114.84 MB** (a 15.5x key storage overhead), and client keys are 128.1 KB vs. 23.4 KB. Ciphertext size exhibits a crossover: for small inputs, TFHE is comparable (3.96 MB vs 4.75 MB), but for larger inputs (256-element faces), TFHE's non-batched representation balloons to 44.22 MB while CKKS remains fixed at 4.75 MB per SIMD ciphertext.

   *Source: [`docs/results/phase10-final-sweep.json`](./results/phase10-final-sweep.json) @ `9b38c1b`.*

   > † **Historical, not citable.** error $\le 1.6 \times 10^{-4}$ on phase6_sklearn — introduced in 8537d14; phase10-final-sweep.json records 0.000 for both samples.

2. **Where the cost actually goes (Table B and Cost Proxies):**
   In [`docs/BENCHMARKS.md` Table B](./BENCHMARKS.md#table-b-per-op-type-eval-breakdown-mean-seconds-per-sample), TFHE's latency is dominated by two components:
   - **Multi-block radix linear algebra:** on `phase5_digits` (9-block radix), `Conv2d` took 149.2 s and `Linear` 51.5 s under TFHE. Neither layer contains a table lookup, yet both issue programmable bootstraps: the measured counters record 48,888 PBS in `Conv2d` and 19,137 in `Linear`, from radix carry propagation in the scalar-multiply and add chains (`crates/penumbra-tfhe/tests/measured_pbs.rs` falsifies 'Linear is PBS-free').
   - **`Requant`:** 21.6 s and 7,128 measured PBS, of which the cost proxy counts 108 logical table lookups (`bootstraps: 108`).
   Under CKKS (Option B), BSGS diagonal transforms replace element-wise work with 45–53 rotations per sample on the CNN models; `Conv2d` evaluates in 0.29 s (`phase4_cnn`) to 1.41 s (`phase7_faces`) and `Requant` in 0.12–0.72 s (1–6 polynomial evaluations, 7 depth levels).

   *Source: [`docs/results/phase10-final-sweep.json`](./results/phase10-final-sweep.json) @ `9b38c1b`.*
3. **Threats to validity live for each headline number:**
   - **For the latency ratios (0.8x–162.6x, Table A):**
     - **Threat 1 (SIMD packing):** Option B SIMD packing is what makes the speedup possible; packing one scalar per ciphertext would have degraded CKKS latency by $256\times$ (unmeasured reasoning from `lt_slots = 256`).
     - **Threats 4 & 5 (Implementation & library maturity):** TFHE in Penumbra evaluates each op's independent outputs across CPU cores via `rayon` under the tuned `classic` profile; the CKKS arm was not similarly parallelized because its ops operate on a single packed ciphertext behind a shared scratch arena mutex. Part of the remaining speedup reflects `poulpy`'s optimized NEON assembly HAL (`FFT64Neon`).
     - **Threat 6 (Single machine / HAL):** Measured on an Apple Silicon M3 Pro core; AVX2/AVX-512 numbers on x86-64 will differ.
     - **Threat 8 (Op build charged per sample):** Plaintext weight encoding and BSGS diagonal prep are included in total eval time; in production, precomputing them saves at most 0.013 s per sample under CKKS (Table A, `of which op-build`).
     - **Threat 9 (Toolchain codegen):** Both arms were compiled on nightly Rust; any nightly vs stable codegen disparity affects the TFHE arm.
   - **For the accuracy numbers:**
     - **Threat 2 (TFHE-shaped graph):** The models are quantized with 2-bit activation bottlenecks specifically designed for TFHE lookup tables; feeding this low-bit quantized graph to CKKS handicaps CKKS precision.
     - **Threat 3 (Polynomial degree truncation):** The earlier bound violation on `phase7_faces` sample 1 was an implementation bug in the `Requant` target function, now fixed; `max_poly_degree = 15` remains a genuine threat to validity because the residual error after the fix is still polynomial-approximation error, at post-fix magnitudes (up to 74.0 on `phase7_faces`, 38.0 on digits).
     - **Threat 7 (Small sample batch):** With $N=2$, sample accuracy reflects individual logit noise rather than population accuracy.

4. **Verdict on `PROJECT.md` §2:**
   `PROJECT.md` §2 asserted:
   > *"For small classifiers with ReLU/argmax (MNIST, faces, tabular), **TFHE is the correct choice** — exact, arbitrary activations as lookup tables, no batching needed."*

   **Verdict: The hypothesis does not survive on multi-layer network latency; it survives on exactness and operational simplicity; the shallow-model latency claim is provisional (item 1).**
   On multi-layer convolutional models, CKKS under SIMD tensor packing (Option B) is **one to two orders of magnitude faster** (41x–163x) than multi-block radix TFHE. With BSGS diagonal packing, SIMD batching within a single ciphertext vastly outperforms multi-block carry-propagation arithmetic.

   However, the hypothesis **survives on shallow models and exactness**:
   1. **Provisional** (`ROADMAP.md` Phase 13; stays provisional until Phase 16 produces Criterion-canonical numbers, `docs/PAPER.md` D6). On shallow or linear models where radix bit-width can be minimized to 6 blocks (`phase2_logreg`), TFHE achieves **sub-second inference (0.52 s)** that matches or outperforms CKKS (0.65 s) without polynomial approximation or SIMD rotation overhead. Criterion reverses this ordering (TFHE 488.82 ms vs CKKS 350.44 ms); see [Logreg timing reconciliation](./BENCHMARKS.md#logreg-timing-reconciliation).
   2. TFHE requires zero polynomial approximation, has zero drift across layers, and guarantees bit-exact agreement with the integer specification. CKKS introduces bounded approximation error on multi-layer models (max error 4.0 to 74.0 in integer units across CNNs), requiring declared bounds and headroom analysis.
   3. TFHE server keys are **114.84 MB**, feasible for deployment on constrained nodes; CKKS requires **1,782.50 MB** (1.74 GiB) of Galois rotation keys for the BSGS baby-step/giant-step steps.


5. **Tree Ensembles (`Compare` and sum-of-comparisons lowering):**
   TFHE evaluates decision tree ensembles bit-for-bit exactly via `scalar_ge_parallelized` comparison PBSs (`phase8_trees`: 67 PBSs total across 5 trees†, ~10.4 s latency†, no Requant/activation lookups). Under CKKS, `Compare` is implemented at the op level via a continuous smoothed step polynomial approximation over a plaintext linear map (`ckks_golden_ops.rs`). However, for a 4-node tree graph with chained sharp step functions (`split_cmp` and `leaf_sel`), the multiplicative level budget required (`(1 + 4) * 30 * 2 + 30 * 2 = 360` bits) exceeds the depth budget capacity of 330 bits at `k = 360`, `log_delta = 30`, `max_poly_degree = 15`. This demonstrates an asymmetric scheme tradeoff: TFHE evaluates sharp discrete comparisons and branching logic natively with zero error and no Requant/activation lookups, whereas CKKS cannot chain two sharp step functions without parameter widening.

   > † **Historical, not citable.** 67 PBSs total across 5 trees and ~10.4 s latency — introduced in f12d63d; no committed results file.
## Scope

This study compares two backends on **Penumbra's existing supported operations and committed
models**. It is not a general survey of FHE schemes, not a claim about CKKS or TFHE outside
this workload class, and not an attempt at feature completeness in either backend beyond what
the comparison requires (`PROJECT.md` §18).
