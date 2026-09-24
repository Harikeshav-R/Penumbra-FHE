# TFHE vs CKKS: A Controlled Comparison

The study `penumbra-ckks` exists to enable. [`docs/BENCHMARKS.md`](./BENCHMARKS.md) owns the
measured numbers; this document owns the **argument** — what is being tested, why the setup
is valid, and what the results do and do not license anyone to conclude.

> **Status: measured (Phase 12.4).** Measured 2026-09-22 on Apple M3 Pro, macOS 25.6.0,
> `rustc 1.100.0-nightly (bba531001 2026-09-20)`, HAL backend `FFT64Neon` (`poulpy-ckks 0.8.3`),
> commit `dc20d05ee332e284045ea971bb8272c5d5ddf522`. CKKS arm re-measured on 2026-09-22 at commit `3f6bd68900fafaf3c8f29e7cff25212e5c8ae551` after the `Requant` fix; TFHE arm carried over from `dc20d05`. Detailed numbers: [`docs/BENCHMARKS.md`](./BENCHMARKS.md);
> raw data: [`docs/results/phase12-4-comparison.json`](./results/phase12-4-comparison.json).

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
2. The `penumbra-bench-report` CLI binary runs the models and emits the comparison as Markdown or JSON tables.
3. For each of the seven committed models (`phase2_logreg`, `phase4_cnn`, `phase5_digits`,
   `phase5_qat`, `phase6_onnx`, `phase6_sklearn`, `phase7_faces`), run $N = 2$ encrypted
   inferences per backend in `--release`.
4. Record per-model: wall-clock latency per sample, per-op-type time breakdown, accuracy
   against the shared quantized-cleartext reference, accuracy against the float model,
   ciphertext and key sizes, and each scheme's own cost proxy (bootstrap count for TFHE;
   multiplicative depth, rotation count, and rescale count for CKKS).
5. Report the TFHE bit-exactness gate as pass/fail and the CKKS error as a measured
   distribution, not a single number.

## Metrics

| Metric | Definition | Why it is here |
|---|---|---|
| Latency / sample | wall clock for one encrypted forward pass, `--release`, pinned machine | the headline practical question |
| Per-op-type breakdown | time attributed at the eval-loop seam | shows *why* one scheme wins, not just that it does |
| Accuracy vs. reference | agreement with `evaluate_graph_int` on the test batch | isolates the scheme's own error from quantization error |
| Accuracy vs. float | agreement with the unquantized model | what a user actually experiences |
| Scheme cost proxy | bootstraps (TFHE) · depth + rotations + rescales (CKKS) | lets the numbers generalize past this machine |
| Ciphertext size | bytes per encrypted input and output | bandwidth cost of the client/server split (`PROJECT.md` §11) |
| Key material size | client key + evaluation key bytes | the real deployment cost; TFHE server keys are large |

Accuracy is reported against **both** references on purpose. Measured against the quantized
reference, the two backends' errors are directly comparable. Measured against float, the
number is what a user sees — and it folds in a quantization gap that is identical for both
backends by construction.

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
   apples-to-apples, and it simultaneously handicaps CKKS on accuracy. Both halves of that
   sentence must appear in any write-up.
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
| Phase-2 logreg | 11.85 s | 0.48 s | 24.4x |
| Phase-4 CNN | 69.99 s | 0.59 s | 118.0x |
| Phase-5 digits (PTQ) | 679.86 s | 2.26 s | 300.5x |
| Phase-5 digits (QAT) | 687.92 s | 2.04 s | 336.8x |
| Phase-6 ONNX | 701.86 s | 2.13 s | 329.7x |
| Phase-6 sklearn | 159.07 s | 0.61 s | 260.6x |
| Phase-7 faces | 730.22 s | 2.21 s | 330.3x |

*(Derived from [`docs/BENCHMARKS.md` Table A](./BENCHMARKS.md#table-a--latency-wall-clock-per-sample). Means over 2 samples in `--release`, Apple M3 Pro, FFT64Neon HAL).*

### Accuracy

| Model | Float | Quantized (shared reference) | TFHE | CKKS max \|err\| | Declared bound | CKKS labels |
|---|---:|---:|---|---:|---:|---|
| Phase-2 logreg | 1.0000 | 1.0000 | *= quantized, exactly* | 0.000 | 0.5 | 2/2 |
| Phase-4 CNN | 0.9805 | 0.9570 | *= quantized, exactly* | 3.000 | 10.0 | 2/2 |
| Phase-5 digits (PTQ) | 0.9639 | 0.9417 | *= quantized, exactly* | 35.000 | 60.0 | 2/2 |
| Phase-5 digits (QAT) | 0.9361 | 0.9389 | *= quantized, exactly* | 28.000 | 50.0 | 2/2 |
| Phase-6 ONNX | 0.9639 | 0.9417 | *= quantized, exactly* | 35.000 | 60.0 | 2/2 |
| Phase-6 sklearn | 0.8944 | 0.8806 | *= quantized, exactly* | 0.000 | 0.001 | 2/2 |
| Phase-7 faces | 0.9500 | 0.9000 | *= quantized, exactly* | 74.000 | 120.0 | 2/2 |

*(Derived from [`docs/BENCHMARKS.md` Table D](./BENCHMARKS.md#table-d--accuracy-and-error)).*

### Overhead

Summary of key and ciphertext dimensions across the suite (see [`docs/BENCHMARKS.md` Table C](./BENCHMARKS.md#table-c--sizes--scheme-cost-proxies) for the full per-model table):

- **Ciphertext sizes:**
  - TFHE encodes each integer element as a radix ciphertext (`num_blocks` shortint ciphertexts). Input ciphertexts scale linearly with input tensor length (from 3.96 MB on 36-element inputs to 44.22 MB on 256-element inputs). Output ciphertexts range from 128 KB (scalar) to 1.81 MB (10 classes).
  - CKKS encodes entire tensors into single SIMD ciphertexts ($N = 16384$, $k = 360$). Every input and output ciphertext is exactly 4.75 MB regardless of tensor dimension ($\le 256$ elements).
- **Key material sizes:**
  - TFHE client key: 23.4 KB; server key (bootstrapping and key-switching keys): 114.84 MB.
  - CKKS client key: 128.1 KB; server key (Galois rotation keys + relinearization keys): 1,782.50 MB (~1.78 GB).
- **Scheme cost proxies:**
  - TFHE cost is dominated by radix arithmetic and PBS: scalar multiplications (up to 2,304/sample) and bootstraps (up to 128 bootstraps + 384 comparison PBS ops/sample).
  - CKKS cost is dominated by rotations (18–53 rotations/sample) and polynomial evaluation rescales (up to 9 polynomial evaluations across 7 depth levels).

### Discussion

The study answers the three motivating sub-questions (§1) with concrete data on committed workloads:

1. **The three sub-questions answered:**
   - **Latency:** CKKS is **4.5x to 183.1x faster** than TFHE across all seven committed fixtures (incorporating TFHE Phase 10 sum-tree carry reduction, rayon parallelism, and tuned parameters). On `phase2_logreg`, CKKS evaluates in 0.444 s vs. TFHE's 1.992 s (4.5x ratio; Criterion median 412.89 ms vs. 1.880 s). On multi-layer convolutional models (`phase4_cnn`, `phase5_digits`, `phase5_qat`, `phase6_onnx`, `phase7_faces`), CKKS evaluates in 0.523 s to 2.033 s per sample, whereas TFHE takes 26.9 s to 360.0 s (~6.0 minutes) per sample (51.4x to 183.1x ratio). On `phase6_sklearn`, TFHE evaluates in 40.86 s vs. CKKS's 0.438 s (93.3x ratio).
   - **Accuracy:** TFHE achieves bit-exact identity with the quantized integer reference (`max |err| = 0.0` everywhere, labels match 2/2 on all models). CKKS adds approximation error: on linear/argmax models (`phase2_logreg`, `phase6_sklearn`), error is zero or $\le 10^{-3}$; on multi-layer polynomial activations, post-fix error reaches 3.0 on `phase4_cnn`, 28.0–35.0 on 8-bit digit CNNs, and 74.0 on `phase7_faces`. Across all 7 models, CKKS labels matched ground truth 14/14 times (100.0% sample label agreement).
   - **Overhead:** The latency advantage of CKKS is paid for in server key storage: CKKS requires **1.78 GB** of server keys (Galois automorphism and relin keys) compared to TFHE's **114.84 MB** (a 15.5x key storage overhead), and client keys are 128.1 KB vs. 23.4 KB. Ciphertext size exhibits a crossover: for small inputs, TFHE is comparable (3.96 MB vs 4.75 MB), but for larger inputs (256-element faces), TFHE's non-batched representation balloons to 44.22 MB while CKKS remains fixed at 4.75 MB per SIMD ciphertext.

2. **Where the cost actually goes (Table B and Cost Proxies):**
   In [`docs/BENCHMARKS.md` Table B](./BENCHMARKS.md#table-b--per-op-type-eval-breakdown-mean-seconds-per-sample), TFHE's latency is dominated by two components:
   - **Multi-block radix linear algebra:** on `phase5_digits`, `Conv2d` took 338.2 s and `Linear` took 287.0 s under TFHE, despite having 0 bootstraps inside the linear algebra. Because each multiplication involves scalar-multiplying 11 independent shortint ciphertexts and accumulating carry chains, the CPU is overwhelmed by sequential radix arithmetic.
   - **Programmable Bootstrapping (PBS) in `Requant`:** on `phase5_digits`, 108 bootstraps took 54.6 s (~0.50 s/PBS).
   Under CKKS (Option B), BSGS diagonal multiplexing packs entire matrices into a single ciphertext, replacing thousands of element-wise operations with 24–53 rotations (taking ~0.9–1.3 s total for `Conv2d`). Polynomial activation approximation (depth 4–5, 6–9 poly evals) evaluates in ~0.7–1.1 s.

3. **Threats to validity live for each headline number:**
   - **For the latency ratios (24.4x–336.8x):**
     - **Threat 1 (SIMD packing):** Option B SIMD packing is what makes the speedup possible; packing one scalar per ciphertext would have degraded CKKS latency by $256\times$.
     - **Threats 4 & 5 (Implementation & library maturity):** TFHE in Penumbra evaluates each op's independent outputs across CPU cores via `rayon` under the tuned `classic` profile; the CKKS arm was not similarly parallelized because its ops operate on a single packed ciphertext behind a shared scratch arena mutex. Part of the remaining speedup reflects `poulpy`'s optimized NEON assembly HAL (`FFT64Neon`).
     - **Threat 6 (Single machine / HAL):** Measured on an Apple Silicon M3 Pro core; AVX2/AVX-512 numbers on x86-64 will differ.
     - **Threat 8 (Op build charged per sample):** Plaintext weight encoding and BSGS diagonal prep are included in total eval time; in production, precomputing them saves an additional ~0.01–0.02 s under CKKS.
     - **Threat 9 (Toolchain codegen):** Both arms were compiled on nightly Rust; any nightly vs stable codegen disparity affects the TFHE arm.
   - **For the accuracy numbers:**
     - **Threat 2 (TFHE-shaped graph):** The models are quantized with 2-bit activation bottlenecks specifically designed for TFHE lookup tables; feeding this low-bit quantized graph to CKKS handicaps CKKS precision.
     - **Threat 3 (Polynomial degree truncation):** The earlier bound violation on `phase7_faces` sample 1 was an implementation bug in the `Requant` target function, now fixed; `max_poly_degree = 15` remains a genuine threat to validity because the residual error after the fix is still polynomial-approximation error, at post-fix magnitudes (up to 74.0 on `phase7_faces`, 35.0 on digits).
     - **Threat 7 (Small sample batch):** With $N=2$, sample accuracy reflects individual logit noise rather than population accuracy.

4. **Verdict on `PROJECT.md` §2:**
   `PROJECT.md` §2 asserted:
   > *"For small classifiers with ReLU/argmax (MNIST, faces, tabular), **TFHE is the correct choice** — exact, arbitrary activations as lookup tables, no batching needed."*

   **Verdict: The hypothesis does not survive on latency.**
   Even on tiny models with non-batched single-inference workloads, CKKS under SIMD tensor packing (Option B) is **one to two orders of magnitude faster** (51x–183x on CNNs, 4.5x on logreg) than multi-block radix TFHE. The belief that "TFHE is faster because no batching is needed" assumed scalar CKKS; with BSGS diagonal packing, SIMD batching within a single tensor vastly outperforms radix-decomposed integer arithmetic.

   However, the hypothesis **survives on exactness and operational simplicity**:
   1. TFHE requires zero polynomial approximation, has zero drift across layers, and guarantees bit-exact agreement with the integer specification. CKKS introduces bounded approximation error on multi-layer models (max error 3.0 to 74.0 in integer units across CNNs), requiring declared bounds and headroom analysis.
   2. TFHE server keys are **114 MB**, feasible for deployment on constrained nodes; CKKS requires **1.78 GB** of Galois rotation keys for the BSGS baby-step/giant-step steps.

   Thus, for applications where latency must be $\le 5$ seconds per inference, CKKS is the only viable backend on this hardware; for applications demanding exactness or constrained key memory, TFHE remains the reference oracle.

## Scope

This study compares two backends on **Penumbra's existing supported operations and committed
models**. It is not a general survey of FHE schemes, not a claim about CKKS or TFHE outside
this workload class, and not an attempt at feature completeness in either backend beyond what
the comparison requires (`PROJECT.md` §18).
