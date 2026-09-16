# TFHE vs CKKS: A Controlled Comparison

The study `penumbra-ckks` exists to enable. [`docs/BENCHMARKS.md`](./BENCHMARKS.md) owns the
measured numbers; this document owns the **argument** — what is being tested, why the setup
is valid, and what the results do and do not license anyone to conclude.

> **Status: pre-measurement.** Method and threats are fixed in advance, deliberately. Results
> are empty until Phase 12.4 (`ROADMAP.md`).

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

1. Both backends are registered with `penumbra-bench` and driven through the same entry point.
2. For each committed model (Phase-2 logreg through Phase-7 faces), run *N* encrypted
   inferences per backend in `--release`.
3. Record per-model: wall-clock latency per sample, per-op-type time breakdown, accuracy
   against the shared quantized-cleartext reference, accuracy against the float model,
   ciphertext and key sizes, and each scheme's own cost proxy (bootstrap count for TFHE;
   multiplicative depth, rotation count, and rescale count for CKKS).
4. Report the TFHE bit-exactness gate as pass/fail and the CKKS error as a measured
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

1. **SIMD packing.** CKKS's advantage is slot batching. If `penumbra-ckks` ships one value per
   ciphertext, the comparison measures a deliberately hobbled CKKS and its latency numbers
   are close to meaningless. The packing decision is an open fork in
   [`docs/BACKENDS.md`](./BACKENDS.md#open-design-forks); whichever way it lands must be
   disclosed prominently here. **This is the most serious threat on the list.**
2. **The graph is quantized for TFHE.** Penumbra caps activations at a single 2-bit block
   because a programmable bootstrap is only feasible over a narrow value
   (`docs/QUANTIZATION.md`). CKKS has no such constraint and would ordinarily run at much
   higher precision. Feeding it the TFHE-shaped graph is what makes the comparison
   apples-to-apples, and it simultaneously handicaps CKKS on accuracy. Both halves of that
   sentence must appear in any write-up.
3. **Polynomial degree is a free parameter.** CKKS accuracy and latency trade against each
   other continuously via approximation degree. A single degree is one point on a curve;
   reporting the curve, or at minimum the chosen degree and its justification, is required.
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

## Results

*Empty pending Phase 12.4. Populate from [`docs/BENCHMARKS.md`](./BENCHMARKS.md); do not
duplicate numbers here — cite them.*

### Latency

| Model | TFHE / sample | CKKS / sample | Ratio |
|---|---|---|---|
| Phase-2 logreg | TBD | TBD | TBD |
| Phase-4 CNN | TBD | TBD | TBD |
| Phase-5 digits (PTQ) | TBD | TBD | TBD |
| Phase-5 digits (QAT) | TBD | TBD | TBD |
| Phase-7 faces | TBD | TBD | TBD |

### Accuracy

| Model | Float | Quantized (shared reference) | TFHE | CKKS | CKKS error vs. reference |
|---|---|---|---|---|---|
| Phase-2 logreg | 1.00 | 1.00 | *= quantized, exactly* | TBD | TBD |
| Phase-4 CNN | 0.98 | 0.96 | *= quantized, exactly* | TBD | TBD |
| Phase-5 digits (PTQ) | ~0.96 | ~0.94 | *= quantized, exactly* | TBD | TBD |
| Phase-5 digits (QAT) | ~0.94 | ~0.94 | *= quantized, exactly* | TBD | TBD |
| Phase-7 faces | 0.95 | 0.90 | *= quantized, exactly* | TBD | TBD |

### Overhead

| Model | Backend | Ciphertext / input | Key material | Cost proxy |
|---|---|---|---|---|
| — | — | TBD | TBD | TBD |

### Discussion

*To be written against the measured results. It must address: which sub-question each number
answers, which threats above are live for that number, and whether `PROJECT.md` §2's claim
survives.*

## Scope

This study compares two backends on **Penumbra's existing supported operations and committed
models**. It is not a general survey of FHE schemes, not a claim about CKKS or TFHE outside
this workload class, and not an attempt at feature completeness in either backend beyond what
the comparison requires (`PROJECT.md` §18).
