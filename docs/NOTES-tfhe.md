# TFHE Notes

Working notes on the `tfhe-rs` primitives Penumbra builds on, the parameter profile, and
empirical cost. Closes the Phase-1 spike deliverable (`ROADMAP` Phase 1) and informs the
bit-width-budget design (`PROJECT.md` §9).

## Parameter profiles

Penumbra exposes five named 128-bit secure parameter profiles at `MESSAGE_BITS = 2` (`TfheProfile`),
holding security constant at $\ge 128$ bits ($p_{\text{fail}} \le 2^{-128}$) across all variants
(`AGENTS.md` §7: never trade security for speed):

1. **`classic` (default):** `PARAM_MESSAGE_2_CARRY_2_KS_PBS`. Classic PBS with TUniform noise.
   $p_{\text{fail}} = 2^{-129.581}$, algorithmic cost ~ 113. Smallest server key (114.84 MB),
   lowest serial latency on multi-core machines when combined with outer rayon parallelism.
2. **`gaussian`:** `PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128`. Classic PBS with discrete-Gaussian noise.
   $p_{\text{fail}} \le 2^{-128}$, algorithmic cost ~ 113, server key 121.89 MB.
3. **`multibit2`:** `V1_8_PARAM_MULTI_BIT_GROUP_2_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128`. Multi-bit PBS,
   grouping factor 2. $p_{\text{fail}} = 2^{-140.341}$, algorithmic cost ~ 188, server key 373.15 MB.
4. **`multibit3`:** `V1_8_PARAM_MULTI_BIT_GROUP_3_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128`. Multi-bit PBS,
   grouping factor 3. $p_{\text{fail}} = 2^{-128.235}$, algorithmic cost ~ 143, server key 392.31 MB.
5. **`multibit4`:** `V1_8_PARAM_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128`. Multi-bit PBS,
   grouping factor 4. $p_{\text{fail}} = 2^{-134.345}$, algorithmic cost ~ 100, server key 302.07 MB.

### Lattice security estimation (Phase 15)

Concrete classical lattice security for the default `classic` profile (`PARAM_MESSAGE_2_CARRY_2_KS_PBS`)
was estimated using the pinned `malb/lattice-estimator` (commit `53da598`) under SageMath 10.6,
evaluating the MATZOV reduction cost model (`RC.MATZOV`) and GSA shape model (`GSA`) under unbounded samples ($m=\infty$):

- **LWE input:** $n = 918, q = 2^{64}, \text{Binary secret}, \text{TUniform}(45)$.
  - Primal uSVP: $\approx 2^{141.3}$ operations ($\beta=393, d=1660$)
  - Primal BDD: $\approx 2^{138.9}$ operations ($\beta=383, \eta=413, d=1688$)
  - Dual lattice: $\approx 2^{145.2}$ operations ($\beta=403, d=1740$)
  - Dual hybrid: $\approx 2^{134.9}$ operations ($\beta=366$) — **bottleneck attack: 134.9 bits**
  - Coded BKW: $\approx 2^{209.3}$ operations
- **GLWE input:** $n = 2048, q = 2^{64}, \text{Binary secret}, \text{TUniform}(17)$ (modeled as unstructured LWE at $n = \text{rank} \times N = 2048$).
  - Primal uSVP: $\approx 2^{137.5}$ operations ($\beta=374, d=3901$)
  - Primal BDD: $\approx 2^{136.5}$ operations ($\beta=369, \eta=404, d=3952$)
  - Dual lattice: $\approx 2^{139.6}$ operations ($\beta=378, d=4010$)
  - Dual hybrid: $\approx 2^{134.8}$ operations ($\beta=360$) — **bottleneck attack: 134.8 bits**
- **Overall TFHE security:** **134.8 bits** classical security (exceeds the 128-bit floor).

Artifact: [`docs/results/phase15-security-estimates.json`](./results/phase15-security-estimates.json),
reproducible via `python examples/security/run.py`.
Server-key sizes: `phase10-param-sweep-<profile>.json` (committed in `9b38c1b`), binary MB.

All multi-bit profiles enable `with_deterministic_execution()`: without deterministic execution,
multi-bit blind rotation reduces across threads in non-deterministic order, causing ciphertext bytes
to differ run-to-run. Decrypted values are exact either way, but reproducible bytes keep published
artifacts stable.

- **Message space:** 2 bits per block (`message_modulus = 4`), with a 2-bit carry buffer.
  Every profile shares `keys::MESSAGE_BITS = 2`, so a radix integer of `num_blocks` blocks holds
  `num_blocks × 2` bits of value (`keys::radix_capacity_bits`) identically across profiles.
  `num_blocks` represents the model-level radix ceiling (central budget); each tensor is sized
  to its derived bit width (Phase 14).
- **Representation:** values are **signed** radix integers (`SignedRadixCiphertext`).
  Weights and logits are naturally signed; signed avoids a zero-point-offset dance and
  generalizes to conv accumulators.
## The two primitives everything composes from

1. **Plaintext-weight arithmetic (cheap, no PBS)** — `scalar_mul_parallelized`,
   `scalar_add_parallelized`, `add_parallelized`. The `Linear`/`Conv` core: encrypted data
   combined with plaintext weights. `i64` scalars work directly (they implement
   `ScalarMultiplier` + `DecomposableInto`).
2. **Programmable bootstrapping / LUT (expensive)** — at the `shortint` block level,
   `generate_lookup_table(f)` + `apply_lookup_table`. The `Activation`/`Requant` core, and
   internally what `scalar_ge_parallelized` (the `Argmax` comparison) uses.

## Empirical cost (the bit-width budget lever)

> **Historical, not citable.** Phase-1 golden-test timings; introduced in 459ff56 (2026-06-22); no committed results file.

Measured on the Phase-2 golden test (`cargo test --release`, `num_blocks = 8` ⇒ 16-bit
signed radix, 64-feature single-logit `Linear → Argmax`):

| Quantity | Value |
|---|---|
| `Activation` LUT over the 4-value message space (4 PBS) | < 1 s total |
| Full `Linear → Argmax` inference, **per sample** | ~30 s |
| keygen + a single signed round-trip | < 1 s |

The per-sample cost is dominated by radix MAC carry-propagation bootstraps in addition and
multiplication. Phase 10 landed rayon parallelism across independent outputs (cutting `phase2_logreg` eval latency 2.13x, 4.238 s → 1.992 s, and `phase4_cnn` 1.45x, 38.912 s → 26.899 s; `phase10-tfhe-sweep.json @ 78f5db7` vs `phase10-parallel-tuned-sweep.json @ 9b38c1b`) and verified that the classic profile provides the
lowest latency under parallel execution. The radix width lever is Phase 14: shrinking radix width
per tensor to the minimum needed by derived bit widths (see [Per-tensor radix width (Phase 14)](#per-tensor-radix-width-phase-14)).

> ⚠️ Always benchmark in `--release`. Debug `tfhe-rs` is orders of magnitude slower and the
> numbers are meaningless (`docs/DEVELOPMENT.md`).

## Per-tensor radix width (Phase 14)

### Linear-path spike (Phase 14.0)

Before implementing full per-tensor widths, Phase 14.0 ran a micro-benchmark spike comparing four MAC variants on isolated neurons from `examples/mnist/phase5_digits_fixture.json` (`conv0` 9-term kernel with 8 non-zero weights, `in_bits = 3`, `acc_blocks = 7`; and `linear2` 108-term row 0 with 38 distinct non-zero weight groups, `in_bits = 2`, `acc_blocks = 9`; model ceiling `num_blocks = 9`):

1. **(a) Baseline:** today's `evaluate_weighted_mac` at model ceiling `num_blocks` (9 blocks).
2. **(b) Widen-once:** inputs resized to `acc_blocks` once before weighted MAC.
3. **(b′) Progressive widening:** groups summed at their minimum width `value_blocks(in_bits + ceil_log2(k), acc_blocks)`, multiplied, and resized to `acc_blocks`.
4. **(c) Deferred carries:** preshifted radix copies and block shifts combined via `unchecked_sum_ciphertexts_vec_parallelized` with deferred carry propagation.

Measured with 7 timed runs per variant after 1 warm-up (11 threads, Apple M3 Pro, commit `74b4b63`, recorded in `docs/results/phase14-spike-mac.json`):

| Neuron | Variant | Median (s) | Min (s) | Max (s) | PBS | Widen (s) | Widen PBS | Exact |
|---|---|---|---|---|---|---|---|---|
| `conv0` | a (baseline) | 1.367 | 1.358 | 1.374 | 402 | 0.000 | 0 | true |
| `conv0` | b (widen-once) | 1.065 | 1.051 | 1.073 | 286 | 0.176 | 64 | true |
| `conv0` | b′ (progressive) | 0.926 | 0.878 | 1.120 | 152 | 0.179 | 64 | true |
| `conv0` | c (deferred) | 0.730 | 0.709 | 0.737 | 168 | 0.189 | 64 | true |
| `linear2` | a (baseline) | 10.466 | 10.358 | 10.533 | 3174 | 0.000 | 0 | true |
| `linear2` | b (widen-once) | 10.527 | 10.132 | 10.658 | 3174 | 0.296 | 108 | true |
| `linear2` | b′ (progressive) | 6.126 | 5.985 | 6.822 | 1017 | 0.292 | 108 | true |
| `linear2` | c (deferred) | 4.050 | 4.023 | 4.089 | 970 | 0.301 | 108 | true |

**Decision rules:**
- **R1:** $S(v) = \text{median}_{\text{conv}}(v) + \text{median}_{\text{linear}}(v)$. $S(b) = 11.592\text{ s}$, $S(b′) = 7.052\text{ s}$. $S(b) / S(b′) = 1.6437 \ge 1.05 \implies$ **b′ wins**.
- **R2:** $S(\text{winner}) / S(c) = 7.052 / 4.781 = 1.4752 \ge 1.20 \implies$ **deferred carries included**.
- **Final decision:** `mac=b' deferred=yes`.

### Width rules & representation

Layer-2 bit-width analysis bounds each tensor value's magnitude: $|v| < 2^{\text{bits}}$. With $2$ bits per `shortint` radix block (`MESSAGE_BITS = 2`) and model ceiling $N = \text{num\_blocks}$:

- **`value_blocks(bits, cap)`** $= \lceil(\text{bits} + 1) / 2\rceil$, clamped to $[1, \text{cap}]$. Adds 1 sign bit. Used for inputs and all intermediate tensors except linear accumulators.
- **`signed_blocks(bits, cap)`** $= \lceil\text{bits} / 2\rceil$, clamped to $[1, \text{cap}]$. Used for `Linear` and `Conv2d` accumulators. *Proof:* bit-growth rule $\max(\text{sum\_bits}, \text{bias\_bits}) + 2$ yields $|\text{acc}| < 2^{m+1}$, fitting signed range $[-2^{m+1}, 2^{m+1}-1]$ in $m + 2 = \text{out\_bits}$ signed bits without an extra sign bit.
- **`scalar_blocks(v, cap)`** $= \text{value_blocks}(\text{magnitude\_bits}(v), \text{cap})$. Sizing for plaintext comparison thresholds and clamp bounds.
- **Input working width:** $\text{tensor\_blocks}(\text{cts}, \text{bits}, N) = \max_{x \in \text{cts}} \min(\text{actual\_blocks}(x), \text{value\_blocks}(\text{bits}, N))$. Graph inputs arrive encrypted at model ceiling $N$ and are trimmed for free ($0$ PBS) to their derived width on arrival.
- **Resizing:** `resize(sk, ct, blocks)` returns `Cow::Borrowed` when widths match; trimming costs $0$ PBS, while sign-extension costs $1$ PBS (`cast_to_signed`).
- **Ceiling:** every width is bounded above by `num_blocks`. When clamping bites, behaviour is identical to the global-radix baseline.


### Server-side input trimming

Under the D14 Amendment, client-side encryption remains uniform: clients encrypt input tensors at the model-level radix ceiling `num_blocks` without needing graph-wide per-tensor width knowledge. On the server side, input ciphertexts are trimmed to each consuming operation's required width (e.g. `WidthCache::build` in `Linear`/`Conv2d` or `resize_tensor` in `Requant`/`Add`) via `sk.cast_to_signed(ct, target_blocks)`. Because dropping higher radix blocks costs 0 PBS and allocates no crypto noise, trimming is computationally free. Performing trimming lazily at the operator boundary preserves the backend-neutral Layer-2 eval loop contract (`penumbra-core`) with zero scheme-specific entrypoint branching.
### Core seam (`Backend::build_op_with_bits`)

To inform Layer-1 op construction of Layer-2 derived bit widths without breaking backend neutrality or leaking crypto into Layer 2, `penumbra-core` added a default-implemented method:

```rust
fn build_op_with_bits(
    &self,
    spec: &OpSpec,
    input_bits: &[usize],
    output_bits: &[usize],
) -> Result<Box<dyn Op<Self>>, String> {
    self.build_op(spec)
}
```

The topological eval loop passes each node its derived input/output bit widths in declaration order. Backends that ignore bit widths (e.g. CKKS) take the default; TFHE wraps the op with `WithWidths` to size radix blocks per tensor. Clients still encrypt inputs at `num_blocks`; the server trims oversized ciphertexts on arrival for free ($0$ PBS).

### Before / after benchmark results (Phase 14)

Measured across all 12 models in the test suite on an Apple M3 Pro (11 threads, 2 samples per model).
Sources: pre-fix baseline `docs/results/phase14-tfhe-baseline.json` (`488f76c`) vs post-fix `docs/results/phase14-tfhe-per-tensor.json` (`b13d0af`):

| Model | Baseline Eval (s) | Post-fix Eval (s) | Speedup | Baseline PBS | Post-fix PBS | PBS Reduction | Conv2d PBS Δ | Linear PBS Δ | Requant PBS Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `phase2_logreg` | 0.439 | 0.354 | **1.24x** | 137 | 91 | -33.6% | - | -46 | - |
| `phase4_cnn` | 25.440 | 17.484 | **1.46x** | 9,104 | 6,134 | -32.6% | -1920 | -1082 | +0 |
| `phase5_digits` | 215.055 | 87.534 | **2.46x** | 75,153 | 31,663 | -57.9% | -31293 | -11333 | -864 |
| `phase5_qat` | 165.394 | 81.166 | **2.04x** | 56,470 | 28,558 | -49.4% | -24804 | -3216 | +108 |
| `phase6_onnx` | 215.041 | 88.239 | **2.44x** | 75,153 | 31,663 | -57.9% | -31293 | -11333 | -864 |
| `phase6_sklearn` | 41.131 | 15.422 | **2.67x** | 13,376 | 4,704 | -64.8% | - | -8672 | - |
| `phase7_faces` | 373.256 | 141.570 | **2.64x** | 128,571 | 50,559 | -60.7% | -59648 | -18492 | +128 |
| `phase8_trees` | 10.914 | 5.200 | **2.10x** | 4,156 | 1,699 | -59.1% | - | -2390 | - |
| `phase8_branch` | 134.025 | 40.967 | **3.27x** | 47,202 | 14,157 | -70.0% | - | -32117 | -464 |
| `phase8_bn_cnn` | 204.966 | 100.924 | **2.03x** | 70,897 | 34,686 | -51.1% | -31968 | -4531 | +144 |
| `phase8_gap_cnn` | 406.451 | 180.198 | **2.26x** | 139,388 | 65,629 | -52.9% | -71496 | -4071 | +256 |
| `phase11_tabular_mlp` | 31.042 | 10.509 | **2.95x** | 10,351 | 3,615 | -65.1% | - | -6744 | +8 |
| **Total / Overall** | **1823.155** | **769.566** | **2.37x** | **629,958** | **273,158** | **-56.6%** | | | |

Every model preserves bit-for-bit exactness (`max_abs_err` unchanged across all runs).

## CI implication

The golden test runs in the release CI job. Because per-sample FHE cost is high, the
committed test batch is kept small (still covering both classes) so the job finishes in a
few minutes rather than tens of minutes. The batch size lives in
`examples/mnist/train_quantize_export.py` (`N_TEST`).
