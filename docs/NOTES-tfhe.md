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

All multi-bit profiles enable `with_deterministic_execution()`: without deterministic execution,
multi-bit blind rotation reduces across threads in non-deterministic order, causing ciphertext bytes
to differ run-to-run. Decrypted values are exact either way, but reproducible bytes keep published
artifacts stable.

- **Message space:** 2 bits per block (`message_modulus = 4`), with a 2-bit carry buffer.
  Every profile shares `keys::MESSAGE_BITS = 2`, so a radix integer of `num_blocks` blocks holds
  `num_blocks × 2` bits of value (`keys::radix_capacity_bits`) identically across profiles.
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

Measured on the Phase-2 golden test (`cargo test --release`, `num_blocks = 8` ⇒ 16-bit
signed radix, 64-feature single-logit `Linear → Argmax`):

| Quantity | Value |
|---|---|
| `Activation` LUT over the 4-value message space (4 PBS) | < 1 s total |
| Full `Linear → Argmax` inference, **per sample** | ~30 s |
| keygen + a single signed round-trip | < 1 s |

The per-sample cost is dominated by radix MAC carry-propagation bootstraps in addition and
multiplication. Phase 10 landed rayon parallelism across independent outputs (cutting `Linear`
latency by 2.1x and `Conv2d` latency by 1.5x) and verified that the classic profile provides the
lowest latency under parallel execution. An open lever is Phase 10 task 5: shrinking `num_blocks`
per layer to the minimum the accumulator actually needs.

> ⚠️ Always benchmark in `--release`. Debug `tfhe-rs` is orders of magnitude slower and the
> numbers are meaningless (`docs/DEVELOPMENT.md`).

## CI implication

The golden test runs in the release CI job. Because per-sample FHE cost is high, the
committed test batch is kept small (still covering both classes) so the job finishes in a
few minutes rather than tens of minutes. The batch size lives in
`examples/mnist/train_quantize_export.py` (`N_TEST`).
