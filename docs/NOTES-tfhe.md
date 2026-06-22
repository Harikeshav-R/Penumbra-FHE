# TFHE Notes

Working notes on the `tfhe-rs` primitives Penumbra builds on, the parameter profile, and
empirical cost. Closes the Phase-1 spike deliverable (`ROADMAP` Phase 1) and informs the
bit-width-budget design (`PROJECT.md` §9).

## Parameter profile

- **Profile:** `PARAM_MESSAGE_2_CARRY_2_KS_PBS` — the `tfhe-rs` default secure profile.
  Exposed as `keys::DEFAULT_PARAMS`; no other knob in Phase 2 (`PROJECT.md` §12).
- **Message space:** 2 bits per block (`message_modulus = 4`), with a 2-bit carry buffer.
  So a radix integer of `num_blocks` blocks holds `num_blocks × 2` bits of value
  (`keys::MESSAGE_BITS`, `keys::radix_capacity_bits`).
- **Representation:** values are **signed** radix integers (`SignedRadixCiphertext`).
  Weights and logits are naturally signed; signed avoids a zero-point-offset dance and
  generalizes to conv accumulators. (The plan flagged unsigned+offset as the alternative;
  signed won on simplicity.)

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

The per-sample cost is dominated by the `Linear` op: 64 scalar-muls + 64 adds on an 8-block
radix, then a 16-bit comparison. This is consistent with `PROJECT.md` §16's "seconds per
inference, research/prototype territory" — and a reminder that **runtime ≈ bootstraps +
radix width**. Two obvious future levers (Phase 10): parallelize the per-element work
(`rayon`), and shrink `num_blocks` to the minimum the accumulator actually needs.

> ⚠️ Always benchmark in `--release`. Debug `tfhe-rs` is orders of magnitude slower and the
> numbers are meaningless (`docs/DEVELOPMENT.md`).

## CI implication

The golden test runs in the release CI job. Because per-sample FHE cost is high, the
committed test batch is kept small (still covering both classes) so the job finishes in a
few minutes rather than tens of minutes. The batch size lives in
`examples/mnist/train_quantize_export.py` (`N_TEST`).
