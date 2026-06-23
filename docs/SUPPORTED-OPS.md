# Supported Operators

This is the authoritative list of operators Penumbra-FHE's runtime implements, and the
bit-width growth rule each one declares (`PROJECT.md` §9). It must always match what the
runtime actually accepts (`AGENTS.md` §5) — when you add an op, update this file in the
same change (the canonical "add an op" path, `AGENTS.md` §4).

Notation: a value is carried as a **signed radix integer** of `num_blocks` blocks; under
the default profile each block holds `MESSAGE_BITS = 2` bits, so the radix capacity is
`num_blocks × 2` bits. `Linear`/`Conv` are *cheap* (plaintext-weight arithmetic, no
bootstrap); `Activation`/`Requant`/`Compare` are *expensive* (one programmable bootstrap
per value). Runtime ≈ number of bootstraps (`PROJECT.md` §5).

## Phase 2 — the narrow waist (`Linear`, `Activation`, `Argmax`)

| Op | Covers | TFHE realization | Bit-width rule (`output_bits`) |
|---|---|---|---|
| `Linear` | dense layers, logistic/linear regression | `Σ (ciphertext × plaintext weight) + bias` — scalar-mul + adds, **no PBS** | `input_bits + weight_bits + ceil(log2 N) + 1` (N = fan-in; `+1` guard for the bias) |
| `Activation` | ReLU, sigmoid, any 1-input function | apply a lookup table via **PBS** on a narrow (≤ `MESSAGE_BITS`-bit) block | `output_bits` of the table (independent of input width; kept small to stay LUT-able) |
| `Argmax` | classification head (2-class) | threshold a single logit: `z ≥ threshold` → encrypted `0/1` (a comparison; LUT-backed) | `1` (a single class bit) |

### Notes & current limits

- **`Argmax` is the 2-class special case** (ROADMAP Phase 2): a threshold on one logit.
  Because a 2-class sigmoid/softmax is monotone, the label is a comparison and needs no
  wide-domain LUT and no `Requant`. A true `>2`-class argmax (pairwise `max`/`gt`) is a
  later phase.
- **`Activation` operates on a narrow value.** A PBS over a wide accumulator is infeasible
  (`PROJECT.md` §9); a wide accumulator must be `Requant`-ed down first. `Requant` is Phase
  4, so in Phase 2 activations are applied only on small values.
- **Bit-width budget is checked, not yet auto-managed.** `eval::check_bit_width_budget`
  refuses to run a model whose declared accumulator exceeds the radix capacity, naming the
  offending op (`AGENTS.md` §1.3, §1.4). Automatic `Requant` *insertion* that would prevent
  the overflow is Phase 4.

## Planned (later phases)

| Op | Phase | Notes |
|---|---|---|
| `Conv2d` | 4 | MACs vs plaintext kernel weights; reuses the `Linear` cheap pattern |
| `Pool` | 4 | average pool (adds); max pool (LUT/compare) |
| `Requant` | 4 | rescale a wide accumulator → small int via LUT; enables multi-layer models |
| `Add` | 4 | ciphertext addition (residuals) |
| `Concat` / branching | 8 | multi-input graphs; true topological eval |
