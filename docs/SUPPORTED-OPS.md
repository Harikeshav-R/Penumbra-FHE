# Supported Operators

This is the authoritative list of operators Penumbra-FHE's runtime implements, and the
bit-width growth rule each one declares (`PROJECT.md` §9). It must always match what the
runtime actually accepts (`AGENTS.md` §5) — when you add an op, update this file in the
same change (the canonical "add an op" path, `AGENTS.md` §4).

As of Phase 3 these ops are **driven by the serialized IR**: each is an `op_type` variant of
the IR `OpSpec` (see [`docs/IR-SPEC.md`](./IR-SPEC.md)). The op names below are exactly the
`op_type` tags the IR accepts; the cross-language conformance test keeps this list and the
runtime's `OpSpec` enum in sync.

Notation: a value is carried as a **signed radix integer** of `num_blocks` blocks; under
the default profile each block holds `MESSAGE_BITS = 2` bits, so the radix capacity is
`num_blocks × 2` bits. `Linear`/`Conv` are *cheap* (plaintext-weight arithmetic, no
bootstrap); `Activation`/`Requant`/`Compare` are *expensive* (one programmable bootstrap
per value). Runtime ≈ number of bootstraps (`PROJECT.md` §5).

## Phase 2 — the narrow waist (`Linear`, `Activation`, `Argmax`)

| Op | Covers | TFHE realization | Bit-width rule (`output_bits`) |
|---|---|---|---|
| `Linear` | dense layers, logistic/linear regression | `Σ (ciphertext × plaintext weight) + bias` — scalar-mul + adds, **no PBS** | `max(sum_bits, bias_bits) + 2`, where `sum_bits = input_bits + weight_bits + ceil(log2 N)` (N = fan-in) and `bias_bits` is the bias magnitude width; the `+2` is one carry from the bias add **and** one sign bit (`AGENTS.md` §1.3) |
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
- **Bit-width budget is checked, not yet auto-managed.** `eval::check_graph_bit_width_budget`
  propagates per-tensor widths through the IR graph (via `propagate_bit_widths`) and refuses
  to run a model whose declared accumulator exceeds the radix capacity, naming the offending
  node (`AGENTS.md` §1.3, §1.4). Automatic `Requant` *insertion* that would prevent the
  overflow is Phase 4.

## Phase 4 — multi-layer CNN ops (`Add`, `Requant`, `Pool`, `Conv2d`)

| Op | Covers | TFHE realization | Bit-width rule (`output_bits`) |
|---|---|---|---|
| `Requant` | rescale a wide accumulator → small int (enables multi-layer models) | `clamp(max(x >> shift, 0), 0, 2^out_bits-1)`: arithmetic shift + ReLU + radix-level saturate, then a single-block **PBS** (resets noise) | `out_bits` (≤ `MESSAGE_BITS`, independent of input width; must not under-count the clamp LUT) |
| `Pool` | average / max pooling in CNNs | per-channel window reduction over the flat map: `avg` = sum (`add_parallelized`, **no PBS**); `max` = pairwise `max` (comparison PBSs, expensive) | `avg`: `input_bits + ceil(log2 k)` (k = window size); `max`: `input_bits` (selection never grows magnitude) |
| `Add` | residuals / skip connections | element-wise ciphertext addition of **two** input tensors — `add_parallelized`, **no PBS** | `max(a_bits, b_bits) + 1` (one carry; the wider operand's sign bit covers the result) |

### Notes — Phase 4

- **`Requant` is the primitive that unlocks multi-layer models** (`PROJECT.md` §9). A
  `Linear`/`Conv2d` accumulator grows ~`log2(N)` bits per layer; a PBS is feasible only over
  a narrow value, so the wide accumulator is shifted down (a **power-of-two** rescale chosen
  by the quantization service), ReLU'd, saturated **at the radix level** so the value truly
  fits one `MESSAGE_BITS`-wide block, then passed through a single-block clamp LUT. It is a
  **fused ReLU+requant**: the output is non-negative (what the single-block PBS path
  requires, and what conv→ReLU produces anyway). Non-power-of-two scales are Phase 5.
- **`Pool` shares the spatial layout convention.** The flat `CtVec` is read as a
  channel-major, row-major `[channels][in_h][in_w]` tensor — element `(c, y, x)` at
  `c*in_h*in_w + y*in_w + x`. `Conv2d` emits this same layout, so `Conv2d → Pool` needs no
  reshape. `avg` mode emits the window **sum** and leaves the `1/k` to the next `Requant`'s
  shift, keeping pooling PBS-free; the headline CNN uses `avg`. No padding in Phase 4.
- **`Add` is the first multi-input op.** Its node carries **two** entries in `inputs`; the
  list order is the merge order (addition is commutative, so order is immaterial to the
  result, but the contract is uniform with future multi-input ops). The eval loop resolves a
  node's inputs in declared order and dispatches `Op::eval_n`; single-input ops keep working
  through the default `eval_n` (`AGENTS.md` §1.2 — the loop never special-cases an op).

## Planned (later phases)

| Op | Phase | Notes |
|---|---|---|
| `Pool` | 4 | average pool (window sum, rescale deferred to `Requant`); max pool (LUT/compare) |
| `Conv2d` | 4 | MACs vs plaintext kernel weights; reuses the `Linear` cheap pattern |
| `Concat` / branching | 8 | multi-input graphs; true topological eval |
