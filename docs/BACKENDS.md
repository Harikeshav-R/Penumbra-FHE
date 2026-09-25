# FHE Backends

Penumbra runs the **same** IR graph on more than one FHE scheme. This document defines the
backend boundary: what a backend must provide, what it may never touch, how correctness is
judged per scheme, and how to add one.

Read [`PROJECT.md`](../PROJECT.md) §4–§6 first for the narrow-waist architecture this extends.

| Backend | Crate | Scheme | Library | Arithmetic | Status |
|---|---|---|---|---|---|
| `tfhe` | `penumbra-tfhe` | TFHE / CGGI | [`tfhe-rs`](https://github.com/zama-ai/tfhe-rs) | exact, small signed integers | reference implementation (Phase 12.1) |
| `ckks` | `penumbra-ckks` | CKKS | [`poulpy-ckks`](https://github.com/phantomzone-org/poulpy) | approximate reals, SIMD-batched | implemented (Phase 12.2) |
The second backend exists to make a **controlled comparison** possible, not to make Penumbra
a general multi-scheme framework (`PROJECT.md` §1, §18). Everything below is in service of
that: two schemes, one graph, one harness, one set of numbers.

## Two waists, not one

The original narrow waist (`PROJECT.md` §4) stops the *op vocabulary* from growing as use
cases multiply. A second waist, **below** it, stops the *op implementations* from growing as
schemes multiply.

```
┌─ Layer 3: MODEL ADAPTERS (grows per use case — NO crypto here) ─────┐
│  MNIST CNN │ face classifier │ tabular MLP │ ...                     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  ◀── waist 1: the stable IR
┌─ Layer 2: IR + OP REGISTRY + EVAL LOOP (fixed, backend-neutral) ────┐
│  a graph of ~8 op types, walked once: Linear, Conv2d, Requant, ...   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  ◀── waist 2: the `Backend` trait
┌─ Layer 1: FHE BACKENDS (pluggable — one per scheme) ────────────────┐
│   penumbra-tfhe (tfhe-rs)      │      penumbra-ckks (poulpy-ckks)    │
│   exact ints, LUT via PBS      │      approx reals, polynomials      │
└──────────────────────────────────────────────────────────────────────┘
```

Layer 2 is written once and knows nothing about either scheme. As of Phase 12.1,
`crates/penumbra-core` contains **zero** cryptographic dependencies, and `crates/penumbra-tfhe`
realizes the `Backend` trait against `tfhe-rs`. The `runtime` crate serves as a backward-compatible
facade re-exporting both crates.

### The discipline, stated both ways

> **A new use case only ever adds a Layer-3 adapter. It never touches Layers 1–2.**
>
> **A new backend only ever adds a Layer-1 crate. It never touches Layers 2–3.**

The litmus tests are duals of each other:

- If adding **face recognition** forces a crypto edit, the *op vocabulary* leaked
  (`AGENTS.md` §1.2).
- If adding **CKKS** forces an IR change, an op-vocabulary change, or a per-scheme branch in
  the eval loop, the *backend abstraction* leaked. The fix is a more general trait method or
  a loud "unsupported on this backend" — never a scheme check inside Layer 2.

**Backend parity** is what makes the comparison scientifically valid: both backends consume
the same IR file, run the same models, and are measured by the same harness. Two
independently built pipelines would produce differences the comparison could not attribute
to the schemes.

## The backend contract

The trait is derived from what the op implementations **actually call today** — not from
what a general FHE API might look like. Every row below is a real call site.

### Evaluation primitives (server side, public key only)

| Primitive | Used by | `tfhe-rs` today | CKKS realization |
|---|---|---|---|
| trivial zero | `linear.rs:66`, `conv2d.rs:104` | `create_trivial_zero_radix` | trivial (noiseless) encoding of 0 |
| ct + ct | `linear.rs:69`, `conv2d.rs:129`, `pool.rs:115`, `add.rs:56` | `add_parallelized` | native homomorphic add |
| ct × plaintext scalar | `linear.rs:68`, `conv2d.rs:128`, `requant.rs:215` | `scalar_mul_parallelized` | plaintext multiply (consumes one level) |
| ct + plaintext scalar | `linear.rs:73`, `conv2d.rs:135`, `requant.rs:221` | `scalar_add_parallelized` | plaintext add |
| ct ≥ plaintext scalar | `argmax.rs:33` | `scalar_ge_parallelized` | **no exact analogue** — polynomial sign/step |
| max(ct, ct) | `pool.rs:123` | `max_parallelized` | polynomial max |
| max(ct, 0) — ReLU | `requant.rs:209` | `scalar_max_parallelized` | polynomial ReLU |
| min(ct, k) — saturate | `requant.rs:228` | `scalar_min_parallelized` | polynomial clamp |
| ct ≫ k | `requant.rs:226` | `scalar_right_shift_parallelized` | rescale / plaintext scaling |
| apply univariate f | `activation.rs:104-115`, `requant.rs:193-232` | `generate_lookup_table` + `apply_lookup_table` (**PBS**) | polynomial approximation of the same tabulated f |

### Key and boundary primitives (client side & preflight)

| Primitive | Used by | `tfhe-rs` today | CKKS realization |
|---|---|---|---|
| keygen | `keys.rs:58` | `gen_keys_radix` | `keygen(&params)` |
| encrypt | `encrypt.rs:21` | `ck.encrypt_signed` | `encrypt(ck, input)` |
| decrypt | `encrypt.rs:35`, `:44` | `ck.decrypt_signed` | `decrypt_vec(ck, out)` / `decrypt_label` |
| key wire serialize | `keys.rs:45`, `:96` | `TaggedKey` over `bincode` | `TaggedKey` over `bincode` |
| ciphertext (de)serialize | `wire.rs` / `encrypt.rs` | `TaggedCts` over `bincode` (shared Layer 2) | `TaggedCts` over `bincode` (shared Layer 2) |
| budget preflight | `backend.rs:27` | `check_graph_bit_width_budget` | `check_graph_depth_budget` |
| op cost proxy | `ops/*.rs` | analytic `Op::cost` | analytic `Op::cost` |
### The two hard rows

Everything above except the last two rows of the evaluation table maps onto CKKS cleanly.
The exceptions are the interesting ones:

1. **`apply univariate f`** is TFHE's superpower. A programmable bootstrap applies an
   *arbitrary* function to a ciphertext and refreshes noise in the same operation, exactly,
   for a fixed cost. CKKS has no such operation. `Activation` and `Requant` — the two ops
   that dominate TFHE runtime — must be **polynomial approximations** under CKKS
   (`poulpy-ckks` provides an `approximation` module for the fitting). This is the single
   largest semantic difference between the backends and the most interesting thing the
   comparison measures.
2. **`ct ≥ scalar`** (`Argmax`) is a LUT under the hood. Under CKKS it is a polynomial step
   function, and a poor one at low degree. Note that Penumbra's multi-class models already
   decrypt logits and argmax **client-side** (`docs/BENCHMARKS.md`, Phase-4 onward), so this
   only affects the 2-class `Argmax` head.

> A backend that cannot implement an op must say so **at load time, naming the op and the
> node** (`AGENTS.md` §1.4) — never approximate it silently, and never fall back to another
> backend.

## Correctness: one invariant, two comparators

The reference is **always** the same: the quantized-cleartext forward pass computed by
`python/penumbra/reference.py` (`evaluate_graph_int`). What changes per backend is only how
the encrypted result is compared to it.

| Backend | Comparator | Rationale |
|---|---|---|
| `tfhe` | **equality, bit-for-bit** | TFHE is exact. Any discrepancy is a quantization or implementation bug — never crypto noise. |
| `ckks` | **within a declared, per-model error bound**, with the measured error always reported | CKKS is approximate by construction. The bound is committed alongside the model and asserted in CI; exceeding it is a bug (scale, level, or polynomial degree) first, noise second. |

Two consequences worth stating plainly:

- The TFHE invariant is **not weakened**. It remains a hard gate (`AGENTS.md` §1.1).
- CKKS is **never** compared against a different or friendlier reference. Comparing it to a
  float model instead of the quantized one would make its accuracy look better and the
  comparison meaningless.

Under CKKS the PBS-free ops (`Linear`, `Conv2d`, `Pool`, `Add`) may well round-trip to the
exact integers at a generous scale. That is worth **reporting as a diagnostic** — it isolates
approximation error to the nonlinearities — but it is not a CI gate, because it is a
property of the chosen scale rather than of the implementation.

## Cost models

The two schemes are fast and slow at opposite things. There is no single cost proxy.
Cost proxies are derived analytically via `Op::cost(&self, input_lens: &[usize]) -> Vec<(&'static str, u64)>`
so the measurement code in `penumbra-core` and `penumbra-bench` stays completely scheme-neutral:

### TFHE Cost Counters

| Counter | Meaning |
|---|---|
| `bootstraps` | explicit programmable bootstraps issued (`apply_lookup_table`) |
| `cmp_pbs_ops` | PBS-bearing comparison/shift primitives invoked (`scalar_max`, `scalar_min`, `scalar_right_shift`, `scalar_ge`, `max`); each costs >= 1 internal PBS |
| `scalar_mul` | ciphertext x plaintext scalar primitive invocations |
| `scalar_add` | ciphertext + plaintext scalar primitive invocations |
| `ct_add` | ciphertext + ciphertext addition primitive invocations |
| `pbs` | **measured** programmable bootstraps read from `tfhe`'s `pbs-stats` counter, including the carry-propagation bootstraps that radix `add`/`scalar_mul`/`scalar_add` issue internally (`tfhe-1.8.1` `integer/server_key/radix_parallel/add.rs:221`) |

The analytic counters remain the *comparable* proxy across backends; `pbs` is TFHE-only ground truth.

### CKKS Cost Counters

| Counter | Meaning |
|---|---|
| `rotations` | slot rotations the BSGS linear map plan performs (`PreparedLinearMap::rotation_count`) |
| `rescales` | rescales consumed — one per linear-map evaluation, one per polynomial level |
| `poly_evals` | polynomial approximations evaluated |
| `depth_levels` | multiplicative levels consumed (realized depth) |

*Note on depth:* `depth_levels` is the **realized** depth of the fitted polynomial (`precision_at_depth(..).depth`),
whereas `check_graph_depth_budget` budgets the **worst case** `bsgs_eval_depth(max_poly_degree, MinDepth)`.
Realized <= budgeted is expected, not a bug.

| | TFHE (`penumbra-tfhe`) | CKKS (`penumbra-ckks`) |
|---|---|---|
| Cost proxy | **runtime ≈ number of bootstraps** | **runtime ≈ multiplicative depth × (rotations + rescales)** |
| Linear ops | cheap — plaintext-weight scalar-mul + add, no PBS | cheap *and* SIMD-batched across slots |
| Nonlinear ops | one PBS per activation — dominates runtime | polynomial evaluation — consumes depth, forces rescales |
| Scaling with width | linear in element count (one ciphertext per value) | amortized across slots until slots run out |
| The lever | reduce PBS count | reduce depth; pack more values per ciphertext |

`PROJECT.md` §5's "runtime ≈ number of bootstraps" is therefore a **TFHE** statement, not a
project-wide one. Where the docs use it as a universal rule, read it as scoped to the TFHE
backend.

## Resource budgets

Both schemes have a hard, centrally-enforced budget that fails loudly with the offending
layer named (`AGENTS.md` §1.3). They are not the same budget.

| | TFHE | CKKS |
|---|---|---|
| The budget | radix capacity: `num_blocks × MESSAGE_BITS` bits | multiplicative depth / level budget, and scale precision |
| What consumes it | accumulator growth (`b + log2(N)`) | every ciphertext-plaintext multiply and every polynomial degree |
| What restores it | `Requant` (a PBS) narrows back to `MESSAGE_BITS` | rescale, or bootstrapping |
| Enforced by | `Backend::check_graph_budget` (radix check) | `Backend::check_graph_budget` (depth/scale check) |
| Overflow symptom | silently wrong ciphertext | precision collapse, then noise |

The bit-width tracker in Layer 2 (`propagate_bit_widths`, `eval.rs:182`) is scheme-neutral
integer arithmetic and is reused by both — it describes the *quantized graph*, not TFHE. What
differs is the capacity it is checked against.

## Adding a backend (the canonical path)

The counterpart to the "add an op" path (`CONTRIBUTING.md`, `AGENTS.md` §4). Every step ships
in the same change:

1. **New crate** under `crates/`, depending on `penumbra-core` and exactly one FHE library.
2. **Implement `Backend`** — the primitives table above. Nothing else in the workspace changes.
3. **Declare the op matrix**: for each op in the vocabulary, implement it or reject it loudly
   at load time with the op and node named.
4. **Declare the comparator** — exact, or a per-model error bound — and add correctness tests
   against `reference.py`'s output at that comparator, for the committed fixtures.
5. **Register with the harness** (`penumbra-bench`): implement `check_graph_budget`, `serialize_client_key`,
   `serialize_server_key`, and `Op::cost`, then add the backend constructor to `available_backends()` so
   the backend is measured by the same code as every other backend.
6. **Docs**: this file's tables, `docs/SUPPORTED-OPS.md` support columns, and a
   `docs/NOTES-<scheme>.md` spike record.

If step 2 cannot be done without touching `penumbra-core`, stop — that is an architectural
fork (`AGENTS.md` §3.2), not a routine addition.

## Open design forks

Recorded here rather than silently decided (`AGENTS.md` §3.2). Each needs a decision before
or during Phase 12.

### 1. SIMD packing — the biggest one

CKKS's entire advantage is that one ciphertext holds thousands of slots. Penumbra's inter-op
currency is `CtVec` — conceptually one ciphertext per scalar value.

- **Option A — keep one value per ciphertext.** Trivial to implement against the existing op
  code; makes the backend a near-mechanical translation. But it throws away the *only* thing
  CKKS is better at, and the resulting comparison measures a hobbled CKKS.
- **Option B — pack a tensor into slots.** Uses the scheme as intended; `Linear`/`Conv2d`
  become rotation-and-sum patterns. Substantially more implementation work and a real risk of
  the packing layout leaking upward into Layer 2.

**Decision (settled in Phase 12.0): Option B (tensor slot packing).** Empirical benchmarks in
the Phase-12.0 spike (`crates/spike-ckks`, `docs/NOTES-ckks.md`) showed that evaluating an
activation layer across 128 packed slots in a single ciphertext requires only ~0.26 ms in
`--release`. Evaluating scalar ciphertexts one value at a time (Option A) would require 128
independent polynomial evaluations, making inference over $100\times$ slower and artificially
crippling CKKS. Whichever is chosen must be stated in `docs/COMPARISON.md`'s threats-to-validity
section.

### 2. What `Requant` means under CKKS

`Requant` exists because TFHE accumulators must be narrowed back into a single 2-bit block.
CKKS has no such constraint — its analogue is a rescale, which is nearly free.

- **Option A — keep it as a fused ReLU + rescale.** Preserves graph shape exactly; the ReLU
  half becomes a polynomial, the rescale half becomes a genuine CKKS rescale.
- **Option B — split it**: apply the ReLU polynomial, treat the fixed-point rescale as a
  no-op absorbed into the scale bookkeeping.

**Decision (settled in Phase 12.2): Option A.** Keeps both backends walking an identical node
list, which is what backend parity requires (`AGENTS.md` §1.2). The continuous piecewise-linear
function `f(t) = clamp((max(t, 0)·mult + round_bias) / 2^shift, 0, max_val)` is fitted over
the active clamp interval `[-t_sat, t_sat]`.

### 3. How the approximation knob is exposed

Polynomial degree trades accuracy against depth against latency — it is CKKS's central
tuning parameter, with no TFHE counterpart. `PROJECT.md` §12 commits to *one* crypto override
knob per backend.

**Decision (settled in Phase 9 / 12.2):** Each backend exposes exactly one parameter profile knob
via `CryptoProfile`:
- **TFHE:** `TfheProfile` — named profiles `"classic"`, `"gaussian"`, `"multibit2"`, `"multibit3"`, `"multibit4"` (all 128-bit secure at message=2/carry=2).
- **CKKS:** `max_poly_degree` (`CkksParams::max_poly_degree`) — the maximum polynomial degree any op may fit.

The per-backend profile exposes this single lever while quantization remains an automated library
service.
## Backend selection

Selection is not parameter exposure. Users pick a **named backend**; they never touch
`tfhe-rs` or `poulpy` parameters directly (`PROJECT.md` §12, `AGENTS.md` §6).

> Keys and ciphertext are **not** portable between backends. The wire format is tagged with a
> backend scheme header (`"tfhe"` or `"ckks"`); feeding mismatched material fails loudly at
> load time with an actionable message naming both backends (`AGENTS.md` §1.4).

## See also

- [`docs/NOTES-tfhe.md`](./NOTES-tfhe.md) — the TFHE parameter profile and measured costs.
- [`docs/NOTES-ckks.md`](./NOTES-ckks.md) — the CKKS parameter profile, the `poulpy` crate
  map, and the toolchain caveats.
- [`docs/COMPARISON.md`](./COMPARISON.md) — the two-backend study: hypothesis, method,
  threats to validity.
- [`docs/BENCHMARKS.md`](./BENCHMARKS.md) — the measured numbers.
