# Performance & Tuning Guide

> Guidance for optimizing encrypted inference latency, throughput, and memory across Penumbra-FHE's
> backends. For measured numbers on committed models, see [`docs/BENCHMARKS.md`](./BENCHMARKS.md); for the
> backend architecture and contracts, see [`docs/BACKENDS.md`](./BACKENDS.md); for quantization math and
> scales, see [`docs/QUANTIZATION.md`](./QUANTIZATION.md).

---

## 1. The Cost Model, Per Backend

Encrypted inference latency in homomorphic encryption is dominated by cryptographic maintenance
operations rather than raw arithmetic. The two backends exhibit fundamentally different cost
mechanisms:

### TFHE Backend (`penumbra-tfhe`)

- **Primary cost:** Programmable Bootstrapping (PBS). In shortint radix arithmetic,
  `runtime ≈ number of bootstraps` ([`PROJECT.md`](../PROJECT.md) §5).
- **Radix carry propagation:** Each integer tensor element is decomposed across `num_blocks` radix
  blocks (`MESSAGE_BITS = 2` bits per block). While scalar additions and plaintext-weight scalar
  multiplications are homomorphic linear combinations, accumulating multi-block integers requires
  carry propagation across block boundaries. Each carry propagation step issues approximately 2 PBS
  operations per block.
- **Scaling behavior:** Evaluation cost scales roughly quadratically with `num_blocks`: wider
  accumulators require more blocks, which lengthens every carry-propagation chain and increases the
  number of PBS operations per matrix multiply.
- **Lookup tables (LUTs):** Standalone nonlinearities and `Requant` operations issue 1 PBS per
  output block. For 2-bit activations (`act_bits = 2`), a `Requant` consumes exactly 1 PBS per
  activation element.

### CKKS Backend (`penumbra-ckks`)

- **Primary cost:** Multiplicative depth, homomorphic rotations, and rescales.
- **SIMD tensor packing (Option B):** Entire tensors are packed into single ciphertexts
  ($N = 16384$, $k = 360$, `lt_slots = 256`). Linear operations (`Linear`, `Conv2d`, `Pool(avg)`)
  are evaluated as Baby-Step Giant-Step (BSGS) diagonal transforms, converting thousands of
  scalar operations into $2\sqrt{N_{\text{diag}}}$ Galois automorphism rotations.
- **Polynomial approximations:** Continuous activations and `Requant` floor functions are
  approximated by Chebyshev or minimax polynomials. Evaluating a degree-$D$ polynomial consumes
  $\lceil \log_2(D+1) \rceil$ multiplicative depth levels and a sequence of ciphertext-ciphertext
  multiplications and rescales.
- **Scaling behavior:** Latency is independent of tensor length up to the slot capacity (256 slots),
  scaling primarily with the number of matrix diagonals and polynomial evaluation depth.

---

## 2. TFHE Knobs

### Parameter Profiles (`TfheProfile`)

Penumbra provides five named 128-bit-secure parameter profiles in `crates/penumbra-tfhe/src/keys.rs`:

| Profile | Noise Distribution | Blind Rotation Grouping | p-fail | Server Key Size | Relative Speed |
|---|---|---|---:|---:|---:|
| `classic` (**tuned default**) | TUniform | 1-bit (Standard PBS) | $2^{-129.58}$ | 114.84 MB | **1.00x** |
| `gaussian` | Gaussian | 1-bit (Standard PBS) | $2^{-128.00}$ | 121.89 MB | 0.98x |
| `multibit2` | TUniform | Group 2 | $2^{-140.34}$ | 373.15 MB | 0.48x |
| `multibit3` | TUniform | Group 3 | $2^{-128.24}$ | 392.31 MB | 0.64x |
| `multibit4` | TUniform | Group 4 | $2^{-134.35}$ | 302.07 MB | 0.91x |

**Why `classic` is the default:** Multi-bit PBS parameter sets trade higher serial algorithmic cost
for internal multi-threaded blind rotation. In Penumbra, outer `rayon` parallelization over
independent tensor elements and channels already saturates all available physical CPU cores. Multi-bit
profiles induce thread contention against the outer pool, resulting in slower wall-clock execution
while requiring 2.6x to 3.4x larger evaluation keys (302–392 MB vs 114.84 MB).

### Radix Width (`num_blocks`)

`num_blocks` is **never set by hand**; it is derived by `penumbra.bitwidth.minimal_num_blocks` to fit
the widest accumulator or transient Requant peak in the graph. Reducing `num_blocks` is the single
most effective lever for accelerating TFHE inference. Three parameters control it:

1. **`input_bits`:** Graph input integer width. Reducing input bits from 4 to 2 or 3 directly
   narrows initial accumulators.
2. **Per-layer `n_bits` (`weight_bits`):** Configured via `Model.quantize(..., n_bits=[w1, w2])`.
   Allows deeper or less sensitive layers to use fewer weight bits while preserving precision where
   critical.
3. **`max_mult_bits`:** Caps the `Requant` fixed-point multiplier `mult`. In multi-channel networks,
   unconstrained PTQ multipliers can reach 31, creating transient multiply peaks
   ($\max(x, 0) \times \text{mult} + \text{round\_bias}$) of 21 bits that force an 11-block radix.
   Capping `max_mult_bits = 1` limits the multiplier to $\le 1$, compressing transient peaks into
   16–18 bits and shedding 2 to 3 radix blocks (saving 30% to 53% of all PBS operations).

#### Binding Constraints on Committed Models

| Model | Pre-Minimization Binding Constraint | Minimized Plan | Num Blocks Reduction |
|---|---|---|---:|
| `phase2_logreg` | Linear accumulator: 16 bits (4 in + 4 w + 6 fan-in + 2 guard) | `in=2, w=[2]` | 8 → 6 blocks (3.82x speedup) |
| `phase4_cnn` | Conv2d accumulator: 14 bits (4 in + 4 w + 4 fan-in + 2 guard) | `in=4, w=[4]` | 7 blocks (accuracy bound) |
| `phase5_digits` | `conv0__requant` internal peak: 21 bits (`mult` up to 31) | `in=3, w=[5, 6], mult=1` | 11 → 9 blocks (1.42x speedup) |
| `phase5_qat` | `conv0__requant` internal peak: 21 bits (`mult` up to 31) | `in=3, w=[5, 4], mult=1` | 11 → 8 blocks (1.88x speedup) |
| `phase6_onnx` | `conv0__requant` internal peak: 21 bits (`mult` up to 31) | `in=3, w=[5, 6], mult=1` | 11 → 9 blocks (1.42x speedup) |
| `phase6_sklearn` | `linear0` output: 20 bits (4 in + 8 w + 6 fan-in + 2 guard) | `in=4, w=[8]` | 10 blocks (accuracy bound) |
| `phase7_faces` | `conv0__requant` internal peak: 21 bits (`mult` up to 29) | `in=4, w=[6, 6]` | 11 blocks (accuracy bound) |

### Activation Bit-Width (`act_bits`)

Post-Requant activations are capped at `MESSAGE_BITS = 2` (values in $[0, 3]$). This is a hard
architectural ceiling: shortint lookup tables evaluate over single radix blocks. Higher activation
precisions are not representable without multi-block PBS decomposition, which is prohibitively costly.

### Parallelism (`rayon`)

The Layer-2 graph walker automatically evaluates independent output elements in parallel using
Rayon:
- `Conv2d`: parallelized across output spatial positions and output channels.
- `Pool`: parallelized across pooled spatial locations and channels.
- `Requant`: parallelized across tensor elements.
- `Linear`: multi-term radix additions evaluated via parallelized tree reduction.

### Graph-Level Optimizations (`penumbra-core::optimize`)

The rule-based optimizer in `penumbra_core::optimize` implements algebraic rewrite passes:
- **R1 (Requant-Activation Fusion):** Fuses a Requant immediately followed by a stateless
  Activation into a single compound Requant LUT.
- **R2 (Activation-Activation Fusion):** Combines consecutive LUT activations into one lookup table.
- **R3 (Requant-Requant Chain Collapse):** Collapses cascaded rescales into a single Requant node.

*Note on current fixtures:* Committed models fuse activations into `Requant` during export via the
quantization service, so these passes act as verification identities on committed graphs while
remaining available for external or unoptimized IR graphs.

---

## 3. CKKS Knobs

### Polynomial Degree (`max_poly_degree`)

The primary user-facing knob for the CKKS backend is `max_poly_degree` in `CkksParams`
(default `15`).
- Under BSGS `SplitStrategy::MinDepth`, degree 15 requires multiplicative depth 4.
- Increasing degree reduces polynomial approximation error on ReLU and floor functions, but consumes
  additional depth levels and requires larger scaling parameters or ciphertext moduli.
- Decreasing degree saves depth and latency but increases approximation error, risking label
  misclassifications.

### Slot Packing (Option B)

CKKS models pack 2D tensors into single 1D SIMD slots:
- Tensors are flattened in channel-major, row-major order up to `lt_slots = 256`.
- Linear layers generate BSGS diagonal masks. Matrix diagonals are computed in parallel during op
  construction and applied with homomorphic Galois rotations.
- Precomputing BSGS plaintext diagonals once per model saves ~10–20 ms per sample.

### Depth and Rescale Budget

- Each multiplication consumes 1 level of scale budget ($q_i \approx \Delta$).
- The default profile provides 330 bits of budget across $N = 16384$, sufficient for depth-7
  convolutional networks without requiring homomorphic bootstrapping.

---

## 4. Bit-Width Minimization Workflow

To find the smallest viable precision assignment for a model without guessing, use the deterministic
coordinate descent search in `penumbra.quantization.minimize`:

```bash
# Run minimization search on the training split (prints trace, does not write fixture):
uv run python examples/mnist/train_quantize_export.py --minimize
uv run python examples/mnist/cnn_export.py --minimize
uv run --extra ml --system-certs python examples/mnist/real_digits_export.py --minimize
uv run --extra ml --system-certs python examples/mnist/qat_export.py --minimize
uv run --extra ml --system-certs python examples/mnist/onnx_export.py --minimize
uv run --extra ml --system-certs python examples/mnist/sklearn_export.py --minimize
uv run --extra ml --system-certs python examples/faces/olivetti_export.py --minimize
```

### Search Rules

1. **Knob order:** Descends `max_mult_bits`, then `input_bits`, then `weight_bits` (per layer).
2. **Accuracy floor:** Rejects any candidate whose training-split accuracy drops below
   $\text{baseline} - 0.01$ (1.0 absolute percentage point tolerance).
3. **Monotonicity guard:** Rejects any candidate where `num_blocks` increases, even if accuracy is
   acceptable.
4. **Training split selection:** Bit-plans are selected exclusively on calibration and training
   data to prevent test-set leakage. The reported fixture accuracy is evaluated on the independent
   test split.
5. **No manual adjustment:** If the search returns the baseline unchanged, the fixture constants
   are preserved as measured-and-unchanged.

---

## 5. Benchmarking and the Regression Gate

Penumbra enforces deterministic performance gates in CI via `penumbra-bench`.

### Running Benchmarks

```bash
# Evaluate a single model on TFHE:
cargo run -p penumbra-bench --release --bin penumbra-bench-report -- \
  --models phase2_logreg --backends tfhe --samples 1

# Check against the committed regression baseline:
cargo run -p penumbra-bench --release --bin penumbra-bench-report -- \
  --models phase2_logreg,phase4_cnn,phase6_sklearn --backends tfhe --samples 1 \
  --baseline crates/penumbra-bench/baselines/tfhe-classic.json
```

### What the Gate Checks (and What It Does Not)

- **Gated (Deterministic):**
  - Radix capacity (`num_blocks`)
  - Ground-truth PBS count (`measured_totals.pbs`)
  - Operational cost proxies (`cost_proxy` counters: `bootstraps`, `cmp_pbs_ops`, `scalar_mul`, etc.)
  - Wire sizes (client key bytes, server key bytes, input ciphertext bytes, output ciphertext bytes)
  - Label correctness (`labels_matched == labels_checked`)
- **Not Gated (Non-Deterministic):**
  - Wall-clock seconds: Excluded intentionally. CI runner hardware exhibits high variance that
    creates flaky gates. The deterministic cost model guarantees that constant PBS and key size
    corresponds to invariant execution cost.

### Regenerating the Baseline

When an architectural change intentionally alters operator implementations or precision:

```bash
cargo run -p penumbra-bench --release --bin penumbra-bench-report -- \
  --models phase2_logreg,phase4_cnn,phase6_sklearn --backends tfhe --samples 1 \
  --write-baseline crates/penumbra-bench/baselines/tfhe-classic.json
```

---

## 6. Architectural Decisions Recorded

### 1. Binary IR Format: Declined Based on Profiling

`ROADMAP.md` Phase 10 conditioned introducing a binary format on profiling evidence. Profiling
demonstrated that:
- JSON IR load times across all models are **0.38 ms to 0.58 ms**.
- IR loading represents **< 0.084%** of evaluation time (and < 0.002% on CNN models).
- Fixture files are compact (5 KB – 28 KB).

Introducing a binary IR format would add schema complexity without measurable performance gain.
Penumbra retains its backend-neutral, human-readable JSON IR format (`SCHEMA_VERSION = 0.6.0`).

### 2. Tree Models: Deferred to Phase 8

Decision tree evaluation requires ciphertext-ciphertext comparisons (`CmpGte`) and conditional
selection (`Select`), which are not part of the current op vocabulary. Benchmarking tree models is
deferred to Phase 8 (`ROADMAP.md` §P8), which owns the Tree-to-IR compiler.

---

## 7. Security Invariant: Never Trade Security for Speed

`AGENTS.md` §7 and `ROADMAP.md` §P10 define a hard constraint:

> **Never trade away security for speed.** Security level is a non-negotiable hard gate.

- All supported TFHE profiles guarantee $\ge 128$-bit classical security ($p_{\text{fail}} \le 2^{-128}$).
- CKKS parameters ($N = 16384, k = 360$) provide $\ge 128$-bit classical security under the
  LWE/RLWE estimator.
- Comparing backends at mismatched security levels is invalid (`docs/COMPARISON.md`). Optimizations
  must operate strictly within the committed 128-bit security envelope.
