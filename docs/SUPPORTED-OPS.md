# Supported Operators

This is the authoritative list of operators Penumbra-FHE's runtime implements, and the
bit-width growth rule each one declares (`PROJECT.md` §9). It must always match what the
runtime actually accepts (`AGENTS.md` §5) — when you add an op, update this file in the
same change (the canonical "add an op" path, `AGENTS.md` §4).

As of Phase 3 these ops are **driven by the serialized IR**: each is an `op_type` variant of
the IR `OpSpec` (see [`docs/IR-SPEC.md`](./IR-SPEC.md)). The op names below are exactly the
`op_type` tags the IR accepts; the cross-language conformance test keeps this list and the
runtime's `OpSpec` enum in sync.

The op vocabulary is **the same for every backend** — a scheme never gets an op of its own.
What differs is how each op is realized and what it costs; see
[Backend support](#backend-support) below and [`docs/BACKENDS.md`](./BACKENDS.md).
Validated end-to-end models: [`docs/MODEL-ZOO.md`](./MODEL-ZOO.md).

Notation (**TFHE backend**, the reference): a value is carried as a **signed radix integer**
of `num_blocks` blocks; under the default profile each block holds `MESSAGE_BITS = 2` bits, so
the radix capacity is `num_blocks × 2` bits. `Linear`/`Conv` are *cheap* (plaintext-weight
arithmetic, no bootstrap); `Activation`/`Requant`/`Compare` are *expensive* (one programmable
bootstrap per value). Runtime ≈ number of bootstraps (`PROJECT.md` §5). The "TFHE realization"
and bit-width columns in the tables below describe this backend specifically.

## Phase 2 — the narrow waist (`Linear`, `Activation`, `Argmax`)

| Op | Covers | TFHE realization | Bit-width rule (`output_bits`) |
|---|---|---|---|
| `Linear` | dense layers, logistic/linear regression | `Σ (ciphertext × plaintext weight) + bias` — scalar-mul + adds, **no PBS** | `max(sum_bits, bias_bits) + 2`, where `sum_bits = input_bits + weight_bits + ceil(log2 N)` (N = fan-in) and `bias_bits` is the bias magnitude width; the `+2` is one carry from the bias add **and** one sign bit (`AGENTS.md` §1.3) |
| `Activation` | ReLU, tanh, GELU, leaky ReLU, hardswish, elu, hard sigmoid, sigmoid, any 1-input function | apply a lookup table via **PBS** on a narrow (≤ `MESSAGE_BITS`-bit) block | `output_bits` of the table (independent of input width; kept small to stay LUT-able) |
| `Argmax` | classification head (2-class) | threshold a single logit: `z ≥ threshold` → encrypted `0/1` (a comparison; LUT-backed) | `1` (a single class bit) |

### Notes & current limits

- **`Argmax` is the 2-class special case** (ROADMAP Phase 2): a threshold on one logit.
  Because a 2-class sigmoid/softmax is monotone, the label is a comparison and needs no
  wide-domain LUT and no `Requant`. A true `>2`-class argmax (pairwise `max`/`gt`) is a
  later phase.
- **`Activation` operates on a narrow value.** A PBS over a wide accumulator is infeasible
  (`PROJECT.md` §9); a wide accumulator must be `Requant`-ed down first (see `Requant` below).
- **Bit-width budget is checked _and_ auto-managed (as of Phase 4).** The runtime's
  `eval::check_graph_bit_width_budget` propagates per-tensor widths through the IR graph (via
  `propagate_bit_widths`) and refuses to run a model whose declared accumulator exceeds the
  radix capacity, naming the offending node (`AGENTS.md` §1.3, §1.4). The Python compile pass
  `penumbra.insert_requants` (mirroring those width rules, kept in lockstep by the bit-width
  conformance test) now **automatically inserts `Requant` nodes** between accumulator layers so
  multi-layer models stay within budget.

## Phase 4 — multi-layer CNN ops (`Add`, `Requant`, `Pool`, `Conv2d`)

| Op | Covers | TFHE realization | Bit-width rule (`output_bits`) |
|---|---|---|---|
| `Conv2d` | convolutional layers in CNNs | `Σ (ciphertext × plaintext kernel weight) + bias` at every spatial position — scalar-mul + adds, **no PBS** (the `Linear` pattern shared across positions) | `max(sum_bits, bias_bits) + 2` with fan-in `N = in_channels·kernel_h·kernel_w` (same form as `Linear`) |
| `Requant` | rescale a wide accumulator → small int (enables multi-layer models) | `clamp(((max(x, clamp_lo)·mult + round_bias) >> shift) + zero_point, 0, 2^out_bits-1)`: signed floor (`clamp_lo`, default `0` = fused ReLU) + fixed-point multiply-then-round-shift + activation domain offset (`zero_point`, default `0`) + radix-level saturate, then a single-block **PBS** (resets noise). `mult`/`round_bias` (default `1`/`0` = legacy pure shift) approximate an arbitrary scale ratio `mult/2^shift`; the multiply is a cheap plaintext scalar-mul (no extra PBS). **Per-channel (IR 0.6.0):** an optional `mults`/`shifts`/`round_biases` + `channel_size` overlay applies a distinct rescale per output channel (flat element `idx` → channel `idx/channel_size`) for per-channel weight quantization; the shared `clamp_lut`/`out_bits` are unchanged, so PBS count is identical. **Signed floor + offset (IR 0.8.0):** `clamp_lo ≤ 0` and `zero_point ≥ 0` allow signed accumulators to narrow into the single-block LUT domain with sign intact. | output: `out_bits` (≤ `MESSAGE_BITS`); internal peak `(max(pos_max·mult + round_bias, neg_mag·mult))` checked against radix capacity |
| `Pool` | average / max pooling in CNNs | per-channel window reduction over the flat map: `avg` = sum (`add_parallelized`, **no PBS**); `max` = pairwise `max` (comparison PBSs, expensive); optional symmetric `padding` (IR 0.10.0): out-of-range taps skipped — `avg` sums in-bounds taps, `max` ignores padded taps | `avg`: `input_bits + ceil(log2 k)` (k = window size); `max`: `input_bits` (selection never grows magnitude) |
| `Add` | residuals / skip connections | element-wise ciphertext addition of **two** input tensors — `add_parallelized`, **no PBS** | `max(a_bits, b_bits) + 1` (one carry; the wider operand's sign bit covers the result) |

### Notes — Phase 4

- **`Requant` is the primitive that unlocks multi-layer models** (`PROJECT.md` §9). A
  `Linear`/`Conv2d` accumulator grows ~`log2(N)` bits per layer; a PBS is feasible only over
  a narrow value, so the wide accumulator is floored at `clamp_lo` (default `0` = fused ReLU),
  rescaled by a **fixed-point multiplier** `mult/2^shift` (chosen by the quantization service to
  approximate the real scale ratio — `mult = 1` recovers the original power-of-two shift),
  round-bias-added, offset by `zero_point`, then saturated **at the radix level** so the value
  truly fits one `MESSAGE_BITS`-wide block and passed through a single-block clamp LUT. For a ReLU,
  it is a **fused ReLU+requant**: the output is non-negative (what the single-block PBS path
  requires, and what conv→ReLU produces anyway). For non-ReLU activations (IR 0.8.0), `clamp_lo < 0`
  and `zero_point > 0` narrow signed values into the LUT domain without clipping negatives. The
  `mult` multiply is a cheap plaintext scalar-mul (no extra PBS), but it widens the value before the
  shift, so the bit-width tracker enforces an **internal-peak** budget in addition to the
  output-width budget.
- **`Conv2d` and `Pool` share one spatial layout.** The flat `CtVec` is read as a
  channel-major, row-major `[channels][in_h][in_w]` tensor — element `(c, y, x)` at
  `c*in_h*in_w + y*in_w + x`. `Conv2d` produces this layout and `Pool` consumes it, so
  `Conv2d → Pool` needs no reshape. `Conv2d` weights are row-major
  `[out_channels][in_channels*kernel_h*kernel_w]` (one flattened kernel per output channel)
  and its zero padding is *virtual* (padded taps contribute nothing, no ciphertext zeros are
  materialized).
- **`Pool` `avg` emits the integer window sum.** The float layer is the true mean, and the
  `1/k` is carried in the output tensor's quantization scale. It folds into the next layer's
  weight/bias quantization, so pooling stays PBS-free. As of IR 0.10.0 `Pool` supports symmetric
  virtual `padding` (`padding < window`). `count_include_pad=0` with padding is rejected at load,
  because border windows would need a per-position divisor.
- **`Add` is the first multi-input op.** Its node carries **two** entries in `inputs`; the
  list order is the merge order (addition is commutative, so order is immaterial to the
  result, but the contract is uniform with future multi-input ops). The eval loop resolves a
  node's inputs in declared order and dispatches `Op::eval_n`; single-input ops keep working
  through the default `eval_n` (`AGENTS.md` §1.2 — the loop never special-cases an op).

## Phase 8 — tree ensembles (`Compare`)

| Op | Covers | TFHE realization | Bit-width rule (`output_bits`) |
|---|---|---|---|
| `Compare` | decision trees, random forests, XGBoost | `out[i] = (x[indices[i]] >= thresholds[i]) ? 1 : 0` via `scalar_ge_parallelized` | `1` bit regardless of input width |

### 4-stage tree ensemble lowering

Tree ensembles lower through `penumbra.adapters.from_sklearn` and `from_xgboost` without ciphertext × ciphertext multiplies:

| Stage | Node | Op | Semantics |
|---|---|---|---|
| 1 | `split_cmp` | `Compare` | $b_g = [x[\text{feature}_g] \ge T_g]$ — split evaluations across all trees |
| 2 | `leaf_score` | `Linear` | $\text{score}_l = \sum_{g \in \text{path}(l)} (\pm 1) \cdot b_g + \|\text{left}(l)\|$ — attains max $\text{depth}_l$ iff every condition on path holds |
| 3 | `leaf_sel` | `Compare` | $[\text{score}_l \ge \text{depth}_l]$ — one-hot leaf indicator |
| 4 | `logits` | `Linear` | $\sum_l V[c][l] \cdot \text{leaf\_sel}[l] + \text{bias}_c$ — class logits |

### Integer threshold formulas

Given per-feature scale $s_j$ and client-side $x_{\text{int}} = \text{clip}(\text{round}(x / s_j), 0, 2^{\text{input\_bits}} - 1)$, continuous thresholds are converted and clamped into $[0, 2^{\text{input\_bits}}]$:
- scikit-learn ($x > t$): $T = \lfloor t / s_j \rfloor + 1$
- XGBoost ($x \ge c$): $T = \lceil c / s_j \rceil$

## Phase 8 — branching graphs (`Concat`, `Split`)

| Op | Covers | TFHE realization | Bit-width rule (`output_bits`) |
|---|---|---|---|
| `Concat` | channel-axis merge in branching DAGs | ciphertext moves, **no PBS** | `max(input_bits)` |
| `Split` | channel-axis segmentation in branching DAGs | contiguous segmentation, **no PBS** | `input_bits` (preserves input width across all outputs) |

## Phase 8 — pooling (padding, global average pool)

- **Global average pooling (`GlobalAveragePool`)** lowers directly to `Pool(avg)` with a spatial window covering the full feature map (`pool_h = in_h, pool_w = in_w, stride = 1`).
- **Symmetric virtual padding (IR 0.10.0):** `Pool` supports symmetric padding `padding < min(pool_h, pool_w)`. Out-of-range taps are skipped: `avg` sums the in-bounds taps, while `max` ignores padded taps (matching $-\infty$ padding). The window must fit the padded input (`pool_h <= in_h + 2*padding`).
- **Quantization scale rule:** `Pool(avg)` integer op emits the window sum; the float layer is the true mean. The `1/k` factor ($k = \text{pool\_h} \times \text{pool\_w}$) is carried in the output tensor's quantization scale (`out_scale = in_scale / k`) and folds into downstream weight/bias quantization, keeping pooling PBS-free.
- **`Pool(max)` with padding** is implemented on the TFHE backend (`cmp_pbs_ops`); max pooling remains unsupported on CKKS.
- **Bit-width growth rule is unchanged:** at most $k$ taps are summed, so `avg` produces $\text{input\_bits} + \lceil \log_2(k) \rceil$, and `max` preserves $\text{input\_bits}$.

## Phase 8 — per-op coverage

Every Phase 8 op/feature is tracked end-to-end across the stack:

| Op / feature | Registry / entry point | TFHE | CKKS | Bit-width rule | Golden tests |
|---|---|---|---|---|---|
| `Compare` | `python/penumbra/adapters/trees.py` (`from_sklearn`, `from_xgboost`) | `crates/penumbra-tfhe/src/ops/compare.rs` | `crates/penumbra-ckks/src/ops/mod.rs` `Compare` (tree graphs rejected by the depth budget) | `1` | `runtime/tests/golden_trees.rs`, `ckks_golden_ops.rs::ckks_fhe_compare_matches_cleartext`, `ckks_unsupported_ops.rs::tree_ensemble_chained_compares_rejected_by_depth_budget` |
| `Activation` (Tanh, LeakyRelu, HardSwish, Gelu, Elu, HardSigmoid, Sigmoid) | `op_registry.py` entries | `crates/penumbra-tfhe/src/ops/activation.rs` | `crates/penumbra-ckks/src/ops/polymap.rs` `fit_activation` | table `output_bits` | `golden_tanh_mlp.rs`, `ckks_golden_tanh_mlp.rs`, `ckks_golden_ops.rs::ckks_fhe_activation_lut_matches_cleartext`, `tests/test_activation_lut.py` |
| Signed `Requant` (`clamp_lo`, `zero_point`) | `Model.quantize` | `ops/requant.rs` | `polymap.rs` `fit_requant` | `out_bits` + internal peak | `runtime/tests/golden_requant_signed.rs`, `ckks_golden_ops.rs::ckks_fhe_requant_non_identity_clamp_lut_matches_reference` |
| `Concat` | registry `Concat` | `ops/concat.rs` | `ops/mod.rs` `Concat` | `max(input_bits)` | `golden_branch_mlp.rs`, `ckks_golden_branch_mlp.rs` |
| `Split` | registry `Split` | `ops/split.rs` | `ops/mod.rs` `Split` | `input_bits` | same as `Concat` |
| Residual `Add` | registry `Add` | `ops/add.rs` | `ops/add.rs` | `max(a,b)+1` | `runtime/tests/golden_add.rs`, `golden_branch_mlp.rs`, `ckks_golden_ops.rs::ckks_fhe_add_matches_cleartext` |
| `BatchNormalization` (load-time fold) | `python/penumbra/quantization/batchnorm.py` | no runtime op | no runtime op | n/a | `tests/test_batchnorm_fold.py`, `golden_bn_cnn.rs`, `ckks_golden_bn_cnn.rs` |
| `Pool` padding + GAP | registry `MaxPool` / `AveragePool` / `GlobalAveragePool` | `ops/pool.rs` | `ops/matvec.rs` `avg_pool_matrix` (max rejected) | unchanged | `golden_pool.rs` padded tests, `golden_gap_cnn.rs`, `ckks_golden_ops.rs::ckks_fhe_padded_pool_avg_matches_cleartext`, `ckks_golden_gap_cnn.rs` |

## ONNX front door (Phase 6)

`penumbra.load_onnx("model.onnx")` is the front door: it parses an ONNX graph, **validates every
node at load time** (failing loudly, all problems at once — `AGENTS.md` §1.4), and **lowers** it
to a `penumbra.Model` that flows through the unchanged Phase-5 `quantize()`/`export()`. It emits
**nothing new at the IR layer** — every ONNX op maps onto the existing internal ops above — so
the golden invariant holds by construction (`AGENTS.md` §1.1, §1.2).

Supported ONNX opset range (ai.onnx domain): **11–22**. A model exported against an opset outside
this window is rejected at load time (re-export in range).

The table below is the authoritative **ONNX op → internal op** mapping the loader's registry
(`python/penumbra/op_registry.py`) accepts; `tests/test_supported_ops_doc.py` asserts this list
equals `op_registry.supported_onnx_ops()` exactly, so doc and validator never drift.

| ONNX op | Internal op | Attribute constraints |
|---|---|---|
| `Gemm` | `Linear` | `transA=0`; `alpha=1.0`; `beta=1.0` (loader resolves `transB`) |
| `MatMul` | `Linear` | 2-D operands; a following constant-`Add` folds into the bias |
| `Conv` | `Conv2d` | `group=1`; `dilations=[1,1]`; symmetric equal `pads`; square strides; 2-D kernel |
| `Relu` | `Activation` | must follow an accumulator (fused into its `Requant`); not terminal |
| `Tanh` | `Activation` | must follow an accumulator (Conv/Gemm/MatMul) and feed one |
| `LeakyRelu` | `Activation` | `alpha` (default 0.01); must follow an accumulator and feed one |
| `HardSwish` | `Activation` | must follow an accumulator (Conv/Gemm/MatMul) and feed one |
| `Gelu` | `Activation` | `approximate` in `{'none', 'tanh'}`; must follow an accumulator and feed one |
| `Elu` | `Activation` | `alpha` (default 1.0); must follow an accumulator and feed one |
| `HardSigmoid` | `Activation` | `alpha` (default 0.2), `beta` (default 0.5); must follow an accumulator and feed one |
| `Sigmoid` | `Activation` | none; terminal Sigmoid is dropped (argmax-invariant), mid-graph lowers to `Activation` LUT |
| `MaxPool` | `Pool` (`max`) | symmetric pads `[p,p,p,p]` with `p < kernel` (padded taps ignored); no `auto_pad`; `ceil_mode=0`; `dilations=[1,1]`; uniform 2-D kernel/stride |
| `AveragePool` | `Pool` (`avg`) | symmetric pads `[p,p,p,p]` with `p < kernel`; `count_include_pad=1` when `p > 0`; no `auto_pad`; `ceil_mode=0`; uniform 2-D kernel/stride (IR emits the window sum; `1/k` carried in the scale) |
| `GlobalAveragePool` | `Pool` (`avg`) | kernel = full spatial size (IR emits the window sum; `1/k` carried in the scale) |
| `Add` | `Add / Linear bias fold` | constant add folds into preceding accumulator bias; residual add (both activations) lowers to internal `Add` |
| `Concat` | `Concat` | `axis=1` (or resolves to 1); non-concatenated dimensions must match |
| `Split` | `Split` | `axis=1` (or resolves to 1); sizes from attribute, constant input, or equal division |
| `BatchNormalization` | `(folded into preceding Conv2d/Linear)` | `training_mode=0`; constant scale/B/mean/var; must follow a Conv/Gemm/MatMul |
| `Reshape` | dropped (layout no-op) | must not reorder the flat channel-major vector |
| `Flatten` | dropped (layout no-op) | — |
| `Transpose` | dropped (layout no-op) | `perm` must not change flat element order (else rejected) |
| `Cast` | dropped (layout no-op) | `to` must be a floating type (int/bool cast rejected) |
| `Softmax` | dropped (terminal) | must be the graph-output node; client argmaxes the wide logits |
| `LogSoftmax` | dropped (terminal) | must be the graph-output node |
| `ArgMax` | dropped (terminal) | must be the graph-output node |

### Notes — Phase 6

- **What "any ONNX model" means (bounded).** The front door accepts a model iff (1) every node
  is in the table above, (2) the graph is a directed acyclic graph (DAG) from input to output,
  (3) it quantizes acceptably, and (4) it is small enough to run in feasible FHE time
  (`PROJECT.md` §10, §16). Anything else fails loudly at `load_onnx()`.
- **Branching graphs are supported.** Residual `Add` (both operands activations), `Concat`, `Split`,
  and fan-out connections are supported in both the IR walker and the `Model.quantize` compile pass
  as of Phase 8.
- **Batch normalization is folded at load time.** Inference-time `BatchNormalization` (constant parameters)
  folds directly into preceding `Conv`/`Gemm`/`MatMul` accumulator weights and bias with zero runtime cost.
- **Terminal classifier tails are dropped, not lowered.** `Softmax`/`LogSoftmax`/`Sigmoid`/`ArgMax`
  at the graph output are argmax-invariant (Penumbra leaves logits wide and argmaxes client-side,
  `PROJECT.md` §11), so they emit no op. A *non-terminal* one is a real activation and is rejected.
- **Layout ops fold to nothing.** The runtime carries a flat channel-major vector and is
  shape-blind, so `Reshape`/`Flatten` (and an order-preserving `Transpose`) between a conv and a
  dense layer are identity on the wire. A genuinely reordering `Transpose` is rejected. A `Cast`
  to a floating type is likewise an identity on the real-valued wire (exporters routinely emit one
  at the input — skl2onnx casts the input to float) and folds away; a cast to an int/bool type
  changes the value and is rejected.
- **No new dependency, no new crypto.** `onnx>=1.16` is already a core dep; the loader is pure
  Python (NumPy + `onnx`). It does not touch `runtime/` or the IR schema.
- **Two frameworks, one front door ("train anywhere").** The same `load_onnx` lowers a **PyTorch**
  CNN (`examples/mnist/onnx_export.py` → `Conv2d → Requant → Linear`) and a **scikit-learn** linear
  classifier (`examples/mnist/sklearn_export.py`, exported with `skl2onnx` → a single `Linear`, the
  leading `Cast` folded away) with no framework-specific code — the ONNX waist is the only
  integration point. Each has a committed fixture and an FHE bit-for-bit golden gate
  (`runtime/tests/golden_onnx.rs`, `golden_sklearn.rs`). Note: `skl2onnx` lowers
  `LogisticRegression`/`MLPClassifier` to the `ai.onnx.ml` custom-op domain (`LinearClassifier`,
  `ZipMap`) with a two-output graph — outside the supported subset; a no-hidden-layer regressor
  exports as a clean single-output `ai.onnx` graph, which is the supported shape.

## Backend support

Every backend implements the vocabulary above, or **rejects an op loudly at load time**
naming the op, the node, and the backend (`AGENTS.md` §1.4). An op is never silently
approximated, and the vocabulary never forks per backend (`AGENTS.md` §1.2).

| Op | `tfhe` (reference) | `ckks` (Phase 12) | CKKS realization |
|---|---|---|---|
| `Linear` | ✅ exact, no PBS | ✅ approximate within declared bound | BSGS diagonal transform + plaintext adds; spends one level (`log_delta` bits) |
| `Conv2d` | ✅ exact, no PBS | ✅ approximate within declared bound | lowered to plaintext im2col matrix; evaluated via BSGS diagonal transform |
| `Pool` (`avg`) | ✅ exact, no PBS | ✅ approximate within declared bound | 0/1 matrix over in-bounds taps (padding skipped); BSGS diagonal transform |
| `Pool` (`max`) | ✅ exact, comparison PBS | ❌ rejected at load time | polynomial sign approximation depth exceeds level budget; use Pool(avg) or TFHE backend |
| `Add` | ✅ exact, no PBS | ✅ approximate within declared bound | native ciphertext-ciphertext add (zero depth) |
| `Activation` | ✅ exact, one PBS | ✅ approximate within declared bound | exact Chebyshev interpolating polynomial over LUT domain; zero fit error on integer inputs |
| `Requant` | ✅ exact, one PBS | ✅ approximate within declared bound | continuous polynomial ramp approximation + scale/shift; per-channel via 0/1 mask multiply; evaluates composed clamp_lut polynomial when non-identity |
| `Compare` | ✅ exact, comparison PBS | ⚠️ op implemented; chained sharp steps exceed level budget for tree graph (needs 360 bits vs 330 budget capacity) | plaintext linear map (gather - threshold) + continuous smoothed step polynomial approximation |
| `Concat` | ✅ exact, no PBS | ✅ approximate within declared bound | 0/1 selection linear map per segment, summed (spends one level) |
| `Split` | ✅ exact, no PBS | ✅ approximate within declared bound | 0/1 window linear map per segment (spends one level) |

Two notes that explain the whole column:

- **CKKS has no lookup table and no programmable bootstrap.** Every op marked *approximate*
  above is a fitted polynomial. That is the single largest semantic difference between the
  backends, and the most interesting thing the scheme comparison measures
  ([`docs/COMPARISON.md`](./COMPARISON.md)).
- **The bit-width rules in the tables above still apply**, because they describe the
  *quantized graph* rather than TFHE. What differs is the budget they are checked against:
  radix capacity for TFHE, multiplicative depth and scale precision for CKKS
  (`PROJECT.md` §9).

## Planned (later phases)

| Op | Phase | Notes |
|---|---|---|
| `>2`-class `Argmax` (in-FHE) | later | pairwise `max`/`gt` over a score vector; Phase 4 decrypts the logits and argmaxes client-side |
