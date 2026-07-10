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
| `Requant` | rescale a wide accumulator → small int (enables multi-layer models) | `clamp((max(x,0)·mult + round_bias) >> shift, 0, 2^out_bits-1)`: ReLU + fixed-point multiply-then-round-shift + radix-level saturate, then a single-block **PBS** (resets noise). `mult`/`round_bias` (default `1`/`0` = legacy pure shift) approximate an arbitrary scale ratio `mult/2^shift`; the multiply is a cheap plaintext scalar-mul (no extra PBS). **Per-channel (IR 0.6.0):** an optional `mults`/`shifts`/`round_biases` + `channel_size` overlay applies a distinct rescale per output channel (flat element `idx` → channel `idx/channel_size`) for per-channel weight quantization; the shared `clamp_lut`/`out_bits` are unchanged, so PBS count is identical. | output: `out_bits` (≤ `MESSAGE_BITS`). **internal peak**: `max(x,0)·mult + round_bias` (max over channels) must also fit the radix (checked separately) — a too-large multiplier overflows mid-op even when input and output fit. |
| `Pool` | average / max pooling in CNNs | per-channel window reduction over the flat map: `avg` = sum (`add_parallelized`, **no PBS**); `max` = pairwise `max` (comparison PBSs, expensive) | `avg`: `input_bits + ceil(log2 k)` (k = window size); `max`: `input_bits` (selection never grows magnitude) |
| `Add` | residuals / skip connections | element-wise ciphertext addition of **two** input tensors — `add_parallelized`, **no PBS** | `max(a_bits, b_bits) + 1` (one carry; the wider operand's sign bit covers the result) |

### Notes — Phase 4

- **`Requant` is the primitive that unlocks multi-layer models** (`PROJECT.md` §9). A
  `Linear`/`Conv2d` accumulator grows ~`log2(N)` bits per layer; a PBS is feasible only over
  a narrow value, so the wide accumulator is ReLU'd, rescaled by a **fixed-point multiplier**
  `mult/2^shift` (chosen by the quantization service to approximate the real scale ratio —
  `mult = 1` recovers the original power-of-two shift), round-bias-added, then saturated **at
  the radix level** so the value truly fits one `MESSAGE_BITS`-wide block and passed through a
  single-block clamp LUT. It is a **fused ReLU+requant**: the output is non-negative (what the
  single-block PBS path requires, and what conv→ReLU produces anyway). The `mult` multiply is a
  cheap plaintext scalar-mul (no extra PBS), but it widens the value before the shift, so the
  bit-width tracker enforces an **internal-peak** budget (`max(x,0)·mult + round_bias` must fit
  the radix) in addition to the output-width budget.
- **`Conv2d` and `Pool` share one spatial layout.** The flat `CtVec` is read as a
  channel-major, row-major `[channels][in_h][in_w]` tensor — element `(c, y, x)` at
  `c*in_h*in_w + y*in_w + x`. `Conv2d` produces this layout and `Pool` consumes it, so
  `Conv2d → Pool` needs no reshape. `Conv2d` weights are row-major
  `[out_channels][in_channels*kernel_h*kernel_w]` (one flattened kernel per output channel)
  and its zero padding is *virtual* (padded taps contribute nothing, no ciphertext zeros are
  materialized).
- **`Pool` `avg` mode emits the window sum** and leaves the `1/k` to the next `Requant`'s
  shift, keeping pooling PBS-free; the headline CNN uses `avg`. `Pool` has no padding in
  Phase 4.
- **`Add` is the first multi-input op.** Its node carries **two** entries in `inputs`; the
  list order is the merge order (addition is commutative, so order is immaterial to the
  result, but the contract is uniform with future multi-input ops). The eval loop resolves a
  node's inputs in declared order and dispatches `Op::eval_n`; single-input ops keep working
  through the default `eval_n` (`AGENTS.md` §1.2 — the loop never special-cases an op).

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
| `MaxPool` | `Pool` (`max`) | `pads=0`; `ceil_mode=0`; uniform 2-D kernel/stride |
| `AveragePool` | `Pool` (`avg`) | `pads=0`; `ceil_mode=0`; uniform 2-D kernel/stride (emits window **sum**) |
| `GlobalAveragePool` | `Pool` (`avg`) | kernel = full spatial size (emits window **sum**) |
| `Add` | `Linear` bias | fold-only: one operand must be a constant initializer (residual `Add` is Phase 8) |
| `Reshape` | dropped (layout no-op) | must not reorder the flat channel-major vector |
| `Flatten` | dropped (layout no-op) | — |
| `Transpose` | dropped (layout no-op) | `perm` must not change flat element order (else rejected) |
| `Softmax` | dropped (terminal) | must be the graph-output node; client argmaxes the wide logits |
| `LogSoftmax` | dropped (terminal) | must be the graph-output node |
| `Sigmoid` | dropped (terminal) | must be the graph-output node |
| `ArgMax` | dropped (terminal) | must be the graph-output node |

### Notes — Phase 6

- **What "any ONNX model" means (bounded).** The front door accepts a model iff (1) every node
  is in the table above, (2) the graph is a **single linear chain** from input to output (a chain
  of `Conv`/`Gemm`/`MatMul` accumulators, each optionally ReLU'd, plus pooling, ending in a wide
  logit head), (3) it quantizes acceptably, and (4) it is small enough to run in feasible FHE time
  (`PROJECT.md` §10, §16). Anything else fails loudly at `load_onnx()`.
- **Branching is Phase 8.** Residual `Add` (both operands activations), `Concat`, and any fan-out
  are rejected — the Rust eval loop already walks DAGs, but `Model.quantize`'s compile pass is
  linear-chain-only, and widening it is Phase-8 work.
- **Terminal classifier tails are dropped, not lowered.** `Softmax`/`LogSoftmax`/`Sigmoid`/`ArgMax`
  at the graph output are argmax-invariant (Penumbra leaves logits wide and argmaxes client-side,
  `PROJECT.md` §11), so they emit no op. A *non-terminal* one is a real activation and is rejected.
- **Layout ops fold to nothing.** The runtime carries a flat channel-major vector and is
  shape-blind, so `Reshape`/`Flatten` (and an order-preserving `Transpose`) between a conv and a
  dense layer are identity on the wire. A genuinely reordering `Transpose` is rejected.
- **No new dependency, no new crypto.** `onnx>=1.16` is already a core dep; the loader is pure
  Python (NumPy + `onnx`). It does not touch `runtime/` or the IR schema.

## Planned (later phases)

| Op | Phase | Notes |
|---|---|---|
| `Concat` / branching | 8 | multi-input graphs; true topological eval |
| `BatchNormalization` | 8 | fold into the preceding Conv/Linear at quantize time |
| non-ReLU activation LUTs (`Tanh`, `Sigmoid` mid-graph, GELU, …) | 8 | single-input LUTs; un-stub the standalone-`Activation` path |
| `>2`-class `Argmax` (in-FHE) | later | pairwise `max`/`gt` over a score vector; Phase 4 decrypts the logits and argmaxes client-side |
