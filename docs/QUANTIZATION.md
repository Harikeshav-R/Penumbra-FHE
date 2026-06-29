# Quantization

TFHE computes on **small integers**, so Penumbra-FHE must turn a float model into a low-bit
integer model — weights, activations, accumulators — before it can run encrypted. This is
"~80% of the engineering effort and where ML accuracy lives or dies" (`PROJECT.md` §8). The
library owns it as a **service**: you supply a trained model plus calibration data, and the
library produces the int weights, scales, lookup tables, and the IR graph the runtime walks —
**you never compute a scale by hand** (`PROJECT.md` §12).

> **The golden invariant.** TFHE is *exact*. The FHE output equals the **quantized-cleartext**
> output **bit-for-bit** (`AGENTS.md` §1.1). Any discrepancy is a quantization or implementation
> bug, never crypto noise. So there is no separate "FHE accuracy": once a model is quantized, its
> FHE accuracy *is* its quantized-cleartext accuracy, and the golden tests guarantee it.

## The one-call path

```python
import penumbra as fhe

model = fhe.Model([
    fhe.Conv2d(weight=w1, in_h=28, in_w=28, in_channels=1),
    fhe.Activation(lambda x: max(x, 0.0)),      # ReLU
    fhe.Pool("avg", in_h=26, in_w=26, channels=8, pool_h=2, pool_w=2, stride=2),
    fhe.Linear(weight=w2, bias=b2),
])
model.quantize(calibration_data, n_bits=4)       # float graph -> int IR, no manual scales
model.export("model.fhe")                         # serialize for the runtime
```

`quantize` runs the whole pipeline:

1. **Calibrate** — run `calibration_data` through the float layers, observing each accumulator's
   output range so the rescale targets the *typical* magnitude, not the worst case.
2. **Quantize** each layer's weights/bias (symmetric, signed) into the int IR.
3. **Fuse activations into Requant** — a `Conv2d`/`Linear` followed by a ReLU `Activation` becomes
   `accumulator → Requant(ReLU + rescale)`; the Requant is a *fused ReLU+rescale*, so the ReLU
   costs no extra op.
4. **Insert Requants + size the radix** — splice the requants, then search the smallest
   `num_blocks` that fits every tensor *and* every Requant's internal multiply peak.
5. **Self-verify** — run the quantized-integer reference (the golden oracle) on calibration
   samples to confirm the graph actually evaluates; a scale/wiring bug fails *here*, inside
   `quantize`, with an actionable message — not as a confusing Rust golden violation later.

## PTQ vs QAT

**Post-Training Quantization (PTQ)** quantizes an already-trained float model using calibration
data to choose scales. It is the default and the easiest path; it works well when the model
tolerates low-bit integers. This is what `Model.quantize` does.

**Quantization-Aware Training (QAT)** trains *with* quantization simulated in the loop, so the
model learns weights that survive low-bit rounding. It recovers most of the accuracy PTQ loses on
harder models. Penumbra wraps **Brevitas** (`penumbra.quantization.qat`) rather than shipping its
own quantizer; the QAT path exports to the **same int-graph form** PTQ produces, so the runtime
and the golden invariant are identical either way. Brevitas (and PyTorch) are an **optional
dependency** (`pip install penumbra-fhe[ml]`); the core PTQ path needs only NumPy. See
`examples/mnist/` for both a PTQ and a QAT example.

## Choosing `n_bits`

`n_bits` is the working integer width for weights and inputs. Smaller is faster and cheaper but
loses accuracy; the bit-width budget (below) caps how large it can be.

- **Start at `n_bits=4`** for weights/inputs. It is the project's default and fits the small-model
  target.
- **Activations are narrower still** (`act_bits`, default 2): a post-Requant activation must fit a
  *single radix block* (`≤ MESSAGE_BITS = 2` bits), because a programmable bootstrap is only
  feasible over a narrow value. This is the central reason multi-layer models need requantization.
- If accuracy is unacceptable, raise `n_bits` (watch the budget), enable **per-channel** weight
  scales, or move to **QAT** — in that order of effort.

Use the **accuracy + SQNR harness** to decide:

```python
from penumbra.quantization import accuracy_report, layer_sqnr_report

print(accuracy_report(float_predict, quant_predict, x_test, y_test))   # float vs quantized + gap
print(layer_sqnr_report(float_layer_outputs, quant_layer_outputs))     # per-layer dB; low = worst
```

The lowest-SQNR layers are the ones worth more bits or per-channel scales.

## Per-tensor vs per-channel scales

Per-tensor uses **one scale per weight tensor**; per-channel uses **one scale per output channel**
(`per_channel=True`). Per-channel keeps a small-magnitude output channel from being crushed by a
large-magnitude one — the main accuracy lever per-tensor leaves on the table — and it stays
entirely within the existing backend (each output row carries its own integer scale, folded into
the int weights/bias and a single shared Requant). **Weights are always symmetric** (zero-point
free); start per-tensor and turn on per-channel for weights if a layer's SQNR is low.

## The bit-width budget (why requantization exists)

A value is carried as a **signed radix integer** of `num_blocks` blocks; each block holds
`MESSAGE_BITS = 2` bits, so the radix holds `num_blocks × 2` bits (`PROJECT.md` §9). A
`Linear`/`Conv2d` summing `N` products of `b`-bit values produces an accumulator needing
~`b + log2(N)` bits — it **grows every layer**. A bootstrap (the only way to apply an activation)
is feasible only over a narrow single-block value, so between accumulator layers the wide value
must be **requantized back down**. The library does this automatically (`penumbra.insert_requants`)
and **errors loudly, naming the layer**, if a model cannot fit its radix.

## Symmetric vs asymmetric (why activations stay symmetric here)

All quantization in Penumbra is **symmetric** (zero-point free). Asymmetric quantization adds a
zero-point `z` so a value is `s·(q − z)`, which lets a quantizer capture a one-sided range without
wasting codes — most useful for post-ReLU activations, which are non-negative.

It is deliberately **not** implemented, for an evidence-backed reason specific to this design:
activations here are produced by the **fused-ReLU Requant**, whose output is already a
*non-negative* value saturated into the full `[0, 2^act_bits − 1]` block (measured on the
real-digit example: the post-Requant values span exactly `0..3` and pile up at `0`). A zero-point
cannot expand a range that is already maximal and already starts at zero, so it would recover
essentially no accuracy — the loss is the coarse 4-level (2-bit) resolution, which a zero-point
does not change. Adding asymmetric activations would also fold a `−z·Σw` correction into each
downstream `Linear`/`Conv` bias, a cross-term that is easy to get subtly wrong and would risk the
golden invariant for ~no gain. It becomes worthwhile only alongside a *non-ReLU / signed*
activation path (which does not exist yet); it is deferred until then. **Weights** are symmetric
unconditionally (the standard choice; asymmetric weights have no benefit).

## The Requant rescale (multiply-then-round-shift)

`Requant` realizes the rescale as a **fixed-point multiplier**:

```
requant(x) = clamp( (max(x, 0) * mult + round_bias) >> shift, 0, 2^out_bits - 1 )
```

An arbitrary real scale ratio `M = (s_in · s_w) / s_out` is approximated by `mult / 2^shift` with
round-to-nearest (`round_bias`). This is **bit-exact in both the FHE and cleartext domains** —
every step (ReLU, multiply, add, arithmetic shift, clamp) has an exact integer counterpart, so the
golden invariant holds — and the multiply is a cheap plaintext scalar-mul, so the PBS count is
unchanged. `mult = 1, round_bias = 0` is the legacy pure power-of-two shift.

Two consequences worth knowing:

- The multiplier **widens** the value to `max(x,0)·mult + round_bias` *before* the shift narrows
  it. That transient peak must still fit the radix, so the bit-width tracker enforces an
  **internal-peak budget** on top of the output-width budget. `choose_requant_params` caps the
  multiplier width (`max_mult_bits`, default 5) so this stays in budget; an over-large multiplier
  fails loudly.
- The Requant is a **fused ReLU+rescale**: its output is non-negative (what the single-block PBS
  path requires, and what conv→ReLU produces anyway).

## Generating LUTs in the integer domain

A non-ReLU activation (sigmoid, GELU, …) is realized as a **lookup table** applied by a bootstrap.
The library generates it (`penumbra.quantization.make_activation_lut`) in the **quantized-integer
domain** consistent with the chosen scales: for each integer input value `v`, dequantize
(`x = v · in_scale`), apply the float function, requantize (`q = round(fn(x) / out_scale)`), clamp
to one block. Getting the scales right here is the whole game — an off-by-scale silently wrecks
accuracy (`PROJECT.md` §8). The generator validates the table against the backend's hard
constraints (full message-space coverage, every entry `< 2^MESSAGE_BITS`) so a bad table fails in
Python *before* it reaches a PBS, not as wrong-but-confident ciphertext.

## The accuracy/speed tradeoff, honestly

- **Speed ≈ number of bootstraps.** Linear/Conv with plaintext weights are cheap; activations and
  requants are bootstraps and dominate runtime. Fewer/narrower requants → faster.
- **Smaller `n_bits` and smaller `num_blocks`** are faster but lose accuracy; the budget caps how
  small you can go without overflow.
- **Latency is seconds-to-minutes per inference** — research/prototype territory, not real-time
  serving (`PROJECT.md` §16). See `docs/BENCHMARKS.md` for measured numbers.

## Verification invariant (the safety net)

Always: **FHE output == quantized-cleartext output, bit-for-bit.** The quantization service must
never break it. Three layers enforce this:

1. `Model.quantize` **self-verifies** by running the integer oracle on calibration samples.
2. The example fixtures commit the integer oracle's `expected_logits`/`expected_labels`, guarded by
   fast NumPy tests.
3. The Rust **golden tests** run the actual FHE forward pass and assert it equals those committed
   integers (`cargo test --release`).

If FHE ever disagrees with the quantized cleartext, debug the **cleartext quantized path first** —
it is almost always an indexing, scale, or bit-width bug, never the crypto.
