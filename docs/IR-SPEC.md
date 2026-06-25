# IR Specification

The **Intermediate Representation (IR)** is Penumbra-FHE's backbone (`PROJECT.md` §7): a
serializable, directed graph of op nodes that the Python front end emits and the Rust
runtime consumes. A new use case is a new IR graph — never a backend edit (`AGENTS.md` §1.2).

This document is the authoritative schema. It is defined **in lockstep** on both sides:
[`python/penumbra/ir.py`](../python/penumbra/ir.py) ↔
[`runtime/src/ir.rs`](../runtime/src/ir.rs). Any change to the format updates *both* sides,
bumps [`SCHEMA_VERSION`](#versioning), updates the [conformance test](#conformance), and
updates this file — all in the **same change** (`AGENTS.md` §5). A schema-version bump is a
breaking change (`AGENTS.md` §8).

## Wire format

JSON, to start — human-inspectable and easy to debug. A compact binary format is a later,
profiling-driven decision (ROADMAP Phase 10) and an architectural fork to raise before
implementing (`AGENTS.md` §3.2). Do **not** add a binary format or compression before then.

## Versioning

`SCHEMA_VERSION` is a string constant hardcoded identically in `ir.py` and `ir.rs`
(currently **`"0.4.0"`**). On load, both sides check it and **fail loudly** on a mismatch
(`AGENTS.md` §1.4) — the version field is the forward-compatibility gate.

> **Forward-compat note.** Neither side uses `deny_unknown_fields` on the graph/node
> containers: an unknown *field* within a matching schema version is tolerated so a newer
> emitter can add metadata without breaking an older reader that has already matched the
> version. Strictness is enforced at the version field, not per-field.

## Schema

### `Graph` (root)

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Must equal `SCHEMA_VERSION`. |
| `num_blocks` | int | The shared radix width — the central **bit-width budget** (`PROJECT.md` §9). Every ciphertext in the model has this many `shortint` blocks; capacity is `num_blocks × MESSAGE_BITS` bits (`MESSAGE_BITS = 2` under the default profile). |
| `input_bits` | int | Declared bit-width of the encrypted model input; seeds the bit-width tracker. |
| `inputs` | [string] | Names of the graph's input tensors. |
| `outputs` | [string] | Names of the graph's output tensors. |
| `nodes` | [Node] | Op nodes, in a valid topological order (see [Node ordering](#node-ordering)). |

### `Node`

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Human-readable node label (used in error messages and `inspect`). |
| `inputs` | [string] | Tensor names this node reads, **in order**. Most ops are single-input; `Add` is multi-input (two operands) and the list order *is* the merge order. |
| `outputs` | [string] | Tensor names this node writes. (All current ops are single-output.) |
| `op` | OpSpec | The op payload — see below. |

### `OpSpec` (op payload)

The op payload is a **nested, internally-tagged object** keyed on `op_type` — the JSON the
Rust `#[serde(tag = "op_type")]` enum expects. It is deliberately *not* `serde(flatten)`ed
into the node: flatten disables `deny_unknown_fields` and has round-trip bugs with
internally-tagged enums. An unknown `op_type` fails loudly (`unknown variant 'BatchNorm',
expected one of 'Linear', 'Conv2d', 'Activation', 'Argmax', 'Requant', 'Pool', 'Add'`).

The supported ops match [`docs/SUPPORTED-OPS.md`](./SUPPORTED-OPS.md) (kept in sync, and
tested via the conformance test). The op fields mirror the runtime ops but live in IR-land
so the ops themselves stay serialization-free.

| `op_type` | Fields | Notes |
|---|---|---|
| `"Linear"` | `weights: [[int]]` (row-major `[out][in]`), `bias: [int]` (one per row), `weight_bits: int` | Dense layer / logistic-regression head. `weights.len() == bias.len()` and all rows equal width, validated at load. |
| `"Conv2d"` | `weights: [[int]]` (row-major `[out_channels][in_channels*kernel_h*kernel_w]`), `bias: [int]` (one per output channel), `weight_bits: int`, `in_h, in_w, in_channels, kernel_h, kernel_w, stride, padding: int` | 2-D convolution vs plaintext kernel. Input/output flat tensors use the channel-major, row-major layout (shared with `Pool`); zero padding is virtual. Kernel-width = fan-in and one-bias-per-channel validated at load. |
| `"Activation"` | `lut: [int]` (indexed by input value), `output_bits: int` | Single-input LUT via PBS over a narrow domain. |
| `"Argmax"` | `threshold: int` | 2-class threshold: label `1` iff `z ≥ threshold`. |
| `"Requant"` | `shift: int` (power-of-two rescale), `out_bits: int` (≤ `MESSAGE_BITS`), `clamp_lut: [int]` (`2^MESSAGE_BITS` entries, each `< 2^MESSAGE_BITS`) | Rescale a wide accumulator → narrow non-negative value: `clamp(max(x >> shift, 0), 0, 2^out_bits - 1)` (fused ReLU+requant). LUT length / range and `out_bits ≤ MESSAGE_BITS` validated at load. |
| `"Pool"` | `mode: string` (`"avg"`\|`"max"`), `in_h, in_w, channels, pool_h, pool_w, stride: int` | Spatial pooling over a flattened **channel-major, row-major** `[channels][in_h][in_w]` map. `avg` emits the window sum (the `/k` is deferred to `Requant`); `max` is pairwise max. Mode and window-fits-input validated at load. |
| `"Add"` | *(none)* | Element-wise addition of **two** input tensors (residuals). The node carries two `inputs`; the payload is the bare `{"op_type": "Add"}`. Multi-input — see [Node](#node). |

## Node ordering

Nodes are stored and evaluated **in the order they appear** in `nodes`. That order is
*trusted but validated* to be a valid topological order: each node's input tensors must
already exist (a graph input or an earlier node's output), output names must be unique (no
silent overwrite), and every declared graph output must be produced. Any violation fails
loudly, naming the offending node.

The runtime does **not** compute the order itself (no Kahn's-algorithm sort). True
topological sorting for branching graphs is deferred to Phase 8; the current models are
linear chains where the emitted order *is* the evaluation order. This keeps the eval loop
minimal while the format already supports the general `inputs`/`outputs` edge shape.

## Worked example — the Phase-2 model

`Linear → Argmax`, a 2-class logistic-regression classifier. Input tensor `x`, the dense
layer produces `logit`, the threshold produces the output `label`:

```json
{
  "schema_version": "0.4.0",
  "num_blocks": 8,
  "input_bits": 4,
  "inputs": ["x"],
  "outputs": ["label"],
  "nodes": [
    {
      "name": "fc",
      "inputs": ["x"],
      "outputs": ["logit"],
      "op": { "op_type": "Linear", "weights": [[7, 7, "..."]], "bias": [-1478], "weight_bits": 4 }
    },
    {
      "name": "head",
      "inputs": ["logit"],
      "outputs": ["label"],
      "op": { "op_type": "Argmax", "threshold": 0 }
    }
  ]
}
```

The committed [`examples/mnist/phase2_fixture.json`](../examples/mnist/phase2_fixture.json)
embeds this graph under a top-level `"graph"` key, alongside sibling **test metadata**
(`test_inputs`, `expected_labels`, `scales`, `accuracy`, and a standalone `activation` LUT).
Test vectors and quantization provenance are *not* part of the portable IR — they belong to
the fixture, not the model.

Inspect any model or fixture with the debug command:

```bash
cd runtime && cargo run --bin inspect ../examples/mnist/phase2_fixture.json
# prints each node, its inputs→outputs, the propagated per-tensor bit-widths,
# and whether the model fits the radix capacity.
```

## Conformance

The cross-language conformance test keeps the two definitions honest (`AGENTS.md` §5). The
two CI jobs run in parallel and never invoke each other, so the **committed IR file is the
meeting point**: Python emits → committed fixture → Rust consumes.

- **Python** ([`tests/test_ir_conformance.py`](../tests/test_ir_conformance.py)) asserts the
  IR round-trips (`from_json(to_json(g)) == g`) and that the committed `graph` is exactly
  what `ir.py` emits today (the drift guard).
- **Rust** ([`runtime/tests/ir_conformance.rs`](../runtime/tests/ir_conformance.rs))
  deserializes the committed `graph` into the typed `Graph`, asserting the schema version
  and the expected `Linear → Argmax` structure.

If you change the IR, regenerate the fixture
(`cd python && uv run python ../examples/mnist/train_quantize_export.py`) so the committed
artifact stays current — the drift guard fails otherwise.
