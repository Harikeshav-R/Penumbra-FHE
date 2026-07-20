# Faces — the abstraction-validation example

A second, different use case that runs **with zero edits to the Rust backend** (Layers 1–2).
This is the proof that the narrow waist holds (`PROJECT.md` §4, `ROADMAP.md` Phase 7): adding
face recognition is adding a *graph*, never adding *crypto*.

- **Task:** **closed-set face recognition** — "is this one of the N enrolled people?" — over the
  first 8 identities of the Olivetti faces dataset (AT&T "Database of Faces": 40 people, 10
  grayscale 64×64 images each). A fixed-output softmax head, very FHE-friendly. Open-set
  embedding + distance matching is a documented stretch goal, not the starting point
  (`PROJECT.md` §11).
- **The validation:** this runs through the *existing* `load_onnx → quantize → IR → encrypted
  inference` pipeline. The committed IR graph is `Conv2d → Requant(fused ReLU) → Linear` — the
  **same op vocabulary** the digit CNN lowers to, with no new IR op. The git diff that added this
  example touches **no `runtime/src/ops/` and no `eval.rs`** (`AGENTS.md` §1.2). If it had, the
  abstraction leaked — the fix would be a more general op, not a use-case hack.

## The model

```
downsample 64×64 → 16×16     (4×4 block-mean — NumPy preprocessing, NOT a graph op)
Conv2d(1 → 8, 3×3, stride 4)  →  ReLU  →  Linear(128 → 8 identity logits)
```

Exported from PyTorch as `Conv → Relu → Flatten → Gemm`; `load_onnx` folds the `Flatten`,
resolves the `Gemm`'s `transB`/bias, and lowers it to `[Conv2d, Activation, Linear]`. The 8
logits are the graph output; the client decrypts and argmaxes them (`PROJECT.md` §11).

The 16×16 downsample is deliberate: FHE cost ≈ number of bootstraps, here
`CONV_CH × OUT_H × OUT_W = 8 × 4 × 4 = 128` requant PBS/sample — comparable to the digit CNN's
108, so a committed golden sample stays feasible (minutes, not hours). This example contains
**no cryptography** — only a model graph and quantized weights; the crypto lives entirely in the
`runtime/` crate.

## Results (honest, not headline)

| Metric | Value |
|---|---|
| Float accuracy | 0.95 |
| Quantized accuracy | 0.90 (gap 0.05) |
| Radix | 11 blocks (22-bit signed) |
| Bootstraps / sample | 128 |

Full numbers and methodology are in [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md). The gap is
the cost of capping activations at a single 2-bit block (`MESSAGE_BITS`, the hard backend limit)
on an 8-way task from tiny 16×16 inputs.

## Regenerating

The `.onnx` model and the fixture are **committed**, so CI never trains or downloads anything —
it reads only the JSON (the hermetic-fixture discipline). Regenerate only when the example
changes. Torch + scikit-learn are the optional `ml` extra; **Olivetti downloads once (~4 MB)** to
`~/scikit_learn_data` and is then cached (unlike `load_digits`, which ships with scikit-learn):

```bash
cd python && uv run --extra ml --system-certs python ../examples/faces/olivetti_export.py
```

Run the tests:

```bash
# Fast hermetic guard (every CI run; core onnx dep, no torch):
cd python && uv run pytest ../tests/test_faces_fixture.py

# Inspect the per-tensor bit-widths without running FHE:
cd runtime && cargo run --release --bin inspect ../examples/faces/phase7_faces_fixture.json

# The FHE bit-for-bit gate (minutes/sample; #[ignore]d, run explicitly):
cd runtime && cargo test --release --test golden_faces -- --ignored --nocapture
```
