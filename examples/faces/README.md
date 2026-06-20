# Faces — the abstraction-validation example

A second, different use case that must run **with zero edits to the Rust backend**
(Layers 1–2). This is the proof that the narrow waist holds (`PROJECT.md` §4,
ROADMAP.md Phase 7).

- Target: **closed-set face classification** (is this one of N enrolled people?) — a
  fixed-output small CNN, very FHE-friendly. Open-set embedding + distance matching is a
  documented stretch goal, not the starting point (`PROJECT.md` §11).
- **The validation:** this must run through the *existing* `load_onnx → quantize → IR →
  encrypted inference` pipeline. If it requires editing `runtime/src/ops/` or `eval.rs`,
  the abstraction leaked — fix the abstraction, not the use case (`AGENTS.md` §1.2).

_Scaffold placeholder; filled in at Phase 7._
