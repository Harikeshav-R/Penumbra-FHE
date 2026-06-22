# AGENTS.md — Guidelines for AI Agents Working on Penumbra-FHE

> This file governs how AI coding agents work in this repository. Read it before
> touching anything. It encodes the project owner's working preferences and the
> non-negotiable invariants of the architecture.
>
> **Required reading, in order:** [`PROJECT.md`](./PROJECT.md) (architecture & rationale),
> [`ROADMAP.md`](./ROADMAP.md) (task-level build plan). This file does not repeat them —
> it tells you how to behave while executing them.

---

## 0. TL;DR — the rules that matter most

1. **Plan before you build.** For any non-trivial task, present a detailed plan and
   **wait for approval** before writing code. Work in a tight loop — check in at each
   meaningful step. (§3)
2. **The golden invariant is sacred and non-negotiable.** FHE output == quantized-cleartext
   output, **bit-for-bit**. Every op/feature touching eval ships with a passing golden test
   in the same change. Never weaken or skip it — even if asked. (§1, §4)
3. **New use case ⇒ new graph, never new crypto.** If a task forces edits to Layer-1
   (`runtime/src/ops/`, `eval.rs`) to support a new model, the abstraction leaked. **Stop
   and flag it** — fix the abstraction, not the use case. This is a hard gate. (§1)
4. **IR changes touch both sides + bump version, same change.** `python/penumbra/ir.py` ↔
   `runtime/src/ir.rs` stay in lockstep with the conformance test and schema-version field.
   Hard rule. (§5)
5. **Architectural forks: present options, wait.** Don't silently pick a design for genuine
   forks (IR LUT representation, per-tensor vs per-channel scales, topo-sort strategy).
   Surface tradeoffs + a recommendation, then wait. (§3)
6. **Test-first. Format + lint clean before "done."** rustfmt + clippy, ruff + black; treat
   warnings as errors. (§6)

---

## 1. Non-negotiable invariants (hard gates)

These are inviolable. If a request would break one, **do not proceed silently** — explain
the conflict and stop for the owner's decision. "The user asked" does not override these;
flag the conflict first.

### 1.1 The golden exactness invariant
> **FHE output must equal the quantized-cleartext output, bit-for-bit.**

TFHE is *exact*. Any discrepancy between FHE and the quantized-cleartext reference is a
**quantization or implementation bug — never crypto noise**. This is your truth oracle.

- Every model, every op, every phase must satisfy it.
- Wire it into CI; never let it regress.
- When FHE ≠ cleartext, debug the **cleartext quantized path first** — it is almost always
  an indexing, scale, or bit-width bug, not the crypto.

### 1.2 The narrow-waist discipline
> **A new use case only ever adds a Layer-3 adapter (or just a new ONNX file). It never
> touches Layers 1–2.**

- Layer 1 (TFHE backend, `runtime/src/ops/`, `eval.rs`, `keys.rs`, `encrypt.rs`) and
  Layer 2 (IR + op registry) are written once and stay stable across use cases.
- Litmus test: *if adding face recognition forces a crypto-backend edit, the abstraction
  leaked.* The correct fix is a **more general op** or a **missing registry entry**, never
  a use-case-specific hack in the backend.
- If you believe a backend edit is genuinely required (e.g. a legitimately new primitive),
  treat it as an architectural fork (§3) and escalate — don't just do it.

### 1.3 Bit-width budget is enforced centrally
- Each op declares how it grows bit-width; the library inserts `Requant` automatically and
  **errors/warns loudly** when precision exceeds the LUT/PBS budget, naming the offending
  layer.
- Keep activation bit-widths small (≤ 6–8 bits). Accumulator overflow is the #1 bug in
  multi-layer models — the bit-width tracker must be correct.

### 1.4 Fail loudly, early
- Unsupported ops and infeasible bit-widths are caught at **compile/load time** with
  **actionable messages** (e.g. `"operator X (node 'name') not supported"`), never
  mysteriously at runtime. List *all* problems at once where feasible, not one at a time.

---

## 2. Project context & intent

- **What this is:** a serious, releasable, Apache-2.0 open-source library for encrypted ML
  inference on ONNX models via `tfhe-rs`. Hold to a production-ish quality bar *within* the
  stated research/prototype scope (latency is seconds-per-inference; this is not real-time
  serving — set expectations honestly, don't over-promise).
- **What this is NOT:** a compiler, a general FHE framework, an LLM tool, or a supporter of
  arbitrary ONNX graphs. See `PROJECT.md` §1, §16.
- **Owner profile:** strong software engineer (Rust/Python/ML), **new to FHE**. Explain
  crypto-specific reasoning (TFHE, PBS, LUTs, parameter choices) where it informs a
  decision; don't over-explain general engineering. Teach the crypto as you go.

---

## 3. How to work (process)

### 3.1 Plan first, tight loop
- For **any non-trivial task**, produce a **detailed implementation plan** and **wait for
  approval before writing code**. The plan should name the files you'll touch, the op/IR
  changes, the tests you'll add, and how the golden invariant is preserved.
- Work in a **tight loop**: prefer frequent check-ins at each meaningful step over large
  autonomous runs. Confirm direction before moving to the next substantial step.
- Small, obvious, reversible changes (a typo, a doc tweak, a one-line fix) don't need a
  formal plan — just do them and report.

### 3.2 Architectural forks: present options, wait
When you hit a genuine design fork, **do not silently pick one**. Surface:
- the options, with concise tradeoffs,
- your recommended option and why,

then **wait for the decision**. Canonical examples: how to represent LUTs in the IR,
per-tensor vs per-channel scales, eval-loop topological-ordering strategy, JSON vs binary
IR format, PyO3 boundary design. If `PROJECT.md`/`ROADMAP.md` already imply an answer,
follow it and say so; only escalate genuinely novel forks.

### 3.3 Reporting style
- Give **detailed walkthroughs** when reporting work: what changed, *why*, the tradeoffs
  considered, how the invariants were preserved, and clear next steps.
- Reference code as `file_path:line` so it's clickable.

### 3.4 Roadmap as guide (not a rigid gate)
- Follow the **spirit and dependency graph** of `ROADMAP.md` (P0→P11). Respect that later
  phases build on earlier foundations.
- Reasonable out-of-order work is fine **when it unblocks progress** — but call it out, and
  never claim a phase's exit criteria are met when they aren't.
- Each phase should still end in a **working, demoable artifact** — favor end-to-end slices
  over half-finished horizontal layers.

---

## 4. Testing (test-first, golden always)

- **Test-first.** Every new op or feature ships with its tests **in the same change**. No
  feature is "done" without them.
- **Golden test required** for anything touching eval (ops, IR, eval loop, quantization):
  the FHE-vs-quantized-cleartext exactness test must run and pass before work is complete.
- **The canonical extension path for a new op** (also the CONTRIBUTING story):
  1. registry entry (ONNX → internal op mapping),
  2. Rust implementation against `tfhe-rs` primitives,
  3. bit-width growth rule,
  4. golden test asserting FHE == quantized-cleartext,
  5. docs update (`docs/SUPPORTED-OPS.md`).
- Maintain the **cross-language IR conformance test** (Python emits IR → Rust loads → assert
  agreement) whenever the IR changes.
- Build the Rust runtime in **`--release`** for any timing/latency work — debug FHE is
  misleadingly slow. Correctness checks can run in debug.
- Property/fuzz idea to lean on: random small models → assert FHE == quantized-cleartext.

---

## 5. IR & cross-language sync (hard rule)

The IR is the product's backbone and spans two languages. Treat drift as a defect.

- Any change to the IR **must, in the same change**:
  1. update **both** `python/penumbra/ir.py` and `runtime/src/ir.rs`,
  2. **bump the schema-version field**,
  3. update/extend the **conformance test**,
  4. update `docs/IR-SPEC.md`.
- Any new supported op **must** update `docs/SUPPORTED-OPS.md` so the documented list always
  matches what the validator actually accepts (this is itself testable — keep it true).
- Wire format is **JSON first** (human-inspectable, easy to debug). Do not introduce a binary
  format before it's warranted by profiling (Phase 10) — and that's an architectural fork (§3).

---

## 6. Code style & quality

- **Before declaring work done**, run and fix:
  - **Rust:** `cargo fmt` + `cargo clippy` — treat clippy warnings as errors.
  - **Python:** `ruff` + `black` — treat ruff warnings as errors.
- Match the surrounding code's idiom, naming, and comment density. Write comments that
  explain *why* (especially crypto/bit-width reasoning), not *what*.
- Keep the public API surface clean and small (see `PROJECT.md` §12): ship a secure default
  crypto-param profile; expose **one** override knob; never make users choose raw `tfhe-rs`
  parameters or compute scales by hand. Quantization is a **library service**, not the
  user's problem.

---

## 7. Dependencies & tooling

- **Named stack is preferred.** Default to the tools the docs specify:
  - Python: **`uv`** for env/deps (project standard — **not** poetry/pip/conda), `onnx`,
    `numpy`, `brevitas` (don't write your own quantizer), `pytest`, PyTorch/sklearn/XGBoost
    for examples.
  - Rust: **`tfhe-rs`** (high-level `integer`/`shortint` API), `serde`/`serde_json`, `PyO3`
    (for bindings, Phase 9), `rayon` (parallelism, Phase 10).
- Small, uncontroversial dependencies are OK without asking, but **flag anything notable**.
  Swapping out a named tool requires approval.
- Start from the `tfhe-rs` **default secure parameter profile**. Don't hand-roll crypto
  parameters; tuning within a fixed security level is a later, opt-in optimization (Phase 10)
  — and never trade away security for speed.

---

## 8. Git & version control

- **Work on a feature branch.** Never commit directly to `main` and never push without being
  asked. Commit logical units of work to the branch as you go.
- **Branch names follow Conventional Branch format:** `<type>/<short-kebab-description>`,
  where `<type>` matches the Conventional Commits types below — e.g. `feat/conv2d-op`,
  `fix/accumulator-overflow`, `chore/scaffold-repo`, `docs/ir-spec`.
- **Commit messages follow [Conventional Commits](https://www.conventionalcommits.org):**
  `<type>(<optional-scope>): <imperative description>`. Keep each commit scoped to one
  logical change.
  - **Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`.
  - **Suggested scopes** for this repo: `runtime`, `python`, `ir`, `ops`, `quant`, `onnx`,
    `ci`, `docs`, `examples`.
  - Breaking changes: append `!` after the type/scope (e.g. `feat(ir)!:`) and add a
    `BREAKING CHANGE:` footer. **An IR schema-version bump (§5) is a breaking change** —
    mark it as such.
  - Examples: `feat(ops): add Conv2d against plaintext weights`,
    `fix(quant): correct per-channel scale indexing`, `chore(ci): add clippy to PR gate`.
- **Never add AI/agent authorship attribution anywhere** — no `Co-Authored-By`, no
  "Generated with…", no agent signatures or credits in commit messages, PR bodies, file
  headers, code comments, or docs. Commits and content are authored as the project owner.
- Don't commit generated artifacts: Rust `target/`, Python `__pycache__/`/venvs, ONNX
  artifacts, `*.fhe` files. Keep `.gitignore` honest.
- Confirm before any irreversible or outward-facing action (force-push, history rewrite,
  publishing a package, deleting files you didn't create).

---

## 9. Repository map (where things go)

```
penumbra-fhe/
├── PROJECT.md / ROADMAP.md / AGENTS.md   # architecture / plan / these rules
├── python/penumbra/                      # Layer 3 + quantization + ONNX loader + IR emitter
│   ├── onnx_loader.py · op_registry.py · ir.py
│   ├── quantization/ · adapters/ · client.py
├── runtime/src/                          # Layer 1 + 2: tfhe-rs backend (DO NOT edit per use case)
│   ├── keys.rs · ir.rs · ops/ · eval.rs · encrypt.rs
├── examples/{mnist,faces}/               # use cases — graphs only, no crypto
├── tests/                                # golden exactness, conformance, unsupported-op
└── docs/                                 # DEVELOPMENT, IR-SPEC, SUPPORTED-OPS, QUANTIZATION, ...
```

- **`runtime/src/ops/` and `eval.rs` are the protected zone** (§1.2). Edits there for a new
  *use case* are a red flag. Edits there to add a *genuinely new primitive op* are legitimate
  but go through the §4 extension path + §3 fork review.
- Layer-3 work (new models/use cases) lives in `python/penumbra/adapters/`, `examples/`, and
  registry entries — not in the backend.

---

## 10. Quick self-check before you say "done"

- [ ] Did I plan and get approval for non-trivial work? (§3.1)
- [ ] Does the golden test (FHE == quantized-cleartext, bit-for-bit) pass? (§1.1, §4)
- [ ] Did this avoid Layer-1 crypto edits for a new use case? If not, did I flag it? (§1.2)
- [ ] If the IR changed: both sides updated, version bumped, conformance test + IR-SPEC
      updated? (§5)
- [ ] New op? registry + Rust impl + bit-width rule + golden test + SUPPORTED-OPS doc. (§4)
- [ ] Bit-width budget respected; over-budget fails loudly with a named layer? (§1.3, §1.4)
- [ ] `cargo fmt`/`clippy` and `ruff`/`black` clean? (§6)
- [ ] On a feature branch, not main; no unwanted artifacts committed? (§8)
- [ ] Reported with a detailed walkthrough: what, why, tradeoffs, next steps? (§3.3)
