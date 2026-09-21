# AGENTS.md — Guidelines for AI Agents Working on Penumbra-FHE

> This file governs how AI coding agents work in this repository. Read it before
> touching anything. It encodes the project owner's working preferences and the
> non-negotiable invariants of the architecture.
>
> **Required reading, in order:** [`PROJECT.md`](./PROJECT.md) (architecture & rationale),
> [`ROADMAP.md`](./ROADMAP.md) (task-level build plan), and — before any backend work —
> [`docs/BACKENDS.md`](./docs/BACKENDS.md) (the backend boundary and per-backend
> correctness). This file does not repeat them — it tells you how to behave while executing
> them.

---

## 0. TL;DR — the rules that matter most

1. **Plan before you build.** For any non-trivial task, present a detailed plan and
   **wait for approval** before writing code. Work in a tight loop — check in at each
   meaningful step. (§3)
2. **The golden invariant is sacred and non-negotiable.** Every backend is judged against the
   *same* quantized-cleartext reference: **TFHE bit-for-bit; CKKS within its declared error
   bound**. Every op/feature touching eval ships with a passing golden test in the same
   change. Never weaken or skip it — even if asked. (§1, §4)
3. **New use case ⇒ new graph, never new crypto. New backend ⇒ new crate, never new IR.**
   If a task forces Layer-1 edits to support a new *model*, or Layer-2/3 edits to support a
   new *scheme*, the abstraction leaked. **Stop and flag it** — fix the abstraction, not the
   use case. This is a hard gate. (§1)
4. **IR changes touch both sides + bump version, same change.** `python/penumbra/ir.py` ↔
   `runtime/src/ir.rs` stay in lockstep with the conformance test and schema-version field.
   The IR is **backend-neutral**: a change motivated by one scheme is a fork, not a bump.
   Hard rule. (§5)
5. **Architectural forks: present options, wait.** Don't silently pick a design for genuine
   forks (IR LUT representation, per-tensor vs per-channel scales, topo-sort strategy, CKKS
   slot packing). Surface tradeoffs + a recommendation, then wait. (§3)
6. **Test-first. Format + lint clean before "done."** rustfmt + clippy, ruff + black; treat
   warnings as errors. (§6)
7. **Flag upstream breakage, never work around it quietly.** `poulpy-ckks`'s API is
   explicitly "subject to change." When it breaks something, log it in
   `docs/NOTES-ckks.md`'s instability table. (§7)

---

## 1. Non-negotiable invariants (hard gates)

These are inviolable. If a request would break one, **do not proceed silently** — explain
the conflict and stop for the owner's decision. "The user asked" does not override these;
flag the conflict first.

### 1.1 The golden invariant — one reference, one comparator per backend
> **Encrypted output must match the quantized-cleartext output: for TFHE, bit-for-bit; for
> CKKS, within the model's declared error bound.**

The **reference never changes**: `python/penumbra/reference.py`'s `evaluate_graph_int`, the
quantized-integer forward pass. Only the comparator is per-backend. That is precisely what
makes the two backends comparable (`docs/COMPARISON.md`).

- **TFHE is *exact*.** Any discrepancy is a **quantization or implementation bug — never
  crypto noise**. This is your truth oracle, and it is not weakened by the existence of a
  second backend.
- **CKKS is *approximate* by construction.** Its gate is a declared, committed, per-model
  error bound, and the measured error is **always reported**, never just compared. Exceeding
  the bound is still a bug first — scale, level, or polynomial degree — and noise second.
- Every model, every op, every phase must satisfy the invariant *at its backend's comparator*.
- Wire it into CI; never let it regress.
- When the encrypted result disagrees, debug the **cleartext quantized path first** — it is
  almost always an indexing, scale, or bit-width bug, not the crypto.
- **Never** compare CKKS against a friendlier reference (e.g. the float model) to make its
  numbers look better. See `docs/BACKENDS.md`.

### 1.2 The narrow-waist discipline (both directions)
> **A new use case only ever adds a Layer-3 adapter (or just a new ONNX file). It never
> touches Layers 1–2.**
>
> **A new backend only ever adds a Layer-1 crate. It never touches Layers 2–3.**

- Layer 1 is the set of **backend crates** (`penumbra-tfhe`, `penumbra-ckks`) — ops, keys,
  encrypt/decrypt against one FHE library. Layer 2 (IR + op registry + eval loop, in
  `penumbra-core`) is written once, is **backend-neutral**, and stays stable across both use
  cases and schemes.
- Litmus test, use case: *if adding face recognition forces a crypto-backend edit, the
  abstraction leaked.* The correct fix is a **more general op** or a **missing registry
  entry**, never a use-case-specific hack in a backend.
- Litmus test, backend: *if adding CKKS forces an IR change, an op-vocabulary change, or a
  per-scheme branch in the eval loop, the backend abstraction leaked.* The correct fix is a
  more general trait method, or a loud "unsupported on this backend" — never a scheme check
  inside Layer 2.
- **Backend parity:** both backends consume the same IR, run the same models, and are
  measured by the same harness. This is a correctness requirement, not a nicety — it is what
  makes the comparison valid.
- If you believe a Layer-1 or Layer-2 edit is genuinely required (e.g. a legitimately new
  primitive), treat it as an architectural fork (§3) and escalate — don't just do it.

### 1.3 Resource budgets are enforced centrally, per backend
- Each op declares how it grows bit-width; the library inserts `Requant` automatically and
  **errors/warns loudly** when precision exceeds the budget, naming the offending layer.
- The budget is **per-backend**: TFHE's is the radix capacity
  (`num_blocks × MESSAGE_BITS`, the LUT/PBS limit); CKKS's is the multiplicative
  depth/level budget and scale precision. The bit-width tracker itself is scheme-neutral —
  it describes the quantized graph — but the capacity it is checked against is not.
- Keep activation bit-widths small (≤ 6–8 bits). Accumulator overflow is the #1 bug in
  multi-layer models — the bit-width tracker must be correct.

### 1.4 Fail loudly, early
- Unsupported ops and infeasible bit-widths are caught at **compile/load time** with
  **actionable messages** (e.g. `"operator X (node 'name') not supported"`), never
  mysteriously at runtime. List *all* problems at once where feasible, not one at a time.
- This extends to backends. An op a backend cannot implement must be rejected at load time
  naming the op, the node, and the backend — never silently approximated, and never quietly
  routed to a different backend.
- Key/ciphertext material is **not** portable across backends. A mismatch must produce an
  actionable message, not a deserialization panic.

---

## 2. Project context & intent

- **What this is:** a serious, releasable, Apache-2.0 open-source library for encrypted ML
  inference on ONNX models, over pluggable FHE backends (`tfhe-rs` today, `poulpy-ckks`
  next). Hold to a production-ish quality bar *within* the stated research/prototype scope
  (latency is seconds-per-inference; this is not real-time serving — set expectations
  honestly, don't over-promise).
- **What this is NOT:** a compiler, a general FHE framework, an LLM tool, or a supporter of
  arbitrary ONNX graphs. See `PROJECT.md` §1, §16. The second backend exists to enable a
  **controlled scheme comparison** (`PROJECT.md` §18, `docs/COMPARISON.md`) — not to make
  Penumbra scheme-general. Don't chase backend feature completeness past what the comparison
  needs.
- **Owner profile:** strong software engineer (Rust/Python/ML), **new to FHE**. Explain
  crypto-specific reasoning (TFHE, PBS, LUTs, CKKS scale/level/depth, polynomial
  approximation, parameter choices) where it informs a decision; don't over-explain general
  engineering. Teach the crypto as you go.

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
IR format, PyO3 boundary design, and every fork listed in `docs/BACKENDS.md` (CKKS slot
packing, what `Requant` means under CKKS, how the approximation-degree knob is exposed).
If `PROJECT.md`/`ROADMAP.md`/`docs/BACKENDS.md` already imply an answer, follow it and say
so; only escalate genuinely novel forks.

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
  the encrypted-vs-quantized-cleartext test must run and pass, **at each affected backend's
  comparator** (§1.1), before work is complete.
- **The canonical extension path for a new op** (also the CONTRIBUTING story):
  1. registry entry (ONNX → internal op mapping),
  2. Rust implementation **in every backend** — or a loud, load-time "unsupported on this
     backend" (§1.4); never a silent approximation,
  3. bit-width growth rule (scheme-neutral) + each backend's budget consequence,
  4. golden test asserting the invariant at each backend's comparator,
  5. docs update (`docs/SUPPORTED-OPS.md`, and `docs/BACKENDS.md` if the trait changed).
- **The canonical extension path for a new backend** is in `docs/BACKENDS.md`. Adding one
  must not touch Layers 2–3 (§1.2).
- Maintain the **cross-language IR conformance test** (Python emits IR → Rust loads → assert
  agreement) whenever the IR changes.
- Build the Rust runtime in **`--release`** for any timing/latency work — debug FHE is
  misleadingly slow, for `poulpy` as much as for `tfhe-rs`. Correctness checks can run in
  debug.
- **Cross-backend timing comparisons must come from the shared harness** (`penumbra-bench`),
  on one pinned machine. Numbers produced by two different measurement paths are not
  comparable and must not be reported as a result (`docs/COMPARISON.md`).
- Property/fuzz idea to lean on: random small models → assert the invariant per backend.

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
- **The IR is backend-neutral, and that is load-bearing.** Both backends deserialize the
  *same* graph; it is what makes the scheme comparison valid (`docs/COMPARISON.md`). Some
  payload fields are historically TFHE-shaped (`num_blocks`, `Requant.clamp_lut`,
  `Activation.lut`) — a second backend **reinterprets** them, it does not get its own fields.
  Any pressure to add scheme-tagged fields or a scheme discriminator is an **architectural
  fork** (§3.2), not a routine schema bump. See `docs/IR-SPEC.md`.
- Wire format is **JSON first** (human-inspectable, easy to debug). Do not introduce a binary
  format before it's warranted by profiling (Phase 10) — and that's an architectural fork (§3).
  This applies to the IR; keys and ciphertext are already binary (`bincode`) and are **not**
  portable across backends (§1.4).

---

## 6. Code style & quality

- **Before declaring work done**, run and fix:
  - **Rust:** `cargo fmt` + `cargo clippy` — treat clippy warnings as errors.
  - **Python:** `ruff` + `black` — treat ruff warnings as errors.
- Match the surrounding code's idiom, naming, and comment density. Write comments that
  explain *why* (especially crypto/bit-width reasoning), not *what*.
- Keep the public API surface clean and small (see `PROJECT.md` §12): ship a secure default
  crypto-param profile **per backend**; expose **one** override knob per backend; never make
  users choose raw `tfhe-rs` or `poulpy` parameters or compute scales by hand. Quantization is
  a **library service**, not the user's problem. Choosing a *named backend* is selection, not
  parameter exposure — it doesn't count against the one-knob budget.

---

## 7. Dependencies & tooling

- **Named stack is preferred.** Default to the tools the docs specify:
  - Python: **`uv`** for env/deps (project standard — **not** poetry/pip/conda), `onnx`,
    `numpy`, `brevitas` (don't write your own quantizer), `pytest`, PyTorch/sklearn/XGBoost
    for examples.
  - Rust, TFHE backend: **`tfhe-rs`** (high-level `integer`/`shortint` API).
  - Rust, CKKS backend: **`poulpy-ckks`** (pure Rust — deliberately *not* OpenFHE/SEAL
    bindings, which would drag in a C++ toolchain), plus the matching `poulpy-cpu-*` HAL
    backend for the target architecture. **Pin the exact version**; the API is explicitly
    "subject to change" (§0 rule 7, `docs/NOTES-ckks.md`).
  - Rust, shared: `serde`/`serde_json`, `bincode`, `PyO3` (for bindings, Phase 9), `rayon`
    (parallelism, Phase 10), `criterion` (the comparison harness, Phase 12).
- Small, uncontroversial dependencies are OK without asking, but **flag anything notable**.
  Swapping out a named tool requires approval.
- Start from each library's **default secure parameter profile** — `tfhe-rs`'s default, and
  `poulpy-ckks`'s `presets`. Don't hand-roll crypto parameters; tuning within a fixed
  security level is a later, opt-in optimization (Phase 10) — and never trade away security
  for speed. The two backends' security levels must **match**, or the comparison is unfair.
- ⚠️ **Toolchain caveat:** `poulpy` pins a nightly toolchain upstream and pulls `libm`'s
  `unstable-float`; Penumbra is stable ≥ 1.83. Whether the workspace needs nightly is a
  blocking question for the Phase-12.0 spike — see `docs/NOTES-ckks.md`. The `poulpy-cpu-*`
  HAL backend is also architecture-specific (`-arm` on Apple Silicon, `-avx` on x86-64).

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
  - **Suggested scopes** for this repo: `runtime`, `core`, `tfhe`, `ckks`, `bench`, `python`,
    `ir`, `ops`, `quant`, `onnx`, `ci`, `docs`, `examples`.
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

Target layout after the Phase-12.1 workspace refactor (`ROADMAP.md`):

```
penumbra-fhe/
├── PROJECT.md / ROADMAP.md / AGENTS.md   # architecture / plan / these rules
├── python/penumbra/                      # Layer 3 + quantization + ONNX loader + IR emitter
│   ├── onnx_loader.py · op_registry.py · ir.py
│   ├── quantization/ · adapters/ · client.py
├── crates/                               # the Rust workspace
│   ├── penumbra-core/                    # Layer 2 — BACKEND-NEUTRAL (DO NOT put crypto here)
│   │   └── ir.rs · eval.rs · bitwidth · the `Backend` trait
│   ├── penumbra-tfhe/                    # Layer 1 — tfhe-rs backend
│   │   └── keys.rs · ops/ · encrypt.rs
│   ├── penumbra-ckks/                    # Layer 1 — poulpy-ckks backend (Phase 12)
│   └── penumbra-bench/                   # the shared comparison harness
├── examples/{mnist,faces}/               # use cases — graphs only, no crypto
├── tests/                                # golden exactness, conformance, unsupported-op
└── docs/                                 # BACKENDS, COMPARISON, IR-SPEC, SUPPORTED-OPS, ...
```

Until that refactor lands, all Rust code lives in `runtime/src/` (`keys.rs · ir.rs · ops/ ·
eval.rs · encrypt.rs`) and the mapping is: `ir.rs`/`eval.rs` → `penumbra-core`; everything
else → `penumbra-tfhe`.

- **The backend crates are the protected zone** (§1.2). Edits in `penumbra-tfhe/ops/` or
  `penumbra-ckks/` for a new *use case* are a red flag. Edits to add a *genuinely new
  primitive op* are legitimate but go through the §4 extension path + §3 fork review.
- **`penumbra-core` is the doubly-protected zone.** It must contain no crypto and no scheme
  branch. A change there motivated by one backend is an architectural fork (§3.2).
- Layer-3 work (new models/use cases) lives in `python/penumbra/adapters/`, `examples/`, and
  registry entries — not in any backend.

---

## 10. Quick self-check before you say "done"

- [ ] Did I plan and get approval for non-trivial work? (§3.1)
- [ ] Does the golden test pass **at the right comparator** for every backend I touched —
      TFHE bit-for-bit, CKKS within its declared bound, error reported? (§1.1, §4)
- [ ] Did this avoid Layer-1 crypto edits for a new use case? If not, did I flag it? (§1.2)
- [ ] Did this avoid Layer-2/3 edits for a backend change? Do both backends still consume the
      same IR and run under the same harness (backend parity)? (§1.2)
- [ ] If the IR changed: both sides updated, version bumped, conformance test + IR-SPEC
      updated — and is the change still **backend-neutral**? (§5)
- [ ] New op? registry + impl (or loud "unsupported") in *every* backend + bit-width rule +
      golden test per comparator + SUPPORTED-OPS doc. (§4)
- [ ] Resource budget respected for the backend in question; over-budget fails loudly with a
      named layer? (§1.3, §1.4)
- [ ] Did `poulpy` break anything? If so, is it logged in `docs/NOTES-ckks.md` rather than
      worked around silently? (§0 rule 7, §7)
- [ ] Any cross-backend numbers from the shared harness on one pinned machine, never from two
      measurement paths? (§4, `docs/COMPARISON.md`)
- [ ] `cargo fmt`/`clippy` and `ruff`/`black` clean? (§6)
- [ ] On a feature branch, not main; no unwanted artifacts committed? (§8)
- [ ] Reported with a detailed walkthrough: what, why, tradeoffs, next steps? (§3.3)

---

## Agent skills

### Issue tracker

GitHub Issues via `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`CONTEXT.md` and `docs/adr/` at repo root). See `docs/agents/domain.md`.
