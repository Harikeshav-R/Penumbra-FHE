# Phase 13 Evidence Audit Standards Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the two standards-axis code review findings on branch `docs/phase-13-evidence-audit`: clarify speedup ratio calculation precision in benchmark latency tables and reconcile unit naming (MB vs MiB) across documentation.

**Architecture:** Maintain mathematical fidelity to underlying JSON fixtures (`docs/results/phase10-final-sweep.json`) while eliminating apparent arithmetic discrepancies by adding explicit ratio-derivation notes to Table A in `docs/BENCHMARKS.md` and the Latency table in `docs/COMPARISON.md`. Standardize unit naming in `docs/PAPER.md` to `MB` (`1 MB = 2^20 B`) to match `penumbra-bench` and the rest of the documentation suite.

**Tech Stack:** Markdown, MkDocs Material (`uv run mkdocs build --strict`).

**Spec:** Code review findings from dual-axis review of branch `docs/phase-13-evidence-audit` diffed against `main` (merge-base `98f728740336cd14cb806e6da6709ba2fea7e3b8`).

## Global Constraints

- Preserve all existing Phase 13 evidence tracing citations (`commit` @ `hash`, fixture JSON paths).
- No modifications to any backend crates (`crates/penumbra-tfhe`, `crates/penumbra-ckks`) or `crates/penumbra-core`.
- Maintain strict anchor integrity: `uv run mkdocs build --strict` must pass with zero warnings.
- Adhere to Conventional Commits: `<type>(<scope>): <imperative description>`.
- No AI/agent attribution in commit messages or file bodies.

## Review Focus

- Table A in `docs/BENCHMARKS.md`: Readers dividing displayed rounded table figures ($27.149 / 0.660 = 41.135$ and $178.933 / 1.170 = 152.934$) should immediately understand why $41.2\times$ and $153.0\times$ are printed.
- Latency table in `docs/COMPARISON.md`: The same ratio derivation note must be present for consistency.
- `docs/PAPER.md`: Key material memory units must match the `penumbra-bench` binary MB convention ($1\text{ MB} = 2^{20}\text{ B}$) documented in `docs/BENCHMARKS.md` and `docs/COMPARISON.md`.
- Cross-references: Anchors in `docs/COMPARISON.md` and `docs/BENCHMARKS.md` must not break.

---

### Task 1: Clarify Speedup Ratio Precision in Latency Tables

**Files:**
- Modify: `docs/BENCHMARKS.md:251-270`
- Modify: `docs/COMPARISON.md:130-145`

**Interfaces:**
- Consumes: `docs/results/phase10-final-sweep.json` (`phase4_cnn` TFHE mean: 27.14926s, CKKS mean: 0.65947s $\to 41.168\times \approx 41.2\times$; `phase5_qat` TFHE mean: 178.93322s, CKKS mean: 1.16971s $\to 152.972\times \approx 153.0\times$).
- Produces: Clear, unambiguous explanatory footnotes under Table A in `docs/BENCHMARKS.md` and the Latency table in `docs/COMPARISON.md`.

- [ ] **Step 1: Inspect current Table A footer in `docs/BENCHMARKS.md`**

Verify line 268 of `docs/BENCHMARKS.md`:
```markdown
*Source: [`docs/results/phase10-final-sweep.json`](./results/phase10-final-sweep.json) @ `9b38c1b`.*
```

- [ ] **Step 2: Update `docs/BENCHMARKS.md` Table A footnote**

Add an explicit note explaining the full-precision ratio calculation:
```markdown
*Source: [`docs/results/phase10-final-sweep.json`](./results/phase10-final-sweep.json) @ `9b38c1b`. Ratios are calculated from unrounded mean evaluation times in the source JSON (e.g. 27.1493 s / 0.6595 s = 41.2x; 178.9332 s / 1.1697 s = 153.0x); dividing the 3-decimal rounded table values yields 41.1x and 152.9x due to intermediate rounding.*
```

- [ ] **Step 3: Update `docs/COMPARISON.md` Latency table note**

In `docs/COMPARISON.md:139`, append the ratio explanation:
```markdown
*(Source: [`docs/results/phase10-final-sweep.json`](./results/phase10-final-sweep.json) @ `9b38c1b`, via [`docs/BENCHMARKS.md` Table A](./BENCHMARKS.md#table-a-latency-wall-clock-per-sample). Means over N = 2 samples in `--release`, Apple M3 Pro, FFT64Neon HAL; each mean includes the first, cold sample. Ratios derive from unrounded JSON means (e.g. 41.2x, 153.0x). The logreg ordering is contradicted by Criterion — see [Logreg timing reconciliation](./BENCHMARKS.md#logreg-timing-reconciliation).)*
```

- [ ] **Step 4: Verify markdown rendering and anchor integrity**

Run: `uv run mkdocs build --strict`
Expected: Build passes with 0 warnings.

- [ ] **Step 5: Commit changes**

```bash
git add docs/BENCHMARKS.md docs/COMPARISON.md
git commit -m "docs(bench): clarify full-precision derivation of speedup ratios in latency tables"
```

---

### Task 2: Standardize Binary Memory Units in `docs/PAPER.md`

**Files:**
- Modify: `docs/PAPER.md:82`

**Interfaces:**
- Consumes: Convention defined in `docs/BENCHMARKS.md:399` ("Sizes use the binary units penumbra-bench prints (1 KB = 1,024 B, 1 MB = 2^20 B)") and `docs/COMPARISON.md:168` ("TFHE client key: 23.4 KB; server key (bootstrapping and key-switching keys): 114.84 MB. CKKS client key: 128.1 KB; server key (Galois rotation keys + relinearization keys): 1,782.50 MB (1.74 GiB).").
- Produces: Harmonized unit representation across all documents citing server key sizes.

- [ ] **Step 1: Check current line 82 in `docs/PAPER.md`**

Line 82 currently reads:
```markdown
Keys: TFHE server key 114.84 MiB; CKKS server key 1,782.50 MiB (1.74 GiB).
```

- [ ] **Step 2: Update `docs/PAPER.md:82` to match repo convention**

Update line 82 to:
```markdown
Keys: TFHE server key 114.84 MB; CKKS server key 1,782.50 MB (1.74 GiB).
```

- [ ] **Step 3: Verify consistency across all documentation**

Check that `grep -n "114.84" docs/*.md` produces consistent `MB` formatting across `docs/BENCHMARKS.md`, `docs/COMPARISON.md`, and `docs/PAPER.md`.

- [ ] **Step 4: Verify mkdocs build**

Run: `uv run mkdocs build --strict`
Expected: Build passes with 0 warnings.

- [ ] **Step 5: Commit changes**

```bash
git add docs/PAPER.md
git commit -m "docs(paper): harmonize server key size units to match penumbra-bench convention"
```

---

### Task 3: Final Verification and Documentation Check

**Files:**
- Check: All modified documentation files (`docs/BENCHMARKS.md`, `docs/COMPARISON.md`, `docs/PAPER.md`).

- [ ] **Step 1: Run strict mkdocs build**

Run: `uv run mkdocs build --strict`
Expected: Clean build, 0 warnings.

- [ ] **Step 2: Inspect git diff**

Run: `git diff HEAD~2..HEAD`
Verify:
1. Explanatory footnotes are clear and accurate.
2. Units in `docs/PAPER.md` match `docs/BENCHMARKS.md` and `docs/COMPARISON.md`.
3. No formatting or syntax regressions.
