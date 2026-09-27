# Paper Plan: A Controlled TFHE vs CKKS Comparison

This is the working record for the **paper track**: a research paper comparing TFHE and CKKS for
encrypted ML inference, with Penumbra-FHE as the experimental platform. It holds the goal, the
state of the evidence when the track started, prior work, every design decision with its
reasoning, and the guardrails. `ROADMAP.md` Phases 13–19 carry out this plan task by task.

- **Origin:** a design review with the owner on 2026-09-27, starting from `main` at `ec06e73`
  (v1.0.0 tagged, Phases 0–12 complete).
- **Authority:** decisions D1–D21 below are **settled**. Only the owner can reopen one. If new
  evidence contradicts a decision, stop and raise it. Do not quietly deviate.
- **Owner context:** the owner is an undergraduate with no supervising professor, and new to FHE
  (`AGENTS.md` §2). Several decisions below (external reader, venue, arXiv endorsement) follow
  from that.
- **Scope boundary:** the paper's LaTeX source lives in a **separate repository**, not this one
  (D20). This repo holds only library changes, measurements, results JSON, and this plan.

---

## 1. Goal and contribution

**Goal:** a measurement paper comparing TFHE (`tfhe-rs`, backend `penumbra-tfhe`) and CKKS
(`poulpy-ckks`, backend `penumbra-ckks`) on the same ML inference workloads.

**Contribution (D1):** a **controlled measurement study**. The contribution is the method:

1. both schemes consume **the same quantized IR graph** (identical integer weights, scales, and
   LUTs);
2. both are judged against **one reference**, `evaluate_graph_int` in
   `python/penumbra/reference.py`, with one comparator per scheme: TFHE bit-for-bit, CKKS within a
   declared bound;
3. both are measured by **one harness** (`penumbra-bench`) on one pinned machine.

This design lets the paper **split CKKS's accuracy loss into quantization error and polynomial
approximation error**, because quantization is identical on both sides. No prior multi-backend
work isolates this (§3). The latency gap between schemes is **not** the novel part: it confirms a
known result (Viand et al., §3).

"When to pick which scheme" guidance is a **section of the discussion**, not a standalone
contribution.

---

## 2. Starting evidence (state at `ec06e73`)

Every fact below was checked against the repo during the design review. Line numbers refer to
that commit.

### 2.1 What exists

- **Models on both backends:** `phase2_logreg`, `phase4_cnn`, `phase5_digits` (PTQ),
  `phase5_qat`, `phase6_onnx`, `phase6_sklearn`, `phase7_faces`. Inputs are tiny: sklearn 8×8
  digits and Olivetti faces downsampled to 16×16.
- **TFHE only:** `phase8_trees` (5 trees, breast cancer). Under CKKS it exceeds the leveled
  depth budget: it needs 360 bits against 330 available. CKKS uses **no bootstrapping** anywhere.
- **TFHE parameters:** `PARAM_MESSAGE_2_CARRY_2_KS_PBS` (`classic` profile), tfhe-rs 1.8.1,
  `MESSAGE_BITS = 2`. Signed radix ciphertexts, 6–11 blocks per model.
- **CKKS parameters:** poulpy-ckks 0.8.3, N = 16384 (8192 slots), k = 360, log Δ = 30, budget 330
  bits, `lt_slots = 256`, uniform ternary secret, `max_poly_degree = 15`. Packing is "Option B":
  one tensor per ciphertext with BSGS (baby-step/giant-step) diagonal transforms. Each tensor uses
  at most 256 of the 8192 slots.
- **Security claims:** 128-bit on both, established **by different methods**. TFHE relies on
  tfhe-rs's own analysis; CKKS on a lookup in the HomomorphicEncryption.org table (log q ≤ 438
  for N = 16384).
- **Machine:** Apple M3 Pro, poulpy HAL `FFT64Neon`, nightly rustc (required by poulpy), rayon
  across all cores.
- **Existing write-up:** `docs/COMPARISON.md` has a hypothesis, method, 9 threats to validity,
  results, and a verdict. `docs/BENCHMARKS.md` Tables A–D hold the numbers.

### 2.2 Headline numbers at the start

Latency per sample (`docs/BENCHMARKS.md` Table A): CKKS is 41–163× faster on the CNNs. Examples:
digits 222.3 s (TFHE) vs 1.37 s (CKKS); faces 370.6 s vs 2.30 s. TFHE is faster on logreg,
0.521 s vs 0.647 s, **but see defect 2 below**.

Accuracy: TFHE is bit-exact everywhere. CKKS max |err| after the floor-midpoint fix
(commit `7d04993`), on the Phase-10 minimized fixtures (`docs/results/phase10-final-sweep.json`):
cnn 4.0, QAT 10, digits/ONNX 38, faces 74, in quantized-integer logit units. On the older
fixtures (`docs/results/phase12-4-comparison.json`) the post-fix values are cnn 3, QAT 28,
digits/ONNX 35, faces 74.

Keys: TFHE server key 114.84 MiB; CKKS server key 1,782.50 MiB (1.74 GiB).

### 2.3 Defects found in the evidence (fixed by Phase 13 onward)

1. **N = 2 test samples per model** (`docs/COMPARISON.md:194`). That is not enough to support
   any accuracy claim.
2. **The two timing paths disagree on logreg, and flip the verdict.**
   - Table A (the per-sample report, N = 2): TFHE 0.521 s, CKKS 0.647 s.
   - Criterion (`docs/BENCHMARKS.md:242`, 10 samples): TFHE 488.8 ms, 95% CI [470, 510];
     CKKS 350.4 ms, 95% CI [347, 354].
   - The CIs do not overlap. The "TFHE wins on shallow models" verdict
     (`docs/COMPARISON.md:204`) rests on the weaker path. The cause is not yet known.
3. **Stale CKKS error numbers** in `docs/NOTES-ckks.md`: digits 188, faces 105, cnn 15.3. These
   predate the floor-midpoint fix in `crates/penumbra-ckks/src/ops/polymap.rs` (commit
   `7d04993`; `3f6bd68`, recorded as the CKKS rerun HEAD in `phase12-4-comparison.json`, is its docs-only parent). Current values are in §2.2.
4. **Wrong prose in `docs/COMPARISON.md:180`.** It says TFHE linear layers have "0 bootstraps";
   the measured counters show tens of thousands. It also quotes 338 s / 287 s for digits
   Conv2d/Linear, from a different run than Table B's 149.2 s / 51.5 s.
5. **CKKS bounds were declared after measurement.** In `crates/penumbra-ckks/src/bounds.rs:7-13`:
   digits bound 60 vs 38 measured, faces 120 vs 74, QAT 15 vs 10. A bound fitted to the test set
   tests nothing.
6. **No relative error metric and no label-flip rate.** Errors are absolute logit units, which
   can't be compared across models.

### 2.4 Why TFHE is slow here (the implementation-vs-scheme confound)

- A linear layer is one `evaluate_weighted_mac` per output neuron
  (`crates/penumbra-tfhe/src/ops/mod.rs:63-96`): `sum_ciphertexts_parallelized` per group of
  equal weights, then one `scalar_mul_parallelized` per distinct weight. Rayon runs over output
  neurons and pixels.
- **Radix width is global per model** (`Graph::num_blocks`, `crates/penumbra-core/src/ir.rs:43`),
  sized to the widest accumulator. On digits every value, including 2-bit ReLU outputs, is an
  11-block (22-bit) signed radix ciphertext.
- **Carry propagation dominates the bootstraps.** Measured PBS on digits (Table B,
  `docs/BENCHMARKS.md:254-256`): Conv2d 48,888, Linear 19,137, Requant 7,128. About 90% are carry
  PBS inside linear layers, not activation lookups. `crates/penumbra-tfhe/tests/measured_pbs.rs`
  pins this down ("Linear is not PBS-free").
- **Per-tensor widths need no IR change.** `penumbra_core::propagate_bit_widths`
  (`crates/penumbra-core/src/bitwidth.rs:191`) already gives every tensor's bit width.
- **Changing the message size would change the IR.** `MESSAGE_BITS = 2` is baked into
  `penumbra-core` and IR validation (`crates/penumbra-core/src/ir.rs:387-411`: `clamp_lut` must
  have exactly 4 entries). Moving to `MESSAGE_4_CARRY_4` is therefore a Layer-2 change made for a
  backend reason, which `AGENTS.md` §1.2 forbids.

---

## 3. Prior work and positioning

| Work | What it compares | Held constant? | Relevance |
|---|---|---|---|
| HE-MAN, Nocker et al. 2023 ([arXiv 2302.08260](https://arxiv.org/abs/2302.08260)) | ONNX models on Concrete (TFHE) and TenSEAL (CKKS); MNIST CryptoNets/LeNet-5; LFW faces | **No.** The Concrete path is integer-quantized with exact PBS ReLU. The TenSEAL path is float with degree-3 polynomial activations. Two separate tools. | Closest prior work, with nearly the same use cases as Penumbra (digits + faces). Its accuracy gaps (LeNet-5: 98.4% Concrete vs 78.9% TenSEAL) mix quantization and approximation error. |
| Viand et al., SoK: FHE compilers ([IEEE S&P 2021](https://arxiv.org/abs/2101.07078)) | Many compilers/schemes on shared workloads | Workloads and λ = 128 | Already reports "BFV/CKKS beats TFHE by orders of magnitude on linear work; TFHE wins on branching". |
| HEIR, Ali et al. 2025 ([arXiv 2508.11095](https://arxiv.org/abs/2508.11095)) | Multi-scheme compiler (CKKS, BFV, CGGI) | Separate lowering paths per scheme, not one quantized IR | Shows the field is moving to multi-scheme tooling. Not a controlled comparison. |
| PEGASUS (Lu et al., S&P 2021), CHIMERA (Boura et al., JMC 2020), LOHEN (Nam et al., USENIX Sec 2025, [arXiv 2504.17785](https://arxiv.org/abs/2504.17785)) | Hybrids that switch between CKKS and TFHE/FHEW | n/a | Report CKKS depth problems on step functions, consistent with the tree finding. |
| TT-TFHE, Benamira et al. ([arXiv 2302.01584](https://arxiv.org/abs/2302.01584)) | TFHE-friendly nets | n/a | MNIST ~4.4 s with 18 MB RAM. Motivates reporting memory. |

**Calibration against published state of the art** (the TFHE side is far behind, the CKKS side
is not):

- CKKS: Penumbra 1.37 s on an 8×8 digits CNN. EVA/SEAL runs a 28×28 MNIST CNN in ~0.6 s on 56
  cores ([PLDI 2020](https://doi.org/10.1145/3385412.3386023)). Same order of magnitude.
- TFHE: Penumbra 222 s on the same tiny CNN. Concrete-ML runs NN-20 on full 28×28 MNIST in
  ~1 s: 21.17 s in [ePrint 2021/091](https://eprint.iacr.org/2021/091.pdf), divided by the "21×
  faster" stated in [Zama's July 2024 post](https://www.zama.org/post/making-fhe-faster-for-ml-beating-our-previous-paper-benchmarks-with-concrete-ml),
  on hpc7a. Note that Concrete-ML there uses 6-bit precision, roundPBS, and `p_error = 0.1`, so it
  is **approximate**, not bit-exact.
- Rough conclusion [INFERENCE: different hardware and models]: Penumbra's TFHE is about two
  orders of magnitude off the state of the art. As it stands, the 41–163× headline measures radix
  bookkeeping more than it measures TFHE.

**Positioning statement:** no prior work runs *identical quantized weights* through both schemes
against *one reference*. The paper's novelty is that control and the error split it makes
possible. It does **not** claim novelty for "CKKS is faster on linear work", and it does not claim
to be a general FHE survey.

---

## 4. Decision register

Each entry: the decision, then why. Rejected alternatives are listed where they could come back.

### Framing

- **D1: Paper type.** A controlled measurement study, with scheme-selection guidance in the
  discussion. *Why:* HE-MAN already published "ONNX system with two FHE backends", so a systems
  paper would be thin. The controls are the plausible novelty. *Rejected:* systems paper;
  standalone guidance paper.
- **D2: Claim level.** Claims are about the schemes, backed by cost counts (PBS for TFHE;
  rotations, rescales, depth for CKKS), **only after** the TFHE linear path is fixed (D14). Title
  and abstract are worded at the library level ("tfhe-rs radix-integer vs poulpy-ckks under a
  shared IR"). If the fix doesn't close the gap, latency claims stay at the library level. *Why:*
  §2.4 and §3 show the current gap mostly measures radix carry handling.
- **D17: Before/after TFHE and the stopping rule.** The TFHE fix is time-boxed (about 2 weeks:
  spike plus implementation), then frozen whatever the gain. **Both** TFHE results (pre-fix and
  post-fix) go in the paper, as the finding "one implementation choice within a scheme moved
  latency N×, against an M× gap between schemes". *Why:* this directly answers the reviewer
  objection and is useful to anyone reading scheme comparisons. *Rejected:* optimizing until
  "competitive", which never ends.

### Measurement protocol

- **D3: Sample protocol.** Use TFHE's exactness.
  - TFHE: its full-test-set accuracy **equals** the quantized-cleartext accuracy, which is
    computed in cleartext. Encrypted TFHE runs cover only latency statistics plus a bit-exact
    spot-check of **n ≈ 30** samples per model.
  - CKKS: its error depends on the sample, so it runs **encrypted over the full test sets**
    (~360 digits, 80–200 faces, ~114 breast cancer; at ~1–2 s/sample this is cheap).
  - *Rejected:* encrypting every TFHE test sample (~30 h for digits alone).
- **D6: Canonical timing path.** First find out why the two logreg timings disagree (defect 2).
  Then **Criterion is the only source of headline latency**: median plus 95% CI, thread count
  pinned and stated. The per-sample report path is kept only for per-op breakdowns and cost
  proxies, labelled as such. *Why:* `AGENTS.md` §4 forbids comparing numbers from two
  measurement paths, and here the two paths reverse the conclusion.
- **D7: CKKS error bounds and metrics.**
  - Bounds are declared **before testing**, from a calibration split kept separate from the test
    set. The rule is p99 of calibration |err| × a margin that is fixed before any test run. The
    bounds live in `crates/penumbra-ckks/src/bounds.rs` with a derivation comment.
  - The primary metric is the **label-flip rate** vs the quantized reference.
  - Also reported: the |err| distribution **relative to the top-2 logit margin** (median, p95,
    max).
  - *Why:* defect 5 (post-hoc bounds) and defect 6 (no relative metric). This is a Layer-1 test
    constant, not an IR or reference change.
- **D10: Headline metric.** Per-sample latency **plus peak server memory (RSS)**, measured by the
  shared harness for both backends. CKKS slot utilization (≤ 256 of 8192) is reported as a stated
  limitation. The throughput gain from batching is described but **not** claimed, because it is
  not measured. *Why:* memory is where TFHE plausibly wins (compare TT-TFHE); a throughput number
  would be unmeasured.
- **D11: Trees.** Included as a finding: TFHE evaluates them exactly (67 PBS, ~10 s); **leveled
  CKKS at these parameters** cannot. Always use that wording, never "CKKS cannot". CKKS
  bootstrapping is out of scope.
- **D12: Hardware.** A single M3 Pro, disclosed as a threat to validity. No second machine
  (cost; the cost proxies partly cover platform dependence).
- **D15: External TFHE calibration.** Run **Concrete-ML in exact mode** (no
  `rounding_threshold_bits`, `p_error` near zero) on the same architectures, on the same M3 Pro.
  Report it as a **calibration row outside the controlled comparison**: Concrete-ML has its own
  quantizer and cannot consume Penumbra's IR or be held to its reference. The script lives in the
  paper repo, not here. If Concrete-ML does not install on macOS arm64, record that and cite
  published numbers instead. [Unverified: macOS arm64 support.]
- **D16: Split the TFHE bootstrap count.** Report **lookup PBS** (inherent to the scheme: one per
  logical table lookup in Requant, Activation, Compare, Argmax) separately from **carry PBS**
  (measured total − lookup PBS: radix bookkeeping, an implementation choice). Use the split in
  the cost-proxy table and the figures.
- **D18: The 28×28 MNIST rule.** Add a small 28×28 CNN (a new graph and fixture only, no backend
  change) **if** post-fix TFHE measures **≤ 10 min per sample**. That keeps the D3 protocol
  (~30 spot-check samples + 10 Criterion runs ≈ 7 h) to an overnight run. Otherwise leave it out
  and state the scale limit as a threat.
- **D19: Security estimate.** Run **both** parameter sets through the same
  [lattice-estimator](https://github.com/malb/lattice-estimator) (Sage) and report both
  estimates in the setup table. If they differ (e.g. 128 vs 140), report both numbers and **do
  not retune** (`AGENTS.md` §7).

### Engineering

- **D5: Engineering scope.** v1.0.0 is frozen except for **one** targeted improvement: the TFHE
  linear path (D14). The following are **out of scope** for the paper track: CKKS bootstrapping,
  CKKS multi-sample packing, larger models beyond D18, and an x86 re-run.
- **D14: TFHE linear-path design (a fork under `AGENTS.md` §3.2, resolved).**
  1. **Spike first** (≤ 2 days): a throwaway microbenchmark of one digits `Conv2d` output neuron
     (a 9-term MAC) and one `Linear` output neuron (a 108-term MAC). Variants: (a) baseline,
     with the global 11-block width; (b) **per-tensor radix width**; (c) (b) plus **deferred
     carries**, meaning `unchecked_*` ops and one propagate once the carry space fills. Record
     latency and measured PBS for each.
  2. **Then implement per-tensor radix width** inside `penumbra-tfhe` only:
     - encrypt and keep each tensor at its `propagate_bit_widths` width;
     - sign-extend only where an accumulator needs more room;
     - narrow after `Requant`.
     `Graph::num_blocks` stays as the model-level ceiling and budget check. Add deferred carries
     only if the spike shows a gain of at least 1.2× on the neuron microbenchmark.
  3. **Rejected:** larger message parameter sets such as `MESSAGE_4_CARRY_4`, because they
     change the IR (§2.4). Also rejected: any approximate or probabilistic rounding
     (roundPBS-style with a nonzero error rate), which would break TFHE's bit-exact gate.

  *Why:* every PBS today pays for 11 blocks when 1–5 would do. How much the fix gains is unknown
  (could be 2× or 10×), and it drives D17 and D18, so measure before building.

### Publication

- **D4 / D8: Venue and path.**
  - Post an **IACR ePrint preprint, but only after** the TFHE fix and the reruns (target
    Feb–Mar 2027).
  - Then submit to **WAHC 2027** (the workshop at ACM CCS). The WAHC 2025 format was 12 pages
    ACM sigconf, references included, with no formal artifact evaluation
    ([WAHC 2025](https://homomorphicencryption.org/wahc-2025/)). The WAHC 2026 deadline (2026-07-19)
    has passed. The 2027 deadline is assumed to be around July 2027 [INFERENCE: yearly cadence].
  - PoPETs is the step up (formal artifact evaluation) if the TFHE side becomes competitive and a
    28×28 model is in.
  - *Why post the preprint late:* a preprint claiming "CKKS 160× faster" on a TFHE baseline two
    orders of magnitude slow would be the version people read.
- **Undergraduate logistics:**
  - Peer-reviewed venues do not gate on affiliation. List "student at <university>" or
    "independent researcher".
  - **arXiv needs a personal endorsement** from an established `cs` author, since the January 2026
    policy ([arXiv endorsement](https://info.arxiv.org/help/endorsement.html)). ePrint has no
    endorsement step [from secondary sources only; not verified on IACR's own page].
  - Presenting at WAHC generally needs a registered author on site; look for student
    registration or travel grants.
- **D9: External reader.** Two kinds:
  - (a) a **professor at the owner's university** (crypto, systems, or security), for overall
    judgment and an arXiv endorsement;
  - (b) **narrow upstream questions**: the tfhe-rs/Zama community on "is this the idiomatic
    exact plaintext-weight dot product?", and poulpy's maintainers on the CKKS usage (BSGS
    packing, polynomial approximation, parameters).
  - *Why:* without a supervisor, nobody checks the work before the reviewers do.
- **D13: Artifact.** Tag the paper commit in this repo and archive it on **Zenodo** for a DOI.
  The paper repo carries one script that rebuilds every table and figure from this repo's
  `docs/results/*.json`. State the toolchain exactly: nightly rustc, `poulpy-ckks 0.8.3`,
  `tfhe 1.8.1`.
- **D20: Repository split.** Paper source (LaTeX, ACM sigconf template, figures, the Concrete-ML
  calibration script) lives in a **separate repo**. Library changes follow the normal rules on
  feature branches: golden test, fmt/clippy, ruff/black, Conventional Commits.
- **D21: Timeline.** The phases run in order (ROADMAP 13 → 19). The exception: **the setup and
  related-work sections are written in parallel with Phases 14–16**, because they depend on no
  result. The results and discussion sections wait for Phase 16, so the numbers cannot reshape
  the method after the fact.

---

## 5. Claim-wording rules (for Phase 17)

- Title and abstract: library level (D2). Scheme-level statements appear only where backed by
  cost proxies, and only after the fix.
- "Leveled CKKS at these parameters", never bare "CKKS", for depth-limited results (D11).
- TFHE latency is always shown with its lookup/carry PBS split (D16), and with before/after where
  relevant (D17).
- CKKS accuracy is always shown as a distribution plus label-flip rate, never as one max-error
  number (D7).
- Differences within small-sample noise are not findings (`docs/COMPARISON.md` threat 7).
- Never compare CKKS against the float model to flatter it (`AGENTS.md` §1.1). Float accuracy is
  reported alongside, never used as the comparator.
- No AI or agent authorship attribution anywhere (`AGENTS.md` §8).

---

## 6. Guardrails for this track

These restate `AGENTS.md` for the concrete temptations of this track:

- **No IR change, no `penumbra-core` change for a backend reason.** In particular, do not change
  `MESSAGE_BITS` and do not add per-tensor width fields to the IR. Widths are derived inside
  `penumbra-tfhe` from `propagate_bit_widths`.
- **TFHE stays bit-exact.** The golden tests stay green, unedited, through Phase 14.
- **No parameter retuning** to match security estimates or to win benchmarks.
- **One measurement path.** Every paper number comes from `penumbra-bench` on the pinned M3 Pro
  and is traceable to a committed JSON file in `docs/results/` plus a commit hash.
- **Keep the pre-fix baseline.** `docs/results/phase12-4-comparison.json` and the other pre-fix
  JSON files are the "before" numbers. Never overwrite them; write new result files.
- **Scope discipline (D5).** Anything outside D5's single improvement is a new decision for the
  owner.

---

## 7. Phase map

| ROADMAP phase | Content | Decisions |
|---|---|---|
| 13 | Evidence audit: doc fixes + logreg timing reconciliation | defects 1–4, D6 (investigation) |
| 14 | TFHE spike + per-tensor radix width | D14, D17 |
| 15 | Harness & protocol hardening | D3, D6, D7, D10, D16, D19 |
| 16 | Full runs & external calibration | D3, D11, D15, D17, D18 |
| 17 | Paper writing (separate repo; setup/related work start alongside 14) | D1, D2, D20, D21, §5 |
| 18 | External review | D9 |
| 19 | Publication & artifact | D4/D8, D13 |

## 8. Open unknowns (to be resolved by the phases, not assumed)

- The cause of the logreg timing disagreement (Phase 13).
- The actual gain from per-tensor radix width (Phase 14 spike).
- Whether post-fix TFHE fits the 28×28 rule (Phase 16).
- Whether Concrete-ML runs on macOS arm64 (Phase 16).
- The lattice-estimator results for both parameter sets (Phase 15).
- WAHC 2027 dates and format (check the CFP when it is published).
