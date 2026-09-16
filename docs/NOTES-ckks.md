# CKKS Notes

Working notes on the CKKS primitives the `penumbra-ckks` backend builds on, the parameter
profile, and empirical cost. The sibling of [`NOTES-tfhe.md`](./NOTES-tfhe.md); it closes the
Phase-12.0 spike deliverable (`ROADMAP.md` Phase 12) the same way that document closed
Phase 1.

> **Status: pre-spike.** The parameter profile and cost tables below are placeholders. Nothing
> here is measured yet. Do not cite these numbers anywhere until Phase 12.0 fills them in.

## The library: `poulpy`

[`poulpy`](https://github.com/phantomzone-org/poulpy) is a pure-Rust, backend-agnostic FHE
library by Jean-Philippe Bossuat (author of Lattigo), incubated by PhantomZone. Chosen over
OpenFHE/SEAL bindings because it needs **no C++ toolchain, CMake, or native linking** — it
drops into the existing Cargo workspace exactly the way `tfhe-rs` already does. Google's HEIR
project is integrating it as a backend, which is third-party validation rather than
self-reported activity.

The workspace splits into a hardware abstraction layer and pluggable implementations:

| Crate | Role | Do we depend on it? |
|---|---|---|
| `poulpy-hal` | layouts + trait-based hardware abstraction with open extension points | transitively |
| `poulpy-core` | scheme-agnostic Module-LWE arithmetic (LWE, GLWE, GGLWE, GGSW) | transitively |
| `poulpy-ckks` | backend-agnostic leveled CKKS: encoding, evaluator, polynomial evaluation, bootstrapping | **yes, directly** |
| `poulpy-bin-fhe` | binary/gate-level FHE | no |
| `poulpy-cpu-ref` | reference CPU implementation — correctness and validation, not performance | as a portable fallback |
| `poulpy-cpu-arm` | NEON/ASIMD accelerated, AArch64 | **yes, on Apple Silicon** |
| `poulpy-cpu-avx` | AVX2/FMA accelerated, x86-64 | **yes, on x86-64 (incl. CI)** |
| `poulpy-cpu-avx512` | AVX-512 accelerated, x86-64 | no |
| `poulpy-cpu-rayon` | shared Rayon executor for the multithreaded backend variants | undecided (Phase 10) |

Backend selection in `poulpy` is by **direct crate dependency plus a generic parameter** —
code is written generically over the backend type and the chosen dependency determines
behavior. It is *not* a Cargo feature flag on `poulpy-ckks` itself. (Penumbra's own
`ckks` feature is a separate, Penumbra-side thing.)

`poulpy-hal`'s open extension points are also the documented seam for a custom SIMD backend
later: implement the `Backend` trait and wire up the HAL operation families. **Out of scope
for Phase 12** — but do not design the integration in a way that forecloses it, which in
practice means: stay generic over the HAL backend, never hardcode `poulpy-cpu-arm`.

### Version pinning

Pinned to **`poulpy-ckks` 0.8.3** (Apache-2.0, published 2026-09-09). Its own documentation
says: *"this is the first iteration of the CKKS crate: the evaluator is functional and tested,
but the public API is still subject to change."* Note that 0.6.0 was yanked.

> Pin the exact version in `Cargo.lock` and **do not chase upstream changes mid-implementation**
> (`ROADMAP.md` Phase 12 pitfalls). If the API breaks something, log it in the table below
> rather than quietly working around it — that is information the next person needs.

### API-instability log

| Date | Version | What broke | How it was handled |
|---|---|---|---|
| — | 0.8.3 | *(pinned; nothing logged yet)* | — |

## ⚠️ Toolchain and platform prerequisites

**These are blocking questions for the Phase-12.0 spike — resolve them before any refactor.**

1. **Nightly Rust may be required.** The `poulpy` repository pins `nightly-2026-05-14` in its
   `rust-toolchain.toml`, and `poulpy-ckks` takes a *normal* (not dev-only) dependency on
   `libm` with the `unstable-float` feature. Penumbra is `rust-version = "1.83"` on stable and
   CI uses `dtolnay/rust-toolchain@stable`. If nightly turns out to be required, the options
   are a workspace-level `rust-toolchain.toml`, or gating `penumbra-ckks` behind an optional
   feature so the TFHE path stays stable-buildable. **Record the answer here.**
2. **The HAL backend is architecture-specific.** `poulpy-cpu-avx` is AVX2/FMA — x86-64 only.
   The primary development machine is Apple Silicon (aarch64), where the correct choice is
   `poulpy-cpu-arm` (NEON); CI runs `ubuntu-latest` (x86-64). Any benchmark must pin **one
   machine and one HAL backend**, or the numbers are not comparable across runs
   (`docs/COMPARISON.md`).
3. **Two FHE libraries in one dependency graph.** `tfhe` 1.6 and the `poulpy` crates must
   coexist in the workspace lockfile. Verify in the spike; a version conflict here is much
   cheaper to discover before the refactor than after.

## Parameter profile

*To be filled by Phase 12.0. The CKKS analogue of `NOTES-tfhe.md`'s
`PARAM_MESSAGE_2_CARRY_2_KS_PBS` section.*

| Quantity | Value | Notes |
|---|---|---|
| Ring degree (`N`) | TBD | slot count is `N/2` for complex-conjugate packing |
| Scale (`log_delta`) | TBD | precision per level; the accuracy/depth lever |
| Level budget (`log_budget`) | TBD | multiplicative depth before bootstrapping is needed |
| Slot kind (`SlotsKind`) | TBD | how a tensor maps onto slots — see the packing fork in `docs/BACKENDS.md` |
| Security level | TBD | must match the TFHE profile's level, or the comparison is unfair |

`poulpy-ckks` ships ready-made parameter sets in its `presets` module. Start there, exactly
as the TFHE side starts from the `tfhe-rs` default secure profile (`AGENTS.md` §7). Never
hand-roll crypto parameters.

## The primitives everything composes from

Structured to mirror `NOTES-tfhe.md`'s "two primitives" framing, because the contrast is the
whole point.

1. **Plaintext arithmetic and adds (cheap, SIMD-batched)** — add, sub, negate, plaintext
   multiply, plaintext add. The `Linear`/`Conv2d` core, as on the TFHE side, except one
   operation covers many values at once. Rotations and conjugation make cross-slot reductions
   possible.
2. **Rescale (cheap, but spends the budget)** — the CKKS analogue of narrowing an
   accumulator. Consumes a level.
3. **Polynomial evaluation (the expensive one, and the approximate one)** — CKKS has **no
   programmable bootstrap and no lookup table**. Every nonlinearity — ReLU, the `Requant`
   clamp, `Argmax`'s comparison — becomes a fitted polynomial. `poulpy-ckks`'s
   `approximation` module handles the fitting and precision selection.
4. **Bootstrapping (very expensive)** — `ModUp`, `CoeffsToSlots`, `SlotsToCoeffs`, `EvalMod`,
   plus a `PaCo` variant. Needed only if a model exceeds the level budget. For Penumbra's
   small models the goal is to **stay leveled** and never bootstrap; if a model needs it,
   that is itself a comparison finding worth reporting.

The mapping from Penumbra's op vocabulary onto these primitives is tabulated in
[`docs/BACKENDS.md`](./BACKENDS.md#the-backend-contract).

## Empirical cost

*To be filled by Phase 12.0 and Phase 12.4, measured by the shared harness
(`penumbra-bench`), in `--release`, on the pinned machine and HAL backend.*

| Quantity | Value |
|---|---|
| keygen + a single round-trip | TBD |
| one packed `Linear` (spike operation) | TBD |
| one ReLU polynomial at the chosen degree | TBD |
| full Phase-2 logreg inference, per sample | TBD |
| full Phase-4 CNN inference, per sample | TBD |

> ⚠️ Always benchmark in `--release`. The rule that makes debug `tfhe-rs` numbers meaningless
> applies to `poulpy` too (`docs/DEVELOPMENT.md`).

## Accuracy notes

CKKS is approximate by construction, so unlike the TFHE backend there is no bit-exactness to
lean on as a truth oracle. The correctness gate is a **declared, per-model error bound**
against the same quantized-cleartext reference, with the measured error always reported
(`docs/BACKENDS.md`, "one invariant, two comparators").

Two sources of error must be kept apart when debugging:

- **Scale/level error** — too small a scale, or too many rescales, degrades every value
  uniformly. Shows up even on PBS-free ops.
- **Approximation error** — the polynomial's fit to ReLU/clamp, concentrated at the
  nonlinearities and worst near the kink at zero.

A useful diagnostic: run a graph containing only `Linear`/`Conv2d`/`Pool`/`Add` and check
whether it round-trips to the exact integers. If it does, all remaining error is
approximation error. This is reported, not gated (`docs/BACKENDS.md`).

## CI implication

*To be settled in Phase 12.1.* Two open items:

- The Rust CI job runs `cargo clippy --all-targets --all-features` and
  `cargo test --release --all-features`. With a `tfhe`/`ckks` feature pair, `--all-features`
  enables **both backends at once** — either the features must be designed to coexist, or the
  CI invocation changes.
- If nightly is required (prerequisite 1 above), the CKKS job needs its own toolchain step,
  and the TFHE job should stay on stable so a poulpy toolchain problem cannot take down the
  reference backend's gate.
