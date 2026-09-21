# CKKS Notes

Working notes on the CKKS primitives the `penumbra-ckks` backend builds on, the parameter
profile, and empirical cost. The sibling of [`NOTES-tfhe.md`](./NOTES-tfhe.md); it closes the
Phase-12.0 spike deliverable (`ROADMAP.md` Phase 12) the same way that document closed
Phase 1.

> **Status: Phase-12.0 spike complete.** The parameter profile, primitives, and cost tables
> below were measured on Apple Silicon (AArch64, `poulpy-cpu-arm` / `FFT64Neon`) using
> `poulpy-ckks` 0.8.3 (`crates/spike-ckks`).

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
| 2026-09-21 | 0.8.3 | `test_suite` and test helpers gated behind `test-utils` feature | Enabled `features = ["test-utils"]` on `poulpy-ckks` in spike |

## ⚠️ Toolchain and platform prerequisites

**Resolved in Phase 12.0:**

1. **Nightly Rust is required.** `poulpy-hal 0.8.3` declares `#![feature(associated_type_defaults)]`
   and `poulpy-ckks 0.8.3` declares `#![feature(f128)]`. It fails on stable Rust with
   `error[E0554]: #![feature] may not be used on the stable release channel`.
   **Resolution:** In accordance with the "Nightly creep" invariant (`AGENTS.md`), the CKKS spike
   lives in `crates/spike-ckks` with its own `rust-toolchain.toml` set to `nightly`. The reference
   TFHE backend in `runtime/` remains on stable Rust (`rust-version = "1.83"`).
2. **The HAL backend is architecture-specific.** On Apple Silicon (AArch64), `poulpy-cpu-arm`
   with NEON acceleration (`FFT64Neon`) is active and verified. In CI and x86-64 environments,
   `poulpy-cpu-avx` (`FFT64Avx`) with `poulpy-cpu-ref` (`FFT64Ref`) as portable fallback is used.
   All Apple Silicon benchmarks are pinned to `FFT64Neon`.
3. **Two FHE libraries in one dependency graph.** `tfhe 1.8.1` and `poulpy-ckks 0.8.3` compile and
   execute concurrently in the same binary (`crates/spike-ckks/tests/spike_correctness.rs`)
   without any symbol clashes or dependency collisions.

## Parameter profile

*Measured in Phase 12.0 with `poulpy-ckks` presets (`FFT64_PARAMS_F64`).*

| Quantity | Value | Notes |
|---|---|---|
| Ring degree (`N`) | 256 (spike) / 1024 / 2048 (full models) | slot count is `N/2` (128 complex slots in spike) |
| Scale (`log_delta`) | 30 bits (`FFT64`) / 40–45 bits (`NTT4x30`) | precision per level; the accuracy/depth lever |
| Level budget (`log_budget`) | 122 bits (`k = 152`) | multiplicative depth headroom before bootstrapping is needed |
| Slot kind (`SlotsKind`) | `SlotsKind::Complex` | packing real values into the real part of conjugate-symmetric slots |
| Security level | 128-bit quantum security (`default_sigma`) | matches the TFHE profile's 128-bit security margin |

`poulpy-ckks` ships ready-made parameter sets in its `presets` module. Start there, exactly
as the TFHE side starts from the `tfhe-rs` default secure profile (`AGENTS.md` §7). Never
hand-roll crypto parameters.

## The primitives everything composes from

Structured to mirror `NOTES-tfhe.md`'s "two primitives" framing, because the contrast is the
whole point.

1. **Plaintext arithmetic and adds (cheap, SIMD-batched)** — `ckks_mul_pt_vec_into`,
   `ckks_add_pt_vec_into`, `ckks_add_pt_const_into`. The `Linear`/`Conv2d` core: encrypted data
   combined with plaintext weights, operating across all slots simultaneously.
2. **Rescale (cheap, but spends the budget)** — the CKKS analogue of narrowing an
   accumulator. Consumes a level (`log_delta` bits).
3. **Polynomial evaluation (the expensive one, and the approximate one)** — CKKS has **no
   programmable bootstrap and no lookup table**. Every nonlinearity — ReLU, the `Requant`
   clamp, `Argmax`'s comparison — becomes a fitted Chebyshev polynomial evaluated via
   `PowerBasis` and Baby-Step Giant-Step (`ckks_eval_poly_real_const_coeffs_from_power_basis`).
4. **Bootstrapping (very expensive)** — `ModUp`, `CoeffsToSlots`, `SlotsToCoeffs`, `EvalMod`,
   plus a `PaCo` variant. Needed only if a model exceeds the level budget. For Penumbra's
   small models the goal is to **stay leveled** and never bootstrap.

The mapping from Penumbra's op vocabulary onto these primitives is tabulated in
[`docs/BACKENDS.md`](./BACKENDS.md#the-backend-contract).

## Empirical cost

*Measured by `crates/spike-ckks` in `--release` on Apple Silicon (AArch64, `FFT64Neon`, 128 slots):*

| Quantity | Value |
|---|---|
| keygen + a single round-trip | ~0.42 ms (368 µs keygen + 49 µs roundtrip) |
| one packed `Linear` (spike operation, 128 slots) | 25.4 µs (~0.025 ms) |
| one ReLU polynomial at degree 3 (128 slots) | 159 µs (~0.16 ms) |
| one ReLU polynomial at degree 7 (128 slots) | 297 µs (~0.30 ms) |
| one ReLU polynomial at degree 15 (128 slots) | 671 µs (~0.67 ms) |
| full Phase-2 logreg inference, per sample | TBD (Phase 12.2 / 12.4) |
| full Phase-4 CNN inference, per sample | TBD (Phase 12.2 / 12.4) |

> ⚠️ Always benchmark in `--release`. Debug FHE is orders of magnitude slower and the numbers
> are meaningless (`docs/DEVELOPMENT.md`).

## Accuracy notes

CKKS is approximate by construction, so unlike the TFHE backend there is no bit-exactness to
lean on as a truth oracle. The correctness gate is a **declared, per-model error bound**
against the same quantized-cleartext reference, with the measured error always reported
(`docs/BACKENDS.md`, "one invariant, two comparators").

Empirical measurements from the Phase-12.0 spike demonstrate the clean decoupling between
cryptographic noise and approximation error:

- **Decryption roundtrip error:** $8.91 \times 10^{-8}$
- **Packed Linear layer error vs cleartext float:** $1.88 \times 10^{-7}$
- **ReLU polynomial evaluation vs true ReLU:**
  - Degree 3: max error $6.25 \times 10^{-2}$ (cryptographic noise contribution: $2.81 \times 10^{-7}$)
  - Degree 7: max error $2.30 \times 10^{-2}$ (cryptographic noise contribution: $1.53 \times 10^{-7}$)
  - Degree 15: max error $9.97 \times 10^{-3}$ (cryptographic noise contribution: $3.68 \times 10^{-7}$)

The cryptographic evaluation noise is on the order of $10^{-7}$, proving that $>99.999\%$ of the
total error is purely mathematical approximation error from the chosen Chebyshev polynomial degree.

## CI implication

- The Rust CI job for `runtime/` continues to run on stable Rust (`dtolnay/rust-toolchain@stable`)
  with `cargo clippy --all-targets --all-features` and `cargo test --release --all-features`.
- A dedicated CI job `ckks-spike` in `.github/workflows/ci.yml` runs on `ubuntu-latest` using
  `dtolnay/rust-toolchain@nightly` in `crates/spike-ckks`.
- This ensures the CKKS backend is tested in CI without introducing toolchain instability to
  the reference TFHE backend.
