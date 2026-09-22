# CKKS Notes

Working notes on the CKKS primitives the `penumbra-ckks` backend builds on, the parameter
profile, and empirical cost. The sibling of [`NOTES-tfhe.md`](./NOTES-tfhe.md); it closes the
Phase-12.0 spike deliverable (`ROADMAP.md` Phase 12) the same way that document closed
Phase 1.

> **Status: Phase-12.2 backend complete.** The parameter profile, primitives, and cost tables
> below were measured on Apple Silicon (AArch64, `poulpy-cpu-arm` / `FFT64Neon`) using
> `poulpy-ckks` 0.8.3 (`crates/penumbra-ckks`).

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
| 2026-09-21 | 0.8.3 | `test_suite` and test helpers gated behind `test-utils` feature | `penumbra-ckks` does not enable `test-utils`; all test-only helpers replaced with public API methods (`ckks_encode_reim_into`, `glwe_secret_fill_ternary_prob`, `glwe_tensor_key_encrypt_sk`, `glwe_automorphism_key_encrypt_sk`) |
| 2026-09-21 | 0.8.3 | `poulpy-ckks::presets` contains only bootstrapping plans, no general parameter preset | `penumbra-ckks::params::DEFAULT_PARAMS` defines its own 128-bit secure parameter profile sized against HomomorphicEncryption.org standard tables |
| 2026-09-21 | 0.8.3 | No poulpy type derives `serde` (`serde` is not a dependency of any poulpy crate) | Hand-rolled wire format over `poulpy_hal::layouts::{WriterTo, ReaderFrom}` wrapped in a serde envelope carrying layout metadata + `CKKSMeta` |
| 2026-09-21 | 0.8.3 | Prepared (DFT-domain) keys implement neither `WriterTo` nor `ReaderFrom` | Persisted standard keys and re-prepared on load via `glwe_automorphism_key_prepare` / `prepare_tensor_key` |
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

*Calibrated in Phase 12.2 (`crates/penumbra-ckks/src/params.rs`, `DEFAULT_PARAMS`).*

| Quantity | Value | Notes |
|---|---|---|
| Ring degree (`N`) | 16384 | 8192 complex slots; 128-bit classical security up to `k <= 438` |
| Torus width (`k`) | 360 bits | accommodates multi-layer depth without bootstrapping |
| Scale (`log_delta`) | 30 bits | precision per level; standard fixed-point scaling factor |
| Level budget (`log_budget`) | 330 bits | `k - log_delta`; multiplicative headroom for entire graphs |
| Transform dimension (`lt_slots`) | 256 | power of two covering the largest committed tensor (faces 256) |
| BSGS giant step | 16 | `sqrt(256)`; fixes automorphism key count independently of model |
| Log sparsity | 5 | `log2(8192) - log2(256)`; native sparse slot embedding |
| Automorphism key count | 30 keys | `15` baby steps + `15` giant steps |
| Gadget decomposition | `base2k = 19`, `dsize = 2`, `rank = 1` | standard key decomposition parameters |
| Secret distribution | uniform ternary (`prob = 2/3`) | standard distribution assumption matching security tables |
| Max polynomial degree | 15 | single override knob (`depth = 4` via BSGS min-depth) |
| Security level | 128-bit classical security | verified against HomomorphicEncryption.org standard table (`log q <= 438`) |
| Single packed ciphertext size | 4.75 MB | 4,980,843 bytes |
| Key generation time | ~2.06 s | Apple Silicon `FFT64Neon` |
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

### Phase-12.2 calibrated model results

*Measured on Apple Silicon (`FFT64Neon`) using `cargo +nightly run -p penumbra-ckks --features ckks --release --example calibrate`:*

| Model / Fixture | Graph Topology | Depth Check | Measured Max \|Err\| | Declared Bound | Match Label |
|---|---|---|---|---|---|
| Phase 6 sklearn | `Linear(64→10)` | PASS | $1.77 \times 10^{-4}$ | `1.0e-3` | YES |
| Phase 2 logreg | `Linear(64→1) → Argmax` | PASS | $3.70 \times 10^{-1}$ | `5.0e-1` | YES |
| Phase 4 cnn | `Conv2d → Requant → Pool(avg) → Linear` | PASS | $1.53 \times 10^{1}$ | `3.0e1` | YES |
| Phase 5 digits | `Conv2d → Requant(per-ch) → Linear` | PASS | $1.88 \times 10^{2}$ | `2.5e2` | YES |
| Phase 5 qat | `Conv2d → Requant(per-ch) → Linear` | PASS | $2.38 \times 10^{2}$ | `3.0e2` | YES |
| Phase 6 onnx | `Conv2d → Requant(per-ch) → Linear` | PASS | $1.88 \times 10^{2}$ | `2.5e2` | YES |
| Phase 7 faces | `Conv2d → Requant(per-ch) → Linear` | PASS | $1.05 \times 10^{2}$ | `1.5e2` | YES |
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
- Dedicated CI jobs `ckks-backend` and `ckks-spike` in `.github/workflows/ci.yml` run on `ubuntu-latest`
  using `dtolnay/rust-toolchain@nightly` to lint and test `penumbra-ckks` and `crates/spike-ckks`.
- This ensures the CKKS backend is tested in CI without introducing toolchain instability to
  the reference TFHE backend.
