# CKKS Standalone Spike (`crates/spike-ckks`)

Standalone exploratory spike for **Phase 12.0** of [`ROADMAP.md`](../../ROADMAP.md).

## Purpose

Proves the cryptographic plumbing for the upcoming `penumbra-ckks` backend against [`poulpy-ckks`](https://github.com/phantomzone-org/poulpy) v0.8.3 before any trait boundary is formalized or workspace refactor is attempted.

## Questions Answered

1. **Toolchain:** `poulpy-ckks 0.8.3` depends on `#![feature(associated_type_defaults)]` (in `poulpy-hal`) and `#![feature(f128)]` (in `poulpy-ckks`), requiring a Nightly Rust toolchain. This spike pins `nightly` in its local `rust-toolchain.toml`, insulating the stable `runtime/` (TFHE) crate from Nightly creep.
2. **Platform & Hardware Acceleration:** Uses `poulpy-cpu-arm` (`FFT64Neon`) on Apple Silicon (AArch64) and `poulpy-cpu-avx` (`FFT64Avx`) with `poulpy-cpu-ref` portable fallback on x86-64.
3. **Coexistence:** `tfhe 1.8.1` and `poulpy-ckks 0.8.3` compile and execute concurrently in the same binary without dependency or symbol collisions.
4. **Packed Operations:** Realizes packed plaintext-weight matrix multiplication (`ckks_mul_pt_vec_into` + `ckks_add_pt_vec_into`) with $< 2 \times 10^{-7}$ error.
5. **Non-Linearities via Polynomials:** Realizes `ReLU` via Chebyshev minimax approximation (`poulpy_ckks::approximation::minimax`) and Baby-Step Giant-Step (BSGS) evaluation.
6. **Error Decoupling:** Proves that $>99.999\%$ of total error in homomorphic evaluation is mathematical approximation error from the polynomial fit, with cryptographic noise remaining bounded $< 10^{-6}$.
7. **Slot-Packing Fork:** Confirms Option B (tensor slot packing) as essential: evaluating 128 elements in one ciphertext takes $\sim 0.26$ ms in `--release`, whereas scalar ciphertexts would be over $100\times$ slower.

## Running

```bash
# Run tests
cargo test --release

# Run benchmark driver
cargo run --release
```
