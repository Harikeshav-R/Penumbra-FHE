#[cfg(target_arch = "aarch64")]
pub type ActiveBackend = poulpy_cpu_arm::FFT64Neon;
#[cfg(all(target_arch = "x86_64", not(target_arch = "aarch64")))]
pub type ActiveBackend = poulpy_cpu_avx::FFT64Avx;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub type ActiveBackend = poulpy_cpu_ref::FFT64Ref;

/// Name of the active HAL backend, reported alongside every measurement:
/// `docs/BENCHMARKS.md` requires it wherever a CKKS number appears, and a silent
/// fall back to the correctness-oriented `FFT64Ref` would make latency meaningless
/// (ROADMAP Phase 12 pitfalls, "Wrong HAL backend").
pub const fn hal_backend_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        "FFT64Neon"
    }
    #[cfg(all(target_arch = "x86_64", not(target_arch = "aarch64")))]
    {
        "FFT64Avx"
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        "FFT64Ref"
    }
}
