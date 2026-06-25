//! `Pool` — spatial pooling over a flattened feature map (`PROJECT.md` §6).
//!
//! Two modes:
//! - **`avg`** — emit the window **sum** (`add_parallelized`, **no PBS**, cheap). The `1/k`
//!   averaging factor is *not* applied here: it is folded into the downstream `Requant`'s
//!   rescale (`PROJECT.md` §9). This keeps pooling PBS-free and lets the quantization service
//!   own all scale math. ROADMAP Phase 4 prefers avg pool for exactly this reason.
//! - **`max`** — pairwise `max_parallelized` over the window. Each comparison is a chain of
//!   block-level PBSs, so this is *expensive*; provided for completeness but not used in the
//!   headline CNN.
//!
//! ## Tensor layout (the spatial-op convention)
//!
//! The flat [`CtVec`] is interpreted as a **channel-major, row-major** tensor of shape
//! `[channels][in_h][in_w]`: element `(c, y, x)` lives at index `c*in_h*in_w + y*in_w + x`.
//! `Conv2d` produces this same layout, so a `Conv2d → Pool` chain needs no reshape. Pooling
//! is per-channel; the output is `[channels][out_h][out_w]` in the same layout, where
//! `out_h = (in_h - pool_h)/stride + 1` and likewise for `out_w`.
//!
//! ## Bit-width growth rule (`PROJECT.md` §9)
//!
//! - `avg` (sum of `k = pool_h*pool_w` terms): `input_bits + ceil(log2 k)`.
//! - `max` (never grows the magnitude): `input_bits`.

use super::{CtVec, EvalCtx, Op};

/// Pooling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolMode {
    /// Average pool, emitted as the window **sum** (the `/k` is deferred to `Requant`).
    Avg,
    /// Max pool (pairwise `max`).
    Max,
}

/// Spatial pooling over a flattened `[channels][in_h][in_w]` feature map.
pub struct Pool {
    pub mode: PoolMode,
    pub in_h: usize,
    pub in_w: usize,
    pub channels: usize,
    pub pool_h: usize,
    pub pool_w: usize,
    pub stride: usize,
}

impl Pool {
    /// Output spatial dims `(out_h, out_w)` for this pool window (no padding — Phase 4).
    fn out_dims(&self) -> (usize, usize) {
        let out_h = (self.in_h - self.pool_h) / self.stride + 1;
        let out_w = (self.in_w - self.pool_w) / self.stride + 1;
        (out_h, out_w)
    }

    /// `ceil(log2(k))` window-sum growth, where `k = pool_h * pool_w`.
    fn sum_growth(&self) -> usize {
        let k = self.pool_h * self.pool_w;
        if k <= 1 {
            0
        } else {
            usize::BITS as usize - (k - 1).leading_zeros() as usize
        }
    }
}

impl Op for Pool {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        // Fail loudly on a layout mismatch before doing any (expensive) crypto (`AGENTS.md`
        // §1.4): the flat vector must be exactly channels*in_h*in_w long.
        assert_eq!(
            inputs.len(),
            self.channels * self.in_h * self.in_w,
            "Pool input length {} != channels*in_h*in_w = {}*{}*{}",
            inputs.len(),
            self.channels,
            self.in_h,
            self.in_w
        );
        assert!(
            self.pool_h > 0 && self.pool_w > 0 && self.stride > 0,
            "Pool window and stride must be positive"
        );
        assert!(
            self.pool_h <= self.in_h && self.pool_w <= self.in_w,
            "Pool window ({}x{}) must fit the input ({}x{})",
            self.pool_h,
            self.pool_w,
            self.in_h,
            self.in_w
        );

        let sk = ctx.sk;
        let (out_h, out_w) = self.out_dims();
        let mut out = Vec::with_capacity(self.channels * out_h * out_w);

        for c in 0..self.channels {
            let base = c * self.in_h * self.in_w;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    // Gather the window's ciphertexts (channel-major, row-major indexing).
                    let mut window = Vec::with_capacity(self.pool_h * self.pool_w);
                    for ky in 0..self.pool_h {
                        for kx in 0..self.pool_w {
                            let y = oy * self.stride + ky;
                            let x = ox * self.stride + kx;
                            window.push(&inputs[base + y * self.in_w + x]);
                        }
                    }

                    let pooled = match self.mode {
                        // Sum the window (no PBS). The `/k` averaging is deferred to Requant.
                        PoolMode::Avg => {
                            let mut acc = window[0].clone();
                            for &ct in &window[1..] {
                                acc = sk.add_parallelized(&acc, ct);
                            }
                            acc
                        }
                        // Pairwise max over the window (expensive: comparison PBSs).
                        PoolMode::Max => {
                            let mut acc = window[0].clone();
                            for &ct in &window[1..] {
                                acc = sk.max_parallelized(&acc, ct);
                            }
                            acc
                        }
                    };
                    out.push(pooled);
                }
            }
        }
        out
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        match self.mode {
            // Summing k terms adds ceil(log2 k) bits; the values stay signed so no extra
            // sign bit beyond what input_bits already carries.
            PoolMode::Avg => input_bits + self.sum_growth(),
            // Max selects one of the inputs — it never grows the magnitude.
            PoolMode::Max => input_bits,
        }
    }
}
