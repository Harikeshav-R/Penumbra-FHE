//! `Pool` — spatial pooling over a flattened feature map (`PROJECT.md` §6).

pub use penumbra_core::ir::PoolMode;

use penumbra_core::ops::Op;

use super::{CtVec, EvalCtx};
use crate::backend::TfheBackend;

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
    fn out_dims(&self) -> (usize, usize) {
        let out_h = (self.in_h - self.pool_h) / self.stride + 1;
        let out_w = (self.in_w - self.pool_w) / self.stride + 1;
        (out_h, out_w)
    }

    fn sum_growth(&self) -> usize {
        let k = self.pool_h * self.pool_w;
        if k <= 1 {
            0
        } else {
            usize::BITS as usize - (k - 1).leading_zeros() as usize
        }
    }
}

impl Op<TfheBackend> for Pool {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
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
                    let mut window = Vec::with_capacity(self.pool_h * self.pool_w);
                    for ky in 0..self.pool_h {
                        for kx in 0..self.pool_w {
                            let y = oy * self.stride + ky;
                            let x = ox * self.stride + kx;
                            window.push(&inputs[base + y * self.in_w + x]);
                        }
                    }

                    let pooled = match self.mode {
                        PoolMode::Avg => {
                            if window.len() == 1 {
                                window[0].clone()
                            } else {
                                sk.sum_ciphertexts_parallelized(window.iter().copied())
                                    .unwrap_or_else(|| window[0].clone())
                            }
                        }
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
            PoolMode::Avg => input_bits + self.sum_growth(),
            PoolMode::Max => input_bits,
        }
    }

    fn cost(&self, _input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        let (out_h, out_w) = self.out_dims();
        let out_elems = (self.channels * out_h * out_w) as u64;
        let k = (self.pool_h * self.pool_w) as u64;
        let ops = if k > 1 { out_elems * (k - 1) } else { 0 };
        let mut counters = Vec::new();
        if ops > 0 {
            match self.mode {
                PoolMode::Avg => counters.push(("ct_add", ops)),
                PoolMode::Max => counters.push(("cmp_pbs_ops", ops)),
            }
        }
        counters
    }
}
