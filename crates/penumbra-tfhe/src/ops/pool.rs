//! `Pool` — spatial pooling over a flattened feature map (`PROJECT.md` §6).

pub use penumbra_core::ir::PoolMode;

use penumbra_core::ops::Op;
use rayon::prelude::*;
use tfhe::integer::SignedRadixCiphertext;

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
    pub padding: usize,
}

/// In-bounds input indices covered by output position `o` along one axis (padded taps skipped).
fn axis_taps(
    o: usize,
    stride: usize,
    padding: usize,
    pool: usize,
    len: usize,
) -> impl Iterator<Item = usize> {
    (0..pool)
        .filter_map(move |k| (o * stride + k).checked_sub(padding))
        .filter(move |&i| i < len)
}

impl Pool {
    fn out_dims(&self) -> (usize, usize) {
        let out_h = (self.in_h + 2 * self.padding - self.pool_h) / self.stride + 1;
        let out_w = (self.in_w + 2 * self.padding - self.pool_w) / self.stride + 1;
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

    fn eval_window(
        &self,
        ctx: &EvalCtx,
        inputs: &CtVec,
        c: usize,
        oy: usize,
        ox: usize,
    ) -> SignedRadixCiphertext {
        let sk = ctx.sk;
        let base = c * self.in_h * self.in_w;
        let mut window = Vec::new();
        for y in axis_taps(oy, self.stride, self.padding, self.pool_h, self.in_h) {
            for x in axis_taps(ox, self.stride, self.padding, self.pool_w, self.in_w) {
                window.push(&inputs[base + y * self.in_w + x]);
            }
        }
        match self.mode {
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
            self.padding < self.pool_h && self.padding < self.pool_w,
            "Pool padding must be smaller than the window"
        );
        assert!(
            self.pool_h <= self.in_h + 2 * self.padding
                && self.pool_w <= self.in_w + 2 * self.padding,
            "Pool window ({}x{}) must fit the padded input ({}x{})",
            self.pool_h,
            self.pool_w,
            self.in_h + 2 * self.padding,
            self.in_w + 2 * self.padding
        );

        let (out_h, out_w) = self.out_dims();
        let plane = out_h * out_w;
        (0..self.channels * plane)
            .into_par_iter()
            .map(|idx| {
                let c = idx / plane;
                let oy = (idx % plane) / out_w;
                let ox = idx % out_w;
                self.eval_window(ctx, inputs, c, oy, ox)
            })
            .collect()
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        match self.mode {
            PoolMode::Avg => input_bits + self.sum_growth(),
            PoolMode::Max => input_bits,
        }
    }

    fn cost(&self, _input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        let (out_h, out_w) = self.out_dims();
        let mut sum_ops: u64 = 0;
        for oy in 0..out_h {
            let rows =
                axis_taps(oy, self.stride, self.padding, self.pool_h, self.in_h).count() as u64;
            for ox in 0..out_w {
                let cols =
                    axis_taps(ox, self.stride, self.padding, self.pool_w, self.in_w).count() as u64;
                let taps = rows * cols;
                if taps > 1 {
                    sum_ops += taps - 1;
                }
            }
        }
        let ops = (self.channels as u64) * sum_ops;
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
