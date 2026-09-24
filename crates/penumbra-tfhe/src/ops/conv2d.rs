//! `Conv2d` — 2-D convolution against **plaintext** kernel weights (`PROJECT.md` §6).
//!
//! Covers CNNs (MNIST, faces). Like [`crate::ops::linear::Linear`], this is the *cheap*
//! regime (`PROJECT.md` §5): the input is encrypted but the kernel is plaintext, so each
//! output is `Σ (ciphertext × plaintext_weight) + plaintext_bias` — scalar-multiplies and
//! additions only, **no programmable bootstrap**.

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;
use tfhe::integer::SignedRadixCiphertext;

use penumbra_core::ops::Op;

use super::{CtVec, EvalCtx};
use crate::backend::TfheBackend;

/// 2-D convolution with plaintext quantized kernel weights.
pub struct Conv2d {
    /// Quantized kernel, row-major `[out_channels][in_channels*kernel_h*kernel_w]`.
    pub weights: Vec<Vec<i64>>,
    /// Quantized bias, one per output channel.
    pub bias: Vec<i64>,
    /// Quantized weight bit-width (magnitude+sign), used by the bit-width growth rule.
    pub weight_bits: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub in_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    fn out_dims(&self) -> (usize, usize) {
        let out_h = (self.in_h + 2 * self.padding - self.kernel_h) / self.stride + 1;
        let out_w = (self.in_w + 2 * self.padding - self.kernel_w) / self.stride + 1;
        (out_h, out_w)
    }

    fn fan_in(&self) -> usize {
        self.in_channels * self.kernel_h * self.kernel_w
    }

    /// Count non-zero in-bounds multiply-accumulate operations.
    ///
    /// The guards here must stay strictly in sync with [`Conv2d::eval`].
    pub fn mac_count(&self) -> u64 {
        let (out_h, out_w) = self.out_dims();
        let mut count = 0u64;
        for kernel in &self.weights {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    for ic in 0..self.in_channels {
                        for ky in 0..self.kernel_h {
                            let iy = (oy * self.stride + ky) as isize - self.padding as isize;
                            for kx in 0..self.kernel_w {
                                let ix = (ox * self.stride + kx) as isize - self.padding as isize;
                                let w = kernel[(ic * self.kernel_h + ky) * self.kernel_w + kx];
                                if w == 0
                                    || iy < 0
                                    || ix < 0
                                    || iy as usize >= self.in_h
                                    || ix as usize >= self.in_w
                                {
                                    continue;
                                }
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        count
    }

    fn primitive_counts(&self) -> (u64, u64) {
        let (out_h, out_w) = self.out_dims();
        let mut total_scalar_mul = 0u64;
        let mut total_ct_add = 0u64;

        for kernel in &self.weights {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut distinct_weights = BTreeSet::new();
                    let mut nonzero_taps = 0u64;

                    for ic in 0..self.in_channels {
                        for ky in 0..self.kernel_h {
                            let iy = (oy * self.stride + ky) as isize - self.padding as isize;
                            for kx in 0..self.kernel_w {
                                let ix = (ox * self.stride + kx) as isize - self.padding as isize;
                                let w = kernel[(ic * self.kernel_h + ky) * self.kernel_w + kx];
                                if w == 0
                                    || iy < 0
                                    || ix < 0
                                    || iy as usize >= self.in_h
                                    || ix as usize >= self.in_w
                                {
                                    continue;
                                }
                                distinct_weights.insert(w);
                                nonzero_taps += 1;
                            }
                        }
                    }

                    total_scalar_mul += distinct_weights.len() as u64;
                    if nonzero_taps > 0 {
                        total_ct_add += nonzero_taps - 1;
                    }
                }
            }
        }
        (total_scalar_mul, total_ct_add)
    }

    fn eval_point(
        &self,
        ctx: &EvalCtx,
        inputs: &CtVec,
        kernel: &[i64],
        bias: i64,
        oy: usize,
        ox: usize,
    ) -> SignedRadixCiphertext {
        let sk = ctx.sk;
        let in_hw = self.in_h * self.in_w;
        let mut groups: BTreeMap<i64, Vec<&SignedRadixCiphertext>> = BTreeMap::new();

        for ic in 0..self.in_channels {
            let in_base = ic * in_hw;
            for ky in 0..self.kernel_h {
                let iy = (oy * self.stride + ky) as isize - self.padding as isize;
                for kx in 0..self.kernel_w {
                    let ix = (ox * self.stride + kx) as isize - self.padding as isize;
                    let w = kernel[(ic * self.kernel_h + ky) * self.kernel_w + kx];
                    if w == 0
                        || iy < 0
                        || ix < 0
                        || iy as usize >= self.in_h
                        || ix as usize >= self.in_w
                    {
                        continue;
                    }
                    let idx = in_base + iy as usize * self.in_w + ix as usize;
                    groups.entry(w).or_default().push(&inputs[idx]);
                }
            }
        }

        let mut group_terms: Vec<SignedRadixCiphertext> = Vec::with_capacity(groups.len());
        for (w, cts) in groups {
            let s = if cts.len() == 1 {
                cts[0].clone()
            } else {
                sk.sum_ciphertexts_parallelized(cts.iter().copied())
                    .expect("non-empty cts group")
            };
            let term = if w == 1 {
                s
            } else {
                sk.scalar_mul_parallelized(&s, w)
            };
            group_terms.push(term);
        }

        let acc = if group_terms.is_empty() {
            sk.create_trivial_zero_radix(ctx.num_blocks)
        } else if group_terms.len() == 1 {
            group_terms.pop().unwrap()
        } else {
            sk.sum_ciphertexts_parallelized(group_terms.iter())
                .expect("non-empty group_terms")
        };

        sk.scalar_add_parallelized(&acc, bias)
    }
}

impl Op<TfheBackend> for Conv2d {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        assert_eq!(
            inputs.len(),
            self.in_channels * self.in_h * self.in_w,
            "Conv2d input length {} != in_channels*in_h*in_w = {}*{}*{}",
            inputs.len(),
            self.in_channels,
            self.in_h,
            self.in_w
        );
        assert_eq!(
            self.weights.len(),
            self.bias.len(),
            "Conv2d must have one bias per output channel: {} kernels vs {} biases",
            self.weights.len(),
            self.bias.len()
        );
        let fan_in = self.fan_in();

        let (out_h, out_w) = self.out_dims();
        let out_channels = self.weights.len();
        for (oc, kernel) in self.weights.iter().enumerate() {
            assert_eq!(
                kernel.len(),
                fan_in,
                "Conv2d kernel row {oc} width ({}) must equal in_channels*kernel_h*kernel_w ({})",
                kernel.len(),
                fan_in
            );
        }

        let plane = out_h * out_w;
        (0..out_channels * plane)
            .into_par_iter()
            .map(|idx| {
                let oc = idx / plane;
                let oy = (idx % plane) / out_w;
                let ox = idx % out_w;
                self.eval_point(ctx, inputs, &self.weights[oc], self.bias[oc], oy, ox)
            })
            .collect()
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        let n = self.fan_in();
        let sum_growth = if n <= 1 {
            0
        } else {
            usize::BITS as usize - (n - 1).leading_zeros() as usize
        };
        let sum_bits = input_bits + self.weight_bits + sum_growth;

        let max_bias = self
            .bias
            .iter()
            .map(|b| b.unsigned_abs())
            .max()
            .unwrap_or(0);
        let bias_bits = crate::keys::magnitude_bits(max_bias as i64);

        sum_bits.max(bias_bits) + 2
    }

    fn cost(&self, _input_lens: &[usize]) -> Vec<(&'static str, u64)> {
        let (scalar_mul, ct_add) = self.primitive_counts();
        let (out_h, out_w) = self.out_dims();
        let out_elems = (self.weights.len() * out_h * out_w) as u64;
        let mut counters = Vec::new();
        if scalar_mul > 0 {
            counters.push(("scalar_mul", scalar_mul));
        }
        if ct_add > 0 {
            counters.push(("ct_add", ct_add));
        }
        if out_elems > 0 {
            counters.push(("scalar_add", out_elems));
        }
        counters
    }
}
