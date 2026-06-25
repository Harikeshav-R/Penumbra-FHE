//! `Conv2d` — 2-D convolution against **plaintext** kernel weights (`PROJECT.md` §6).
//!
//! Covers CNNs (MNIST, faces). Like [`crate::ops::linear::Linear`], this is the *cheap*
//! regime (`PROJECT.md` §5): the input is encrypted but the kernel is plaintext, so each
//! output is `Σ (ciphertext × plaintext_weight) + plaintext_bias` — scalar-multiplies and
//! additions only, **no programmable bootstrap**. Conv is just `Linear` applied at every
//! spatial position with a shared kernel; the implementation is a direct (im2col-style) loop
//! over output positions reusing the same `scalar_mul + add` core.
//!
//! ## Tensor layout (shared with [`crate::ops::pool::Pool`])
//!
//! Input is a flat [`CtVec`] read as **channel-major, row-major** `[in_channels][in_h][in_w]`:
//! element `(c, y, x)` at `c*in_h*in_w + y*in_w + x`. Output is `[out_channels][out_h][out_w]`
//! in the same layout, with `out_h = (in_h + 2*padding - kernel_h)/stride + 1` (likewise
//! `out_w`). Zero padding is *virtual* — padded taps contribute nothing and are skipped (a
//! zero ciphertext times a weight is zero), so no real ciphertext zeros are materialized.
//!
//! ## Weight layout
//!
//! `weights` is row-major `[out_channels][in_channels*kernel_h*kernel_w]` — one flattened
//! kernel per output channel, the in-channel/kernel-row/kernel-col index running fastest in
//! that order. `bias` has one entry per output channel. This mirrors how the quantization
//! service flattens a PyTorch/ONNX `[out_c][in_c][kh][kw]` kernel.
//!
//! ## Bit-width growth rule (`PROJECT.md` §9)
//!
//! Identical to `Linear` with fan-in `N = in_channels * kernel_h * kernel_w`:
//! `max(sum_bits, bias_bits) + 2`, where `sum_bits = input_bits + weight_bits + ceil(log2 N)`
//! and the `+2` is one carry from the bias add plus one sign bit.

use tfhe::integer::SignedRadixCiphertext;

use super::{CtVec, EvalCtx, Op};

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
}

impl Op for Conv2d {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        // Fail loudly on a layout/shape mismatch before any crypto (`AGENTS.md` §1.4).
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

        let sk = ctx.sk;
        let (out_h, out_w) = self.out_dims();
        let in_hw = self.in_h * self.in_w;
        let out_channels = self.weights.len();
        let mut out = Vec::with_capacity(out_channels * out_h * out_w);

        for (oc, (kernel, &b)) in self.weights.iter().zip(&self.bias).enumerate() {
            assert_eq!(
                kernel.len(),
                fan_in,
                "Conv2d kernel row {oc} width ({}) must equal in_channels*kernel_h*kernel_w ({})",
                kernel.len(),
                fan_in
            );
            for oy in 0..out_h {
                for ox in 0..out_w {
                    // Accumulate the MACs for this output position (no PBS). Start from a
                    // trivial encrypted zero sized to the model's radix.
                    let mut acc: SignedRadixCiphertext =
                        sk.create_trivial_zero_radix(ctx.num_blocks);

                    // Walk the kernel taps in weight-layout order: in-channel, then kernel
                    // row, then kernel col (the index that runs fastest).
                    for ic in 0..self.in_channels {
                        let in_base = ic * in_hw;
                        for ky in 0..self.kernel_h {
                            // Signed source row before padding offset.
                            let iy = (oy * self.stride + ky) as isize - self.padding as isize;
                            for kx in 0..self.kernel_w {
                                let ix = (ox * self.stride + kx) as isize - self.padding as isize;
                                let w = kernel[(ic * self.kernel_h + ky) * self.kernel_w + kx];
                                // Skip padded taps (virtual zeros) and zero weights — both
                                // contribute nothing, and skipping zero weights also trims
                                // needless scalar-muls.
                                if w == 0
                                    || iy < 0
                                    || ix < 0
                                    || iy as usize >= self.in_h
                                    || ix as usize >= self.in_w
                                {
                                    continue;
                                }
                                let idx = in_base + iy as usize * self.in_w + ix as usize;
                                let term = sk.scalar_mul_parallelized(&inputs[idx], w);
                                acc = sk.add_parallelized(&acc, &term);
                            }
                        }
                    }

                    // Plaintext bias add (still cheap, no PBS).
                    out.push(sk.scalar_add_parallelized(&acc, b));
                }
            }
        }
        out
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        // Same derivation as `Linear` with fan-in N = in_channels*kernel_h*kernel_w. See
        // `linear.rs` for why the two contributors (summed products vs bias) are max'd and
        // why the guard is `+2` (one carry from the bias add, one sign bit).
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
        let bias_bits = crate::keys::magnitude_bits(max_bias);

        sum_bits.max(bias_bits) + 2
    }
}
