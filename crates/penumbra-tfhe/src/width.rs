#![allow(dead_code)]

use std::borrow::Cow;

use rayon::prelude::*;
use tfhe::integer::{IntegerCiphertext, ServerKey, SignedRadixCiphertext};

use crate::keys::{magnitude_bits, radix_capacity_bits, MESSAGE_BITS};

/// Magnitude semantics: Layer-2 `bits` bound a value's magnitude: `|v| < 2^bits`.
/// With 1 sign bit and `MESSAGE_BITS` per block, `value_blocks` computes
/// `ceil((bits + 1) / MESSAGE_BITS)`, clamped to `[1, cap]`.
pub(crate) fn value_blocks(bits: usize, cap: usize) -> usize {
    (bits + 1).div_ceil(MESSAGE_BITS).max(1).min(cap)
}

/// Signed accumulator proof: rule `max(sum_bits, bias_bits) + 2` gives `|acc| < 2^(m+1)`,
/// which fits in `m + 2 = out_bits` signed bits without needing an extra sign bit.
/// Computes `ceil(bits / MESSAGE_BITS)`, clamped to `[1, cap]`.
pub(crate) fn signed_blocks(bits: usize, cap: usize) -> usize {
    bits.div_ceil(MESSAGE_BITS).max(1).min(cap)
}

/// Sizing for comparisons or clamping against a plaintext scalar `v`.
pub(crate) fn scalar_blocks(v: i64, cap: usize) -> usize {
    value_blocks(magnitude_bits(v), cap)
}

/// Sizing for an input tensor: max over elements of `min(actual_blocks, value_blocks(bits, cap))`.
/// Returns `value_blocks(bits, cap)` for an empty tensor.
pub(crate) fn tensor_blocks(cts: &[SignedRadixCiphertext], bits: usize, cap: usize) -> usize {
    let target = value_blocks(bits, cap);
    if cts.is_empty() {
        return target;
    }
    cts.iter()
        .map(|ct| ct.blocks().len().min(target))
        .max()
        .unwrap_or(target)
}

/// Resize a ciphertext to `blocks` radix blocks.
/// Returns `Cow::Borrowed` when `ct.blocks().len() == blocks`.
/// Trimming is free; sign-extension costs 1 PBS.
pub(crate) fn resize<'a>(
    sk: &ServerKey,
    ct: &'a SignedRadixCiphertext,
    blocks: usize,
) -> Cow<'a, SignedRadixCiphertext> {
    if ct.blocks().len() == blocks {
        Cow::Borrowed(ct)
    } else {
        Cow::Owned(sk.cast_to_signed(ct.clone(), blocks))
    }
}

/// Resize a tensor in parallel using rayon.
pub(crate) fn resize_tensor<'a>(
    sk: &ServerKey,
    cts: &'a [SignedRadixCiphertext],
    blocks: usize,
) -> Vec<Cow<'a, SignedRadixCiphertext>> {
    cts.par_iter().map(|ct| resize(sk, ct, blocks)).collect()
}

/// Sizing carrier passed to node operators.
#[derive(Debug, Clone)]
pub(crate) enum NodeWidths {
    /// Built without derived widths: every tensor at the model ceiling `ctx.num_blocks`.
    Uniform,
    /// Layer-2 widths in node input / output declaration order.
    PerTensor {
        inputs: Vec<usize>,
        outputs: Vec<usize>,
    },
}

impl NodeWidths {
    pub(crate) fn input_bits(&self, i: usize, nb: usize) -> usize {
        match self {
            Self::Uniform => radix_capacity_bits(nb),
            Self::PerTensor { inputs, .. } => inputs
                .get(i)
                .copied()
                .unwrap_or_else(|| panic!("NodeWidths: no width for input {i}")),
        }
    }

    pub(crate) fn output_bits(&self, i: usize, nb: usize) -> usize {
        match self {
            Self::Uniform => radix_capacity_bits(nb),
            Self::PerTensor { outputs, .. } => outputs
                .get(i)
                .copied()
                .unwrap_or_else(|| panic!("NodeWidths: no width for output {i}")),
        }
    }

    /// Extracts `(input_bits, output_bits, acc_blocks)` for single-input, single-output linear operations.
    pub(crate) fn linear_op_blocks(&self, num_blocks: usize) -> (usize, usize, usize) {
        let ib = self.input_bits(0, num_blocks);
        let ob = self.output_bits(0, num_blocks);
        let acc = signed_blocks(ob, num_blocks);
        (ib, ob, acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_asserts() {
        assert_eq!(value_blocks(2, 9), 2);
        assert_eq!(value_blocks(3, 9), 2);
        assert_eq!(value_blocks(0, 9), 1);
        assert_eq!(value_blocks(20, 9), 9);

        assert_eq!(signed_blocks(14, 9), 7);
        assert_eq!(signed_blocks(17, 9), 9);
        assert_eq!(signed_blocks(0, 9), 1);

        assert_eq!(scalar_blocks(0, 9), 1);
        assert_eq!(scalar_blocks(-4, 9), 2);
        assert_eq!(scalar_blocks(8, 9), 3);
    }

    #[test]
    fn test_keyed_resize() {
        let (ck, sk) = crate::keys::keygen(9);
        let ct_neg5 = ck.encrypt_signed(-5);
        let resized = resize(&sk, &ct_neg5, 3);
        assert_eq!(resized.blocks().len(), 3);
        assert_eq!(ck.decrypt_signed::<i64>(&resized), -5);

        let resized_back = resize(&sk, &resized, 9);
        assert_eq!(resized_back.blocks().len(), 9);
        assert_eq!(ck.decrypt_signed::<i64>(&resized_back), -5);

        let ct_pos7 = ck.encrypt_signed(7);
        let resized_pos7 = resize(&sk, &ct_pos7, 2);
        assert_eq!(resized_pos7.blocks().len(), 2);
        assert_eq!(ck.decrypt_signed::<i64>(&resized_pos7), 7);
    }

    #[test]
    fn test_linear_op_blocks() {
        let w = NodeWidths::PerTensor {
            inputs: vec![4],
            outputs: vec![12],
        };
        let (ib, ob, acc) = w.linear_op_blocks(8);
        assert_eq!(ib, 4);
        assert_eq!(ob, 12);
        assert_eq!(acc, signed_blocks(12, 8)); // 12 / 2 = 6 blocks

        let u = NodeWidths::Uniform;
        let (u_ib, u_ob, u_acc) = u.linear_op_blocks(8);
        assert_eq!(u_ib, radix_capacity_bits(8));
        assert_eq!(u_ob, radix_capacity_bits(8));
        assert_eq!(u_acc, 8);
    }
}
