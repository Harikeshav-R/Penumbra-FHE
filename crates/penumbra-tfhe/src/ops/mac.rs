#![cfg_attr(not(test), allow(dead_code))]

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use tfhe::integer::{ServerKey, SignedRadixCiphertext};

use crate::keys::MESSAGE_BITS;
use crate::width::{resize, resize_tensor, value_blocks};

/// Width cache for input ciphertexts at various radix widths.
pub(crate) struct WidthCache<'a> {
    pub(crate) by_width: BTreeMap<usize, Vec<Cow<'a, SignedRadixCiphertext>>>,
}

impl<'a> WidthCache<'a> {
    pub(crate) fn build(
        sk: &ServerKey,
        inputs: &'a [SignedRadixCiphertext],
        widths: &BTreeSet<usize>,
    ) -> Self {
        let mut by_width = BTreeMap::new();
        for &w in widths {
            by_width.insert(w, resize_tensor(sk, inputs, w));
        }
        Self { by_width }
    }

    pub(crate) fn get(&self, width: usize, idx: usize) -> &SignedRadixCiphertext {
        self.by_width
            .get(&width)
            .and_then(|v| v.get(idx))
            .map(|c| c.as_ref())
            .unwrap_or_else(|| panic!("WidthCache: no entry for width {width}, idx {idx}"))
    }
}

/// Compute the unique ciphertext widths needed in [`WidthCache`] for progressive MAC.
pub(crate) fn mac_cache_widths<'g>(
    groups: impl IntoIterator<Item = &'g BTreeMap<i64, Vec<usize>>>,
    in_bits: usize,
    acc_blocks: usize,
) -> BTreeSet<usize> {
    let mut widths = BTreeSet::new();
    for g in groups {
        for idxs in g.values() {
            let k = idxs.len();
            let sum_bits = in_bits + penumbra_core::bitwidth::ceil_log2(k);
            widths.insert(value_blocks(sum_bits, acc_blocks));
        }
    }
    widths
}

/// Evaluates a weighted MAC using progressive widening and deferred carry propagation
/// (the winning variant from the Phase-14 spike).
pub(crate) fn evaluate_weighted_mac(
    sk: &ServerKey,
    cache: &WidthCache,
    groups: &BTreeMap<i64, Vec<usize>>,
    bias: i64,
    in_bits: usize,
    acc_blocks: usize,
) -> SignedRadixCiphertext {
    let mut pos: Vec<SignedRadixCiphertext> = Vec::new();
    let mut neg: Vec<SignedRadixCiphertext> = Vec::new();

    for (&w, idxs) in groups {
        let k = idxs.len();
        let sum_bits = in_bits + penumbra_core::bitwidth::ceil_log2(k);
        let g = value_blocks(sum_bits, acc_blocks);
        let s_g = if idxs.len() == 1 {
            cache.get(g, idxs[0]).clone()
        } else {
            sk.sum_ciphertexts_parallelized(idxs.iter().map(|&i| cache.get(g, i)))
                .expect("non-empty cts group")
        };
        let s = resize(sk, &s_g, acc_blocks).into_owned();

        let u = w.unsigned_abs();
        let mut pre: Vec<SignedRadixCiphertext> = Vec::with_capacity(MESSAGE_BITS);
        pre.push(s);
        for j in 1..MESSAGE_BITS {
            let needed = (0..64).any(|bit| ((u >> bit) & 1) != 0 && (bit % MESSAGE_BITS == j));
            if needed {
                let shifted = sk.unchecked_scalar_left_shift_parallelized(&pre[0], j as u64);
                pre.push(shifted);
            } else {
                pre.push(sk.create_trivial_zero_radix(acc_blocks));
            }
        }

        for bit in 0..64 {
            if ((u >> bit) & 1) != 0 {
                let block_idx = bit / MESSAGE_BITS;
                if block_idx < acc_blocks {
                    let shifted = sk.blockshift(&pre[bit % MESSAGE_BITS], block_idx);
                    if w > 0 {
                        pos.push(shifted);
                    } else {
                        neg.push(shifted);
                    }
                }
            }
        }
    }

    if bias > 0 {
        pos.push(sk.create_trivial_radix::<u64, SignedRadixCiphertext>(bias.unsigned_abs(), acc_blocks));
    } else if bias < 0 {
        neg.push(sk.create_trivial_radix::<u64, SignedRadixCiphertext>(bias.unsigned_abs(), acc_blocks));
    }

    let p = if pos.is_empty() {
        sk.create_trivial_zero_radix(acc_blocks)
    } else {
        sk.unchecked_sum_ciphertexts_vec_parallelized(pos)
            .unwrap_or_else(|| sk.create_trivial_zero_radix(acc_blocks))
    };

    if neg.is_empty() {
        p
    } else {
        let n = sk.unchecked_sum_ciphertexts_vec_parallelized(neg)
            .unwrap_or_else(|| sk.create_trivial_zero_radix(acc_blocks));
        sk.sub_parallelized(&p, &n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_cache_widths() {
        let mut g1 = BTreeMap::new();
        g1.insert(3, vec![0, 1]); // k=2, in_bits=2 => sum_bits = 3 => value_blocks(3, 7) = 2
        g1.insert(-5, vec![2]);   // k=1, in_bits=2 => sum_bits = 2 => value_blocks(2, 7) = 2
        let widths = mac_cache_widths([&g1], 2, 7);
        assert_eq!(widths, [2].into_iter().collect());
    }

    #[test]
    fn test_evaluate_weighted_mac_correctness() {
        let (ck, sk) = crate::keys::keygen(7);
        let raw_inputs: Vec<i64> = vec![3, -1, 2];
        let encrypted: Vec<SignedRadixCiphertext> =
            raw_inputs.iter().map(|&v| ck.encrypt_signed(v)).collect();

        let mut groups = BTreeMap::new();
        groups.insert(3, vec![0, 2]); // 3 * (3 + 2) = 15
        groups.insert(-2, vec![1]);   // -2 * (-1) = 2
        let bias = -5;                // cleartext: 15 + 2 - 5 = 12

        let widths = mac_cache_widths([&groups], 3, 7);
        let cache = WidthCache::build(&sk, &encrypted, &widths);

        let out = evaluate_weighted_mac(&sk, &cache, &groups, bias, 3, 7);
        assert_eq!(ck.decrypt_signed::<i64>(&out), 12);
    }
}
