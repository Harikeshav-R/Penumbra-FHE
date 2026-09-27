#![cfg_attr(not(test), allow(dead_code))]

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use tfhe::integer::{ServerKey, SignedRadixCiphertext};

use crate::keys::{magnitude_bits, MESSAGE_BITS};
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

/// Variant (b): widen-once to acc_blocks.
pub(crate) fn mac_widen_once(
    sk: &ServerKey,
    cache: &WidthCache,
    groups: &BTreeMap<i64, Vec<usize>>,
    bias: i64,
    acc_blocks: usize,
) -> SignedRadixCiphertext {
    let mut group_terms = Vec::with_capacity(groups.len());
    for (&w, idxs) in groups {
        let s = if idxs.len() == 1 {
            cache.get(acc_blocks, idxs[0]).clone()
        } else {
            sk.sum_ciphertexts_parallelized(idxs.iter().map(|&i| cache.get(acc_blocks, i)))
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
        sk.create_trivial_zero_radix(acc_blocks)
    } else if group_terms.len() == 1 {
        group_terms.pop().unwrap()
    } else {
        sk.sum_ciphertexts_parallelized(group_terms.iter())
            .expect("non-empty group_terms")
    };

    sk.scalar_add_parallelized(&acc, bias)
}

/// Variant (b'): progressive widening based on group size.
pub(crate) fn mac_progressive(
    sk: &ServerKey,
    cache: &WidthCache,
    groups: &BTreeMap<i64, Vec<usize>>,
    bias: i64,
    in_bits: usize,
    acc_blocks: usize,
) -> SignedRadixCiphertext {
    let mut group_terms = Vec::with_capacity(groups.len());
    for (&w, idxs) in groups {
        let k = idxs.len();
        let sum_bits = in_bits + penumbra_core::bitwidth::ceil_log2(k);
        let g = value_blocks(sum_bits, acc_blocks);
        let s = if idxs.len() == 1 {
            cache.get(g, idxs[0]).clone()
        } else {
            sk.sum_ciphertexts_parallelized(idxs.iter().map(|&i| cache.get(g, i)))
                .expect("non-empty cts group")
        };
        let p = if w == 1 {
            g
        } else {
            value_blocks(sum_bits + magnitude_bits(w), acc_blocks)
        };
        let s_p = resize(sk, &s, p);
        let term_p = if w == 1 {
            s_p.into_owned()
        } else {
            sk.scalar_mul_parallelized(s_p.as_ref(), w)
        };
        let term = resize(sk, &term_p, acc_blocks).into_owned();
        group_terms.push(term);
    }

    let acc = if group_terms.is_empty() {
        sk.create_trivial_zero_radix(acc_blocks)
    } else if group_terms.len() == 1 {
        group_terms.pop().unwrap()
    } else {
        sk.sum_ciphertexts_parallelized(group_terms.iter())
            .expect("non-empty group_terms")
    };

    sk.scalar_add_parallelized(&acc, bias)
}

/// Variant (c): deferred carry propagation.
pub(crate) fn mac_deferred(
    sk: &ServerKey,
    cache: &WidthCache,
    groups: &BTreeMap<i64, Vec<usize>>,
    bias: i64,
    in_bits: usize,
    acc_blocks: usize,
    progressive: bool,
) -> SignedRadixCiphertext {
    let mut pos: Vec<SignedRadixCiphertext> = Vec::new();
    let mut neg: Vec<SignedRadixCiphertext> = Vec::new();

    for (&w, idxs) in groups {
        let s = if progressive {
            let k = idxs.len();
            let sum_bits = in_bits + penumbra_core::bitwidth::ceil_log2(k);
            let g = value_blocks(sum_bits, acc_blocks);
            let s_g = if idxs.len() == 1 {
                cache.get(g, idxs[0]).clone()
            } else {
                sk.sum_ciphertexts_parallelized(idxs.iter().map(|&i| cache.get(g, i)))
                    .expect("non-empty cts group")
            };
            resize(sk, &s_g, acc_blocks).into_owned()
        } else if idxs.len() == 1 {
            cache.get(acc_blocks, idxs[0]).clone()
        } else {
            sk.sum_ciphertexts_parallelized(idxs.iter().map(|&i| cache.get(acc_blocks, i)))
                .expect("non-empty cts group")
        };

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
mod spike {
    use super::*;
    use crate::width::signed_blocks;
    use std::time::Instant;
    use crate::ops::EvalCtx;

    struct VariantResult {
        variant: &'static str,
        median_secs: f64,
        min_secs: f64,
        max_secs: f64,
        pbs: u64,
        widen_secs: f64,
        widen_pbs: u64,
        exact: bool,
    }

    struct SpikeNeuronData {
        name: &'static str,
        terms: usize,
        in_bits: usize,
        acc_bits: usize,
        acc_blocks: usize,
        nb: usize,
        groups: BTreeMap<i64, Vec<usize>>,
        bias: i64,
        cleartext_sum: i64,
        raw_inputs: Vec<SignedRadixCiphertext>,
        trimmed_inputs: Vec<SignedRadixCiphertext>,
    }

    #[test]
    #[ignore]
    fn spike_mac_variants() {
        let fixture_str = include_str!("../../../../examples/mnist/phase5_digits_fixture.json");
        let fixture_val: serde_json::Value =
            serde_json::from_str(fixture_str).expect("parse fixture json");
        let graph_str = serde_json::to_string(&fixture_val["graph"]).expect("graph json");
        let graph = penumbra_core::ir::Graph::from_json(&graph_str).expect("parse graph");
        let nb = graph.num_blocks;
        assert_eq!(nb, 9, "digits fixture num_blocks must be 9");

        let (ck, sk) = crate::keys::keygen(nb);

        // 1. Conv neuron: node `conv0`, output channel 0 at (oy, ox) = (0, 0)
        let conv0 = graph.nodes.iter().find(|n| n.name == "conv0").expect("conv0");
        let (w_conv, b_conv) = match &conv0.op {
            penumbra_core::ir::OpSpec::Conv2d { weights, bias, .. } => (&weights[0], bias[0]),
            _ => panic!("conv0 is not Conv2d"),
        };
        let in_bits_conv = graph.input_bits; // 3
        let acc_bits_conv =
            penumbra_core::bitwidth::op_spec_output_bits_n(&conv0.op, &[in_bits_conv]); // 14
        let acc_blocks_conv = signed_blocks(acc_bits_conv, nb); // 7

        let test_inputs_0: Vec<i64> =
            serde_json::from_value(fixture_val["test_inputs"][0].clone()).expect("test_inputs[0]");
        let mut conv_groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        let mut conv_cleartext = b_conv;
        for ky in 0..3 {
            for kx in 0..3 {
                let idx = ky * 8 + kx;
                let w = w_conv[ky * 3 + kx];
                if w != 0 {
                    conv_groups.entry(w).or_default().push(idx);
                }
                conv_cleartext += w * test_inputs_0[idx];
            }
        }
        let conv_raw_inputs: Vec<SignedRadixCiphertext> =
            test_inputs_0.iter().map(|&v| ck.encrypt_signed(v)).collect();
        let conv_trim_width = value_blocks(in_bits_conv, nb);
        let conv_trimmed_inputs: Vec<SignedRadixCiphertext> = conv_raw_inputs
            .iter()
            .map(|ct| resize(&sk, ct, conv_trim_width).into_owned())
            .collect();

        let conv_neuron = SpikeNeuronData {
            name: "conv0",
            terms: 9,
            in_bits: in_bits_conv,
            acc_bits: acc_bits_conv,
            acc_blocks: acc_blocks_conv,
            nb,
            groups: conv_groups,
            bias: b_conv,
            cleartext_sum: conv_cleartext,
            raw_inputs: conv_raw_inputs,
            trimmed_inputs: conv_trimmed_inputs,
        };

        // 2. Linear neuron: node `linear2`, row 0 (108 terms), bias[0]
        let linear2 = graph.nodes.iter().find(|n| n.name == "linear2").expect("linear2");
        let (w_lin, b_lin) = match &linear2.op {
            penumbra_core::ir::OpSpec::Linear { weights, bias, .. } => (&weights[0], bias[0]),
            _ => panic!("linear2 is not Linear"),
        };
        let conv0_rq = graph
            .nodes
            .iter()
            .find(|n| n.name == "conv0__requant")
            .expect("conv0__requant");
        let in_bits_lin = match &conv0_rq.op {
            penumbra_core::ir::OpSpec::Requant { out_bits, .. } => *out_bits,
            _ => panic!("conv0__requant is not Requant"),
        }; // 2
        let acc_bits_lin =
            penumbra_core::bitwidth::op_spec_output_bits_n(&linear2.op, &[in_bits_lin]); // 17
        let acc_blocks_lin = signed_blocks(acc_bits_lin, nb); // 9

        let lin_input_vals: Vec<i64> =
            (0..w_lin.len()).map(|i| (i as i64 * 7 + 3) % 4).collect();
        let mut lin_groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        let mut lin_cleartext = b_lin;
        for (i, &w) in w_lin.iter().enumerate() {
            if w != 0 {
                lin_groups.entry(w).or_default().push(i);
            }
            lin_cleartext += w * lin_input_vals[i];
        }
        let lin_raw_inputs: Vec<SignedRadixCiphertext> =
            lin_input_vals.iter().map(|&v| ck.encrypt_signed(v)).collect();
        let lin_trim_width = value_blocks(in_bits_lin, nb);
        let lin_trimmed_inputs: Vec<SignedRadixCiphertext> = lin_raw_inputs
            .iter()
            .map(|ct| resize(&sk, ct, lin_trim_width).into_owned())
            .collect();

        let lin_neuron = SpikeNeuronData {
            name: "linear2",
            terms: 108,
            in_bits: in_bits_lin,
            acc_bits: acc_bits_lin,
            acc_blocks: acc_blocks_lin,
            nb,
            groups: lin_groups,
            bias: b_lin,
            cleartext_sum: lin_cleartext,
            raw_inputs: lin_raw_inputs,
            trimmed_inputs: lin_trimmed_inputs,
        };

        // Runner function for a neuron & variant
        let run_variant = |neuron: &SpikeNeuronData,
                           variant: &'static str,
                           progressive_c: bool|
         -> VariantResult {
            let (widen_secs, widen_pbs, cache) = if variant == "a" {
                (0.0, 0, None)
            } else {
                let needed_widths: BTreeSet<usize> = match variant {
                    "b" => [neuron.acc_blocks].into_iter().collect(),
                    "b'" => neuron
                        .groups
                        .values()
                        .map(|idxs| {
                            let k = idxs.len();
                            let sum_bits = neuron.in_bits + penumbra_core::bitwidth::ceil_log2(k);
                            value_blocks(sum_bits, neuron.acc_blocks)
                        })
                        .collect(),
                    "c" => {
                        if progressive_c {
                            neuron
                                .groups
                                .values()
                                .map(|idxs| {
                                    let k = idxs.len();
                                    let sum_bits =
                                        neuron.in_bits + penumbra_core::bitwidth::ceil_log2(k);
                                    value_blocks(sum_bits, neuron.acc_blocks)
                                })
                                .collect()
                        } else {
                            [neuron.acc_blocks].into_iter().collect()
                        }
                    }
                    _ => unreachable!(),
                };

                tfhe::shortint::server_key::reset_pbs_count();
                let t0 = Instant::now();
                let c = WidthCache::build(&sk, &neuron.trimmed_inputs, &needed_widths);
                let dt = t0.elapsed().as_secs_f64();
                let pbs = tfhe::shortint::server_key::get_pbs_count();
                (dt, pbs, Some(c))
            };

            let eval_once = || -> SignedRadixCiphertext {
                match variant {
                    "a" => {
                        let mut ref_groups: BTreeMap<i64, Vec<&SignedRadixCiphertext>> =
                            BTreeMap::new();
                        for (&w, idxs) in &neuron.groups {
                            let list = ref_groups.entry(w).or_default();
                            for &idx in idxs {
                                list.push(&neuron.raw_inputs[idx]);
                            }
                        }
                        let ctx = EvalCtx::new(&sk, neuron.nb);
                        super::super::evaluate_weighted_mac(&ctx, ref_groups, neuron.bias)
                    }
                    "b" => mac_widen_once(
                        &sk,
                        cache.as_ref().unwrap(),
                        &neuron.groups,
                        neuron.bias,
                        neuron.acc_blocks,
                    ),
                    "b'" => mac_progressive(
                        &sk,
                        cache.as_ref().unwrap(),
                        &neuron.groups,
                        neuron.bias,
                        neuron.in_bits,
                        neuron.acc_blocks,
                    ),
                    "c" => mac_deferred(
                        &sk,
                        cache.as_ref().unwrap(),
                        &neuron.groups,
                        neuron.bias,
                        neuron.in_bits,
                        neuron.acc_blocks,
                        progressive_c,
                    ),
                    _ => unreachable!(),
                }
            };

            // 1 warm-up run
            let warm = eval_once();
            let warm_dec = ck.decrypt_signed::<i64>(&warm);
            assert_eq!(
                warm_dec, neuron.cleartext_sum,
                "warm-up mismatch for variant {variant} on neuron {}",
                neuron.name
            );

            // 7 timed runs
            let mut durations = Vec::with_capacity(7);
            let mut first_pbs = 0;
            for run_idx in 0..7 {
                tfhe::shortint::server_key::reset_pbs_count();
                let t0 = Instant::now();
                let res = eval_once();
                let dt = t0.elapsed().as_secs_f64();
                durations.push(dt);
                if run_idx == 0 {
                    first_pbs = tfhe::shortint::server_key::get_pbs_count();
                }
                let dec = ck.decrypt_signed::<i64>(&res);
                assert_eq!(
                    dec, neuron.cleartext_sum,
                    "timed run {run_idx} mismatch for variant {variant} on neuron {}",
                    neuron.name
                );
            }

            durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
            VariantResult {
                variant,
                median_secs: durations[3],
                min_secs: durations[0],
                max_secs: durations[6],
                pbs: first_pbs,
                widen_secs,
                widen_pbs,
                exact: true,
            }
        };

        // Run variants (a), (b), (b') on both neurons
        let conv_a = run_variant(&conv_neuron, "a", false);
        let conv_b = run_variant(&conv_neuron, "b", false);
        let conv_bprime = run_variant(&conv_neuron, "b'", false);

        let lin_a = run_variant(&lin_neuron, "a", false);
        let lin_b = run_variant(&lin_neuron, "b", false);
        let lin_bprime = run_variant(&lin_neuron, "b'", false);

        // Decision rule R1: S(v) = median_conv(v) + median_linear(v)
        let s_b = conv_b.median_secs + lin_b.median_secs;
        let s_bprime = conv_bprime.median_secs + lin_bprime.median_secs;
        let r1_ratio = s_b / s_bprime;
        let r1_winner = if r1_ratio >= 1.05 { "b'" } else { "b" };
        let progressive_c = r1_winner == "b'";

        // Run variant (c) on the R1 winner
        let conv_c = run_variant(&conv_neuron, "c", progressive_c);
        let lin_c = run_variant(&lin_neuron, "c", progressive_c);

        let s_winner = if r1_winner == "b'" { s_bprime } else { s_b };
        let s_c = conv_c.median_secs + lin_c.median_secs;
        let r2_ratio = s_winner / s_c;
        let r2_deferred = if r2_ratio >= 1.2 { "yes" } else { "no" };

        let threads = rayon::current_num_threads();

        // Print results table
        println!();
        println!(
            "{:<10} {:<8} {:>12} {:>12} {:>12} {:>8} {:>12} {:>10} {:<6}",
            "Neuron", "Variant", "Median (s)", "Min (s)", "Max (s)", "PBS", "Widen (s)", "Widen PBS", "Exact"
        );
        println!("{}", "-".repeat(95));
        for (n_name, results) in [
            ("conv0", [&conv_a, &conv_b, &conv_bprime, &conv_c]),
            ("linear2", [&lin_a, &lin_b, &lin_bprime, &lin_c]),
        ] {
            for r in results {
                println!(
                    "{:<10} {:<8} {:>12.6} {:>12.6} {:>12.6} {:>8} {:>12.6} {:>10} {:<6}",
                    n_name,
                    r.variant,
                    r.median_secs,
                    r.min_secs,
                    r.max_secs,
                    r.pbs,
                    r.widen_secs,
                    r.widen_pbs,
                    r.exact
                );
            }
        }
        println!("{}", "-".repeat(95));
        println!("S(b) = {:.6}s, S(b') = {:.6}s, S(b)/S(b') = {:.4}", s_b, s_bprime, r1_ratio);
        println!("R1 winner: {r1_winner}");
        println!("S(winner) = {:.6}s, S(c) = {:.6}s, S(winner)/S(c) = {:.4}", s_winner, s_c, r2_ratio);
        println!("DECISION: mac={r1_winner} deferred={r2_deferred}");
        println!();

        // JSON output if requested
        if let Ok(out_path) = std::env::var("PENUMBRA_SPIKE_OUT") {
            let commit = std::env::var("PENUMBRA_SPIKE_COMMIT").unwrap_or_default();
            let machine = std::env::var("PENUMBRA_SPIKE_MACHINE").unwrap_or_default();
            let rustc = std::env::var("PENUMBRA_SPIKE_RUSTC").unwrap_or_default();
            let date = std::env::var("PENUMBRA_SPIKE_DATE")
                .unwrap_or_else(|_| "2026-09-27".to_string());

            let neurons_json = vec![
                serde_json::json!({
                    "name": conv_neuron.name,
                    "terms": conv_neuron.terms,
                    "in_bits": conv_neuron.in_bits,
                    "acc_bits": conv_neuron.acc_bits,
                    "acc_blocks": conv_neuron.acc_blocks,
                    "nb": conv_neuron.nb,
                    "variants": [
                        { "variant": conv_a.variant, "median_secs": conv_a.median_secs, "min_secs": conv_a.min_secs, "max_secs": conv_a.max_secs, "pbs": conv_a.pbs, "widen_secs": conv_a.widen_secs, "widen_pbs": conv_a.widen_pbs, "exact": conv_a.exact },
                        { "variant": conv_b.variant, "median_secs": conv_b.median_secs, "min_secs": conv_b.min_secs, "max_secs": conv_b.max_secs, "pbs": conv_b.pbs, "widen_secs": conv_b.widen_secs, "widen_pbs": conv_b.widen_pbs, "exact": conv_b.exact },
                        { "variant": conv_bprime.variant, "median_secs": conv_bprime.median_secs, "min_secs": conv_bprime.min_secs, "max_secs": conv_bprime.max_secs, "pbs": conv_bprime.pbs, "widen_secs": conv_bprime.widen_secs, "widen_pbs": conv_bprime.widen_pbs, "exact": conv_bprime.exact },
                        { "variant": conv_c.variant, "median_secs": conv_c.median_secs, "min_secs": conv_c.min_secs, "max_secs": conv_c.max_secs, "pbs": conv_c.pbs, "widen_secs": conv_c.widen_secs, "widen_pbs": conv_c.widen_pbs, "exact": conv_c.exact },
                    ]
                }),
                serde_json::json!({
                    "name": lin_neuron.name,
                    "terms": lin_neuron.terms,
                    "in_bits": lin_neuron.in_bits,
                    "acc_bits": lin_neuron.acc_bits,
                    "acc_blocks": lin_neuron.acc_blocks,
                    "nb": lin_neuron.nb,
                    "variants": [
                        { "variant": lin_a.variant, "median_secs": lin_a.median_secs, "min_secs": lin_a.min_secs, "max_secs": lin_a.max_secs, "pbs": lin_a.pbs, "widen_secs": lin_a.widen_secs, "widen_pbs": lin_a.widen_pbs, "exact": lin_a.exact },
                        { "variant": lin_b.variant, "median_secs": lin_b.median_secs, "min_secs": lin_b.min_secs, "max_secs": lin_b.max_secs, "pbs": lin_b.pbs, "widen_secs": lin_b.widen_secs, "widen_pbs": lin_b.widen_pbs, "exact": lin_b.exact },
                        { "variant": lin_bprime.variant, "median_secs": lin_bprime.median_secs, "min_secs": lin_bprime.min_secs, "max_secs": lin_bprime.max_secs, "pbs": lin_bprime.pbs, "widen_secs": lin_bprime.widen_secs, "widen_pbs": lin_bprime.widen_pbs, "exact": lin_bprime.exact },
                        { "variant": lin_c.variant, "median_secs": lin_c.median_secs, "min_secs": lin_c.min_secs, "max_secs": lin_c.max_secs, "pbs": lin_c.pbs, "widen_secs": lin_c.widen_secs, "widen_pbs": lin_c.widen_pbs, "exact": lin_c.exact },
                    ]
                }),
            ];

            let json_out = serde_json::json!({
                "meta": {
                    "commit": commit,
                    "date": date,
                    "machine": machine,
                    "rustc": rustc,
                    "threads": threads,
                    "repeats": 7,
                    "fixture": "examples/mnist/phase5_digits_fixture.json"
                },
                "neurons": neurons_json,
                "decision": {
                    "mac": r1_winner,
                    "deferred": r2_deferred,
                    "s_ratio_b_over_bprime": r1_ratio,
                    "s_ratio_winner_over_c": r2_ratio
                }
            });

            std::fs::write(&out_path, serde_json::to_string_pretty(&json_out).unwrap() + "\n")
                .expect("write PENUMBRA_SPIKE_OUT");
            println!("Spike results written to {out_path}");
        }
    }
}
