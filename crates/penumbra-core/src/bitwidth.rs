//! Bit-width growth rules and budget propagation (Layer 2, `PROJECT.md` §9).
//!
//! Centralized scheme-neutral bit-width tracker: describes how accumulator precision
//! grows across nodes in an IR graph, and verifies that intermediate peaks and outputs
//! fit the radix capacity budget.

use std::collections::HashMap;

use crate::ir::{Graph, OpSpec, PoolMode};

/// Number of message bits per radix block in the default TFHE profile.
pub const MESSAGE_BITS: usize = 2;

/// Compute the radix capacity in bits for a given number of blocks.
#[inline]
pub fn radix_capacity_bits(num_blocks: usize) -> usize {
    num_blocks * MESSAGE_BITS
}

/// Bit-width of a signed integer's magnitude (excluding the sign bit).
#[inline]
pub fn magnitude_bits(v: i64) -> usize {
    let u = v.unsigned_abs();
    if u == 0 {
        0
    } else {
        (64 - u.leading_zeros()) as usize
    }
}

/// Compute ceiling of log2 for a positive integer (0 for n <= 1).
#[inline]
pub fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (n as f64).log2().ceil() as usize
    }
}

/// Output bit-width growth for an [`OpSpec`].
pub fn op_spec_output_bits_n(spec: &OpSpec, input_bits: &[usize]) -> usize {
    match spec {
        OpSpec::Linear {
            weights,
            bias,
            weight_bits,
        } => {
            assert_eq!(input_bits.len(), 1, "Linear is a single-input op");
            let in_b = input_bits[0];
            let n = weights.first().map_or(0, Vec::len);
            let sum_growth = ceil_log2(n);
            let sum_bits = in_b + weight_bits + sum_growth;
            let bias_bits = bias.iter().map(|&b| magnitude_bits(b)).max().unwrap_or(0);
            sum_bits.max(bias_bits) + 2
        }
        OpSpec::Conv2d {
            weights: _,
            bias,
            weight_bits,
            in_h: _,
            in_w: _,
            in_channels,
            kernel_h,
            kernel_w,
            stride: _,
            padding: _,
        } => {
            assert_eq!(input_bits.len(), 1, "Conv2d is a single-input op");
            let in_b = input_bits[0];
            let fan_in = in_channels * kernel_h * kernel_w;
            let sum_growth = ceil_log2(fan_in);
            let sum_bits = in_b + weight_bits + sum_growth;
            let bias_bits = bias.iter().map(|&b| magnitude_bits(b)).max().unwrap_or(0);
            sum_bits.max(bias_bits) + 2
        }
        OpSpec::Activation { output_bits, .. } => {
            assert_eq!(input_bits.len(), 1, "Activation is a single-input op");
            *output_bits
        }
        OpSpec::Requant { out_bits, .. } => {
            assert_eq!(input_bits.len(), 1, "Requant is a single-input op");
            *out_bits
        }
        OpSpec::Pool {
            mode,
            pool_h,
            pool_w,
            ..
        } => {
            assert_eq!(input_bits.len(), 1, "Pool is a single-input op");
            let in_b = input_bits[0];
            let k = pool_h * pool_w;
            let parsed_mode = match mode.as_str() {
                "avg" => PoolMode::Avg,
                "max" => PoolMode::Max,
                _ => PoolMode::Avg,
            };
            match parsed_mode {
                PoolMode::Avg => in_b + ceil_log2(k),
                PoolMode::Max => in_b,
            }
        }
        OpSpec::Add {} => {
            assert_eq!(input_bits.len(), 2, "Add is a two-input op");
            input_bits[0].max(input_bits[1]) + 1
        }
        OpSpec::Argmax { .. } => {
            assert_eq!(input_bits.len(), 1, "Argmax is a single-input op");
            1
        }
    }
}

/// Peak internal bit-width the rescale needs before the shift narrows it.
pub fn requant_internal_bits(input_bits: usize, mult: u64, round_bias: u64) -> usize {
    let relu_max: u128 = if input_bits >= 1 {
        (1u128 << (input_bits - 1)) - 1
    } else {
        0
    };
    let intermediate_max = relu_max * mult as u128 + round_bias as u128;
    let magnitude = (u128::BITS - intermediate_max.leading_zeros()) as usize;
    (magnitude + 1).max(input_bits)
}

/// Peak transient internal bit-width for an [`OpSpec`].
pub fn op_spec_internal_bits_n(spec: &OpSpec, input_bits: &[usize]) -> usize {
    match spec {
        OpSpec::Requant {
            mult,
            round_bias,
            mults,
            round_biases,
            ..
        } => {
            assert_eq!(input_bits.len(), 1, "Requant is a single-input op");
            if mults.is_empty() {
                requant_internal_bits(input_bits[0], *mult, *round_bias)
            } else {
                mults
                    .iter()
                    .zip(round_biases)
                    .map(|(&m, &rb)| requant_internal_bits(input_bits[0], m, rb))
                    .max()
                    .expect("per-channel Requant has at least one channel")
            }
        }
        _ => op_spec_output_bits_n(spec, input_bits),
    }
}

/// Propagate per-tensor bit-widths through an IR [`Graph`], seeded by `graph.input_bits`.
pub fn propagate_bit_widths(graph: &Graph) -> Result<HashMap<String, usize>, String> {
    let mut widths: HashMap<String, usize> = graph
        .inputs
        .iter()
        .map(|name| (name.clone(), graph.input_bits))
        .collect();

    for node in &graph.nodes {
        node.op.validate()?;
        if node.inputs.is_empty() || node.outputs.len() != 1 {
            return Err(format!(
                "node '{}' ({}) must have at least one input and exactly one output",
                node.name,
                node.op.op_type()
            ));
        }
        let in_bits: Vec<usize> = node
            .inputs
            .iter()
            .map(|input_name| {
                widths.get(input_name).copied().ok_or_else(|| {
                    format!(
                        "node '{}' reads tensor '{input_name}', which no earlier node produced \
                         and is not a graph input — node order is not a valid topological order",
                        node.name
                    )
                })
            })
            .collect::<Result<_, _>>()?;

        let out_bits = op_spec_output_bits_n(&node.op, &in_bits);
        let output_name = &node.outputs[0];
        if widths.contains_key(output_name) {
            return Err(format!(
                "node '{}' writes tensor '{output_name}', which already exists — tensor names \
                 must be unique",
                node.name
            ));
        }
        widths.insert(output_name.clone(), out_bits);
    }
    Ok(widths)
}

/// Verify an IR [`Graph`]'s declared bit-width budget fits the radix capacity, failing
/// **loudly** before evaluation if not (`AGENTS.md` §1.3, §1.4).
pub fn check_graph_bit_width_budget(graph: &Graph) -> Result<(), String> {
    let capacity = radix_capacity_bits(graph.num_blocks);
    let widths = propagate_bit_widths(graph)?;

    for node in &graph.nodes {
        let name = &node.outputs[0];
        let bits = widths[name];
        if bits > capacity {
            return Err(format!(
                "bit-width budget exceeded at node '{}' (tensor '{name}'): requires {bits} bits \
                 but the radix holds only {capacity} ({} blocks × {} bits). Reduce precision or \
                 widen num_blocks (a Requant here is Phase 4).",
                node.name, graph.num_blocks, MESSAGE_BITS
            ));
        }

        let in_bits: Vec<usize> = node.inputs.iter().map(|n| widths[n]).collect();
        let internal = op_spec_internal_bits_n(&node.op, &in_bits);
        if internal > capacity {
            return Err(format!(
                "bit-width budget exceeded inside node '{}' ({}): its rescale needs {internal} \
                 transient bits (the multiply-then-shift intermediate) but the radix holds only \
                 {capacity} ({} blocks × {} bits). Reduce the requant multiplier, reduce input \
                 precision, or widen num_blocks.",
                node.name,
                node.op.op_type(),
                graph.num_blocks,
                MESSAGE_BITS
            ));
        }
    }
    Ok(())
}

/// Verify a sequence of ops fits the radix capacity.
pub fn check_bit_width_budget<B: crate::backend::Backend + ?Sized>(
    ops: &[Box<dyn crate::ops::Op<B>>],
    input_bits: usize,
    num_blocks: usize,
) -> Result<(), String> {
    let capacity = radix_capacity_bits(num_blocks);
    let mut bits = input_bits;
    for (i, op) in ops.iter().enumerate() {
        let out = op.output_bits(bits);
        if out > capacity {
            return Err(format!(
                "bit-width budget exceeded at op #{i}: requires {out} bits but the radix \
                 holds only {capacity} ({num_blocks} blocks × {MESSAGE_BITS} bits). Reduce precision or \
                 widen num_blocks (a Requant here is Phase 4)."
            ));
        }
        bits = out;
    }
    Ok(())
}
