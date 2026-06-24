//! The graph walker / evaluation loop.
//!
//! Evaluates a model under FHE by dispatching each op to its implementation in
//! [`crate::ops`]. This loop is written **once** and never changes per use case
//! (`PROJECT.md` §4, §7): adding an op never edits the loop, and a new use case never
//! edits any op (`AGENTS.md` §1.2).
//!
//! Runtime ≈ number of bootstraps (`PROJECT.md` §5): linear ops with plaintext weights are
//! cheap; activations/requant are programmable bootstraps and dominate cost.
//!
//! Two entry points share the same dispatch discipline:
//!
//! - [`evaluate`] walks a flat `Vec<Box<dyn Op>>` (the Phase-2 linear-chain primitive); the
//!   golden gate now drives the model through [`evaluate_graph`], but this stays as the
//!   minimal op-walking primitive the graph walker is built on and a direct entry point for
//!   in-code op chains.
//! - [`evaluate_graph`] walks a deserialized IR [`Graph`] (Phase 3): it resolves each
//!   node's named input tensors from an environment map and dispatches through the same
//!   [`Op::eval`]. Node order is taken as given and *validated* to be a valid topological
//!   order — true Kahn's-algorithm sorting for branching graphs is Phase 8.
//!
//! Neither walk's body special-cases an op — that is the point of the [`Op`] trait.

use std::collections::{HashMap, HashSet};

use crate::ir::Graph;
use crate::ops::{CtVec, EvalCtx, Op};

/// Evaluate a linear chain of ops over an encrypted input, threading each op's output into
/// the next. Returns the final encrypted output (e.g. a one-element class-index vector).
pub fn evaluate(ctx: &EvalCtx, ops: &[Box<dyn Op>], input: &CtVec) -> CtVec {
    // We clone the input once so the borrowed `&CtVec` API stays ergonomic for callers.
    // Threading ownership through the loop to elide this clone is a possible optimization,
    // but it's negligible against PBS cost (runtime ≈ number of bootstraps) and not worth an
    // API change here — perf tuning is deferred to Phase 10 (`PROJECT.md` §10).
    let mut acc = input.clone();
    for op in ops {
        acc = op.eval(ctx, &acc);
    }
    acc
}

/// Verify a model's declared bit-width budget fits the radix capacity, failing **loudly**
/// before evaluation if not (`AGENTS.md` §1.3, §1.4).
///
/// Walks the op chain propagating bit-widths from `input_bits`; if any op's output width
/// exceeds what `num_blocks` radix blocks can hold, returns an `Err` naming the offending
/// layer index and the required-vs-available bits. Automatic `Requant` insertion that
/// would *prevent* such overflow is Phase 4; Phase 2 at least refuses to run silently.
pub fn check_bit_width_budget(
    ops: &[Box<dyn Op>],
    input_bits: usize,
    num_blocks: usize,
) -> Result<(), String> {
    let capacity = crate::keys::radix_capacity_bits(num_blocks);
    let mut bits = input_bits;
    for (i, op) in ops.iter().enumerate() {
        let out = op.output_bits(bits);
        if out > capacity {
            return Err(format!(
                "bit-width budget exceeded at op #{i}: requires {out} bits but the radix \
                 holds only {capacity} ({num_blocks} blocks × {} bits). Reduce precision or \
                 widen num_blocks (a Requant here is Phase 4).",
                crate::keys::MESSAGE_BITS
            ));
        }
        bits = out;
    }
    Ok(())
}

/// Evaluate a deserialized IR [`Graph`] under FHE, returning the named output tensors.
///
/// `inputs` maps each `graph.inputs` name to its encrypted [`CtVec`]. We build each node's
/// op from its [`crate::ir::OpSpec`], walk the nodes **in their serialized order**, resolve
/// each node's declared input tensors from a running environment, dispatch through
/// [`Op::eval`], and store the outputs. The result holds every `graph.outputs` tensor.
///
/// The serialized order is trusted but **validated** to be a real topological order, failing
/// loudly (`AGENTS.md` §1.4) on any of: an input tensor not yet produced, an output name
/// that collides with an existing tensor (silent overwrite), a provided-`inputs` key set
/// that doesn't match `graph.inputs`, or a declared graph output that no node produces.
/// Computing the order ourselves (Kahn's algorithm) is deferred to Phase 8, when branching
/// graphs arrive; Phase-2/3 models are linear chains where the emitted order *is* the order.
pub fn evaluate_graph(
    ctx: &EvalCtx,
    graph: &Graph,
    inputs: HashMap<String, CtVec>,
) -> Result<HashMap<String, CtVec>, String> {
    // The provided inputs must be exactly the graph's declared inputs — no missing seed
    // (would surface as an opaque "tensor not produced" later) and no stray extra.
    let declared: HashSet<&str> = graph.inputs.iter().map(String::as_str).collect();
    let provided: HashSet<&str> = inputs.keys().map(String::as_str).collect();
    if declared != provided {
        return Err(format!(
            "graph inputs {:?} do not match the provided input tensors {:?}",
            graph.inputs,
            inputs.keys().collect::<Vec<_>>()
        ));
    }

    let mut env = inputs;
    for node in &graph.nodes {
        let op = node.op.build()?;

        // Every Phase-2/3 op is single-input; reject a graph that wires several tensors into
        // one (multi-input ops like Add/Concat are Phase 4/8 and need a defined merge order).
        if node.inputs.len() != 1 {
            return Err(format!(
                "node '{}' ({}) has {} inputs; the current ops are single-input",
                node.name,
                node.op.op_type(),
                node.inputs.len()
            ));
        }

        let input_name = &node.inputs[0];
        let input_ct = env.get(input_name).ok_or_else(|| {
            format!(
                "node '{}' reads tensor '{input_name}', which no earlier node produced and is \
                 not a graph input — node order is not a valid topological order",
                node.name
            )
        })?;

        let result = op.eval(ctx, input_ct);

        // A node with multiple declared outputs needs a defined split of `result`; the
        // current ops each produce one output tensor. Enforce the 1:1 mapping loudly.
        if node.outputs.len() != 1 {
            return Err(format!(
                "node '{}' ({}) declares {} outputs; the current ops produce one output tensor",
                node.name,
                node.op.op_type(),
                node.outputs.len()
            ));
        }
        let output_name = &node.outputs[0];
        if env.contains_key(output_name) {
            return Err(format!(
                "node '{}' writes tensor '{output_name}', which already exists — tensor names \
                 must be unique (no silent overwrite)",
                node.name
            ));
        }
        env.insert(output_name.clone(), result);
    }

    // Pull out exactly the declared graph outputs; a missing one means the graph promised an
    // output no node produced.
    let mut outputs = HashMap::with_capacity(graph.outputs.len());
    for name in &graph.outputs {
        let ct = env
            .get(name)
            .ok_or_else(|| format!("graph declares output '{name}' but no node produced it"))?;
        outputs.insert(name.clone(), ct.clone());
    }
    Ok(outputs)
}

/// Propagate per-tensor bit-widths through an IR [`Graph`], seeded by `graph.input_bits`.
///
/// Returns the width of every tensor (inputs + each node's outputs) by walking nodes in
/// order and applying each op's [`Op::output_bits`] growth rule (`PROJECT.md` §9). Shared by
/// [`check_graph_bit_width_budget`] and the `inspect` debug binary so the bit-width math has
/// a single source of truth (`AGENTS.md` §1.3).
///
/// Validation mirrors [`evaluate_graph`]'s wiring checks (single input/output, input
/// produced, no overwrite) so a graph that fails here fails there too, before keygen.
///
/// Note (a known Phase-3 seam): `Activation::output_bits` *panics* on a malformed table
/// (declared width below the LUT's true range, or a wider-than-one-block input) rather than
/// returning an `Err`. The Phase-2 `Linear → Argmax` model never reaches that path; a
/// checked Activation budget arrives with Requant in Phase 4.
pub fn propagate_bit_widths(graph: &Graph) -> Result<HashMap<String, usize>, String> {
    let mut widths: HashMap<String, usize> = graph
        .inputs
        .iter()
        .map(|name| (name.clone(), graph.input_bits))
        .collect();

    for node in &graph.nodes {
        let op = node.op.build()?;
        if node.inputs.len() != 1 || node.outputs.len() != 1 {
            return Err(format!(
                "node '{}' ({}) must have exactly one input and one output for the current ops",
                node.name,
                node.op.op_type()
            ));
        }
        let input_name = &node.inputs[0];
        let in_bits = *widths.get(input_name).ok_or_else(|| {
            format!(
                "node '{}' reads tensor '{input_name}', which no earlier node produced and is \
                 not a graph input — node order is not a valid topological order",
                node.name
            )
        })?;
        let out_bits = op.output_bits(in_bits);
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
/// **loudly** before evaluation if not (`AGENTS.md` §1.3, §1.4) — the graph analogue of
/// [`check_bit_width_budget`].
///
/// Walks every tensor's propagated width (via [`propagate_bit_widths`]); if any exceeds what
/// `graph.num_blocks` radix blocks can hold, returns an `Err` naming the offending tensor.
pub fn check_graph_bit_width_budget(graph: &Graph) -> Result<(), String> {
    let capacity = crate::keys::radix_capacity_bits(graph.num_blocks);
    let widths = propagate_bit_widths(graph)?;

    // Report against node outputs (named, ordered) for an actionable message; the input
    // tensors are seeded at `input_bits` and are the user's declared range, not a growth.
    for node in &graph.nodes {
        let name = &node.outputs[0];
        let bits = widths[name];
        if bits > capacity {
            return Err(format!(
                "bit-width budget exceeded at node '{}' (tensor '{name}'): requires {bits} bits \
                 but the radix holds only {capacity} ({} blocks × {} bits). Reduce precision or \
                 widen num_blocks (a Requant here is Phase 4).",
                node.name,
                graph.num_blocks,
                crate::keys::MESSAGE_BITS
            ));
        }
    }
    Ok(())
}
