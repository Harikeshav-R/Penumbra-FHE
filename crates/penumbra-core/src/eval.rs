//! The graph walker / evaluation loop (Layer 2).
//!
//! Evaluates a model under FHE by dispatching each op to its implementation in
//! the active [`crate::backend::Backend`]. This loop is written **once** and never changes
//! per use case or backend (`PROJECT.md` §4, §7): adding an op never edits the loop,
//! and a new use case never edits any op (`AGENTS.md` §1.2).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::backend::{Backend, CtVec, EvalCtx};
use crate::ir::Graph;
use crate::ops::Op;
use crate::profile::{GraphProfile, NodeProfile};
// Re-export bit-width budget functions for backwards compatibility.
pub use crate::bitwidth::{
    check_bit_width_budget, check_graph_bit_width_budget, propagate_bit_widths,
};

/// Evaluate a linear chain of ops over an encrypted input, threading each op's output into
/// the next. Returns the final encrypted output.
pub fn evaluate<B: Backend>(
    ctx: &EvalCtx<B::ServerKey>,
    ops: &[Box<dyn Op<B>>],
    input: &CtVec<B>,
) -> CtVec<B> {
    let mut acc = input.clone();
    for op in ops {
        acc = op.eval(ctx, &acc);
    }
    acc
}

/// Evaluate a deserialized IR [`Graph`] under FHE, returning the named output tensors.
///
/// `inputs` maps each `graph.inputs` name to its encrypted [`CtVec<B>`]. We build each node's
/// op from the active backend ([`Backend::build_op`]), walk the nodes **in their serialized order**,
/// resolve each node's declared input tensors (in order), dispatch through [`Op::eval_n`],
/// and store the output. The result holds every `graph.outputs` tensor.
pub fn evaluate_graph<B: Backend>(
    backend: &B,
    ctx: &EvalCtx<B::ServerKey>,
    graph: &Graph,
    inputs: HashMap<String, CtVec<B>>,
) -> Result<HashMap<String, CtVec<B>>, String> {
    evaluate_graph_inner(backend, ctx, graph, inputs, None)
}

/// Walk the graph exactly as [`evaluate_graph`] does, recording per-node build time,
/// eval time, tensor shapes, and the backend's cost counters into `profile`.
///
/// Instrumented **once**, here, so every backend is measured by the same code
/// (`ROADMAP.md` Phase 12.3, `docs/COMPARISON.md`).
///
/// Note: backend-measured counters (such as TFHE's PBS count) may use process-global atomics,
/// so per-node deltas are exact only while one graph is evaluated at a time in the process —
/// which every entry point does today.
pub fn evaluate_graph_profiled<B: Backend>(
    backend: &B,
    ctx: &EvalCtx<B::ServerKey>,
    graph: &Graph,
    inputs: HashMap<String, CtVec<B>>,
    profile: &mut GraphProfile,
) -> Result<HashMap<String, CtVec<B>>, String> {
    evaluate_graph_inner(backend, ctx, graph, inputs, Some(profile))
}

fn evaluate_graph_inner<B: Backend>(
    backend: &B,
    ctx: &EvalCtx<B::ServerKey>,
    graph: &Graph,
    inputs: HashMap<String, CtVec<B>>,
    mut profile: Option<&mut GraphProfile>,
) -> Result<HashMap<String, CtVec<B>>, String> {
    let declared: HashSet<&str> = graph.inputs.iter().map(String::as_str).collect();
    let provided: HashSet<&str> = inputs.keys().map(String::as_str).collect();
    if declared != provided {
        return Err(format!(
            "graph inputs {:?} do not match the provided input tensors {:?}",
            graph.inputs,
            inputs.keys().collect::<Vec<_>>()
        ));
    }

    let graph = crate::optimize::optimize_graph(graph)?;
    let graph = graph.as_ref();

    if let Some(prof) = &mut profile {
        prof.backend = backend.name();
        prof.nodes.clear();
    }

    let t_total = profile.as_ref().map(|_| Instant::now());

    let mut env = inputs;
    for node in &graph.nodes {
        let t_build = profile.as_ref().map(|_| Instant::now());
        let op = backend
            .build_op(&node.op)
            .map_err(|e| format!("node '{}': {e}", node.name))?;
        let build = t_build.map(|t| t.elapsed()).unwrap_or_default();

        if node.inputs.is_empty() {
            return Err(format!(
                "node '{}' ({}) has no inputs",
                node.name,
                node.op.op_type()
            ));
        }

        let input_cts: Vec<&CtVec<B>> = node
            .inputs
            .iter()
            .map(|input_name| {
                env.get(input_name).ok_or_else(|| {
                    format!(
                        "node '{}' reads tensor '{input_name}', which no earlier node produced \
                         and is not a graph input — node order is not a valid topological order",
                        node.name
                    )
                })
            })
            .collect::<Result<_, _>>()?;

        let input_lens: Vec<usize> = if profile.is_some() {
            input_cts.iter().map(|v| v.len()).collect()
        } else {
            Vec::new()
        };

        let measured_before = profile
            .as_ref()
            .map(|_| backend.measured_counters())
            .unwrap_or_default();
        let t_eval = profile.as_ref().map(|_| Instant::now());
        let result = op.eval_n(ctx, &input_cts);
        let eval = t_eval.map(|t| t.elapsed()).unwrap_or_default();

        if let Some(prof) = &mut profile {
            let measured_after = backend.measured_counters();
            if measured_after.len() != measured_before.len() {
                return Err(format!(
                    "backend '{}' violated the stable-order contract for measured_counters: \
                     before had {} counters, after had {}",
                    backend.name(),
                    measured_before.len(),
                    measured_after.len()
                ));
            }
            let mut measured = Vec::with_capacity(measured_before.len());
            for ((name_b, b), (name_a, a)) in measured_before.iter().zip(&measured_after) {
                if name_b != name_a {
                    return Err(format!(
                        "backend '{}' violated the stable-order contract for measured_counters: \
                         counter name mismatch ('{name_b}' vs '{name_a}')",
                        backend.name()
                    ));
                }
                measured.push((*name_b, a.saturating_sub(*b)));
            }
            let counters = op.cost(&input_lens);
            prof.nodes.push(NodeProfile {
                name: node.name.clone(),
                op_type: node.op.op_type(),
                build,
                eval,
                input_lens,
                output_len: result.len(),
                counters,
                measured,
            });
        }

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

    if let (Some(prof), Some(t_tot)) = (profile, t_total) {
        prof.total = t_tot.elapsed();
    }

    let mut outputs = HashMap::with_capacity(graph.outputs.len());
    for name in &graph.outputs {
        let ct = env
            .get(name)
            .ok_or_else(|| format!("graph declares output '{name}' but no node produced it"))?;
        outputs.insert(name.clone(), ct.clone());
    }
    Ok(outputs)
}
