//! Backend-generic execution helpers for PyO3 in-process inference.

use std::collections::HashMap;

use penumbra_core::backend::{Backend, EvalCtx};
use penumbra_core::ir::Graph;

/// Run evaluation over a batch of encrypted input rows using the public server key.
pub fn run_evaluate_batch<B: Backend>(
    backend: &B,
    graph: &Graph,
    sk: &B::ServerKey,
    batch: Vec<Vec<B::Ciphertext>>,
) -> Result<Vec<Vec<B::Ciphertext>>, String> {
    if graph.inputs.len() != 1 {
        return Err(format!(
            "evaluate expects a single-input graph, got {} inputs {:?}; multi-input graphs are not yet supported by the bridge",
            graph.inputs.len(),
            graph.inputs
        ));
    }
    if graph.outputs.len() != 1 {
        return Err(format!(
            "evaluate expects a single-output graph, got {} outputs {:?}",
            graph.outputs.len(),
            graph.outputs
        ));
    }
    backend.check_graph_budget(graph)?;

    let ctx = EvalCtx::new(sk, graph.num_blocks);
    let input_name = &graph.inputs[0];
    let output_name = &graph.outputs[0];

    let mut out_batch = Vec::with_capacity(batch.len());
    for (i, row_cts) in batch.into_iter().enumerate() {
        let mut env = HashMap::with_capacity(1);
        env.insert(input_name.clone(), row_cts);
        let mut outputs = penumbra_core::eval::evaluate_graph(backend, &ctx, graph, env)
            .map_err(|e| format!("evaluating sample {i}: {e}"))?;
        let out = outputs
            .remove(output_name)
            .ok_or_else(|| format!("sample {i}: graph did not produce output '{output_name}'"))?;
        out_batch.push(out);
    }
    Ok(out_batch)
}

/// Run an all-in-one predict forward pass: keygen once -> encrypt -> evaluate -> decrypt.
pub fn run_predict<B: Backend>(
    backend: &B,
    graph: &Graph,
    rows: &[Vec<i64>],
) -> Result<Vec<Vec<i64>>, String> {
    if graph.inputs.len() != 1 {
        return Err(format!(
            "predict expects a single-input graph, got {} inputs {:?}; multi-input graphs are not yet supported by the bridge",
            graph.inputs.len(),
            graph.inputs
        ));
    }
    if graph.outputs.len() != 1 {
        return Err(format!(
            "predict expects a single-output graph, got {} outputs {:?}",
            graph.outputs.len(),
            graph.outputs
        ));
    }
    backend.check_graph_budget(graph)?;

    let (ck, sk) = backend.keygen(graph.num_blocks);
    let ctx = EvalCtx::new(&sk, graph.num_blocks);
    let input_name = &graph.inputs[0];
    let output_name = &graph.outputs[0];

    let mut outputs = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let mut env = HashMap::with_capacity(1);
        env.insert(input_name.clone(), backend.encrypt(&ck, row));
        let mut res = penumbra_core::eval::evaluate_graph(backend, &ctx, graph, env)
            .map_err(|e| format!("evaluating sample {i}: {e}"))?;
        let produced = res
            .remove(output_name)
            .ok_or_else(|| format!("sample {i}: graph did not produce output '{output_name}'"))?;
        outputs.push(backend.decrypt_vec(&ck, &produced));
    }
    Ok(outputs)
}
