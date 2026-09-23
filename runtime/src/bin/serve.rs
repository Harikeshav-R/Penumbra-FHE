//! `penumbra serve <model.fhe> <server.key> <in.cts> <out.cts>` — the server side.
//!
//! The privacy-critical half of the Phase-9 client/server split (`PROJECT.md` §11,
//! `SECURITY.md`). It is handed **only** the model graph, the **public** server key, and the
//! **encrypted** inputs — never the secret client key and never plaintext. It evaluates the op
//! graph under FHE and writes the encrypted outputs; it structurally *cannot* decrypt, which is
//! the whole privacy claim: the server runs the entire model on ciphertext and learns nothing.
//!
//! A thin CLI over the public API (`evaluate_graph`) — the *same* graph walk the golden tests
//! and `predict` use, touching neither `ops/` nor `eval.rs` (`AGENTS.md` §1.2).

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, deserialize_cts_batch, ensure_num_blocks_match, load_server_key,
    serialize_cts_batch, CtVec, EvalCtx, Graph, TfheBackend,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: serve <model.fhe | fixture.json> <server.key> <in.cts> <out.cts>\n  \
                 evaluates the model on the encrypted inputs and writes encrypted outputs";
    let model_path = args.next().ok_or_else(|| usage.to_string())?;
    let server_path = PathBuf::from(args.next().ok_or_else(|| usage.to_string())?);
    let in_path = PathBuf::from(args.next().ok_or_else(|| usage.to_string())?);
    let out_path = PathBuf::from(args.next().ok_or_else(|| usage.to_string())?);

    // Load the graph (accept a bare graph or a fixture embedding it under "graph", like `predict`).
    let text = std::fs::read_to_string(&model_path)
        .map_err(|e| format!("cannot read {model_path}: {e}"))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{model_path} is not valid JSON: {e}"))?;
    let graph_json = if value.get("graph").is_some() {
        value["graph"].to_string()
    } else {
        text
    };
    let graph = Graph::from_json(&graph_json)?;

    if graph.inputs.len() != 1 {
        return Err(format!(
            "serve expects a single-input graph, got {} inputs {:?}; multi-input graphs are not \
             yet supported by the bridge",
            graph.inputs.len(),
            graph.inputs
        ));
    }
    check_graph_bit_width_budget(&graph)?;

    // The server holds ONLY the public server key (`PROJECT.md` §11) — never the client key.
    let (sk, key_num_blocks, profile) = load_server_key(&server_path)?;
    // A key generated for a different radix width cannot evaluate this model (the "key mismatch"
    // failure mode, ROADMAP Phase 9) — fail loudly before touching ciphertext (`AGENTS.md` §1.4).
    ensure_num_blocks_match(key_num_blocks, graph.num_blocks)?;
    // Evaluate under the profile the key was generated with: a "gaussian" server key is as
    // valid as the default one (`PROJECT.md` §12 — the single TFHE override knob). The profile
    // drives keygen and key serialization only; the graph walk is identical either way.
    let backend = TfheBackend::new(profile);

    let in_bytes = std::fs::read(&in_path)
        .map_err(|e| format!("cannot read ciphertext from {}: {e}", in_path.display()))?;
    let inputs: Vec<CtVec> = deserialize_cts_batch(&in_bytes)
        .map_err(|e| format!("{e} (from {})", in_path.display()))?;

    let ctx = EvalCtx {
        sk: &sk,
        num_blocks: graph.num_blocks,
    };
    let input_name = graph.inputs[0].clone();
    let output_name = graph.outputs[0].clone();

    let mut outputs: Vec<CtVec> = Vec::with_capacity(inputs.len());
    for (i, ct) in inputs.into_iter().enumerate() {
        let mut env = HashMap::new();
        env.insert(input_name.clone(), ct);
        let out = penumbra_core::eval::evaluate_graph(&backend, &ctx, &graph, env)
            .map_err(|e| format!("evaluating sample {i}: {e}"))?;
        let produced = out
            .get(&output_name)
            .ok_or_else(|| format!("sample {i}: graph did not produce output '{output_name}'"))?;
        outputs.push(produced.clone());
    }

    let out_bytes = serialize_cts_batch(&outputs)?;
    std::fs::write(&out_path, out_bytes).map_err(|e| {
        format!(
            "cannot write output ciphertext to {}: {e}",
            out_path.display()
        )
    })?;

    eprintln!(
        "evaluated {} sample(s) under FHE -> {}",
        outputs.len(),
        out_path.display()
    );
    Ok(())
}
