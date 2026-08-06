//! `penumbra predict <file>` — run encrypted inference on an IR model and print the outputs.
//!
//! The runtime side of the Phase-9 subprocess bridge (`PROJECT.md` §15, `ROADMAP.md` Phase 9):
//! the Python front end (`penumbra.client.run_encrypted`) writes a model + a batch of quantized
//! integer inputs, this binary runs the real encrypted forward pass, and Python decrypts the
//! meaning (argmax / label) on its side (`PROJECT.md` §11 — the client owns argmax).
//!
//! It is a thin CLI over the crate's public API — the *same* keygen → encrypt → `evaluate_graph`
//! → decrypt round trip the golden tests use (`runtime/tests/golden_cnn.rs`). It adds **no**
//! crypto and touches neither `ops/` nor `eval.rs`: a new use case is a new graph, never a
//! backend edit (`AGENTS.md` §1.2).
//!
//! Contract:
//!   - Argument: a path to a model file — either a bare IR [`Graph`] (what `Model.export` writes)
//!     or a fixture embedding it under a top-level `"graph"` key (same as `inspect`).
//!   - Stdin: a JSON array of quantized integer input rows, `[[i64, …], …]` — one row per sample,
//!     each already in the graph's input domain (the client quantizes; the server never sees
//!     floats or scales, `PROJECT.md` §11).
//!   - Stdout: `{"outputs": [[i64, …], …]}` — the decrypted graph-output tensor for each sample,
//!     in input order. Raw values (wide logits, or a single 2-class `Argmax` label bit); the
//!     client interprets them.
//!
//! Keygen runs **once** and the key pair is reused across the whole batch — keygen is a
//! per-`num_blocks` cost, not a per-sample one, so a batch amortizes it. Run in `--release`;
//! debug FHE is impractically slow (`docs/DEVELOPMENT.md`).

use std::collections::HashMap;
use std::io::Read;
use std::process::ExitCode;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
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
    let path = std::env::args().nth(1).ok_or_else(|| {
        "usage: predict <model.fhe | fixture.json>\n  reads a JSON batch of quantized integer \
         input rows on stdin and prints {\"outputs\": [[i64, …], …]}"
            .to_string()
    })?;

    let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))?;

    // Accept either a bare graph or a fixture embedding it under "graph" (mirrors `inspect`),
    // so the same command runs an exported model or a committed test fixture.
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{path} is not valid JSON: {e}"))?;
    let graph_json = if value.get("graph").is_some() {
        value["graph"].to_string()
    } else {
        text
    };

    let graph = Graph::from_json(&graph_json)?;

    // A single-input graph is the only shape the front end produces today; reject anything else
    // loudly here rather than let the batch-loop pick `inputs[0]` and silently ignore the rest
    // (`AGENTS.md` §1.4). Multi-input graphs are Phase-8 work.
    if graph.inputs.len() != 1 {
        return Err(format!(
            "predict expects a single-input graph, got {} inputs {:?}; multi-input graphs are \
             not yet supported by the bridge",
            graph.inputs.len(),
            graph.inputs
        ));
    }

    // Fail loudly before any (expensive) keygen if the declared widths can't fit the radix,
    // naming the offending node (`AGENTS.md` §1.3, §1.4).
    check_graph_bit_width_budget(&graph)?;

    // Read the quantized integer input batch from stdin.
    let mut stdin_text = String::new();
    std::io::stdin()
        .read_to_string(&mut stdin_text)
        .map_err(|e| format!("cannot read input batch from stdin: {e}"))?;
    let inputs: Vec<Vec<i64>> = serde_json::from_str(&stdin_text).map_err(|e| {
        format!("stdin is not a JSON array of integer input rows ([[i64, …], …]): {e}")
    })?;

    let input_name = graph.inputs[0].clone();
    let output_name = graph.outputs[0].clone();

    // Keygen once and reuse across the whole batch (keygen is a per-`num_blocks` cost, not
    // per-sample). The client key encrypts/decrypts; the server key drives evaluation.
    let (ck, sk) = keygen(graph.num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks: graph.num_blocks,
    };

    let mut outputs: Vec<Vec<i64>> = Vec::with_capacity(inputs.len());
    for (i, row) in inputs.iter().enumerate() {
        // encrypt -> walk the IR graph -> decrypt the named output tensor. `decrypt_vec` handles
        // a length-1 output (the 2-class `Argmax` label) and a wide logit head uniformly, so the
        // binary stays op-agnostic — the client decides argmax vs. label.
        let mut env = HashMap::new();
        env.insert(input_name.clone(), encrypt(&ck, row));
        let out =
            evaluate_graph(&ctx, &graph, env).map_err(|e| format!("evaluating sample {i}: {e}"))?;
        let produced = out
            .get(&output_name)
            .ok_or_else(|| format!("sample {i}: graph did not produce output '{output_name}'"))?;
        outputs.push(decrypt_vec(&ck, produced));
    }

    let result = serde_json::json!({ "outputs": outputs });
    println!("{result}");
    Ok(())
}
