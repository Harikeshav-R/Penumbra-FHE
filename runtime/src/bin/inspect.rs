//! `penumbra inspect <file>` — print an IR model's op graph + per-tensor bit-widths.
//!
//! A human-inspection aid (ROADMAP Phase 3): it deserializes an IR [`Graph`] and walks the
//! same bit-width tracker the runtime uses ([`propagate_bit_widths`]), so the widths shown
//! are the real ones — no reimplementation of the growth rules (`AGENTS.md` §1.3, single
//! source of truth). It never runs keygen or FHE, so it is instant and needs no `--release`.
//!
//! The argument may be either a bare IR graph file or a fixture/model file that embeds the
//! graph under a top-level `"graph"` key (e.g. `examples/mnist/phase2_fixture.json`); both
//! are accepted so the same command inspects exported models and test fixtures.

use std::process::ExitCode;

use penumbra_fhe_runtime::keys::{radix_capacity_bits, MESSAGE_BITS};
use penumbra_fhe_runtime::{propagate_bit_widths, Graph};

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
        "usage: inspect <model.fhe | fixture.json>\n  prints the IR op graph and per-tensor \
         bit-widths"
            .to_string()
    })?;

    let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))?;

    // Accept either a bare graph or a fixture embedding it under "graph".
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{path} is not valid JSON: {e}"))?;
    let graph_json = if value.get("graph").is_some() {
        value["graph"].to_string()
    } else {
        text
    };

    let graph = Graph::from_json(&graph_json)?;
    let widths = propagate_bit_widths(&graph)?;
    let capacity = radix_capacity_bits(graph.num_blocks);

    println!(
        "schema_version {}  num_blocks {}  capacity {capacity} bits ({} blocks × {MESSAGE_BITS})",
        graph.schema_version, graph.num_blocks, graph.num_blocks
    );
    println!(
        "inputs {:?} @ {} bits   outputs {:?}",
        graph.inputs, graph.input_bits, graph.outputs
    );
    println!("nodes:");

    let mut over_capacity = false;
    for node in &graph.nodes {
        // Widths are keyed by tensor name; show each output tensor's propagated width.
        let parts: Vec<String> = node
            .outputs
            .iter()
            .map(|name| {
                let bits = widths.get(name).copied().unwrap_or(0);
                let flag = if bits > capacity {
                    over_capacity = true;
                    "  [OVER CAPACITY]"
                } else {
                    ""
                };
                format!("{name}={bits} bits{flag}")
            })
            .collect();
        println!(
            "  {:<6} {:<10} {:?} -> {:?}   {}",
            node.name,
            node.op.op_type(),
            node.inputs,
            node.outputs,
            parts.join(", ")
        );
    }

    if over_capacity {
        return Err(format!(
            "bit-width budget EXCEEDED: a tensor needs more than {capacity} bits. Reduce \
             precision or widen num_blocks (a Requant here is Phase 4)."
        ));
    }
    println!("OK: fits the {capacity}-bit budget");
    Ok(())
}
