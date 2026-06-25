//! Golden exactness test — the project's truth oracle (`AGENTS.md` §1.1, ROADMAP Phase 2).
//!
//! > FHE output must equal the quantized-cleartext output, **bit-for-bit**.
//!
//! TFHE is exact, so any mismatch here is a quantization or implementation bug, never
//! crypto noise. This test wires that invariant into CI: it reads the committed Phase-2
//! fixture (`examples/mnist/phase2_fixture.json`), computes the quantized-integer
//! reference *in Rust*, runs the encrypted forward pass through the real op graph, and
//! asserts they agree over a batch.
//!
//! The reference is recomputed here (not just read from `expected_labels`) so that a
//! divergence is localized to the FHE path immediately — "debug the cleartext quantized
//! path first" (`AGENTS.md` §1.1). The fixture's `expected_labels` are additionally
//! checked against this Rust reference, guarding against fixture drift.
//!
//! As of Phase 3 the model is **not hardcoded**: the test deserializes the IR graph from
//! the fixture's `"graph"` key and runs it through `evaluate_graph`. The oracle's
//! parameters are recovered *from that same deserialized graph* (not a separate hardcoded
//! copy), so there is one source of model truth. Cross-language IR conformance lives in
//! `ir_conformance.rs`.
//!
//! Run with `cargo test --release` — debug FHE is impractically slow (`docs/DEVELOPMENT.md`).

use std::collections::HashMap;
use std::path::PathBuf;

use penumbra_fhe_runtime::{
    check_bit_width_budget, check_graph_bit_width_budget, decrypt_label, encrypt, evaluate_graph,
    keygen, Activation, EvalCtx, Graph, Linear, Op, OpSpec,
};
use serde_json::Value;

/// Load the committed Phase-2 fixture as untyped JSON (typed IR structs are Phase 3).
fn load_fixture() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../examples/mnist/phase2_fixture.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn as_i64_vec(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(|x| x.as_i64().expect("expected integer"))
        .collect()
}

/// The quantized-cleartext oracle, in plain integer arithmetic: `w·x + b`, then threshold.
fn cleartext_label(weights: &[i64], bias: i64, input: &[i64], threshold: i64) -> i64 {
    let logit: i64 = weights.iter().zip(input).map(|(&w, &x)| w * x).sum::<i64>() + bias;
    i64::from(logit >= threshold)
}

/// Recover the oracle's parameters from the deserialized IR graph itself, so the executed
/// model and the cleartext oracle share a single source of truth (no re-hardcoded copy).
/// Phase-2 model is `Linear → Argmax` with one weight row / bias / threshold.
fn oracle_params_from_graph(graph: &Graph) -> (Vec<i64>, i64, i64) {
    let mut weights = None;
    let mut bias = None;
    let mut threshold = None;
    for node in &graph.nodes {
        match &node.op {
            OpSpec::Linear {
                weights: w,
                bias: b,
                ..
            } => {
                assert_eq!(w.len(), 1, "Phase-2 logreg has a single weight row");
                weights = Some(w[0].clone());
                bias = Some(b[0]);
            }
            OpSpec::Argmax { threshold: t } => threshold = Some(*t),
            OpSpec::Activation { .. } | OpSpec::Add {} => {}
        }
    }
    (
        weights.expect("graph must contain a Linear node"),
        bias.expect("graph must contain a Linear node"),
        threshold.expect("graph must contain an Argmax node"),
    )
}

/// The headline gate: the encrypted IR graph (`Linear → Argmax`) equals the
/// quantized-cleartext label, bit-for-bit, over the whole committed test batch — run
/// entirely from the serialized IR, no hardcoded model (ROADMAP Phase 3 exit criterion).
#[test]
fn fhe_matches_quantized_cleartext_logreg() {
    let fx = load_fixture();

    // The model is the serialized IR graph. Deserialize it; no in-code op assembly.
    let graph = Graph::from_json(&fx["graph"].to_string()).expect("fixture graph deserializes");

    let test_inputs: Vec<Vec<i64>> = fx["test_inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(as_i64_vec)
        .collect();
    let expected_labels = as_i64_vec(&fx["expected_labels"]);

    // Oracle parameters come from the same graph, not a separate hardcoded copy.
    let (weights, bias, threshold) = oracle_params_from_graph(&graph);
    let input_name = graph.inputs[0].clone();
    let output_name = graph.outputs[0].clone();

    // Fail loudly before any crypto if the budget can't hold the accumulator.
    check_graph_bit_width_budget(&graph).expect("bit-width budget must fit");

    let (ck, sk) = keygen(graph.num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks: graph.num_blocks,
    };

    for (i, input) in test_inputs.iter().enumerate() {
        // Oracle: recompute in Rust integers, and confirm the fixture agrees (no drift).
        let reference = cleartext_label(&weights, bias, input, threshold);
        assert_eq!(
            reference, expected_labels[i],
            "fixture expected_labels[{i}] disagrees with the Rust cleartext oracle"
        );

        // FHE path: encrypt -> walk the IR graph -> decrypt the named output tensor.
        let mut env = HashMap::new();
        env.insert(input_name.clone(), encrypt(&ck, input));
        let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
        let label = decrypt_label(&ck, &out[&output_name]);

        assert_eq!(
            label, reference,
            "GOLDEN VIOLATION at sample {i}: FHE label {label} != quantized-cleartext {reference}"
        );
    }
}

/// Regression for the §1.3 budget soundness bug in `Linear::output_bits` (FINDING #1): the
/// accumulator is `sum_of_products + bias`, a signed quantity, so its width is
/// `max(sum_bits, bias_bits) + 2` — one carry bit from the bias add *and* one sign bit.
///
/// This is a pure bit-width-tracker test: it calls `output_bits`/`check_bit_width_budget`
/// only, with no keygen or FHE eval, so it runs instantly.
#[test]
fn budget_rejects_bias_sum_carry_overflow() {
    // 129 inputs, input_bits = 1, weight_bits = 6, bias = 32767 (15 magnitude bits).
    //   sum_growth = bit_length(128) = 8;  sum_bits = 1 + 6 + 8 = 15
    //   bias_bits  = magnitude_bits(32767) = 15
    // True signed width = max(15, 15) + 2 = 17 > capacity 16 (num_blocks = 8). The buggy
    // `+1` formula declared 16 and wrongly passed, silently wrapping the signed radix.
    let overflowing: Vec<Box<dyn Op>> = vec![Box::new(Linear {
        weights: vec![vec![0i64; 129]],
        bias: vec![32767],
        weight_bits: 6,
    })];
    assert!(
        check_bit_width_budget(&overflowing, 1, 8).is_err(),
        "budget check must reject a 17-bit signed accumulator against a 16-bit radix"
    );

    // Positive control: the committed-fixture shape (weight_bits = 4, 64 inputs,
    // bias = -1478, input_bits = 4, num_blocks = 8) must still fit, so the golden test stays
    // green.
    //   sum_growth = bit_length(63) = 6;  sum_bits = 4 + 4 + 6 = 14
    //   bias_bits  = magnitude_bits(1478) = 11
    // declared = max(14, 11) + 2 = 16 ≤ capacity 16.
    let fits: Vec<Box<dyn Op>> = vec![Box::new(Linear {
        weights: vec![vec![0i64; 64]],
        bias: vec![-1478],
        weight_bits: 4,
    })];
    assert!(
        check_bit_width_budget(&fits, 4, 8).is_ok(),
        "fixture-shaped Linear (declared 16 bits) must fit a 16-bit radix"
    );
}

/// Golden exactness for the `Activation(LUT)` op on its narrow domain: the PBS output must
/// match the cleartext table for every input value (the `hello_fhe` LUT discipline, now
/// through the op interface). The binary decision doesn't use this LUT, so it is proven
/// independently here — and is the forward-compat anchor for Phase-4 post-Requant activations.
#[test]
fn fhe_activation_lut_matches_table() {
    let fx = load_fixture();
    // num_blocks now lives in the IR graph; the activation LUT is sibling test data.
    let num_blocks = fx["graph"]["num_blocks"].as_u64().unwrap() as usize;
    let output_bits = fx["activation"]["output_bits"].as_u64().unwrap() as usize;

    let lut: Vec<u64> = fx["activation"]["lut"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect();
    let test_inputs = as_i64_vec(&fx["activation"]["test_inputs"]);
    let expected = as_i64_vec(&fx["activation"]["expected"]);

    let act = Activation {
        lut: lut.clone(),
        output_bits,
    };

    let (ck, sk) = keygen(num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks,
    };

    for (i, &v) in test_inputs.iter().enumerate() {
        let ct = encrypt(&ck, &[v]);
        let out = act.eval(&ctx, &ct);
        let got = decrypt_label(&ck, &out);
        assert_eq!(
            got, expected[i],
            "GOLDEN VIOLATION: activation LUT on input {v} gave {got}, expected {}",
            expected[i]
        );
    }
}
