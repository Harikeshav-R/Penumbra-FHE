//! Golden exactness for the multi-input `Add` op (`AGENTS.md` §1.1, ROADMAP Phase 4).
//!
//! `Add` is the project's first **multi-input** op (two operand tensors). This test wires a
//! two-input `Add` node into an IR graph, runs it under FHE through `evaluate_graph`, and
//! asserts the decrypted element-wise sum equals the plain-integer sum, bit-for-bit. It also
//! exercises the relaxed eval loop / bit-width tracker (multiple `inputs` per node).
//!
//! Run with `cargo test --release` — debug FHE is impractically slow (`docs/DEVELOPMENT.md`).

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    Node, OpSpec,
};

/// Build a single-node graph `add: (a, b) -> sum` over a small radix.
fn add_graph(num_blocks: usize, input_bits: usize) -> Graph {
    Graph {
        schema_version: penumbra_fhe_runtime::SCHEMA_VERSION.to_string(),
        num_blocks,
        input_bits,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["sum".to_string()],
        nodes: vec![Node {
            name: "add".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["sum".to_string()],
            op: OpSpec::Add {},
        }],
    }
}

/// FHE `Add` over a two-element vector equals the cleartext element-wise sum, bit-for-bit.
#[test]
fn fhe_add_matches_cleartext() {
    // 4 blocks = 8-bit signed radix: plenty for the small operands below, and cheaper than
    // the 8-block logreg fixture so this stays a fast Phase-4 gate.
    let num_blocks = 4;
    let input_bits = 4; // operands in roughly [-8, 7]

    let graph = add_graph(num_blocks, input_bits);

    // input_bits=4 → Add output_bits = max(4,4)+1 = 5 ≤ capacity 8. Must pass the budget.
    check_graph_bit_width_budget(&graph).expect("Add budget must fit");

    let a = vec![3i64, -5];
    let b = vec![4i64, 6];
    let expected: Vec<i64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let (ck, sk) = keygen(num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks,
    };

    let mut env = HashMap::new();
    env.insert("a".to_string(), encrypt(&ck, &a));
    env.insert("b".to_string(), encrypt(&ck, &b));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let got = decrypt_vec(&ck, &out["sum"]);

    assert_eq!(
        got, expected,
        "GOLDEN VIOLATION: FHE Add {got:?} != cleartext sum {expected:?}"
    );
}
