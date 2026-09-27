//! Golden exactness for Concat and Split ops under TFHE (ROADMAP Phase 8 / Phase 11).
//!
//! Verifies that splitting a ciphertext vector into two slices and concatenating them in
//! reversed order matches the cleartext permutation bit-for-bit.

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    Node, OpSpec,
};

fn concat_split_graph(num_blocks: usize, input_bits: usize) -> Graph {
    Graph {
        schema_version: penumbra_fhe_runtime::SCHEMA_VERSION.to_string(),
        num_blocks,
        input_bits,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![
            Node {
                name: "split".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["a".to_string(), "b".to_string()],
                op: OpSpec::Split {
                    sizes: vec![2, 4],
                },
            },
            Node {
                name: "concat".to_string(),
                inputs: vec!["b".to_string(), "a".to_string()],
                outputs: vec!["y".to_string()],
                op: OpSpec::Concat {
                    sizes: vec![4, 2],
                },
            },
        ],
    }
}

/// FHE Concat and Split over a ciphertext vector matches cleartext bit-for-bit.
#[test]
fn fhe_concat_split_matches_cleartext() {
    let num_blocks = 4;
    let input_bits = 4;

    let graph = concat_split_graph(num_blocks, input_bits);
    check_graph_bit_width_budget(&graph).expect("Concat/Split budget must fit");

    let x = vec![1i64, -2, 3, -4, 5, 6];
    let expected = vec![3i64, -4, 5, 6, 1, -2];

    let (ck, sk) = keygen(num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks,
    };

    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &x));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let got = decrypt_vec(&ck, &out["y"]);

    assert_eq!(
        got, expected,
        "GOLDEN VIOLATION: FHE Concat/Split {got:?} != cleartext {expected:?}"
    );
}
