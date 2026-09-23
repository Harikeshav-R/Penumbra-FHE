//! Golden exactness test for graph fusion (ROADMAP Phase 10 Step 2, AGENTS.md §1.1).
//!
//! Asserts that:
//! 1. `optimize_graph` rewrites a `Linear` -> `Requant` -> `Activation` chain into
//!    `Linear` -> `Requant(fused)`.
//! 2. Encrypted execution over the fused graph produces bit-for-bit identical results
//!    to the unfused cleartext reference (mirroring `reference.py`) over multiple inputs.

use std::borrow::Cow;
use std::collections::HashMap;

use penumbra_fhe_runtime::{
    decrypt_vec, encrypt, evaluate_graph, keygen, optimize_graph, EvalCtx, Graph, Node, OpSpec,
    SCHEMA_VERSION,
};

fn reference_unfused_eval(input: &[i64]) -> Vec<i64> {
    assert_eq!(input.len(), 2);
    // 1. Linear: weights [[1, 2], [2, 1]], bias [0, 0]
    let fc0 = input[0] * 1 + input[1] * 2;
    let fc1 = input[0] * 2 + input[1] * 1;

    // 2. Requant: shift 1, mult 1, round_bias 0, out_bits 2, clamp_lut [0, 1, 2, 3]
    let rq = |val: i64| -> i64 {
        let val_mult = val * 1 + 0;
        let shifted = val_mult >> 1;
        let nonneg = shifted.max(0);
        let sat = nonneg.min((1i64 << 2) - 1);
        let clamp_lut = [0i64, 1, 2, 3];
        clamp_lut[sat as usize]
    };
    let rq0 = rq(fc0);
    let rq1 = rq(fc1);

    // 3. Activation: lut [0, 2, 1, 3]
    let act = |val: i64| -> i64 {
        let act_lut = [0i64, 2, 1, 3];
        act_lut[val as usize]
    };
    vec![act(rq0), act(rq1)]
}

#[test]
fn test_golden_fusion_exact_agreement() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 2,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
        nodes: vec![
            Node {
                name: "fc".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["fc_out".to_string()],
                op: OpSpec::Linear {
                    weights: vec![vec![1, 2], vec![2, 1]],
                    bias: vec![0, 0],
                    weight_bits: 2,
                },
            },
            Node {
                name: "rq".to_string(),
                inputs: vec!["fc_out".to_string()],
                outputs: vec!["rq_out".to_string()],
                op: OpSpec::Requant {
                    shift: 1,
                    mult: 1,
                    round_bias: 0,
                    out_bits: 2,
                    clamp_lut: vec![0, 1, 2, 3],
                    mults: vec![],
                    shifts: vec![],
                    round_biases: vec![],
                    channel_size: None,
                },
            },
            Node {
                name: "act".to_string(),
                inputs: vec!["rq_out".to_string()],
                outputs: vec!["out".to_string()],
                op: OpSpec::Activation {
                    lut: vec![0, 2, 1, 3],
                    output_bits: 2,
                },
            },
        ],
    };

    // Assert that optimize_graph rewrites the graph
    let opt = optimize_graph(&graph).expect("optimize_graph succeeds");
    match &opt {
        Cow::Owned(g) => {
            assert_eq!(g.nodes.len(), 2, "3 nodes must fuse to 2 nodes");
            assert_eq!(g.nodes[0].name, "fc");
            assert_eq!(g.nodes[1].name, "rq");
            assert_eq!(g.nodes[1].outputs, vec!["out"]);
            if let OpSpec::Requant { clamp_lut, .. } = &g.nodes[1].op {
                assert_eq!(*clamp_lut, vec![0, 2, 1, 3]);
            } else {
                panic!("second node should be fused Requant");
            }
        }
        Cow::Borrowed(_) => panic!("optimize_graph should have fused Requant and Activation"),
    }

    // Run FHE path over several inputs and compare against cleartext unfused reference
    let (ck, sk) = keygen(graph.num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks: graph.num_blocks,
    };

    let test_inputs = vec![
        vec![0i64, 0],
        vec![1, 0],
        vec![0, 1],
        vec![1, 1],
        vec![2, 1],
        vec![-1, 2],
    ];

    for input in test_inputs {
        let expected = reference_unfused_eval(&input);

        let mut in_map = HashMap::new();
        in_map.insert("x".to_string(), encrypt(&ck, &input));

        // evaluate_graph invokes optimize_graph internally
        let out_map = evaluate_graph(&ctx, &graph, in_map)
            .expect("evaluate_graph should succeed");

        let out_ct = out_map.get("out").expect("output 'out' must be present");
        let got = decrypt_vec(&ck, out_ct);

        assert_eq!(
            got, expected,
            "fused FHE evaluation must equal unfused reference bit-for-bit on input {input:?}"
        );
    }
}
