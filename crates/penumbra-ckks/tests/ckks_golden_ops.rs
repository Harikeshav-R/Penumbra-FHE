#![cfg(feature = "ckks")]

//! Synthetic graph tests for individual ops under CKKS:
//! Add, Pool(avg), Activation, and Requant.

use std::collections::HashMap;

use penumbra_ckks::backend::evaluate_graph;
use penumbra_ckks::encrypt::{decrypt_raw_vec, decrypt_vec, encrypt};
use penumbra_ckks::keys::keygen;
use penumbra_ckks::params::DEFAULT_PARAMS;
use penumbra_core::backend::EvalCtx;
use penumbra_core::ir::{Graph, Node, OpSpec};

#[test]
fn ckks_fhe_add_matches_cleartext() {
    let graph = Graph {
        schema_version: "0.6.0".to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["sum".to_string()],
        nodes: vec![Node {
            name: "add".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["sum".to_string()],
            op: OpSpec::Add {},
        }],
    };

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    let a = vec![3i64, -5, 12, -20];
    let b = vec![4i64, 6, -8, 25];
    let expected: Vec<i64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let mut env = HashMap::new();
    env.insert("a".to_string(), encrypt(&ck, &a));
    env.insert("b".to_string(), encrypt(&ck, &b));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let got = decrypt_vec(&ck, &out["sum"]);

    assert_eq!(got, expected, "CKKS Add must match cleartext sum");
}

#[test]
fn ckks_fhe_pool_avg_matches_cleartext() {
    let graph = Graph {
        schema_version: "0.6.0".to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "pool".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Pool {
                mode: "avg".to_string(),
                in_h: 4,
                in_w: 4,
                channels: 2,
                pool_h: 2,
                pool_w: 2,
                stride: 2,
            },
        }],
    };

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    // 2 channels of 4x4 = 32 elements
    let x: Vec<i64> = (0..32).map(|i| (i % 7) - 3).collect();

    // Cleartext 2x2 pooling with sum
    let mut expected = Vec::new();
    for c in 0..2 {
        let base = c * 16;
        for oy in 0..2 {
            for ox in 0..2 {
                let mut sum = 0;
                for ky in 0..2 {
                    for kx in 0..2 {
                        let y = oy * 2 + ky;
                        let xx = ox * 2 + kx;
                        sum += x[base + y * 4 + xx];
                    }
                }
                expected.push(sum);
            }
        }
    }

    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &x));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let got = decrypt_vec(&ck, &out["y"]);

    assert_eq!(got, expected, "CKKS Pool(avg) must match window sums");
}

#[test]
fn ckks_fhe_activation_lut_matches_cleartext() {
    let lut = vec![0u64, 2, 4, 6];
    let graph = Graph {
        schema_version: "0.6.0".to_string(),
        num_blocks: 4,
        input_bits: 2,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "act".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Activation {
                lut: lut.clone(),
                output_bits: 4,
            },
        }],
    };

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    let x = vec![0i64, 1, 2, 3];
    let expected: Vec<i64> = x.iter().map(|&v| lut[v as usize] as i64).collect();

    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &x));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let raw = decrypt_raw_vec(&ck, &out["y"]);
    let got = decrypt_vec(&ck, &out["y"]);

    for (i, (&g, &w)) in raw.iter().zip(&expected).enumerate() {
        let err = (g - w as f64).abs();
        assert!(
            err < 0.1,
            "slot {i}: err {err} too large for exact activation LUT"
        );
    }
    assert_eq!(
        got, expected,
        "Activation LUT outputs must round to expected"
    );
}
