#![cfg(feature = "ckks")]

//! Synthetic graph tests for individual ops under CKKS:
//! Add, Pool(avg), Activation, and Requant.

use std::collections::HashMap;

use penumbra_ckks::backend::evaluate_graph;
use penumbra_ckks::encrypt::{decrypt_raw_vec, decrypt_vec, encrypt};
use penumbra_ckks::keys::keygen;
use penumbra_ckks::params::DEFAULT_PARAMS;
use penumbra_core::backend::EvalCtx;
use penumbra_core::ir::{Graph, Node, OpSpec, SCHEMA_VERSION};

#[test]
fn ckks_fhe_add_matches_cleartext() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
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
        schema_version: SCHEMA_VERSION.to_string(),
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
        schema_version: SCHEMA_VERSION.to_string(),
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

#[test]
fn ckks_fhe_compare_matches_cleartext() {
    let indices = vec![2, 0, 1, 0];
    let thresholds = vec![5, 3, 10, 4];
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "cmp".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Compare {
                indices: indices.clone(),
                thresholds: thresholds.clone(),
            },
        }],
    };

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    let x = vec![3i64, 9, 5];
    let expected: Vec<i64> = indices
        .iter()
        .zip(&thresholds)
        .map(|(&idx, &t)| if x[idx] >= t { 1 } else { 0 })
        .collect();

    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &x));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let raw = decrypt_raw_vec(&ck, &out["y"]);
    let got = decrypt_vec(&ck, &out["y"]);
    for (i, (&g, &w)) in raw.iter().zip(&expected).enumerate() {
        let err = (g - w as f64).abs();
        assert!(err < 0.55, "slot {i}: err {err} too large for compare step");
    }
    assert_eq!(
        got, expected,
        "Compare outputs must round to expected 0/1 bits"
    );
}

#[test]
fn ckks_fhe_requant_non_identity_clamp_lut_matches_reference() {
    let clamp_lut = vec![0u64, 2, 1, 3];
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 10,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "rq".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Requant {
                shift: 7,
                mult: 1,
                round_bias: 64,
                clamp_lo: 0,
                zero_point: 0,
                out_bits: 2,
                clamp_lut: clamp_lut.clone(),
                mults: vec![],
                shifts: vec![],
                round_biases: vec![],
                channel_size: None,
            },
        }],
    };

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    // Requant: ((v.max(0) + 64) >> 7).clamp(0, 3), then clamp_lut[sat]
    // -256: 64 >> 7 = 0 -> clamp_lut[0] = 0
    //    0: 64 >> 7 = 0 -> clamp_lut[0] = 0
    //  128: 192 >> 7 = 1 -> clamp_lut[1] = 2
    //  256: 320 >> 7 = 2 -> clamp_lut[2] = 1
    //  384: 448 >> 7 = 3 -> clamp_lut[3] = 3
    let x = vec![-256i64, 0, 128, 256, 384];
    let expected = vec![0i64, 0, 2, 1, 3];

    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &x));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let raw = decrypt_raw_vec(&ck, &out["y"]);
    let got = decrypt_vec(&ck, &out["y"]);

    for (i, (&g, &w)) in raw.iter().zip(&expected).enumerate() {
        let err = (g - w as f64).abs();
        assert!(
            err < 0.35,
            "slot {i}: err {err} too large for non-identity Requant"
        );
    }
    assert_eq!(
        got, expected,
        "Requant with non-identity clamp_lut outputs must round to expected"
    );
}
