#![cfg(feature = "ckks")]

//! Synthetic graph tests for individual ops under CKKS:
//! Add, Pool(avg), Activation, Compare, Requant, Linear, Conv2d, Argmax, Concat, and Split.

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
                padding: 0,
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
fn ckks_fhe_padded_pool_avg_matches_cleartext() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 5,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "pool".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Pool {
                mode: "avg".to_string(),
                in_h: 3,
                in_w: 3,
                channels: 1,
                pool_h: 2,
                pool_w: 2,
                stride: 2,
                padding: 1,
            },
        }],
    };

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    // 1 channel of 3x3 = 9 elements: [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    let x: Vec<i64> = (0..9).map(|i| i - 4).collect();
    let expected = vec![-4, -5, 1, 8];

    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &x));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let got = decrypt_vec(&ck, &out["y"]);

    assert_eq!(
        got, expected,
        "CKKS padded Pool(avg) must match window sums"
    );
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

#[test]
fn ckks_fhe_linear_matches_cleartext() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 2,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "fc".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Linear {
                weights: vec![vec![1, -2, 3], vec![0, 4, -1]],
                bias: vec![5, -3],
                weight_bits: 4,
            },
        }],
    };

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    // Arithmetic:
    // y[0] = 3*1 + 1*(-2) + 2*3 + 5 = 3 - 2 + 6 + 5 = 12
    // y[1] = 3*0 + 1*4 + 2*(-1) + (-3) = 0 + 4 - 2 - 3 = -1
    let x = vec![3i64, 1, 2];
    let expected = vec![12i64, -1];

    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &x));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let raw = decrypt_raw_vec(&ck, &out["y"]);
    let got = decrypt_vec(&ck, &out["y"]);

    for (i, (&g, &w)) in raw.iter().zip(&expected).enumerate() {
        let err = (g - w as f64).abs();
        assert!(err < 0.1, "slot {i}: err {err} >= 0.1 tolerance for Linear");
    }
    assert_eq!(got, expected, "CKKS Linear must match cleartext");
}

#[test]
fn ckks_fhe_conv2d_matches_cleartext() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "conv".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Conv2d {
                weights: vec![vec![1, 0, -1, 2]],
                bias: vec![1],
                weight_bits: 3,
                in_h: 3,
                in_w: 3,
                in_channels: 1,
                kernel_h: 2,
                kernel_w: 2,
                stride: 1,
                padding: 0,
            },
        }],
    };

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    // Arithmetic:
    // out(oy,ox) = Σ x[oy+ky][ox+kx] · k[ky][kx] + 1
    // Input 3x3:
    //   [1, 2, 3]
    //   [4, 5, 6]
    //   [7, 8, 9]
    // Kernel 2x2: [[1, 0], [-1, 2]], bias: 1
    //   (0,0): 1*1 + 2*0 + 4*(-1) + 5*2 + 1 = 1 - 4 + 10 + 1 = 8
    //   (0,1): 2*1 + 3*0 + 5*(-1) + 6*2 + 1 = 2 - 5 + 12 + 1 = 10
    //   (1,0): 4*1 + 5*0 + 7*(-1) + 8*2 + 1 = 4 - 7 + 16 + 1 = 14
    //   (1,1): 5*1 + 6*0 + 8*(-1) + 9*2 + 1 = 5 - 8 + 18 + 1 = 16
    let x = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9];
    let expected = vec![8i64, 10, 14, 16];

    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &x));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let raw = decrypt_raw_vec(&ck, &out["y"]);
    let got = decrypt_vec(&ck, &out["y"]);

    for (i, (&g, &w)) in raw.iter().zip(&expected).enumerate() {
        let err = (g - w as f64).abs();
        assert!(err < 0.1, "slot {i}: err {err} >= 0.1 tolerance for Conv2d");
    }
    assert_eq!(got, expected, "CKKS Conv2d must match cleartext");
}

#[test]
fn ckks_fhe_argmax_matches_cleartext() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 16,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "argmax".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Argmax { threshold: 0 },
        }],
    };

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    // prepare_argmax ramps within tau = 1638 of threshold 0.
    // Inputs sit well outside the ramp:
    // -20000 -> 0, -5000 -> 0, 5000 -> 1, 20000 -> 1.
    let test_cases = vec![
        (-20000i64, 0i64),
        (-5000, 0),
        (5000, 1),
        (20000, 1),
    ];

    for (val, expected) in test_cases {
        let mut env = HashMap::new();
        env.insert("x".to_string(), encrypt(&ck, &[val]));

        let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
        let raw = decrypt_raw_vec(&ck, &out["y"]);
        let got = decrypt_vec(&ck, &out["y"]);

        let err = (raw[0] - expected as f64).abs();
        assert!(
            err < 0.55,
            "input {val}: err {err} >= 0.55 tolerance for Argmax"
        );
        assert_eq!(got[0], expected, "input {val}: got != expected for Argmax");
    }
}

#[test]
fn ckks_fhe_concat_split_matches_cleartext() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![
            Node {
                name: "split".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["a".to_string(), "b".to_string()],
                op: OpSpec::Split { sizes: vec![2, 2] },
            },
            Node {
                name: "concat".to_string(),
                inputs: vec!["b".to_string(), "a".to_string()],
                outputs: vec!["y".to_string()],
                op: OpSpec::Concat { sizes: vec![2, 2] },
            },
        ],
    };

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    // Split [1, -2, 3, 4] with sizes [2, 2] -> a=[1, -2], b=[3, 4]
    // Concat [b, a] with sizes [2, 2] -> [3, 4, 1, -2]
    let x = vec![1i64, -2, 3, 4];
    let expected = vec![3i64, 4, 1, -2];

    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &x));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let raw = decrypt_raw_vec(&ck, &out["y"]);
    let got = decrypt_vec(&ck, &out["y"]);

    for (i, (&g, &w)) in raw.iter().zip(&expected).enumerate() {
        let err = (g - w as f64).abs();
        assert!(err < 0.1, "slot {i}: err {err} >= 0.1 tolerance for Concat/Split");
    }
    assert_eq!(got, expected, "CKKS Concat/Split must match cleartext");
}

#[test]
fn ckks_fhe_per_channel_requant_matches_reference() {
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
                shift: 0,
                mult: 1,
                round_bias: 0,
                clamp_lo: 0,
                zero_point: 0,
                out_bits: 2,
                clamp_lut: vec![0, 1, 2, 3],
                mults: vec![1, 3],
                shifts: vec![7, 8],
                round_biases: vec![64, 128],
                channel_size: Some(2),
            },
        }],
    };

    let params = DEFAULT_PARAMS;
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let ctx = EvalCtx::new(&sk, 8);

    // Arithmetic:
    // min(((max(v,0)·m + rb) >> s), 3), with m, rb, s taken per channel idx/2.
    // Domain-aligned shifts (7, 8) match the CKKS Chebyshev approximation domain [-513, 513].
    // Channel 0 (indices 0, 1): mult=1, shift=7, round_bias=64
    //   idx 0: v = 128 -> max(128,0)*1 + 64 = 192; 192 >> 7 = 1; min(1, 3) = 1
    //   idx 1: v = 384 -> max(384,0)*1 + 64 = 448; 448 >> 7 = 3; min(3, 3) = 3
    // Channel 1 (indices 2, 3): mult=3, shift=8, round_bias=128
    //   idx 2: v = 85  -> max(85,0)*3 + 128 = 383; 383 >> 8 = 1; min(1, 3) = 1
    //   idx 3: v = 256 -> max(256,0)*3 + 128 = 896; 896 >> 8 = 3; min(3, 3) = 3
    let x = vec![128i64, 384, 85, 256];
    let expected = vec![1i64, 3, 1, 3];

    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &x));

    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let raw = decrypt_raw_vec(&ck, &out["y"]);
    let got = decrypt_vec(&ck, &out["y"]);

    for (i, (&g, &w)) in raw.iter().zip(&expected).enumerate() {
        let err = (g - w as f64).abs();
        assert!(
            err < 0.35,
            "slot {i}: err {err} >= 0.35 tolerance for per-channel Requant"
        );
    }
    assert_eq!(
        got, expected,
        "CKKS per-channel Requant must match reference"
    );
}
