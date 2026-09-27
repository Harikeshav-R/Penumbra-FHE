#![cfg(feature = "ckks")]

//! Exercise Backend trait methods and Op<CkksBackend> methods for penumbra-ckks.

use penumbra_ckks::backend::CkksBackend;
use penumbra_ckks::params::DEFAULT_PARAMS;
use penumbra_core::backend::Backend;
use penumbra_core::ir::{Graph, OpSpec, SCHEMA_VERSION};

#[test]
fn test_ckks_backend_metadata_and_build_op() {
    let backend = CkksBackend::new(DEFAULT_PARAMS);
    assert_eq!(backend.name(), "ckks");
    assert!(backend.measured_counters().is_empty());

    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![],
    };
    assert!(backend.check_graph_budget(&graph).is_ok());

    // Linear
    let linear = backend
        .build_op(&OpSpec::Linear {
            weights: vec![vec![1, 2]],
            bias: vec![0],
            weight_bits: 2,
        })
        .unwrap();
    assert_eq!(linear.output_bits(4), 7);
    assert_eq!(linear.internal_bits_n(&[4]), 7);
    assert!(!linear.cost(&[2]).is_empty());

    // Conv2d
    let conv = backend
        .build_op(&OpSpec::Conv2d {
            weights: vec![vec![1, 2, 3, 4]],
            bias: vec![0],
            weight_bits: 2,
            in_h: 2,
            in_w: 2,
            in_channels: 1,
            kernel_h: 2,
            kernel_w: 2,
            stride: 1,
            padding: 0,
        })
        .unwrap();
    assert_eq!(conv.output_bits(4), 8);
    assert!(!conv.cost(&[4]).is_empty());

    // Pool
    let pool = backend
        .build_op(&OpSpec::Pool {
            mode: "avg".to_string(),
            in_h: 2,
            in_w: 2,
            channels: 1,
            pool_h: 2,
            pool_w: 2,
            stride: 1,
            padding: 0,
        })
        .unwrap();
    assert_eq!(pool.output_bits(4), 6);
    let _ = pool.cost(&[4]);

    // Activation
    let act = backend
        .build_op(&OpSpec::Activation {
            lut: vec![0, 1, 2, 3],
            output_bits: 2,
        })
        .unwrap();
    assert_eq!(act.output_bits(2), 2);
    assert!(!act.cost(&[4]).is_empty());

    // Argmax
    let argmax = backend.build_op(&OpSpec::Argmax { threshold: 0 }).unwrap();
    assert_eq!(argmax.output_bits(16), 1);
    assert!(!argmax.cost(&[1]).is_empty());

    // Compare
    let cmp = backend
        .build_op(&OpSpec::Compare {
            indices: vec![0],
            thresholds: vec![5],
        })
        .unwrap();
    assert_eq!(cmp.output_bits(4), 1);
    assert!(!cmp.cost(&[1]).is_empty());

    // Add
    let add = backend.build_op(&OpSpec::Add {}).unwrap();
    assert_eq!(add.output_bits_n(&[4, 5]), 6);
    let _ = add.cost(&[2, 2]);

    // Concat
    let concat = backend
        .build_op(&OpSpec::Concat { sizes: vec![2, 2] })
        .unwrap();
    assert_eq!(concat.output_bits_n(&[4, 4]), 4);
    let _ = concat.cost(&[2, 2]);

    // Split
    let split = backend
        .build_op(&OpSpec::Split { sizes: vec![2, 2] })
        .unwrap();
    assert_eq!(split.output_bits_multi(&[4]), vec![4, 4]);
    let _ = split.cost(&[4]);

    // Requant
    let rq = backend
        .build_op(&OpSpec::Requant {
            shift: 7,
            mult: 1,
            round_bias: 64,
            clamp_lo: 0,
            zero_point: 0,
            out_bits: 2,
            clamp_lut: vec![0, 1, 2, 3],
            mults: vec![],
            shifts: vec![],
            round_biases: vec![],
            channel_size: None,
        })
        .unwrap();
    assert_eq!(rq.output_bits(10), 2);
    assert!(!rq.cost(&[4]).is_empty());
}

#[test]
fn test_ckks_backend_crypto_primitives() {
    let backend = CkksBackend::new(DEFAULT_PARAMS);
    let (ck, sk) = backend.keygen(8);

    let zero = backend.create_trivial_zero(&sk, 8);
    assert_eq!(zero.len(), 0);

    let in_a = vec![3i64];
    let in_b = vec![2i64];
    let ct_a = backend.encrypt(&ck, &in_a);
    let ct_b = backend.encrypt(&ck, &in_b);

    let sum = backend.add(&sk, &ct_a[0], &ct_b[0]);
    assert_eq!(backend.decrypt_vec(&ck, &[sum]), vec![5]);

    let scaled = backend.scalar_mul(&sk, &ct_a[0], 2);
    assert_eq!(backend.decrypt_vec(&ck, &[scaled]), vec![6]);

    let offset = backend.scalar_add(&sk, &ct_a[0], 4);
    assert_eq!(backend.decrypt_vec(&ck, &[offset]), vec![7]);

    let ge = backend.scalar_ge(&sk, &ct_a[0], 2);
    assert_eq!(backend.decrypt_label(&ck, &[ge]), 1);

    let smax = backend.scalar_max(&sk, &ct_a[0], 0);
    assert!(!backend.decrypt_vec(&ck, &[smax]).is_empty());

    let smin = backend.scalar_min(&sk, &ct_a[0], 0);
    assert!(!backend.decrypt_vec(&ck, &[smin]).is_empty());

    let shifted = backend.scalar_right_shift(&sk, &ct_a[0], 1);
    assert!(!backend.decrypt_vec(&ck, &[shifted]).is_empty());

    let lut_out = backend.apply_lut(&sk, &ct_a[0], &[1, 2, 3, 4], 8);
    assert!(!backend.decrypt_vec(&ck, &[lut_out]).is_empty());

    let ct_bytes = backend.serialize_cts(&ct_a).unwrap();
    let deserialized = backend.deserialize_cts(&ct_bytes).unwrap();
    assert_eq!(backend.decrypt_vec(&ck, &deserialized), in_a);

    let ck_bytes = backend.serialize_client_key(&ck, 8).unwrap();
    assert!(!ck_bytes.is_empty());
    let sk_bytes = backend.serialize_server_key(&sk, 8).unwrap();
    assert!(!sk_bytes.is_empty());
}
