//! Tests for TfheBackend's implementation of the Backend trait.

use penumbra_core::backend::Backend;
use penumbra_core::ir::{Graph, OpSpec, SCHEMA_VERSION};
use penumbra_tfhe::backend::TfheBackend;
use penumbra_tfhe::keys::TfheProfile;

#[test]
fn test_tfhe_backend_metadata_and_build_op() {
    let backend = TfheBackend::new(TfheProfile::Classic);
    assert_eq!(backend.name(), "tfhe");
    assert!(!backend.measured_counters().is_empty());

    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![],
    };
    assert!(backend.check_graph_budget(&graph).is_ok());

    // Exercise build_op and op methods for each op
    let linear = backend
        .build_op(&OpSpec::Linear {
            weights: vec![vec![1, 2]],
            bias: vec![0],
            weight_bits: 2,
        })
        .unwrap();
    assert_eq!(linear.output_bits(4), 9);
    assert_eq!(linear.internal_bits_n(&[4]), 9);
    assert!(!linear.cost(&[2]).is_empty());

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
    assert_eq!(conv.output_bits(4), 10);
    assert!(!conv.cost(&[4]).is_empty());

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

    let act = backend
        .build_op(&OpSpec::Activation {
            lut: vec![0, 1, 2, 3],
            output_bits: 2,
        })
        .unwrap();
    assert_eq!(act.output_bits(2), 2);
    assert!(!act.cost(&[4]).is_empty());

    let argmax = backend.build_op(&OpSpec::Argmax { threshold: 0 }).unwrap();
    assert_eq!(argmax.output_bits(16), 1);
    assert!(!argmax.cost(&[1]).is_empty());

    let cmp = backend
        .build_op(&OpSpec::Compare {
            indices: vec![0],
            thresholds: vec![5],
        })
        .unwrap();
    assert_eq!(cmp.output_bits(4), 1);
    assert!(!cmp.cost(&[1]).is_empty());

    let add = backend.build_op(&OpSpec::Add {}).unwrap();
    assert_eq!(add.output_bits_n(&[4, 5]), 6);
    assert!(!add.cost(&[2, 2]).is_empty());

    let concat = backend
        .build_op(&OpSpec::Concat { sizes: vec![2, 2] })
        .unwrap();
    assert_eq!(concat.output_bits_n(&[4, 4]), 4);
    assert!(concat.cost(&[2, 2]).is_empty());

    let split = backend
        .build_op(&OpSpec::Split { sizes: vec![2, 2] })
        .unwrap();
    assert_eq!(split.output_bits_multi(&[4]), vec![4, 4]);
    assert!(split.cost(&[4]).is_empty());

    let rq = backend
        .build_op(&OpSpec::Requant {
            shift: 2,
            mult: 1,
            round_bias: 0,
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
fn test_tfhe_backend_crypto_primitives() {
    let backend = TfheBackend::new(TfheProfile::Classic);
    let nb = 2;
    let (ck, sk) = backend.keygen(nb);

    let zero = backend.create_trivial_zero(&sk, nb);
    assert_eq!(backend.decrypt_vec(&ck, &[zero]), vec![0]);

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

    let max_val = backend.max(&sk, &ct_a[0], &ct_b[0]);
    assert_eq!(backend.decrypt_vec(&ck, &[max_val]), vec![3]);

    let smax = backend.scalar_max(&sk, &ct_a[0], 5);
    assert_eq!(backend.decrypt_vec(&ck, &[smax]), vec![5]);

    let smin = backend.scalar_min(&sk, &ct_a[0], 1);
    assert_eq!(backend.decrypt_vec(&ck, &[smin]), vec![1]);

    let shifted = backend.scalar_right_shift(&sk, &ct_a[0], 1);
    assert_eq!(backend.decrypt_vec(&ck, &[shifted]), vec![1]);

    let lut_out = backend.apply_lut(&sk, &ct_a[0], &[1, 2, 3, 4], nb);
    assert_eq!(backend.decrypt_vec(&ck, &[lut_out]), vec![4]);

    let ct_bytes = backend.serialize_cts(&ct_a).unwrap();
    let deserialized = backend.deserialize_cts(&ct_bytes).unwrap();
    assert_eq!(backend.decrypt_vec(&ck, &deserialized), in_a);

    let ck_bytes = backend.serialize_client_key(&ck, nb).unwrap();
    assert!(!ck_bytes.is_empty());
    let sk_bytes = backend.serialize_server_key(&sk, nb).unwrap();
    assert!(!sk_bytes.is_empty());
}
