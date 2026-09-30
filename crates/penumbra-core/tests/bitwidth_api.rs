//! Public API tests for penumbra-core bitwidth functions.

use penumbra_core::bitwidth::{
    ceil_log2, check_graph_bit_width_budget, magnitude_bits, op_spec_internal_bits_n,
    op_spec_output_bits_multi, op_spec_output_bits_multi_checked, op_spec_output_bits_n,
    op_spec_output_bits_n_checked, propagate_bit_widths, radix_capacity_bits,
};
use penumbra_core::ir::{Graph, Node, OpSpec, SCHEMA_VERSION};

#[test]
fn test_magnitude_and_ceil_log2() {
    assert_eq!(magnitude_bits(0), 0);
    assert_eq!(magnitude_bits(1), 1);
    assert_eq!(magnitude_bits(-1), 1);
    assert_eq!(magnitude_bits(7), 3);
    assert_eq!(magnitude_bits(-8), 4);
    assert_eq!(magnitude_bits(i64::MAX), 63);

    assert_eq!(ceil_log2(0), 0);
    assert_eq!(ceil_log2(1), 0);
    assert_eq!(ceil_log2(2), 1);
    assert_eq!(ceil_log2(3), 2);
    assert_eq!(ceil_log2(4), 2);
    assert_eq!(ceil_log2(8), 3);
    assert_eq!(ceil_log2(9), 4);

    assert_eq!(radix_capacity_bits(4), 8);
    assert_eq!(radix_capacity_bits(8), 16);
}

#[test]
fn test_op_spec_bit_growths() {
    let linear = OpSpec::Linear {
        weights: vec![vec![1, 2, 3, 4]],
        bias: vec![10],
        weight_bits: 3,
    };
    assert_eq!(op_spec_output_bits_n(&linear, &[4]), 11);

    let conv = OpSpec::Conv2d {
        weights: vec![vec![1, 2, 3, 4]],
        bias: vec![5],
        weight_bits: 3,
        in_h: 3,
        in_w: 3,
        in_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride: 1,
        padding: 0,
    };
    assert_eq!(op_spec_output_bits_n(&conv, &[4]), 11);

    let pool_avg = OpSpec::Pool {
        mode: "avg".to_string(),
        in_h: 4,
        in_w: 4,
        channels: 1,
        pool_h: 2,
        pool_w: 2,
        stride: 2,
        padding: 0,
    };
    assert_eq!(op_spec_output_bits_n(&pool_avg, &[4]), 6);

    let pool_max = OpSpec::Pool {
        mode: "max".to_string(),
        in_h: 4,
        in_w: 4,
        channels: 1,
        pool_h: 2,
        pool_w: 2,
        stride: 2,
        padding: 0,
    };
    assert_eq!(op_spec_output_bits_n(&pool_max, &[4]), 4);

    let act = OpSpec::Activation {
        lut: vec![0, 1, 2, 3],
        output_bits: 2,
    };
    assert_eq!(op_spec_output_bits_n(&act, &[4]), 2);

    let argmax = OpSpec::Argmax { threshold: 0 };
    assert_eq!(op_spec_output_bits_n(&argmax, &[16]), 1);

    let cmp = OpSpec::Compare {
        indices: vec![0],
        thresholds: vec![5],
    };
    assert_eq!(op_spec_output_bits_n(&cmp, &[8]), 1);

    let add = OpSpec::Add {};
    assert_eq!(op_spec_output_bits_n(&add, &[4, 6]), 7);

    let concat = OpSpec::Concat { sizes: vec![2, 3] };
    assert_eq!(op_spec_output_bits_n(&concat, &[4, 5]), 5);

    let split = OpSpec::Split { sizes: vec![2, 2] };
    assert_eq!(op_spec_output_bits_n(&split, &[4]), 4);
    assert_eq!(op_spec_output_bits_multi(&split, &[4]), vec![4, 4]);

    let rq = OpSpec::Requant {
        shift: 2,
        mult: 3,
        round_bias: 2,
        clamp_lo: -4,
        zero_point: 2,
        out_bits: 2,
        clamp_lut: vec![0, 1, 2, 3],
        mults: vec![],
        shifts: vec![],
        round_biases: vec![],
        channel_size: None,
    };
    assert_eq!(op_spec_output_bits_n(&rq, &[10]), 2);
    assert_eq!(op_spec_internal_bits_n(&rq, &[10]), 12);

    let rq_pc = OpSpec::Requant {
        shift: 0,
        mult: 1,
        round_bias: 0,
        clamp_lo: 0,
        zero_point: 0,
        out_bits: 2,
        clamp_lut: vec![0, 1, 2, 3],
        mults: vec![1, 3],
        shifts: vec![2, 3],
        round_biases: vec![2, 4],
        channel_size: Some(2),
    };
    assert_eq!(op_spec_output_bits_n(&rq_pc, &[10]), 2);
    assert_eq!(op_spec_internal_bits_n(&rq_pc, &[10]), 12);

    // Checked bit-width API: valid inputs
    assert_eq!(op_spec_output_bits_n_checked(&linear, &[4]).unwrap(), 11);
    assert_eq!(op_spec_output_bits_n_checked(&conv, &[4]).unwrap(), 11);
    assert_eq!(op_spec_output_bits_n_checked(&pool_avg, &[4]).unwrap(), 6);
    assert_eq!(op_spec_output_bits_n_checked(&pool_max, &[4]).unwrap(), 4);
    let pool_fallback = OpSpec::Pool {
        mode: "unknown_mode".to_string(),
        in_h: 4,
        in_w: 4,
        channels: 1,
        pool_h: 2,
        pool_w: 2,
        stride: 2,
        padding: 0,
    };
    assert_eq!(
        op_spec_output_bits_n_checked(&pool_fallback, &[4]).unwrap(),
        6
    );
    assert_eq!(op_spec_output_bits_n_checked(&act, &[4]).unwrap(), 2);
    assert_eq!(op_spec_output_bits_n_checked(&argmax, &[16]).unwrap(), 1);
    assert_eq!(op_spec_output_bits_n_checked(&cmp, &[8]).unwrap(), 1);
    assert_eq!(op_spec_output_bits_n_checked(&add, &[4, 6]).unwrap(), 7);
    assert_eq!(op_spec_output_bits_n_checked(&concat, &[4, 5]).unwrap(), 5);
    assert_eq!(op_spec_output_bits_n_checked(&split, &[4]).unwrap(), 4);
    assert_eq!(
        op_spec_output_bits_multi_checked(&split, &[4]).unwrap(),
        vec![4, 4]
    );
    assert_eq!(op_spec_output_bits_n_checked(&rq, &[10]).unwrap(), 2);
    assert_eq!(op_spec_output_bits_n_checked(&rq_pc, &[10]).unwrap(), 2);

    // Checked bit-width API: input length mismatch error cases
    assert!(op_spec_output_bits_n_checked(&linear, &[4, 4]).is_err());
    assert!(op_spec_output_bits_n_checked(&linear, &[]).is_err());
    assert!(op_spec_output_bits_n_checked(&conv, &[4, 4]).is_err());
    assert!(op_spec_output_bits_n_checked(&pool_avg, &[]).is_err());
    assert!(op_spec_output_bits_n_checked(&pool_max, &[4, 4]).is_err());
    assert!(op_spec_output_bits_n_checked(&act, &[]).is_err());
    assert!(op_spec_output_bits_n_checked(&argmax, &[16, 16]).is_err());
    assert!(op_spec_output_bits_n_checked(&cmp, &[]).is_err());
    assert!(op_spec_output_bits_n_checked(&add, &[4]).is_err());
    assert!(op_spec_output_bits_n_checked(&add, &[4, 6, 8]).is_err());
    assert!(op_spec_output_bits_n_checked(&concat, &[4]).is_err());
    assert!(op_spec_output_bits_n_checked(&split, &[4, 4]).is_err());
    assert!(op_spec_output_bits_multi_checked(&split, &[4, 4]).is_err());
    assert!(op_spec_output_bits_n_checked(&rq, &[]).is_err());
}
#[test]
fn test_check_graph_bit_width_budget_errors() {
    let bad_graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 2,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "fc".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Linear {
                weights: vec![vec![1, 2, 3, 4]],
                bias: vec![0],
                weight_bits: 3,
            },
        }],
    };
    let err = check_graph_bit_width_budget(&bad_graph).unwrap_err();
    assert!(err.contains("bit-width budget exceeded at node 'fc'"));

    let bad_rq_graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 8,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "rq".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Requant {
                shift: 2,
                mult: 3,
                round_bias: 2,
                clamp_lo: 0,
                zero_point: 0,
                out_bits: 2,
                clamp_lut: vec![0, 1, 2, 3],
                mults: vec![],
                shifts: vec![],
                round_biases: vec![],
                channel_size: None,
            },
        }],
    };
    let err = check_graph_bit_width_budget(&bad_rq_graph).unwrap_err();
    assert!(err.contains("transient bits"));

    let unlinked_graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "fc".to_string(),
            inputs: vec!["nonexistent".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Linear {
                weights: vec![vec![1]],
                bias: vec![0],
                weight_bits: 1,
            },
        }],
    };
    assert!(propagate_bit_widths(&unlinked_graph).is_err());
}

#[test]
fn test_propagate_bit_widths_validation_errors() {
    // Node with empty inputs
    let empty_inputs_graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "bad_node".to_string(),
            inputs: vec![],
            outputs: vec!["y".to_string()],
            op: OpSpec::Add {},
        }],
    };
    let err = propagate_bit_widths(&empty_inputs_graph).unwrap_err();
    assert!(err.contains("must have at least one input and at least one output"));

    // Node with empty outputs
    let empty_outputs_graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "bad_node".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec![],
            op: OpSpec::Activation {
                lut: vec![0, 1],
                output_bits: 1,
            },
        }],
    };
    let err = propagate_bit_widths(&empty_outputs_graph).unwrap_err();
    assert!(err.contains("must have at least one input and at least one output"));

    // Node declaring wrong number of outputs (Split with 2 sizes but 1 output name)
    let wrong_outputs_graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "split_node".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Split { sizes: vec![2, 2] },
        }],
    };
    let err = propagate_bit_widths(&wrong_outputs_graph).unwrap_err();
    assert!(err.contains("declares 1 output(s) but produces 2"));

    // Duplicate tensor output name
    let dup_output_graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![
            Node {
                name: "node1".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["t".to_string()],
                op: OpSpec::Activation {
                    lut: vec![0, 1],
                    output_bits: 1,
                },
            },
            Node {
                name: "node2".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["t".to_string()],
                op: OpSpec::Activation {
                    lut: vec![0, 1],
                    output_bits: 1,
                },
            },
        ],
    };
    let err = propagate_bit_widths(&dup_output_graph).unwrap_err();
    assert!(err.contains("already exists — tensor names must be unique"));
}
