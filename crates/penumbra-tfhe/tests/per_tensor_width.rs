use std::collections::HashMap;

use penumbra_core::ir::{Graph, Node, OpSpec, SCHEMA_VERSION};
use penumbra_tfhe::ops::EvalCtx;
use penumbra_tfhe::{deserialize_cts, encrypt, keygen, serialize_cts};
use tfhe::integer::IntegerCiphertext;

#[test]
fn test_per_tensor_radix_widths_and_wire_format() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 8,
        input_bits: 2,
        inputs: vec!["x".to_string()],
        outputs: vec!["fc".to_string(), "rq".to_string(), "out".to_string()],
        nodes: vec![
            Node {
                name: "fc".to_string(),
                op: OpSpec::Linear {
                    weights: vec![vec![1, 2, 3, 1], vec![-1, 1, 2, -2]],
                    bias: vec![1, -3],
                    weight_bits: 3,
                },
                inputs: vec!["x".to_string()],
                outputs: vec!["fc".to_string()],
            },
            Node {
                name: "rq".to_string(),
                op: OpSpec::Requant {
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
                },
                inputs: vec!["fc".to_string()],
                outputs: vec!["rq".to_string()],
            },
            Node {
                name: "out".to_string(),
                op: OpSpec::Linear {
                    weights: vec![vec![1, 1]],
                    bias: vec![0],
                    weight_bits: 2,
                },
                inputs: vec!["rq".to_string()],
                outputs: vec!["out".to_string()],
            },
        ],
    };

    let (ck, sk) = keygen(8);
    let x_cts = encrypt(&ck, &[0, 1, 2, 3]);
    assert_eq!(x_cts[0].blocks().len(), 8);

    let mut env = HashMap::new();
    env.insert("x".to_string(), x_cts);

    let ctx = EvalCtx::new(&sk, 8);
    let results =
        penumbra_tfhe::evaluate_graph(&ctx, &graph, env).expect("evaluate_graph should succeed");

    // fc decrypts [12, -4] with 5 blocks each
    let fc = &results["fc"];
    assert_eq!(fc.len(), 2);
    assert_eq!(fc[0].blocks().len(), 5);
    assert_eq!(fc[1].blocks().len(), 5);
    assert_eq!(ck.decrypt_signed::<i64>(&fc[0]), 12);
    assert_eq!(ck.decrypt_signed::<i64>(&fc[1]), -4);

    // rq decrypts [3, 0] with 2 blocks each (the narrow tensor, not 8)
    let rq = &results["rq"];
    assert_eq!(rq.len(), 2);
    assert_eq!(rq[0].blocks().len(), 2);
    assert_eq!(rq[1].blocks().len(), 2);
    assert_eq!(ck.decrypt_signed::<i64>(&rq[0]), 3);
    assert_eq!(ck.decrypt_signed::<i64>(&rq[1]), 0);

    // out decrypts [3] with 4 blocks
    let out = &results["out"];
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].blocks().len(), 4);
    assert_eq!(ck.decrypt_signed::<i64>(&out[0]), 3);

    // serialize_cts(&results["rq"]) -> deserialize_cts -> still 2 blocks and decrypts [3, 0]
    let wire_bytes = serialize_cts(rq).expect("serialization must succeed");
    let deserialized = deserialize_cts(&wire_bytes).expect("deserialization must succeed");
    assert_eq!(deserialized.len(), 2);
    assert_eq!(deserialized[0].blocks().len(), 2);
    assert_eq!(deserialized[1].blocks().len(), 2);
    assert_eq!(ck.decrypt_signed::<i64>(&deserialized[0]), 3);
    assert_eq!(ck.decrypt_signed::<i64>(&deserialized[1]), 0);
}
