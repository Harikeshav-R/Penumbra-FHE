//! Backend-neutral, semantics-preserving graph-fusion pass (Layer 2).
//!
//! Replaces adjacent single-block lookup-table nodes (`Requant` and `Activation`)
//! with a single composed node, cutting programmable bootstraps under TFHE and
//! reducing polynomial evaluations (and multiplicative depth) under CKKS.
//!
//! Note on node reordering:
//! The only op reordering candidate in current model architectures would be moving a
//! `Requant` past `Pool(avg)` (e.g. 32 requants down to 8 in `phase4_cnn`). However,
//! `Requant` is a fused ReLU + clamp, neither of which commutes with summation.
//! Therefore, reordering `Requant` and `Pool(avg)` is not semantics-preserving and is
//! deliberately not implemented.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use crate::bitwidth::{magnitude_bits, MESSAGE_BITS};
use crate::ir::{Graph, Node, OpSpec};

/// Rewrite `graph` into a semantics-identical graph with fewer bootstrap-bearing nodes.
///
/// Every rule composes single-block integer lookup tables, so the rewrite is exact under
/// TFHE and reduces polynomial evaluations (and therefore depth) under CKKS. Returns
/// `Cow::Borrowed` when no rule fires, so the common path allocates nothing.
pub fn optimize_graph(graph: &Graph) -> Result<Cow<'_, Graph>, String> {
    let mut current: Option<Graph> = None;
    let max_iters = graph.nodes.len();
    let mut iters = 0;

    loop {
        let g = current.as_ref().unwrap_or(graph);
        match step_fuse(g)? {
            Some(rewritten) => {
                iters += 1;
                if iters > max_iters {
                    return Err(format!(
                        "graph optimization loop exceeded iteration cap ({max_iters}) without reaching a fixed point"
                    ));
                }
                current = Some(rewritten);
            }
            None => break,
        }
    }

    if let Some(rewritten) = current {
        crate::bitwidth::propagate_bit_widths(&rewritten)?;
        Ok(Cow::Owned(rewritten))
    } else {
        Ok(Cow::Borrowed(graph))
    }
}

/// Perform at most one fusion step on `g`. Returns `Ok(Some(new_graph))` if a rule fired,
/// or `Ok(None)` if no rule fired.
fn step_fuse(g: &Graph) -> Result<Option<Graph>, String> {
    let mut tensor_consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, node) in g.nodes.iter().enumerate() {
        for input in &node.inputs {
            tensor_consumers
                .entry(input.as_str())
                .or_default()
                .push(idx);
        }
    }
    let graph_outputs: HashSet<&str> = g.outputs.iter().map(String::as_str).collect();

    for p in 0..g.nodes.len() {
        let producer = &g.nodes[p];
        if producer.outputs.len() != 1 {
            continue;
        }
        let intermediate = &producer.outputs[0];
        if graph_outputs.contains(intermediate.as_str()) {
            continue;
        }
        let consumers = match tensor_consumers.get(intermediate.as_str()) {
            Some(c) if c.len() == 1 => c,
            _ => continue,
        };
        let c = consumers[0];
        if c <= p {
            continue;
        }
        let consumer = &g.nodes[c];
        if consumer.inputs.len() != 1 || consumer.outputs.len() != 1 {
            continue;
        }

        // Rule R1: Requant -> Activation => one Requant
        if let (
            OpSpec::Requant {
                shift,
                mult,
                round_bias,
                out_bits,
                clamp_lut,
                mults,
                shifts,
                round_biases,
                channel_size,
            },
            OpSpec::Activation { lut: act_lut, .. },
        ) = (&producer.op, &consumer.op)
        {
            let mut composed = Vec::with_capacity(clamp_lut.len());
            let mut valid = true;
            for &v in clamp_lut {
                let idx = v as usize;
                if idx >= act_lut.len() {
                    valid = false;
                    break;
                }
                let mapped = act_lut[idx];
                if mapped >= (1u64 << MESSAGE_BITS) {
                    valid = false;
                    break;
                }
                composed.push(mapped);
            }
            if valid {
                let max_val = composed.iter().copied().max().unwrap_or(0);
                if magnitude_bits(max_val as i64) <= *out_bits {
                    let fused_op = OpSpec::Requant {
                        shift: *shift,
                        mult: *mult,
                        round_bias: *round_bias,
                        out_bits: *out_bits,
                        clamp_lut: composed,
                        mults: mults.clone(),
                        shifts: shifts.clone(),
                        round_biases: round_biases.clone(),
                        channel_size: *channel_size,
                    };
                    let fused_node = Node {
                        name: producer.name.clone(),
                        inputs: producer.inputs.clone(),
                        outputs: consumer.outputs.clone(),
                        op: fused_op,
                    };
                    let mut new_nodes = g.nodes.clone();
                    new_nodes[p] = fused_node;
                    new_nodes.remove(c);
                    return Ok(Some(Graph {
                        nodes: new_nodes,
                        ..g.clone()
                    }));
                }
            }
        }

        // Rule R2: Activation -> Activation => one Activation
        if let (
            OpSpec::Activation { lut: first_lut, .. },
            OpSpec::Activation {
                lut: second_lut,
                output_bits: second_out_bits,
            },
        ) = (&producer.op, &consumer.op)
        {
            let mut composed = Vec::with_capacity(first_lut.len());
            let mut valid = true;
            for &v in first_lut {
                if (v as usize) >= second_lut.len() {
                    valid = false;
                    break;
                }
                composed.push(second_lut[v as usize]);
            }
            if valid {
                let fused_op = OpSpec::Activation {
                    lut: composed,
                    output_bits: *second_out_bits,
                };
                let fused_node = Node {
                    name: producer.name.clone(),
                    inputs: producer.inputs.clone(),
                    outputs: consumer.outputs.clone(),
                    op: fused_op,
                };
                let mut new_nodes = g.nodes.clone();
                new_nodes[p] = fused_node;
                new_nodes.remove(c);
                return Ok(Some(Graph {
                    nodes: new_nodes,
                    ..g.clone()
                }));
            }
        }

        // Rule R3: Requant -> Requant => one Requant
        if let (
            OpSpec::Requant {
                shift: shift1,
                mult: mult1,
                round_bias: round_bias1,
                out_bits: out_bits1,
                clamp_lut: clamp_lut1,
                mults: mults1,
                shifts: shifts1,
                round_biases: round_biases1,
                channel_size: channel_size1,
            },
            OpSpec::Requant {
                shift: shift2,
                mult: mult2,
                round_bias: round_bias2,
                out_bits: out_bits2,
                clamp_lut: clamp_lut2,
                mults: mults2,
                shifts: shifts2,
                round_biases: round_biases2,
                ..
            },
        ) = (&producer.op, &consumer.op)
        {
            if mults2.is_empty() && shifts2.is_empty() && round_biases2.is_empty() {
                let max_sat = (1u64 << *out_bits2) - 1;
                let mut composed = Vec::with_capacity(clamp_lut1.len());
                let mut valid = true;
                for &u in clamp_lut1 {
                    let val = (u * *mult2 + *round_bias2) >> *shift2;
                    let sat = val.min(max_sat);
                    let sat_idx = sat as usize;
                    if sat_idx >= clamp_lut2.len() {
                        valid = false;
                        break;
                    }
                    let res = clamp_lut2[sat_idx];
                    if res >= (1u64 << MESSAGE_BITS) {
                        valid = false;
                        break;
                    }
                    composed.push(res);
                }
                if valid {
                    let max_val = composed.iter().copied().max().unwrap_or(0);
                    if magnitude_bits(max_val as i64) <= *out_bits1 {
                        let fused_op = OpSpec::Requant {
                            shift: *shift1,
                            mult: *mult1,
                            round_bias: *round_bias1,
                            out_bits: *out_bits1,
                            clamp_lut: composed,
                            mults: mults1.clone(),
                            shifts: shifts1.clone(),
                            round_biases: round_biases1.clone(),
                            channel_size: *channel_size1,
                        };
                        let fused_node = Node {
                            name: producer.name.clone(),
                            inputs: producer.inputs.clone(),
                            outputs: consumer.outputs.clone(),
                            op: fused_op,
                        };
                        let mut new_nodes = g.nodes.clone();
                        new_nodes[p] = fused_node;
                        new_nodes.remove(c);
                        return Ok(Some(Graph {
                            nodes: new_nodes,
                            ..g.clone()
                        }));
                    }
                }
            }
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::SCHEMA_VERSION;
    use std::path::PathBuf;

    #[test]
    fn test_rule_r1_requant_activation_fusion() {
        let graph = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 4,
            input_bits: 4,
            inputs: vec!["x".to_string()],
            outputs: vec!["z".to_string()],
            nodes: vec![
                Node {
                    name: "rq".to_string(),
                    inputs: vec!["x".to_string()],
                    outputs: vec!["y".to_string()],
                    op: OpSpec::Requant {
                        shift: 2,
                        mult: 1,
                        round_bias: 2,
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
                    inputs: vec!["y".to_string()],
                    outputs: vec!["z".to_string()],
                    op: OpSpec::Activation {
                        lut: vec![0, 2, 1, 3],
                        output_bits: 2,
                    },
                },
            ],
        };

        let opt = optimize_graph(&graph).expect("optimization succeeds");
        match opt {
            Cow::Owned(g) => {
                assert_eq!(g.nodes.len(), 1, "two nodes fused into one");
                let node = &g.nodes[0];
                assert_eq!(node.name, "rq", "retains producer name");
                assert_eq!(node.inputs, vec!["x"]);
                assert_eq!(node.outputs, vec!["z"], "downstream tensor name preserved");
                if let OpSpec::Requant {
                    clamp_lut,
                    shift,
                    mult,
                    round_bias,
                    out_bits,
                    ..
                } = &node.op
                {
                    assert_eq!(*shift, 2);
                    assert_eq!(*mult, 1);
                    assert_eq!(*round_bias, 2);
                    assert_eq!(*out_bits, 2);
                    assert_eq!(*clamp_lut, vec![0, 2, 1, 3]);
                } else {
                    panic!("fused node should be Requant");
                }
            }
            Cow::Borrowed(_) => panic!("expected fusion to rewrite graph"),
        }
    }

    #[test]
    fn test_rule_r1_does_not_fire_when_multi_consumer_or_graph_output() {
        // Multi-consumer case
        let graph_multi = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 4,
            input_bits: 4,
            inputs: vec!["x".to_string()],
            outputs: vec!["z1".to_string(), "z2".to_string()],
            nodes: vec![
                Node {
                    name: "rq".to_string(),
                    inputs: vec!["x".to_string()],
                    outputs: vec!["y".to_string()],
                    op: OpSpec::Requant {
                        shift: 2,
                        mult: 1,
                        round_bias: 2,
                        out_bits: 2,
                        clamp_lut: vec![0, 1, 2, 3],
                        mults: vec![],
                        shifts: vec![],
                        round_biases: vec![],
                        channel_size: None,
                    },
                },
                Node {
                    name: "act1".to_string(),
                    inputs: vec!["y".to_string()],
                    outputs: vec!["z1".to_string()],
                    op: OpSpec::Activation {
                        lut: vec![0, 2, 1, 3],
                        output_bits: 2,
                    },
                },
                Node {
                    name: "act2".to_string(),
                    inputs: vec!["y".to_string()],
                    outputs: vec!["z2".to_string()],
                    op: OpSpec::Activation {
                        lut: vec![0, 1, 2, 3],
                        output_bits: 2,
                    },
                },
            ],
        };
        let opt_multi = optimize_graph(&graph_multi).expect("success");
        assert!(matches!(opt_multi, Cow::Borrowed(_)));

        // Graph output case
        let graph_out = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 4,
            input_bits: 4,
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string(), "z".to_string()],
            nodes: vec![
                Node {
                    name: "rq".to_string(),
                    inputs: vec!["x".to_string()],
                    outputs: vec!["y".to_string()],
                    op: OpSpec::Requant {
                        shift: 2,
                        mult: 1,
                        round_bias: 2,
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
                    inputs: vec!["y".to_string()],
                    outputs: vec!["z".to_string()],
                    op: OpSpec::Activation {
                        lut: vec![0, 2, 1, 3],
                        output_bits: 2,
                    },
                },
            ],
        };
        let opt_out = optimize_graph(&graph_out).expect("success");
        assert!(matches!(opt_out, Cow::Borrowed(_)));
    }

    #[test]
    fn test_rule_r2_activation_activation_fusion() {
        let graph = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 4,
            input_bits: 2,
            inputs: vec!["x".to_string()],
            outputs: vec!["z".to_string()],
            nodes: vec![
                Node {
                    name: "act1".to_string(),
                    inputs: vec!["x".to_string()],
                    outputs: vec!["y".to_string()],
                    op: OpSpec::Activation {
                        lut: vec![1, 2, 3, 0],
                        output_bits: 2,
                    },
                },
                Node {
                    name: "act2".to_string(),
                    inputs: vec!["y".to_string()],
                    outputs: vec!["z".to_string()],
                    op: OpSpec::Activation {
                        lut: vec![3, 2, 1, 0],
                        output_bits: 2,
                    },
                },
            ],
        };

        let opt = optimize_graph(&graph).expect("success");
        match opt {
            Cow::Owned(g) => {
                assert_eq!(g.nodes.len(), 1);
                let node = &g.nodes[0];
                assert_eq!(node.name, "act1");
                assert_eq!(node.outputs, vec!["z"]);
                if let OpSpec::Activation { lut, output_bits } = &node.op {
                    assert_eq!(*lut, vec![2, 1, 0, 3]);
                    assert_eq!(*output_bits, 2);
                } else {
                    panic!("expected Activation op");
                }
            }
            Cow::Borrowed(_) => panic!("expected fusion"),
        }
    }

    #[test]
    fn test_rule_r3_requant_requant_fusion() {
        let graph = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 4,
            input_bits: 4,
            inputs: vec!["x".to_string()],
            outputs: vec!["z".to_string()],
            nodes: vec![
                Node {
                    name: "rq1".to_string(),
                    inputs: vec!["x".to_string()],
                    outputs: vec!["y".to_string()],
                    op: OpSpec::Requant {
                        shift: 0,
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
                    name: "rq2".to_string(),
                    inputs: vec!["y".to_string()],
                    outputs: vec!["z".to_string()],
                    op: OpSpec::Requant {
                        shift: 1,
                        mult: 1,
                        round_bias: 0,
                        out_bits: 2,
                        clamp_lut: vec![0, 2, 1, 3],
                        mults: vec![],
                        shifts: vec![],
                        round_biases: vec![],
                        channel_size: None,
                    },
                },
            ],
        };

        let opt = optimize_graph(&graph).expect("success");
        match opt {
            Cow::Owned(g) => {
                assert_eq!(g.nodes.len(), 1);
                let node = &g.nodes[0];
                assert_eq!(node.name, "rq1");
                assert_eq!(node.outputs, vec!["z"]);
                if let OpSpec::Requant { clamp_lut, .. } = &node.op {
                    // v=0: u=0 -> (0*1)>>1 = 0 -> lut2[0] = 0
                    // v=1: u=1 -> (1*1)>>1 = 0 -> lut2[0] = 0
                    // v=2: u=2 -> (2*1)>>1 = 1 -> lut2[1] = 2
                    // v=3: u=3 -> (3*1)>>1 = 1 -> lut2[1] = 2
                    assert_eq!(*clamp_lut, vec![0, 0, 2, 2]);
                } else {
                    panic!("expected Requant op");
                }
            }
            Cow::Borrowed(_) => panic!("expected fusion"),
        }

        // Does not fire if second has per-channel overlay
        let mut graph_per_channel = graph.clone();
        if let OpSpec::Requant { mults, .. } = &mut graph_per_channel.nodes[1].op {
            *mults = vec![1];
        }
        let opt_pc = optimize_graph(&graph_per_channel).expect("success");
        assert!(matches!(opt_pc, Cow::Borrowed(_)));
    }

    #[test]
    fn test_fixed_point_chain_collapse() {
        // Requant -> Requant -> Activation collapses to a single Requant
        let graph = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 4,
            input_bits: 4,
            inputs: vec!["x".to_string()],
            outputs: vec!["out".to_string()],
            nodes: vec![
                Node {
                    name: "rq1".to_string(),
                    inputs: vec!["x".to_string()],
                    outputs: vec!["t1".to_string()],
                    op: OpSpec::Requant {
                        shift: 0,
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
                    name: "rq2".to_string(),
                    inputs: vec!["t1".to_string()],
                    outputs: vec!["t2".to_string()],
                    op: OpSpec::Requant {
                        shift: 0,
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
                    inputs: vec!["t2".to_string()],
                    outputs: vec!["out".to_string()],
                    op: OpSpec::Activation {
                        lut: vec![3, 2, 1, 0],
                        output_bits: 2,
                    },
                },
            ],
        };

        let opt = optimize_graph(&graph).expect("success");
        match opt {
            Cow::Owned(g) => {
                assert_eq!(g.nodes.len(), 1, "3 nodes collapse to 1 node");
                assert_eq!(g.nodes[0].name, "rq1");
                assert_eq!(g.nodes[0].outputs, vec!["out"]);
                if let OpSpec::Requant { clamp_lut, .. } = &g.nodes[0].op {
                    assert_eq!(*clamp_lut, vec![3, 2, 1, 0]);
                } else {
                    panic!("expected Requant op");
                }
            }
            Cow::Borrowed(_) => panic!("expected fusion"),
        }
    }

    #[test]
    fn test_phase4_cnn_fixture_returns_borrowed() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let fixture_path = manifest_dir.join("../../examples/mnist/phase4_cnn_fixture.json");
        let text = std::fs::read_to_string(&fixture_path).expect("read phase4 fixture");
        let v: serde_json::Value = serde_json::from_str(&text).expect("parse JSON");
        let graph = Graph::from_json(&v["graph"].to_string()).expect("valid Graph");

        let opt = optimize_graph(&graph).expect("optimize_graph succeeds");
        assert!(
            matches!(opt, Cow::Borrowed(_)),
            "phase4_cnn should return Cow::Borrowed (Finding 3: no fusion sites exist today)"
        );
    }
}
