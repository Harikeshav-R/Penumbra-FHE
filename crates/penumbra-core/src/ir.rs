//! Intermediate Representation (IR) — the serializable op graph (Layer 2).
//!
//! The IR is the product's backbone (`PROJECT.md` §7): a directed graph of op nodes that
//! the Python front end emits (JSON to start) and this runtime consumes *without
//! per-use-case changes*. A new use case is a new graph, never a backend edit
//! (`AGENTS.md` §1.2).
//!
//! Mirrors `python/penumbra/ir.py` — the two definitions **must stay in lockstep**
//! (`AGENTS.md` §5). Any IR change updates both language sides, bumps [`SCHEMA_VERSION`],
//! and updates the cross-language conformance test + `docs/IR-SPEC.md` in the **same
//! change**. A schema-version bump is a breaking change (`AGENTS.md` §8).

use serde::{Deserialize, Serialize};

use crate::bitwidth::MESSAGE_BITS;
use crate::ops::OpSummary;

/// IR wire-format version. Hardcoded identically in `python/penumbra/ir.py`; a mismatch is
/// a breaking change caught loudly at load time (`AGENTS.md` §5, §8).
pub const SCHEMA_VERSION: &str = "0.6.0";

/// Serde default for `Requant.mult`: `1` makes the rescale a pure power-of-two shift.
fn default_requant_mult() -> u64 {
    1
}

/// The root IR object: a directed graph of op nodes in a valid topological order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Graph {
    pub schema_version: String,
    pub num_blocks: usize,
    pub input_bits: usize,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub nodes: Vec<Node>,
}

/// One op in the graph: a name, the tensor names it reads/writes, and its op payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub op: OpSpec,
}

/// Spatial pooling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolMode {
    Avg,
    Max,
}

/// The op payload — the serializable mirror of the runtime ops.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op_type")]
pub enum OpSpec {
    Linear {
        weights: Vec<Vec<i64>>,
        bias: Vec<i64>,
        weight_bits: usize,
    },
    Conv2d {
        weights: Vec<Vec<i64>>,
        bias: Vec<i64>,
        weight_bits: usize,
        in_h: usize,
        in_w: usize,
        in_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    },
    Activation {
        lut: Vec<u64>,
        output_bits: usize,
    },
    Argmax {
        threshold: i64,
    },
    Requant {
        shift: u32,
        #[serde(default = "default_requant_mult")]
        mult: u64,
        #[serde(default)]
        round_bias: u64,
        out_bits: usize,
        clamp_lut: Vec<u64>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        mults: Vec<u64>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        shifts: Vec<u32>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        round_biases: Vec<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        channel_size: Option<usize>,
    },
    Pool {
        mode: String,
        in_h: usize,
        in_w: usize,
        channels: usize,
        pool_h: usize,
        pool_w: usize,
        stride: usize,
    },
    Add {},
}

impl Graph {
    /// Deserialize an IR graph from JSON, validating the schema version loudly.
    pub fn from_json(s: &str) -> Result<Graph, String> {
        let graph: Graph = serde_json::from_str(s).map_err(|e| format!("IR parse error: {e}"))?;
        if graph.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "IR schema-version mismatch: file declares \"{}\" but this runtime expects \
                 \"{SCHEMA_VERSION}\". A schema-version change is breaking (AGENTS.md §5, §8); \
                 regenerate the IR from a matching front end.",
                graph.schema_version
            ));
        }
        Ok(graph)
    }

    /// Serialize this IR graph to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("Graph serialization to JSON is infallible")
    }
}

impl OpSpec {
    pub fn op_type(&self) -> &'static str {
        match self {
            OpSpec::Linear { .. } => "Linear",
            OpSpec::Conv2d { .. } => "Conv2d",
            OpSpec::Activation { .. } => "Activation",
            OpSpec::Argmax { .. } => "Argmax",
            OpSpec::Requant { .. } => "Requant",
            OpSpec::Pool { .. } => "Pool",
            OpSpec::Add { .. } => "Add",
        }
    }

    /// Validate the op specification fields at load time, failing loudly on invalid configuration.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            OpSpec::Linear { weights, bias, .. } => {
                if weights.is_empty() {
                    return Err("Linear op has no weight rows".to_string());
                }
                if weights.len() != bias.len() {
                    return Err(format!(
                        "Linear op has {} weight rows but {} biases; need one bias per row",
                        weights.len(),
                        bias.len()
                    ));
                }
                let width = weights[0].len();
                if let Some((i, row)) = weights.iter().enumerate().find(|(_, r)| r.len() != width) {
                    return Err(format!(
                        "Linear weight row {i} has width {} but row 0 has width {width}; all \
                         rows must match the input length",
                        row.len()
                    ));
                }
            }
            OpSpec::Conv2d {
                weights,
                bias,
                in_h,
                in_w,
                in_channels,
                kernel_h,
                kernel_w,
                stride,
                padding,
                ..
            } => {
                if weights.is_empty() {
                    return Err("Conv2d op has no output channels (empty weights)".to_string());
                }
                if weights.len() != bias.len() {
                    return Err(format!(
                        "Conv2d has {} kernels but {} biases; need one bias per output channel",
                        weights.len(),
                        bias.len()
                    ));
                }
                if *in_h == 0 || *in_w == 0 || *in_channels == 0 {
                    return Err("Conv2d in_h/in_w/in_channels must be positive".to_string());
                }
                if *kernel_h == 0 || *kernel_w == 0 || *stride == 0 {
                    return Err("Conv2d kernel_h/kernel_w/stride must be positive".to_string());
                }
                let fan_in = in_channels * kernel_h * kernel_w;
                if let Some((i, row)) = weights.iter().enumerate().find(|(_, r)| r.len() != fan_in)
                {
                    return Err(format!(
                        "Conv2d kernel row {i} has width {} but in_channels*kernel_h*kernel_w \
                         = {fan_in}; every kernel must match the fan-in",
                        row.len()
                    ));
                }
                if kernel_h > &(in_h + 2 * padding) || kernel_w > &(in_w + 2 * padding) {
                    return Err(format!(
                        "Conv2d kernel ({kernel_h}x{kernel_w}) does not fit the padded input \
                         ({}x{})",
                        in_h + 2 * padding,
                        in_w + 2 * padding
                    ));
                }
            }
            OpSpec::Activation { .. } => {}
            OpSpec::Argmax { .. } => {}
            OpSpec::Requant {
                mult,
                out_bits,
                clamp_lut,
                mults,
                shifts,
                round_biases,
                channel_size,
                ..
            } => {
                let domain = 1usize << MESSAGE_BITS;
                if clamp_lut.len() != domain {
                    return Err(format!(
                        "Requant clamp_lut must have {domain} entries (the \
                         {MESSAGE_BITS}-bit message space); got {}",
                        clamp_lut.len()
                    ));
                }
                if let Some((i, &e)) = clamp_lut
                    .iter()
                    .enumerate()
                    .find(|&(_, &e)| e >= (1u64 << MESSAGE_BITS))
                {
                    return Err(format!(
                        "Requant clamp_lut[{i}] = {e} does not fit one shortint block \
                         (must be < {})",
                        1u64 << MESSAGE_BITS
                    ));
                }
                if *out_bits > MESSAGE_BITS {
                    return Err(format!(
                        "Requant out_bits ({out_bits}) exceeds MESSAGE_BITS ({MESSAGE_BITS}); the narrowed \
                         value must fit a single shortint block"
                    ));
                }
                if *mult == 0 {
                    return Err(
                        "Requant mult must be >= 1 (a fixed-point multiplier; 1 is a pure shift)"
                            .to_string(),
                    );
                }
                let has_pc = !mults.is_empty()
                    || !shifts.is_empty()
                    || !round_biases.is_empty()
                    || channel_size.is_some();
                if has_pc {
                    if mults.is_empty() {
                        return Err(
                            "Requant per-channel overlay present but `mults` is empty; supply one \
                             multiplier per output channel"
                                .to_string(),
                        );
                    }
                    if shifts.len() != mults.len() || round_biases.len() != mults.len() {
                        return Err(format!(
                            "Requant per-channel arrays must be equal length: mults={}, shifts={}, \
                             round_biases={}",
                            mults.len(),
                            shifts.len(),
                            round_biases.len()
                        ));
                    }
                    let cs = channel_size.ok_or_else(|| {
                        "Requant per-channel overlay needs `channel_size` (elements per channel: 1 \
                         for a Linear head, out_h*out_w for a Conv2d)"
                            .to_string()
                    })?;
                    if cs < 1 {
                        return Err("Requant channel_size must be >= 1".to_string());
                    }
                    if let Some((i, &m)) = mults.iter().enumerate().find(|&(_, &m)| m == 0) {
                        return Err(format!(
                            "Requant mults[{i}] = {m} must be >= 1 (a fixed-point multiplier per \
                             channel; 1 is a pure shift)"
                        ));
                    }
                }
            }
            OpSpec::Pool {
                mode,
                in_h,
                in_w,
                channels,
                pool_h,
                pool_w,
                stride,
            } => {
                match mode.as_str() {
                    "avg" | "max" => {}
                    other => {
                        return Err(format!(
                            "Pool mode must be \"avg\" or \"max\"; got {other:?}"
                        ))
                    }
                }
                if *in_h == 0 || *in_w == 0 || *channels == 0 {
                    return Err("Pool in_h/in_w/channels must be positive".to_string());
                }
                if *pool_h == 0 || *pool_w == 0 || *stride == 0 {
                    return Err("Pool pool_h/pool_w/stride must be positive".to_string());
                }
                if pool_h > in_h || pool_w > in_w {
                    return Err(format!(
                        "Pool window ({pool_h}x{pool_w}) must fit the input ({in_h}x{in_w})"
                    ));
                }
            }
            OpSpec::Add {} => {}
        }
        Ok(())
    }

    /// Build an op summary trait object for bit-width validation and conformance checks.
    pub fn build(&self) -> Result<Box<dyn OpSummary>, String> {
        self.validate()?;
        Ok(Box::new(self.clone()))
    }
}

impl OpSummary for OpSpec {
    fn output_bits(&self, input_bits: usize) -> usize {
        self.output_bits_n(&[input_bits])
    }
    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        crate::bitwidth::op_spec_output_bits_n(self, input_bits)
    }
    fn internal_bits_n(&self, input_bits: &[usize]) -> usize {
        crate::bitwidth::op_spec_internal_bits_n(self, input_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_json_round_trip() {
        let graph = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 8,
            input_bits: 4,
            inputs: vec!["x".to_string()],
            outputs: vec!["pred".to_string()],
            nodes: vec![
                Node {
                    name: "linear".to_string(),
                    inputs: vec!["x".to_string()],
                    outputs: vec!["z".to_string()],
                    op: OpSpec::Linear {
                        weights: vec![vec![1, -2, 3]],
                        bias: vec![4],
                        weight_bits: 3,
                    },
                },
                Node {
                    name: "act".to_string(),
                    inputs: vec!["z".to_string()],
                    outputs: vec!["a".to_string()],
                    op: OpSpec::Activation {
                        lut: vec![0, 1, 2, 3],
                        output_bits: 2,
                    },
                },
                Node {
                    name: "argmax".to_string(),
                    inputs: vec!["a".to_string()],
                    outputs: vec!["pred".to_string()],
                    op: OpSpec::Argmax { threshold: 2 },
                },
            ],
        };
        let restored = Graph::from_json(&graph.to_json()).expect("round-trips");
        assert_eq!(graph, restored);
    }

    #[test]
    fn add_op_json_round_trip() {
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
        let restored = Graph::from_json(&graph.to_json()).expect("round-trips");
        assert_eq!(graph, restored);
        assert_eq!(graph.nodes[0].op.op_type(), "Add");
    }

    #[test]
    fn conv2d_op_round_trip_and_validation() {
        let graph = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 8,
            input_bits: 4,
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            nodes: vec![Node {
                name: "conv".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["y".to_string()],
                op: OpSpec::Conv2d {
                    weights: vec![vec![0i64; 9]],
                    bias: vec![0],
                    weight_bits: 4,
                    in_h: 5,
                    in_w: 5,
                    in_channels: 1,
                    kernel_h: 3,
                    kernel_w: 3,
                    stride: 1,
                    padding: 0,
                },
            }],
        };
        let restored = Graph::from_json(&graph.to_json()).expect("round-trips");
        assert_eq!(graph, restored);

        let bad = OpSpec::Conv2d {
            weights: vec![vec![0i64; 8]],
            bias: vec![0],
            weight_bits: 4,
            in_h: 5,
            in_w: 5,
            in_channels: 1,
            kernel_h: 3,
            kernel_w: 3,
            stride: 1,
            padding: 0,
        };
        assert!(bad.build().is_err(), "kernel/fan-in mismatch must fail");
    }

    #[test]
    fn requant_op_round_trip_and_validation() {
        let graph = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 6,
            input_bits: 10,
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            nodes: vec![Node {
                name: "rq".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["y".to_string()],
                op: OpSpec::Requant {
                    shift: 4,
                    mult: 1,
                    round_bias: 0,
                    out_bits: 2,
                    clamp_lut: vec![0, 1, 2, 3],
                    mults: vec![],
                    shifts: vec![],
                    round_biases: vec![],
                    channel_size: None,
                },
            }],
        };
        let restored = Graph::from_json(&graph.to_json()).expect("round-trips");
        assert_eq!(graph, restored);
        let json = graph.to_json();
        assert!(!json.contains("mults"));
        assert!(!json.contains("channel_size"));

        let bad_len = OpSpec::Requant {
            shift: 1,
            mult: 1,
            round_bias: 0,
            out_bits: 2,
            clamp_lut: vec![0, 1, 2],
            mults: vec![],
            shifts: vec![],
            round_biases: vec![],
            channel_size: None,
        };
        assert!(bad_len.build().is_err());

        let bad_entry = OpSpec::Requant {
            shift: 1,
            mult: 1,
            round_bias: 0,
            out_bits: 2,
            clamp_lut: vec![0, 1, 2, 9],
            mults: vec![],
            shifts: vec![],
            round_biases: vec![],
            channel_size: None,
        };
        assert!(bad_entry.build().is_err());

        let zero_mult = OpSpec::Requant {
            shift: 1,
            mult: 0,
            round_bias: 0,
            out_bits: 2,
            clamp_lut: vec![0, 1, 2, 3],
            mults: vec![],
            shifts: vec![],
            round_biases: vec![],
            channel_size: None,
        };
        assert!(zero_mult.build().is_err());
    }

    #[test]
    fn requant_fields_default_when_absent() {
        let json = format!(
            r#"{{"schema_version":"{SCHEMA_VERSION}","num_blocks":6,"input_bits":10,
            "inputs":["x"],"outputs":["y"],"nodes":[{{"name":"rq","inputs":["x"],
            "outputs":["y"],"op":{{"op_type":"Requant","shift":4,"out_bits":2,
            "clamp_lut":[0,1,2,3]}}}}]}}"#
        );
        let graph = Graph::from_json(&json).expect("legacy Requant JSON deserializes");
        match &graph.nodes[0].op {
            OpSpec::Requant {
                mult, round_bias, ..
            } => {
                assert_eq!(*mult, 1);
                assert_eq!(*round_bias, 0);
            }
            other => panic!("expected Requant, got {}", other.op_type()),
        }
    }

    #[test]
    fn requant_with_mult_round_trips() {
        let graph = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 8,
            input_bits: 10,
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            nodes: vec![Node {
                name: "rq".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["y".to_string()],
                op: OpSpec::Requant {
                    shift: 5,
                    mult: 3,
                    round_bias: 16,
                    out_bits: 2,
                    clamp_lut: vec![0, 1, 2, 3],
                    mults: vec![],
                    shifts: vec![],
                    round_biases: vec![],
                    channel_size: None,
                },
            }],
        };
        let restored = Graph::from_json(&graph.to_json()).expect("round-trips");
        assert_eq!(graph, restored);
    }

    #[test]
    fn requant_per_channel_round_trips_and_validates() {
        let graph = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 8,
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
                    out_bits: 2,
                    clamp_lut: vec![0, 1, 2, 3],
                    mults: vec![1, 3],
                    shifts: vec![0, 5],
                    round_biases: vec![0, 16],
                    channel_size: Some(2),
                },
            }],
        };
        let restored = Graph::from_json(&graph.to_json()).expect("round-trips");
        assert_eq!(graph, restored);
        assert!(graph.to_json().contains("channel_size"));
        assert!(graph.nodes[0].op.build().is_ok());

        let bad = OpSpec::Requant {
            shift: 0,
            mult: 1,
            round_bias: 0,
            out_bits: 2,
            clamp_lut: vec![0, 1, 2, 3],
            mults: vec![1, 3],
            shifts: vec![0],
            round_biases: vec![0, 16],
            channel_size: Some(2),
        };
        assert!(bad.build().is_err());

        let no_cs = OpSpec::Requant {
            shift: 0,
            mult: 1,
            round_bias: 0,
            out_bits: 2,
            clamp_lut: vec![0, 1, 2, 3],
            mults: vec![1, 3],
            shifts: vec![0, 5],
            round_biases: vec![0, 16],
            channel_size: None,
        };
        assert!(no_cs.build().is_err());
    }

    #[test]
    fn pool_op_round_trip_and_validation() {
        let graph = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 6,
            input_bits: 5,
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
        let restored = Graph::from_json(&graph.to_json()).expect("round-trips");
        assert_eq!(graph, restored);

        let bad_mode = OpSpec::Pool {
            mode: "median".to_string(),
            in_h: 4,
            in_w: 4,
            channels: 1,
            pool_h: 2,
            pool_w: 2,
            stride: 2,
        };
        assert!(bad_mode.build().is_err());

        let too_big = OpSpec::Pool {
            mode: "max".to_string(),
            in_h: 2,
            in_w: 2,
            channels: 1,
            pool_h: 3,
            pool_w: 3,
            stride: 1,
        };
        assert!(too_big.build().is_err());
    }

    #[test]
    fn from_json_rejects_version_mismatch() {
        let bad = r#"{"schema_version":"0.0.1","num_blocks":8,"input_bits":4,
            "inputs":["x"],"outputs":["y"],"nodes":[]}"#;
        let err = Graph::from_json(bad).expect_err("version mismatch must fail");
        assert!(err.contains("schema-version mismatch"), "got: {err}");
    }

    #[test]
    fn from_json_rejects_unknown_op_type() {
        let bad = format!(
            r#"{{"schema_version":"{SCHEMA_VERSION}","num_blocks":8,"input_bits":4,
            "inputs":["x"],"outputs":["y"],"nodes":[{{"name":"c","inputs":["x"],
            "outputs":["y"],"op":{{"op_type":"BatchNorm"}}}}]}}"#
        );
        let err = Graph::from_json(&bad).expect_err("unknown op must fail");
        assert!(
            err.contains("BatchNorm") || err.contains("unknown variant"),
            "got: {err}"
        );
    }

    #[test]
    fn build_rejects_mismatched_linear() {
        let spec = OpSpec::Linear {
            weights: vec![vec![1, 2], vec![3, 4]],
            bias: vec![0],
            weight_bits: 4,
        };
        assert!(spec.build().is_err());
    }
}
