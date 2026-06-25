//! Intermediate Representation (IR) — the serializable op graph.
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
//!
//! ## Decoupled from the runtime ops (a deliberate design choice)
//!
//! [`OpSpec`] is the *wire format* — it owns (de)serialization and load-time validation,
//! and `build`s the runtime [`Op`] (`crate::ops`). The op structs themselves stay
//! serde-free: they `use tfhe::integer::…` and we do not want the public serialization
//! contract coupled to crypto-adjacent field layout. As Phase-4 ops (`Conv2d`, `Requant`)
//! grow fields the wire format shouldn't echo verbatim, this seam keeps both sides clean.
//!
//! ## Op payload encoding
//!
//! Each node carries its op as a nested, *internally tagged* object keyed on `op_type`
//! (`#[serde(tag = "op_type")]`), **not** `#[serde(flatten)]`: flatten disables
//! `deny_unknown_fields` and has known round-trip bugs with internally-tagged enums. An
//! unknown `op_type` therefore fails loudly for free (`unknown variant 'Conv2d', expected
//! one of 'Linear', 'Activation', 'Argmax'`).

use serde::{Deserialize, Serialize};

use crate::ops::{Activation, Add, Argmax, Conv2d, Linear, Op, Pool, PoolMode, Requant};

/// IR wire-format version. Hardcoded identically in `python/penumbra/ir.py`; a mismatch is
/// a breaking change caught loudly at load time (`AGENTS.md` §5, §8).
pub const SCHEMA_VERSION: &str = "0.4.0";

/// The root IR object: a directed graph of op nodes in a valid topological order.
///
/// `num_blocks` is the central bit-width budget (the radix width every ciphertext shares,
/// `keys::keygen`); `input_bits` is the declared width of the encrypted model input that
/// seeds the bit-width tracker. `inputs`/`outputs` name the graph's boundary tensors.
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
///
/// Phase-2 ops are single-input/single-output; the `Vec`s are general so branching graphs
/// (Phase 8) reuse the same shape without a schema change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub op: OpSpec,
}

/// The op payload — the serializable mirror of the runtime ops (`crate::ops`).
///
/// Internally tagged on `op_type` (see module docs). Carries the same fields as the op
/// structs but lives in IR-land so the ops stay serde-free and validation has a home.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op_type")]
pub enum OpSpec {
    Linear {
        weights: Vec<Vec<i64>>,
        bias: Vec<i64>,
        weight_bits: usize,
    },
    /// 2-D convolution against plaintext kernel weights. `weights` is row-major
    /// `[out_channels][in_channels*kernel_h*kernel_w]`; the input/output flat tensors use the
    /// channel-major, row-major layout shared with `Pool`. See [`crate::ops::Conv2d`].
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
    /// Rescale a wide accumulator down to a narrow, LUT-able value: arithmetic right-shift by
    /// `shift`, ReLU, clamp to `2^out_bits - 1` via `clamp_lut`. See [`crate::ops::Requant`].
    Requant {
        shift: u32,
        out_bits: usize,
        clamp_lut: Vec<u64>,
    },
    /// Spatial pooling over a flattened `[channels][in_h][in_w]` feature map. `mode` is
    /// `"avg"` (window sum, rescale deferred to `Requant`) or `"max"`. See [`crate::ops::Pool`].
    Pool {
        mode: String,
        in_h: usize,
        in_w: usize,
        channels: usize,
        pool_h: usize,
        pool_w: usize,
        stride: usize,
    },
    /// Element-wise addition of two input tensors (residuals). The first **multi-input** op:
    /// its node carries two entries in `inputs`. No payload fields — the operands come from
    /// the graph wiring, not the spec.
    Add {},
}

impl Graph {
    /// Deserialize an IR graph from JSON, validating the schema version loudly.
    ///
    /// Returns `Err` (never panics) on malformed JSON, an unknown `op_type`, or a version
    /// mismatch — all actionable load-time failures (`AGENTS.md` §1.4). Semantic graph
    /// checks (tensor wiring, topo order) live in [`crate::eval`], not here: this is purely
    /// the parse + version gate. Forward-compat is gated by the version field, so we do
    /// **not** `deny_unknown_fields` — a future compatible key must not hard-fail an older
    /// reader that has already matched the version.
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

    /// Serialize the graph to pretty JSON (the human-inspectable wire format, `PROJECT.md`
    /// §7). Infallible: every field is a plain integer/string container.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("IR graph is composed of serializable types")
    }
}

impl OpSpec {
    /// Construct the runtime [`Op`] this spec describes, validating its parameters loudly
    /// *before* any crypto (`AGENTS.md` §1.4) — keygen is slow, so a malformed layer should
    /// fail at load, not after.
    ///
    /// Returns `Err` rather than panicking so the graph loader ([`crate::eval`]) surfaces a
    /// clean, named error. The op structs additionally assert their own invariants in
    /// `eval`/`output_bits` (defense in depth); this just catches them earlier.
    pub fn build(&self) -> Result<Box<dyn Op>, String> {
        match self {
            OpSpec::Linear {
                weights,
                bias,
                weight_bits,
            } => {
                // Mirror the `assert!`s in `Linear::eval`, but as load-time errors: a `zip`
                // mismatch would otherwise silently truncate, and a ragged matrix would
                // panic deep in eval after keygen.
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
                Ok(Box::new(Linear {
                    weights: weights.clone(),
                    bias: bias.clone(),
                    weight_bits: *weight_bits,
                }))
            }
            OpSpec::Conv2d {
                weights,
                bias,
                weight_bits,
                in_h,
                in_w,
                in_channels,
                kernel_h,
                kernel_w,
                stride,
                padding,
            } => {
                // Load-time validation mirroring `Conv2d::eval`'s asserts (`AGENTS.md` §1.4).
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
                // The padded kernel must fit the padded input, or `out_dims` underflows.
                if kernel_h > &(in_h + 2 * padding) || kernel_w > &(in_w + 2 * padding) {
                    return Err(format!(
                        "Conv2d kernel ({kernel_h}x{kernel_w}) does not fit the padded input \
                         ({}x{})",
                        in_h + 2 * padding,
                        in_w + 2 * padding
                    ));
                }
                Ok(Box::new(Conv2d {
                    weights: weights.clone(),
                    bias: bias.clone(),
                    weight_bits: *weight_bits,
                    in_h: *in_h,
                    in_w: *in_w,
                    in_channels: *in_channels,
                    kernel_h: *kernel_h,
                    kernel_w: *kernel_w,
                    stride: *stride,
                    padding: *padding,
                }))
            }
            // Activation/Argmax own their remaining invariants in `eval`/`output_bits`. The
            // Phase-2 model graph is `Linear → Argmax` (the activation LUT is exercised
            // standalone, not in the inference path), so an in-graph Activation is dormant
            // until Phase 4 adds a Requant in front of it.
            OpSpec::Activation { lut, output_bits } => Ok(Box::new(Activation {
                lut: lut.clone(),
                output_bits: *output_bits,
            })),
            OpSpec::Argmax { threshold } => Ok(Box::new(Argmax {
                threshold: *threshold,
            })),
            OpSpec::Requant {
                shift,
                out_bits,
                clamp_lut,
            } => {
                // Mirror the asserts in `Requant::eval`/`output_bits` as load-time errors so a
                // malformed table fails before keygen (`AGENTS.md` §1.4), not deep in eval.
                let domain = 1usize << crate::keys::MESSAGE_BITS;
                if clamp_lut.len() != domain {
                    return Err(format!(
                        "Requant clamp_lut must have {domain} entries (the \
                         {}-bit message space); got {}",
                        crate::keys::MESSAGE_BITS,
                        clamp_lut.len()
                    ));
                }
                if let Some((i, &e)) = clamp_lut
                    .iter()
                    .enumerate()
                    .find(|&(_, &e)| e >= (1u64 << crate::keys::MESSAGE_BITS))
                {
                    return Err(format!(
                        "Requant clamp_lut[{i}] = {e} does not fit one shortint block \
                         (must be < {})",
                        1u64 << crate::keys::MESSAGE_BITS
                    ));
                }
                if *out_bits > crate::keys::MESSAGE_BITS {
                    return Err(format!(
                        "Requant out_bits ({out_bits}) exceeds MESSAGE_BITS ({}); the narrowed \
                         value must fit a single shortint block",
                        crate::keys::MESSAGE_BITS
                    ));
                }
                Ok(Box::new(Requant {
                    shift: *shift,
                    out_bits: *out_bits,
                    clamp_lut: clamp_lut.clone(),
                }))
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
                // Parse the mode string into the typed enum, failing loudly on a typo rather
                // than silently picking a default (`AGENTS.md` §1.4).
                let mode = match mode.as_str() {
                    "avg" => PoolMode::Avg,
                    "max" => PoolMode::Max,
                    other => {
                        return Err(format!(
                            "Pool mode must be \"avg\" or \"max\"; got {other:?}"
                        ))
                    }
                };
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
                Ok(Box::new(Pool {
                    mode,
                    in_h: *in_h,
                    in_w: *in_w,
                    channels: *channels,
                    pool_h: *pool_h,
                    pool_w: *pool_w,
                    stride: *stride,
                }))
            }
            // `Add` has no payload to validate here; its operand-count and equal-length
            // invariants are enforced in `Add::eval_n` (the wiring is the graph's job). The
            // two-input requirement is checked by the eval loop / bit-width tracker.
            OpSpec::Add {} => Ok(Box::new(Add)),
        }
    }

    /// The op-type tag, for human-facing output (`inspect`) and error messages.
    pub fn op_type(&self) -> &'static str {
        match self {
            OpSpec::Linear { .. } => "Linear",
            OpSpec::Conv2d { .. } => "Conv2d",
            OpSpec::Activation { .. } => "Activation",
            OpSpec::Argmax { .. } => "Argmax",
            OpSpec::Requant { .. } => "Requant",
            OpSpec::Pool { .. } => "Pool",
            OpSpec::Add {} => "Add",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal Phase-2 graph (`Linear → Argmax`) round-trips through JSON unchanged.
    #[test]
    fn graph_json_round_trip() {
        let graph = Graph {
            schema_version: SCHEMA_VERSION.to_string(),
            num_blocks: 8,
            input_bits: 4,
            inputs: vec!["x".to_string()],
            outputs: vec!["label".to_string()],
            nodes: vec![
                Node {
                    name: "fc".to_string(),
                    inputs: vec!["x".to_string()],
                    outputs: vec!["logit".to_string()],
                    op: OpSpec::Linear {
                        weights: vec![vec![1, -2, 3]],
                        bias: vec![-1],
                        weight_bits: 4,
                    },
                },
                Node {
                    name: "head".to_string(),
                    inputs: vec!["logit".to_string()],
                    outputs: vec!["label".to_string()],
                    op: OpSpec::Argmax { threshold: 0 },
                },
            ],
        };
        let restored = Graph::from_json(&graph.to_json()).expect("round-trips");
        assert_eq!(graph, restored);
    }

    /// The multi-input `Add` op (two `inputs`, empty payload) round-trips unchanged and
    /// serializes to the bare `{"op_type":"Add"}` the Python `AddSpec` emits.
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

    /// A `Conv2d` node round-trips, and `build` rejects a kernel row whose width disagrees
    /// with `in_channels*kernel_h*kernel_w`.
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
                    weights: vec![vec![0i64; 9]], // 1 in-channel * 3 * 3
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

        // Kernel width (8) disagrees with fan-in 1*3*3 = 9.
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

    /// A `Requant` node round-trips through JSON, and `build` rejects a malformed clamp LUT
    /// (wrong length / out-of-range entry) at load time rather than panicking deep in eval.
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
                    out_bits: 2,
                    clamp_lut: vec![0, 1, 2, 3],
                },
            }],
        };
        let restored = Graph::from_json(&graph.to_json()).expect("round-trips");
        assert_eq!(graph, restored);

        // Wrong LUT length fails to build (must cover the whole 2-bit message space).
        let bad_len = OpSpec::Requant {
            shift: 1,
            out_bits: 2,
            clamp_lut: vec![0, 1, 2],
        };
        assert!(
            bad_len.build().is_err(),
            "short clamp_lut must fail to build"
        );

        // An entry that doesn't fit one block fails to build.
        let bad_entry = OpSpec::Requant {
            shift: 1,
            out_bits: 2,
            clamp_lut: vec![0, 1, 2, 9],
        };
        assert!(
            bad_entry.build().is_err(),
            "out-of-range clamp_lut entry must fail to build"
        );
    }

    /// A `Pool` node round-trips, and `build` rejects an unknown mode / oversized window.
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
        assert!(bad_mode.build().is_err(), "unknown Pool mode must fail");

        let too_big = OpSpec::Pool {
            mode: "max".to_string(),
            in_h: 2,
            in_w: 2,
            channels: 1,
            pool_h: 3,
            pool_w: 3,
            stride: 1,
        };
        assert!(
            too_big.build().is_err(),
            "window larger than the input must fail"
        );
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
        // `BatchNorm` is a still-unsupported op (Conv2d/Pool/Requant/Add became known in
        // Phase 4); use it as the negative case so the test exercises a genuine rejection.
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
            bias: vec![0], // 2 rows, 1 bias
            weight_bits: 4,
        };
        assert!(
            spec.build().is_err(),
            "weights/bias length mismatch must fail to build"
        );
    }
}
