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

use crate::ops::{Activation, Add, Argmax, Linear, Op};

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
    Activation {
        lut: Vec<u64>,
        output_bits: usize,
    },
    Argmax {
        threshold: i64,
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
            OpSpec::Activation { .. } => "Activation",
            OpSpec::Argmax { .. } => "Argmax",
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
            "outputs":["y"],"op":{{"op_type":"Conv2d"}}}}]}}"#
        );
        let err = Graph::from_json(&bad).expect_err("unknown op must fail");
        assert!(
            err.contains("Conv2d") || err.contains("unknown variant"),
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
