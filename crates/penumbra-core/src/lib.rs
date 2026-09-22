//! # Penumbra-FHE Core (Layer 2)
//!
//! The backend-neutral evaluation core of Penumbra-FHE (`PROJECT.md` §4, §13).
//!
//! - **Intermediate Representation ([`ir`]):** Directed graph of op nodes that Python emits
//!   and this runtime consumes without per-use-case changes.
//! - **Graph Walker & Eval Loop ([`eval`]):** Walks an IR [`ir::Graph`] and dispatches nodes
//!   to the active [`backend::Backend`].
//! - **Bit-width Budget Propagation ([`bitwidth`]):** Centralized scheme-neutral bit-width tracker
//!   enforcing capacity constraints.
//! - **Backend Contract ([`backend`]):** The formal [`backend::Backend`] trait connecting
//!   evaluation to pluggable FHE schemes (TFHE, CKKS).

pub mod backend;
pub mod bitwidth;
pub mod eval;
pub mod ir;
pub mod ops;
pub mod profile;

pub use backend::{Backend, CtVec, EvalCtx};
pub use bitwidth::{
    check_bit_width_budget, check_graph_bit_width_budget, propagate_bit_widths,
    radix_capacity_bits, MESSAGE_BITS,
};
pub use eval::{evaluate, evaluate_graph, evaluate_graph_profiled};
pub use ir::{Graph, Node, OpSpec, PoolMode, SCHEMA_VERSION};
pub use ops::{Op, OpSummary};
pub use profile::{GraphProfile, NodeProfile, OpTypeStats};
