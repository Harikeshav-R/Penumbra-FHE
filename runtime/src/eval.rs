//! The graph walker / evaluation loop.
//!
//! Reads a deserialized IR graph ([`crate::ir`]) and evaluates it under FHE by
//! dispatching each node to its op implementation in [`crate::ops`]. This loop is
//! written **once** and never changes per use case (`PROJECT.md` §4, §7).
//!
//! Runtime ≈ number of bootstraps (`PROJECT.md` §5): linear ops with plaintext weights
//! are cheap; activations/requant are programmable bootstraps and dominate cost.
//!
//! TODO(phase-2): hardcoded `Linear → Activation → Argmax` sequence.
//! TODO(phase-3): walk the deserialized IR graph instead of a hardcoded sequence.
//! TODO(phase-8): true topological ordering with intermediate-result storage for
//! branching/multi-input graphs.
