//! Intermediate Representation (IR) deserialization.
//!
//! Mirrors `python/penumbra/ir.py` — the two definitions **must stay in lockstep**
//! (`AGENTS.md` §5). The Python front end emits the IR (JSON to start); this runtime
//! consumes *any* IR and walks the op graph without per-use-case changes.
//!
//! Hard rule (`AGENTS.md` §5): any IR change updates both language sides, bumps the
//! schema-version field, and updates the cross-language conformance test + `docs/IR-SPEC.md`
//! in the **same change**. A schema-version bump is a breaking change (`AGENTS.md` §8).
//!
//! See `PROJECT.md` §7 for the IR design (a directed graph of op nodes carrying op type,
//! inputs, attributes, quantized params, scales/zero-points, and precomputed LUTs).
//!
//! TODO(phase-3): define `serde`-deserializable structs for the op graph + a
//! `SCHEMA_VERSION` constant, mirroring `ir.py`.
