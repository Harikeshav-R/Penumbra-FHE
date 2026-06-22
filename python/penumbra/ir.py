"""Intermediate Representation (IR): data structures + (de)serialization.

The IR is the product's backbone (``PROJECT.md`` §7): a directed graph of op nodes, each
carrying op type, inputs, attributes (kernel size, stride, ...), quantized params (int
weights, bias), scales/zero-points, and — for nonlinear ops — the precomputed lookup
table. The Python front end emits it; the Rust runtime consumes it.

This module **must stay in lockstep** with ``runtime/src/ir.rs`` (``AGENTS.md`` §5). Any
IR change updates both language sides, bumps the schema-version field, and updates the
cross-language conformance test + ``docs/IR-SPEC.md`` in the **same change**. A
schema-version bump is a breaking change (``AGENTS.md`` §8).

Wire format is JSON to start (human-inspectable, easy to debug); a compact binary format
is a later, profiling-driven decision (ROADMAP.md Phase 10), and an architectural fork to
raise before implementing (``AGENTS.md`` §3.2).

TODO(phase-3): define node/graph dataclasses, a ``SCHEMA_VERSION`` constant, and
``to_json()`` / ``from_json()``.
"""
