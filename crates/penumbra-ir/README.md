# penumbra-ir

Typed intermediate representation for Penumbra.

This crate defines the graph structure, operations, and tensor metadata. It acts as the contract between the Python ingestion layer and the Rust backend.

It explicitly **does not** depend on any FHE library or Python bindings.
