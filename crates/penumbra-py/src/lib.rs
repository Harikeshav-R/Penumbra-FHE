//! Python bindings for Penumbra.
//!
//! This crate exposes the Rust core to Python via PyO3. It handles type translation
//! and exposes a pythonic API for the underlying Rust primitives.

use pyo3::prelude::*;

#[pymodule]
fn _bindings(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register Rust functions and classes here
    Ok(())
}
