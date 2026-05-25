//! Python bindings for Penumbra.
//!
//! This crate exposes the Rust core to Python via PyO3. It handles type translation
//! and exposes a pythonic API for the underlying Rust primitives.

use pyo3::prelude::*;

#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[pymodule]
fn _bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register Rust functions and classes here
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
