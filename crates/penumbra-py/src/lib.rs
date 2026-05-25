//! Python bindings for Penumbra.
//!
//! This crate exposes the Rust core to Python via PyO3. It handles type translation
//! and exposes a pythonic API for the underlying Rust primitives.

use penumbra_runtime::{
    add, decrypt, encrypt, keygen, scalar_mul, sub, Ciphertext, ClientKey, PenumbraRuntimeError,
    SecurityParams, ServerKey,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// We need to implement From<PenumbraRuntimeError> for PyErr, but for now we'll just map it directly in the wrappers.
// To follow rule 9.2 (Penumbra-specific exception types), we should import it from `penumbra_fhe.errors` if possible,
// but PyO3 allows creating custom exception types.
// Wait, we can import `penumbra_fhe.errors.PenumbraRuntimeError` dynamically.
fn to_py_err(err: PenumbraRuntimeError, py: Python<'_>) -> PyErr {
    if let Ok(module) = py.import("penumbra_fhe.errors") {
        if let Ok(err_class) = module.getattr("PenumbraRuntimeError") {
            return PyErr::from_value(err_class.call1((err.to_string(),)).unwrap().into_any());
        }
    }
    PyRuntimeError::new_err(err.to_string())
}

/// Security parameters for homomorphic encryption.
#[pyclass(name = "SecurityParams")]
pub struct PySecurityParams(pub SecurityParams);

#[pymethods]
impl PySecurityParams {
    /// Create new security parameters.
    ///
    /// :param rng_seed: A deterministic seed for the random number generator.
    /// :returns: A new SecurityParams instance.
    #[new]
    fn new(rng_seed: u64) -> Self {
        PySecurityParams(SecurityParams { rng_seed })
    }
}

/// The client key used for encryption and decryption.
#[pyclass(name = "ClientKey")]
pub struct PyClientKey(pub ClientKey);

/// The server key used for homomorphic evaluation.
#[pyclass(name = "ServerKey")]
pub struct PyServerKey(pub ServerKey);

/// An encrypted 32-bit unsigned integer.
#[pyclass(name = "Ciphertext", from_py_object)]
#[derive(Clone)]
pub struct PyCiphertext(pub Ciphertext);

#[pymethods]
impl PyCiphertext {
    /// Add two encrypted values.
    ///
    /// :param rhs: The ciphertext to add.
    /// :returns: A new ciphertext containing the sum.
    /// :raises PenumbraRuntimeError: If the server key is not set.
    fn __add__(&self, rhs: &PyCiphertext, py: Python<'_>) -> PyResult<Self> {
        let result = add(&self.0, &rhs.0).map_err(|e| to_py_err(e, py))?;
        Ok(PyCiphertext(result))
    }

    /// Subtract an encrypted value from this one.
    ///
    /// :param rhs: The ciphertext to subtract.
    /// :returns: A new ciphertext containing the difference.
    /// :raises PenumbraRuntimeError: If the server key is not set.
    fn __sub__(&self, rhs: &PyCiphertext, py: Python<'_>) -> PyResult<Self> {
        let result = sub(&self.0, &rhs.0).map_err(|e| to_py_err(e, py))?;
        Ok(PyCiphertext(result))
    }

    /// Multiply this encrypted value by a plaintext scalar.
    ///
    /// :param scalar: The plaintext scalar multiplier.
    /// :returns: A new ciphertext containing the product.
    /// :raises PenumbraRuntimeError: If the server key is not set.
    fn __mul__(&self, scalar: u32, py: Python<'_>) -> PyResult<Self> {
        let result = scalar_mul(&self.0, scalar).map_err(|e| to_py_err(e, py))?;
        Ok(PyCiphertext(result))
    }
}

/// Generate a client and server key pair from security parameters.
///
/// :param params: The security parameters containing the RNG seed.
/// :returns: A tuple containing the (ClientKey, ServerKey).
#[pyfunction]
#[pyo3(name = "keygen")]
fn py_keygen(params: &PySecurityParams, py: Python<'_>) -> PyResult<(PyClientKey, PyServerKey)> {
    let p = SecurityParams {
        rng_seed: params.0.rng_seed,
    };
    let (c, s) = keygen(p).map_err(|e| to_py_err(e, py))?;
    Ok((PyClientKey(c), PyServerKey(s)))
}

/// Encrypt a 32-bit integer using the client key.
///
/// :param val: The integer to encrypt.
/// :param key: The client key.
/// :returns: A new Ciphertext.
#[pyfunction]
#[pyo3(name = "encrypt")]
fn py_encrypt(val: u32, key: &PyClientKey) -> PyCiphertext {
    PyCiphertext(encrypt(val, &key.0))
}

/// Decrypt a ciphertext back into a 32-bit integer.
///
/// :param ct: The ciphertext to decrypt.
/// :param key: The client key.
/// :returns: The decrypted integer.
#[pyfunction]
#[pyo3(name = "decrypt")]
fn py_decrypt(ct: &PyCiphertext, key: &PyClientKey) -> u32 {
    decrypt(&ct.0, &key.0)
}

/// Set the server key for the current thread context.
///
/// :param key: The server key to set globally for the thread.
#[pyfunction]
#[pyo3(name = "set_server_key")]
fn py_set_server_key(key: &PyServerKey) {
    penumbra_runtime::set_server_key(&key.0);
}

/// Get the version of the penumbra-fhe core library.
///
/// :returns: The version string.
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[pymodule]
fn _bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(py_keygen, m)?)?;
    m.add_function(wrap_pyfunction!(py_encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(py_decrypt, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_server_key, m)?)?;

    m.add_class::<PySecurityParams>()?;
    m.add_class::<PyClientKey>()?;
    m.add_class::<PyServerKey>()?;
    m.add_class::<PyCiphertext>()?;

    Ok(())
}
