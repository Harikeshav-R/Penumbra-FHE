use crate::crypto::Ciphertext;
use crate::error::PenumbraRuntimeError;

/// Add two encrypted values. Requires server key to be set.
///
/// # Parameters
/// * `lhs` - The first ciphertext.
/// * `rhs` - The second ciphertext.
///
/// # Returns
/// A new ciphertext containing the homomorphically evaluated sum.
///
/// # Errors
/// Can return a `PenumbraRuntimeError` if the operation fails.
///
/// # Example
/// ```
/// use penumbra_runtime::{keygen, set_server_key, encrypt, add, decrypt, SecurityParams};
/// let params = SecurityParams { rng_seed: 42 };
/// let (client_key, server_key) = keygen(params).unwrap();
/// set_server_key(&server_key);
/// let ct1 = encrypt(5, &client_key);
/// let ct2 = encrypt(10, &client_key);
/// let res = add(&ct1, &ct2).unwrap();
/// assert_eq!(decrypt(&res, &client_key), 15);
/// ```
pub fn add(lhs: &Ciphertext, rhs: &Ciphertext) -> Result<Ciphertext, PenumbraRuntimeError> {
    Ok(Ciphertext(&lhs.0 + &rhs.0))
}

/// Subtract two encrypted values. Requires server key to be set.
///
/// # Parameters
/// * `lhs` - The ciphertext to subtract from.
/// * `rhs` - The ciphertext to subtract.
///
/// # Returns
/// A new ciphertext containing the homomorphically evaluated difference.
///
/// # Errors
/// Can return a `PenumbraRuntimeError` if the operation fails.
///
/// # Example
/// ```
/// use penumbra_runtime::{keygen, set_server_key, encrypt, sub, decrypt, SecurityParams};
/// let params = SecurityParams { rng_seed: 42 };
/// let (client_key, server_key) = keygen(params).unwrap();
/// set_server_key(&server_key);
/// let ct1 = encrypt(15, &client_key);
/// let ct2 = encrypt(5, &client_key);
/// let res = sub(&ct1, &ct2).unwrap();
/// assert_eq!(decrypt(&res, &client_key), 10);
/// ```
pub fn sub(lhs: &Ciphertext, rhs: &Ciphertext) -> Result<Ciphertext, PenumbraRuntimeError> {
    Ok(Ciphertext(&lhs.0 - &rhs.0))
}

/// Multiply an encrypted value by a plaintext scalar. Requires server key to be set.
///
/// # Parameters
/// * `lhs` - The ciphertext to multiply.
/// * `scalar` - The plaintext scalar to multiply by.
///
/// # Returns
/// A new ciphertext containing the homomorphically evaluated product.
///
/// # Errors
/// Can return a `PenumbraRuntimeError` if the operation fails.
///
/// # Example
/// ```
/// use penumbra_runtime::{keygen, set_server_key, encrypt, scalar_mul, decrypt, SecurityParams};
/// let params = SecurityParams { rng_seed: 42 };
/// let (client_key, server_key) = keygen(params).unwrap();
/// set_server_key(&server_key);
/// let ct = encrypt(5, &client_key);
/// let res = scalar_mul(&ct, 3).unwrap();
/// assert_eq!(decrypt(&res, &client_key), 15);
/// ```
pub fn scalar_mul(lhs: &Ciphertext, scalar: u32) -> Result<Ciphertext, PenumbraRuntimeError> {
    Ok(Ciphertext(&lhs.0 * scalar))
}
