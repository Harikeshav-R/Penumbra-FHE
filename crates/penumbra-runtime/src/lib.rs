//! FHE runtime and primitive operations for Penumbra.
//!
//! This crate wraps TFHE-rs to provide encrypted tensor operations, key generation,
//! encryption/decryption, and the execution engine for compiled graphs.

pub mod crypto;
pub mod error;
pub mod ops;

pub use crypto::{
    decrypt, encrypt, keygen, set_server_key, Ciphertext, ClientKey, SecurityParams, ServerKey,
};
pub use error::PenumbraRuntimeError;
pub use ops::{add, scalar_mul, sub};
