use thiserror::Error;

/// Errors that can occur within the Penumbra runtime.
#[derive(Debug, Error)]
pub enum PenumbraRuntimeError {
    /// Error during key generation.
    #[error("Failed to generate keys: {0}")]
    KeyGen(String),
    /// Error during encryption.
    #[error("Encryption failed: {0}")]
    Encryption(String),
    /// Error during decryption.
    #[error("Decryption failed: {0}")]
    Decryption(String),
    /// Error during an FHE operation.
    #[error("Operation failed: {0}")]
    Operation(String),
}
