use crate::error::PenumbraRuntimeError;
use tfhe::prelude::*;
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
use tfhe::{
    ClientKey as TfheClientKey, ConfigBuilder, FheUint32, Seed, ServerKey as TfheServerKey,
};

/// Security parameters for key generation.
///
/// Contains the seed for deterministic key generation.
pub struct SecurityParams {
    /// The deterministic seed for the PRNG.
    pub rng_seed: u64,
}

/// A wrapper around TFHE-rs `ClientKey`.
pub struct ClientKey(pub(crate) TfheClientKey);

/// A wrapper around TFHE-rs `ServerKey`.
pub struct ServerKey(pub(crate) TfheServerKey);

/// A wrapper around TFHE-rs `FheUint32`.
#[derive(Clone)]
pub struct Ciphertext(pub(crate) FheUint32);

/// Generate client and server keys using the given security parameters.
///
/// # Parameters
/// * `params` - The security parameters including the RNG seed.
///
/// # Returns
/// A tuple containing the `ClientKey` and `ServerKey`.
///
/// # Example
/// ```
/// use penumbra_runtime::{keygen, SecurityParams};
/// let params = SecurityParams { rng_seed: 42 };
/// let (client_key, server_key) = keygen(params).unwrap();
/// ```
pub fn keygen(params: SecurityParams) -> Result<(ClientKey, ServerKey), PenumbraRuntimeError> {
    let config = ConfigBuilder::with_custom_parameters(PARAM_MESSAGE_2_CARRY_2_KS_PBS).build();
    let seed = Seed(params.rng_seed as u128);

    let client_key = TfheClientKey::generate_with_seed(config, seed);
    let server_key = TfheServerKey::new(&client_key);

    Ok((ClientKey(client_key), ServerKey(server_key)))
}

/// Set the global server key for the current thread.
///
/// This must be called before any homomorphic operations are performed.
///
/// # Parameters
/// * `key` - The server key to set for the thread.
///
/// # Example
/// ```
/// use penumbra_runtime::{keygen, set_server_key, SecurityParams};
/// let params = SecurityParams { rng_seed: 42 };
/// let (_, server_key) = keygen(params).unwrap();
/// set_server_key(&server_key);
/// ```
pub fn set_server_key(key: &ServerKey) {
    tfhe::set_server_key(key.0.clone());
}

/// Encrypt a 32-bit unsigned integer.
///
/// # Parameters
/// * `val` - The value to encrypt.
/// * `key` - The client key used for encryption.
///
/// # Returns
/// A `Ciphertext` representing the encrypted value.
///
/// # Example
/// ```
/// use penumbra_runtime::{keygen, encrypt, SecurityParams};
/// let params = SecurityParams { rng_seed: 42 };
/// let (client_key, _) = keygen(params).unwrap();
/// let ct = encrypt(5, &client_key);
/// ```
pub fn encrypt(val: u32, key: &ClientKey) -> Ciphertext {
    let encrypted = FheUint32::encrypt(val, &key.0);
    Ciphertext(encrypted)
}

/// Decrypt a ciphertext into a 32-bit unsigned integer.
///
/// # Parameters
/// * `ct` - The ciphertext to decrypt.
/// * `key` - The client key used for decryption.
///
/// # Returns
/// The decrypted 32-bit integer.
///
/// # Example
/// ```
/// use penumbra_runtime::{keygen, encrypt, decrypt, SecurityParams};
/// let params = SecurityParams { rng_seed: 42 };
/// let (client_key, _) = keygen(params).unwrap();
/// let ct = encrypt(5, &client_key);
/// assert_eq!(decrypt(&ct, &client_key), 5);
/// ```
pub fn decrypt(ct: &Ciphertext, key: &ClientKey) -> u32 {
    ct.0.decrypt(&key.0)
}
