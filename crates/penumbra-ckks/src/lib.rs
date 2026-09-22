//! # Penumbra-FHE CKKS backend (Layer 1)
//!
//! Requires the `ckks` feature and a nightly toolchain; see the crate manifest.

#[cfg(feature = "ckks")]
pub mod backend;
#[cfg(feature = "ckks")]
pub mod bounds;
#[cfg(feature = "ckks")]
pub mod encrypt;
#[cfg(feature = "ckks")]
pub mod hal;
#[cfg(feature = "ckks")]
pub mod keys;
#[cfg(feature = "ckks")]
pub mod ops;
#[cfg(feature = "ckks")]
pub mod params;

#[cfg(feature = "ckks")]
pub use backend::{check_graph_depth_budget, evaluate_graph, CkksBackend};
#[cfg(feature = "ckks")]
pub use encrypt::{
    decrypt_label, decrypt_raw_vec, decrypt_vec, deserialize_cts, deserialize_cts_batch, encrypt,
    serialize_cts, serialize_cts_batch, CkksCt, CtVec, TaggedCts,
};
#[cfg(feature = "ckks")]
pub use hal::hal_backend_name;
#[cfg(feature = "ckks")]
pub use keys::{
    client_key_bytes, keygen, load_client_key, load_server_key, save_client_key, save_server_key,
    server_key_bytes, CkksClientKey, CkksServerKey, SCHEME_CKKS,
};
#[cfg(feature = "ckks")]
pub use params::{slots, CkksParams, DEFAULT_PARAMS};
