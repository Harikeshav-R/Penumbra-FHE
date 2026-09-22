//! The formal `Backend` trait (Waist 2) and evaluation context.
//!
//! Penumbra separates the backend-neutral graph evaluation core (Layer 2) from
//! the pluggable FHE cryptographic backends (Layer 1) via this trait (see `PROJECT.md` §4,
//! §6 and `docs/BACKENDS.md`).
//!
//! A backend provides:
//! 1. Evaluation primitives executed by ML ops (server-side, public server key only).
//! 2. Op instantiation ([`Backend::build_op`]).
//! 3. Key management and client-side encryption/decryption boundaries.

use crate::ir::{Graph, OpSpec};
use crate::ops::Op;

/// An encrypted tensor flowing between ops for a given backend.
pub type CtVec<B> = Vec<<B as Backend>::Ciphertext>;

/// Shared, read-only evaluation context handed to every op during graph execution.
pub struct EvalCtx<'a, K> {
    /// Public evaluation key enabling plaintext-weight arithmetic and bootstrapping.
    pub sk: &'a K,
    /// Central bit-width budget (radix width shared across the model under TFHE).
    pub num_blocks: usize,
}

impl<'a, K> EvalCtx<'a, K> {
    pub fn new(sk: &'a K, num_blocks: usize) -> Self {
        Self { sk, num_blocks }
    }
}

/// The formal FHE Backend contract (`docs/BACKENDS.md`).
///
/// Derived from what the op implementations actually call. Every primitive is a real call site.
pub trait Backend: 'static + Send + Sync {
    /// The ciphertext representation for this backend.
    type Ciphertext: Clone + Send + Sync;
    /// The public evaluation / server key.
    type ServerKey: Send + Sync;
    /// The secret client key.
    type ClientKey: Send + Sync;

    /// Unique identifier for this backend scheme (e.g. `"tfhe"` or `"ckks"`).
    fn name(&self) -> &'static str;

    /// Build a runnable op implementation for this backend from an [`OpSpec`].
    ///
    /// Fails loudly at load time if an op is unsupported on this backend (`AGENTS.md` §1.4).
    fn build_op(&self, spec: &OpSpec) -> Result<Box<dyn Op<Self>>, String>;

    /// Validate a graph against this backend's resource budget before any crypto runs,
    /// naming the offending node on failure (`AGENTS.md` §1.3, §1.4).
    ///
    /// This is the shared preflight seam: TFHE checks radix bit-width capacity, CKKS checks
    /// multiplicative depth and scale. Layer 2 knows only that a budget exists.
    fn check_graph_budget(&self, graph: &Graph) -> Result<(), String>;

    // --- Server-side evaluation primitives ----------------------------------------------------

    /// Create a trivial (plaintext) encryption of zero sized to `num_blocks`.
    fn create_trivial_zero(&self, sk: &Self::ServerKey, num_blocks: usize) -> Self::Ciphertext;

    /// Homomorphic ciphertext-ciphertext addition (`ct + ct`).
    fn add(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        b: &Self::Ciphertext,
    ) -> Self::Ciphertext;

    /// Ciphertext-plaintext scalar multiplication (`ct * scalar`).
    fn scalar_mul(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext;

    /// Ciphertext-plaintext scalar addition (`ct + scalar`).
    fn scalar_add(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext;

    /// Ciphertext-scalar comparison (`ct >= threshold`).
    fn scalar_ge(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        threshold: i64,
    ) -> Self::Ciphertext;

    /// Elementwise ciphertext maximum (`max(ct1, ct2)`).
    fn max(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        b: &Self::Ciphertext,
    ) -> Self::Ciphertext;

    /// Scalar maximum (`max(ct, scalar)`, e.g. ReLU at radix level).
    fn scalar_max(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext;

    /// Scalar minimum / clamp ceiling (`min(ct, scalar)`).
    fn scalar_min(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext;

    /// Arithmetic right shift (`ct >> shift`).
    fn scalar_right_shift(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        shift: u32,
    ) -> Self::Ciphertext;

    /// Apply a univariate lookup table function (`apply univariate f` via PBS in TFHE).
    fn apply_lut(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        lut: &[u64],
        num_blocks: usize,
    ) -> Self::Ciphertext;

    // --- Client-side key and boundary primitives ----------------------------------------------

    /// Generate a fresh client and server key pair for the given `num_blocks` capacity.
    fn keygen(&self, num_blocks: usize) -> (Self::ClientKey, Self::ServerKey);

    /// Encrypt a slice of signed integer inputs into ciphertexts.
    fn encrypt(&self, ck: &Self::ClientKey, input: &[i64]) -> Vec<Self::Ciphertext>;

    /// Decrypt a single-element output ciphertext to a scalar label.
    fn decrypt_label(&self, ck: &Self::ClientKey, out: &[Self::Ciphertext]) -> i64;

    /// Decrypt an array of output ciphertexts to integer values.
    fn decrypt_vec(&self, ck: &Self::ClientKey, out: &[Self::Ciphertext]) -> Vec<i64>;

    /// Serialize ciphertexts to a tagged byte buffer.
    fn serialize_cts(&self, cts: &[Self::Ciphertext]) -> Result<Vec<u8>, String>;

    /// Deserialize ciphertexts from a tagged byte buffer, verifying the scheme tag.
    fn deserialize_cts(&self, bytes: &[u8]) -> Result<Vec<Self::Ciphertext>, String>;

    /// Serialize the client secret key to its tagged wire bytes (for size accounting).
    fn serialize_client_key(
        &self,
        ck: &Self::ClientKey,
        num_blocks: usize,
    ) -> Result<Vec<u8>, String>;

    /// Serialize the public server/evaluation key to its tagged wire bytes.
    fn serialize_server_key(
        &self,
        sk: &Self::ServerKey,
        num_blocks: usize,
    ) -> Result<Vec<u8>, String>;
}
