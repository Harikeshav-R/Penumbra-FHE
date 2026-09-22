//! # Penumbra-FHE TFHE Backend (Layer 1)
//!
//! The reference TFHE/CGGI backend for Penumbra-FHE, built against `tfhe-rs` (`PROJECT.md` §4, §13).
//!
//! - **Key management ([`keys`]):** `keygen`, param profiles, client/server key persistence.
//! - **Encrypt/decrypt ([`encrypt`]):** client-side encryption and decryption boundaries.
//! - **Op implementations ([`ops`]):** TFHE implementations of the narrow-waist op vocabulary.
//! - **Backend realization ([`backend`]):** [`backend::TfheBackend`] implementing [`penumbra_core::backend::Backend`].

pub mod backend;
pub mod encrypt;
pub mod keys;
pub mod ops;

pub use backend::TfheBackend;
pub use encrypt::{
    decrypt_label, decrypt_vec, deserialize_cts, deserialize_cts_batch, encrypt, serialize_cts,
    serialize_cts_batch, CtVec, TaggedCts,
};
pub use keys::{
    client_key_bytes, ensure_num_blocks_match, keygen, load_client_key, load_server_key,
    magnitude_bits, radix_capacity_bits, save_client_key, save_server_key, server_key_bytes,
    DEFAULT_PARAMS, MESSAGE_BITS, SCHEME_TFHE,
};
pub use ops::{Activation, Add, Argmax, Conv2d, EvalCtx, Linear, Op, Pool, PoolMode, Requant};

/// Evaluate a linear chain of ops over an encrypted input using the TFHE backend.
pub fn evaluate(
    ctx: &ops::EvalCtx,
    ops: &[Box<dyn ops::Op>],
    input: &encrypt::CtVec,
) -> encrypt::CtVec {
    let mut acc = input.clone();
    for op in ops {
        acc = ops::Op::eval(&**op, ctx, &acc);
    }
    acc
}

/// Verify a model's declared bit-width budget fits the radix capacity under the TFHE backend.
pub fn check_bit_width_budget(
    ops: &[Box<dyn ops::Op>],
    input_bits: usize,
    num_blocks: usize,
) -> Result<(), String> {
    let capacity = keys::radix_capacity_bits(num_blocks);
    let mut bits = input_bits;
    for (i, op) in ops.iter().enumerate() {
        let out = ops::Op::output_bits(&**op, bits);
        if out > capacity {
            return Err(format!(
                "layer {i} output bit-width {out} exceeds radix capacity {capacity} \
                 ({num_blocks} blocks of {} bits). Model will overflow.",
                keys::MESSAGE_BITS
            ));
        }
        bits = out;
    }
    Ok(())
}

/// Evaluate an IR [`penumbra_core::ir::Graph`] over encrypted input tensors using the TFHE backend.
pub fn evaluate_graph(
    ctx: &ops::EvalCtx,
    graph: &penumbra_core::ir::Graph,
    inputs: std::collections::HashMap<String, encrypt::CtVec>,
) -> Result<std::collections::HashMap<String, encrypt::CtVec>, String> {
    penumbra_core::eval::evaluate_graph(&backend::TfheBackend, ctx, graph, inputs)
}
