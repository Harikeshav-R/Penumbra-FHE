//! # Penumbra-FHE Runtime (Facade Crate)
//!
//! Re-exports [`penumbra_core`] (Layer 2: IR, Op trait, eval loop, bitwidth tracking)
//! and [`penumbra_tfhe`] (Layer 1: TFHE backend against `tfhe-rs`).
//!
//! Preserves 100% backward compatibility for existing binaries and integration tests.

pub use penumbra_core::bitwidth::{check_graph_bit_width_budget, propagate_bit_widths};
pub use penumbra_core::ir::{Graph, Node, OpSpec, PoolMode, SCHEMA_VERSION};
pub use penumbra_core::ops::OpSummary;

pub use penumbra_tfhe::backend::TfheBackend;
pub use penumbra_tfhe::check_bit_width_budget;
pub use penumbra_tfhe::encrypt::{
    decrypt_label, decrypt_vec, deserialize_cts, deserialize_cts_batch, encrypt, serialize_cts,
    serialize_cts_batch, CtVec,
};
pub use penumbra_tfhe::keys::{
    ensure_num_blocks_match, keygen, load_client_key, load_server_key, magnitude_bits,
    radix_capacity_bits, save_client_key, save_server_key, DEFAULT_PARAMS, MESSAGE_BITS,
    SCHEME_TFHE,
};
pub use penumbra_tfhe::ops::{Activation, Add, Argmax, Conv2d, EvalCtx, Linear, Op, Pool, Requant};
pub use penumbra_tfhe::{evaluate, evaluate_graph};

pub mod encrypt {
    pub use penumbra_tfhe::encrypt::*;
}

pub mod keys {
    pub use penumbra_tfhe::keys::*;
}

pub mod ops {
    pub use penumbra_core::ops::OpSummary;
    pub use penumbra_tfhe::ops::*;
}

pub mod ir {
    pub use penumbra_core::ir::*;
}

pub mod eval {
    pub use penumbra_core::bitwidth::{check_graph_bit_width_budget, propagate_bit_widths};
    pub use penumbra_tfhe::{check_bit_width_budget, evaluate, evaluate_graph};
}
