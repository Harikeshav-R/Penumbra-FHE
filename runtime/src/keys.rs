//! Key generation and crypto-parameter profiles.
//!
//! The client holds the secret key (encrypt/decrypt); the server holds the public
//! evaluation/server key (enables bootstrapping) plus the plaintext model weights.
//! See `PROJECT.md` §11.
//!
//! Policy (`PROJECT.md` §12, `AGENTS.md` §7): ship a single secure default parameter
//! profile and expose at most **one** override knob. Never surface raw `tfhe-rs`
//! parameters to end users. Phase 2 exposes none — everything runs on [`DEFAULT_PARAMS`].
//!
//! ## The `integer` API and the bit-width budget
//!
//! Penumbra builds on the `tfhe-rs` high-level `integer` API: a value is a **radix**
//! ciphertext made of `num_blocks` small `shortint` blocks. Under [`DEFAULT_PARAMS`] each
//! block carries [`MESSAGE_BITS`] bits of message, so a radix integer holds
//! `num_blocks * MESSAGE_BITS` bits. `num_blocks` is therefore the single, central
//! **bit-width budget** handle (`PROJECT.md` §9): the library derives it from a model's
//! widest accumulator; it is never a user-facing crypto parameter.

use tfhe::integer::{gen_keys_radix, RadixClientKey, ServerKey};
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
use tfhe::shortint::ClassicPBSParameters;

/// The single secure default parameter profile (`PROJECT.md` §12, `AGENTS.md` §7).
///
/// `PARAM_MESSAGE_2_CARRY_2_KS_PBS` is the `tfhe-rs` default secure profile: a 2-bit
/// message space with a 2-bit carry buffer per block. Small integers are the working
/// range (`PROJECT.md` §9); we never hand-roll crypto parameters.
pub const DEFAULT_PARAMS: ClassicPBSParameters = PARAM_MESSAGE_2_CARRY_2_KS_PBS;

/// Message bits carried by a single radix block under [`DEFAULT_PARAMS`].
///
/// `message_modulus` is 2^2 = 4 for this profile, i.e. 2 usable message bits per block.
/// A radix integer of `n` blocks thus holds `n * MESSAGE_BITS` bits of value. This is the
/// arithmetic behind every bit-width-budget check (`PROJECT.md` §9, `AGENTS.md` §1.3).
pub const MESSAGE_BITS: usize = 2;

/// Generate a `(client_key, server_key)` pair over [`DEFAULT_PARAMS`] for radix integers
/// of `num_blocks` blocks.
///
/// - The **client** keeps `RadixClientKey` (the secret key) for encrypt/decrypt.
/// - The **server** holds `ServerKey` (the public evaluation key) and runs the encrypted
///   forward pass; it never sees plaintext (`PROJECT.md` §11).
///
/// `num_blocks` fixes the radix width — the central bit-width budget (`PROJECT.md` §9).
/// All ciphertexts in one model must share it so they are arithmetic-compatible.
pub fn keygen(num_blocks: usize) -> (RadixClientKey, ServerKey) {
    gen_keys_radix(DEFAULT_PARAMS, num_blocks)
}

/// Bits of value a radix integer of `num_blocks` blocks can hold under [`DEFAULT_PARAMS`].
///
/// Used by the bit-width-budget check to fail loudly *before* an accumulator would
/// overflow the chosen radix (`AGENTS.md` §1.3, §1.4).
pub fn radix_capacity_bits(num_blocks: usize) -> usize {
    num_blocks * MESSAGE_BITS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keygen + a signed radix round-trip over the default profile. Confirms the chosen
    /// `num_blocks` actually carries the values we claim (`radix_capacity_bits`).
    #[test]
    fn keygen_signed_roundtrip() {
        let num_blocks = 8; // 16-bit signed range under DEFAULT_PARAMS
        let (ck, _sk) = keygen(num_blocks);

        assert_eq!(radix_capacity_bits(num_blocks), 16);

        for v in [-1234i64, -1, 0, 1, 4321] {
            let ct = ck.encrypt_signed(v);
            let got: i64 = ck.decrypt_signed(&ct);
            assert_eq!(got, v, "signed radix round-trip must be exact");
        }
    }
}
