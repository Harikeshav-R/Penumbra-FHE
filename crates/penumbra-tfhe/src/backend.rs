//! The TFHE Backend implementation (Layer 1).
//!
//! Realizes the [`penumbra_core::backend::Backend`] trait against `tfhe-rs` primitives.

use penumbra_core::backend::Backend;
use penumbra_core::ir::{OpSpec, PoolMode};
use penumbra_core::ops::Op;
use tfhe::integer::{IntegerCiphertext, RadixClientKey, ServerKey, SignedRadixCiphertext};
use tfhe::shortint::Ciphertext;

use crate::ops::{Activation, Add, Argmax, Conv2d, Linear, Pool, Requant};

/// The concrete TFHE / CGGI backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct TfheBackend;

impl Backend for TfheBackend {
    type Ciphertext = SignedRadixCiphertext;
    type ServerKey = ServerKey;
    type ClientKey = RadixClientKey;

    fn name(&self) -> &'static str {
        crate::keys::SCHEME_TFHE
    }

    fn build_op(&self, spec: &OpSpec) -> Result<Box<dyn Op<Self>>, String> {
        spec.validate()?;
        match spec {
            OpSpec::Linear {
                weights,
                bias,
                weight_bits,
            } => Ok(Box::new(Linear {
                weights: weights.clone(),
                bias: bias.clone(),
                weight_bits: *weight_bits,
            })),
            OpSpec::Conv2d {
                weights,
                bias,
                weight_bits,
                in_h,
                in_w,
                in_channels,
                kernel_h,
                kernel_w,
                stride,
                padding,
            } => Ok(Box::new(Conv2d {
                weights: weights.clone(),
                bias: bias.clone(),
                weight_bits: *weight_bits,
                in_h: *in_h,
                in_w: *in_w,
                in_channels: *in_channels,
                kernel_h: *kernel_h,
                kernel_w: *kernel_w,
                stride: *stride,
                padding: *padding,
            })),
            OpSpec::Activation { lut, output_bits } => Ok(Box::new(Activation {
                lut: lut.clone(),
                output_bits: *output_bits,
            })),
            OpSpec::Argmax { threshold } => Ok(Box::new(Argmax {
                threshold: *threshold,
            })),
            OpSpec::Requant {
                shift,
                mult,
                round_bias,
                out_bits,
                clamp_lut,
                mults,
                shifts,
                round_biases,
                channel_size,
            } => Ok(Box::new(Requant {
                shift: *shift,
                mult: *mult,
                round_bias: *round_bias,
                out_bits: *out_bits,
                clamp_lut: clamp_lut.clone(),
                mults: mults.clone(),
                shifts: shifts.clone(),
                round_biases: round_biases.clone(),
                channel_size: channel_size.unwrap_or(0),
            })),
            OpSpec::Pool {
                mode,
                in_h,
                in_w,
                channels,
                pool_h,
                pool_w,
                stride,
            } => {
                let pool_mode = match mode.as_str() {
                    "avg" => PoolMode::Avg,
                    "max" => PoolMode::Max,
                    other => {
                        return Err(format!(
                            "Pool mode must be \"avg\" or \"max\"; got {other:?}"
                        ))
                    }
                };
                Ok(Box::new(Pool {
                    mode: pool_mode,
                    in_h: *in_h,
                    in_w: *in_w,
                    channels: *channels,
                    pool_h: *pool_h,
                    pool_w: *pool_w,
                    stride: *stride,
                }))
            }
            OpSpec::Add {} => Ok(Box::new(Add)),
        }
    }

    fn create_trivial_zero(&self, sk: &Self::ServerKey, num_blocks: usize) -> Self::Ciphertext {
        sk.create_trivial_zero_radix(num_blocks)
    }

    fn add(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        b: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        sk.add_parallelized(a, b)
    }

    fn scalar_mul(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext {
        sk.scalar_mul_parallelized(a, scalar)
    }

    fn scalar_add(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext {
        sk.scalar_add_parallelized(a, scalar)
    }

    fn scalar_ge(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        threshold: i64,
    ) -> Self::Ciphertext {
        let ge = sk.scalar_ge_parallelized(a, threshold);
        ge.into_radix(a.blocks().len(), sk)
    }

    fn max(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        b: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        sk.max_parallelized(a, b)
    }

    fn scalar_max(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext {
        sk.scalar_max_parallelized(a, scalar)
    }

    fn scalar_min(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext {
        sk.scalar_min_parallelized(a, scalar)
    }

    fn scalar_right_shift(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        shift: u32,
    ) -> Self::Ciphertext {
        sk.scalar_right_shift_parallelized(a, shift)
    }

    fn apply_lut(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        lut: &[u64],
        num_blocks: usize,
    ) -> Self::Ciphertext {
        let shortint_sk = sk.as_ref();
        let table = lut.to_vec();
        let lookup_table =
            shortint_sk.generate_lookup_table(move |v| *table.get(v as usize).unwrap_or(&0));
        let mapped: Ciphertext = shortint_sk.apply_lookup_table(&a.blocks()[0], &lookup_table);
        let mut blocks = Vec::with_capacity(num_blocks);
        blocks.push(mapped);
        for _ in 1..num_blocks {
            blocks.push(shortint_sk.create_trivial(0));
        }
        SignedRadixCiphertext::from(blocks)
    }

    fn keygen(&self, num_blocks: usize) -> (Self::ClientKey, Self::ServerKey) {
        crate::keys::keygen(num_blocks)
    }

    fn encrypt(&self, ck: &Self::ClientKey, input: &[i64]) -> Vec<Self::Ciphertext> {
        crate::encrypt::encrypt(ck, input)
    }

    fn decrypt_label(&self, ck: &Self::ClientKey, out: &[Self::Ciphertext]) -> i64 {
        crate::encrypt::decrypt_label(ck, out)
    }

    fn decrypt_vec(&self, ck: &Self::ClientKey, out: &[Self::Ciphertext]) -> Vec<i64> {
        crate::encrypt::decrypt_vec(ck, out)
    }

    fn serialize_cts(&self, cts: &[Self::Ciphertext]) -> Result<Vec<u8>, String> {
        crate::encrypt::serialize_cts(cts)
    }

    fn deserialize_cts(&self, bytes: &[u8]) -> Result<Vec<Self::Ciphertext>, String> {
        crate::encrypt::deserialize_cts(bytes)
    }
}
