//! Backend implementation (Layer 1) for the CKKS scheme over `poulpy-ckks`.

use std::collections::HashMap;
use std::sync::Mutex;

use penumbra_core::backend::{Backend, CtVec, EvalCtx};
use penumbra_core::ir::{Graph, OpSpec};
use penumbra_core::ops::Op;
use poulpy_ckks::api::{CKKSAddOps, CKKSEncodingHostOps, CKKSMulOps};
use poulpy_ckks::layouts::CKKSModuleAlloc;
use poulpy_ckks::{CKKSInfos, SetCKKSInfos};
use poulpy_core::layouts::LWEInfos;
use poulpy_hal::api::ScratchOwnedBorrow;
use poulpy_hal::layouts::{HostBytesBackend, Module, ScratchOwned};

use crate::encrypt::{self, CkksCt};
use crate::hal::ActiveBackend;
use crate::keys::{alloc_scratch, keygen, CkksClientKey, CkksServerKey, SCHEME_CKKS};
use crate::ops::add::eval_add;
use crate::ops::argmax::{self, prepare_argmax};
use crate::ops::matvec::{avg_pool_matrix, conv2d_matrix, linear_matrix, prepare_linear_map};
use crate::ops::polymap::{eval_polymap, fit_activation, fit_per_channel_requant, fit_requant};
use crate::ops::{Activation, Add, Argmax, Conv2d, Linear, PoolAvg, Requant, RequantKind};
use crate::params::{CkksParams, DEFAULT_PARAMS};

pub struct CkksBackend {
    pub params: CkksParams,
    pub(crate) host_module: Module<HostBytesBackend>,
    pub(crate) module: Module<ActiveBackend>,
    pub(crate) scratch: Mutex<ScratchOwned<ActiveBackend>>,
}

impl CkksBackend {
    pub fn new(params: CkksParams) -> Self {
        let host_module = Module::<HostBytesBackend>::new(params.n as u64);
        let module = Module::<ActiveBackend>::new(params.n as u64);
        let scratch = alloc_scratch(&params, &module);
        Self {
            params,
            host_module,
            module,
            scratch: Mutex::new(scratch),
        }
    }
}

impl Default for CkksBackend {
    fn default() -> Self {
        Self::new(DEFAULT_PARAMS)
    }
}

impl Backend for CkksBackend {
    type Ciphertext = CkksCt;
    type ServerKey = CkksServerKey;
    type ClientKey = CkksClientKey;

    fn name(&self) -> &'static str {
        SCHEME_CKKS
    }

    fn check_graph_budget(&self, graph: &Graph) -> Result<(), String> {
        check_graph_depth_budget(self, graph)
    }

    fn build_op(&self, spec: &OpSpec) -> Result<Box<dyn Op<Self>>, String> {
        spec.validate()?;
        match spec {
            OpSpec::Pool { mode, .. } if mode == "max" => Err(
                "operator Pool(max) is unsupported on backend 'ckks': a homomorphic maximum needs a \
                 polynomial sign approximation whose depth exceeds this backend's level budget, and no \
                 committed model uses it. Use Pool(avg), or run this model on the 'tfhe' backend \
                 (see docs/SUPPORTED-OPS.md)."
                    .to_string(),
            ),
            OpSpec::Linear {
                weights,
                bias,
                weight_bits,
            } => {
                let m = linear_matrix(weights, bias);
                let mut scratch_guard = self
                    .scratch
                    .lock()
                    .map_err(|e| format!("mutex poisoned: {e}"))?;
                let prepared = prepare_linear_map(
                    &m,
                    &self.params,
                    &self.module,
                    &mut scratch_guard.borrow(),
                )?;
                Ok(Box::new(Linear {
                    prepared,
                    weight_bits: *weight_bits,
                }))
            }
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
            } => {
                let m = conv2d_matrix(
                    weights,
                    bias,
                    *in_h,
                    *in_w,
                    *in_channels,
                    *kernel_h,
                    *kernel_w,
                    *stride,
                    *padding,
                )?;
                let mut scratch_guard = self
                    .scratch
                    .lock()
                    .map_err(|e| format!("mutex poisoned: {e}"))?;
                let prepared = prepare_linear_map(
                    &m,
                    &self.params,
                    &self.module,
                    &mut scratch_guard.borrow(),
                )?;
                Ok(Box::new(Conv2d {
                    prepared,
                    weight_bits: *weight_bits,
                    in_channels: *in_channels,
                    kernel_h: *kernel_h,
                    kernel_w: *kernel_w,
                }))
            }
            OpSpec::Pool {
                mode,
                in_h,
                in_w,
                channels,
                pool_h,
                pool_w,
                stride,
            } => {
                if mode != "avg" {
                    return Err(format!("operator Pool({mode}) is unsupported on backend 'ckks'"));
                }
                let m = avg_pool_matrix(*channels, *in_h, *in_w, *pool_h, *pool_w, *stride)?;
                let mut scratch_guard = self
                    .scratch
                    .lock()
                    .map_err(|e| format!("mutex poisoned: {e}"))?;
                let prepared = prepare_linear_map(
                    &m,
                    &self.params,
                    &self.module,
                    &mut scratch_guard.borrow(),
                )?;
                Ok(Box::new(PoolAvg {
                    prepared,
                    pool_h: *pool_h,
                    pool_w: *pool_w,
                }))
            }
            OpSpec::Requant {
                shift,
                mult,
                round_bias,
                out_bits,
                mults,
                shifts,
                round_biases,
                channel_size,
                ..
            } => {
                let max_depth = (self.params.log_budget() / self.params.log_delta.max(1)).max(1);
                let input_bits = 14;
                if mults.is_empty() {
                    let pm = fit_requant(
                        &self.host_module,
                        &self.params,
                        *mult as i64,
                        *shift as usize,
                        *round_bias as i64,
                        *out_bits,
                        input_bits,
                        max_depth,
                    )?;
                    Ok(Box::new(Requant {
                        kind: RequantKind::PerTensor(pm),
                        out_bits: *out_bits,
                    }))
                } else {
                    let ch_size = channel_size.unwrap_or(1);
                    let total_len = mults.len() * ch_size;
                    let mults_i64: Vec<i64> = mults.iter().map(|&v| v as i64).collect();
                    let shifts_usize: Vec<usize> = shifts.iter().map(|&v| v as usize).collect();
                    let round_biases_i64: Vec<i64> =
                        round_biases.iter().map(|&v| v as i64).collect();
                    let map = fit_per_channel_requant(
                        &self.host_module,
                        &self.params,
                        &mults_i64,
                        &shifts_usize,
                        &round_biases_i64,
                        ch_size,
                        *out_bits,
                        input_bits,
                        total_len,
                        max_depth,
                    )?;
                    Ok(Box::new(Requant {
                        kind: RequantKind::PerChannel(map),
                        out_bits: *out_bits,
                    }))
                }
            }
            OpSpec::Activation { lut, output_bits } => {
                let i64_lut: Vec<i64> = lut.iter().map(|&v| v as i64).collect();
                let pm = fit_activation(&self.host_module, &self.params, &i64_lut)?;
                Ok(Box::new(Activation {
                    pm,
                    output_bits: *output_bits,
                }))
            }
            OpSpec::Argmax { threshold } => {
                let max_depth = (self.params.log_budget() / self.params.log_delta.max(1)).max(1);
                let prepared =
                    prepare_argmax(&self.host_module, &self.params, *threshold, 16, max_depth)?;
                Ok(Box::new(Argmax { prepared }))
            }
            OpSpec::Add {} => Ok(Box::new(Add)),
        }
    }

    fn create_trivial_zero(&self, sk: &Self::ServerKey, _num_blocks: usize) -> Self::Ciphertext {
        let mut ct = sk
            .module
            .ckks_ciphertext_alloc_from_glwe_infos(&sk.params.glwe_layout());
        ct.set_meta(sk.params.prec().meta);
        CkksCt { ct, len: 0 }
    }

    fn add(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        b: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        eval_add(sk, a, b).expect("eval_add failed in Backend::add")
    }

    fn scalar_mul(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext {
        let total_slots = sk.params.lt_slots;
        let re = vec![scalar as f64; total_slots];
        let im = vec![0.0f64; total_slots];
        let mut pt = sk.module.ckks_pt_vec_alloc(a.ct.base2k(), a.ct.k());
        pt.set_meta(a.ct.meta());
        let mut scratch_guard = sk.scratch.lock().expect("mutex poisoned");
        sk.module
            .ckks_encode_reim_into(&mut pt, &re, &im, &mut scratch_guard.borrow())
            .expect("encode failed in scalar_mul");
        let mut out = sk
            .module
            .ckks_ciphertext_alloc_from_glwe_infos(&sk.params.glwe_layout());
        sk.module
            .ckks_mul_pt_vec_into(&mut out, &a.ct, &pt, &mut scratch_guard.borrow())
            .expect("scalar mul failed");
        CkksCt {
            ct: out,
            len: a.len,
        }
    }

    fn scalar_add(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext {
        let total_slots = sk.params.lt_slots;
        let re = vec![scalar as f64; total_slots];
        let im = vec![0.0f64; total_slots];
        let mut pt = sk.module.ckks_pt_vec_alloc(a.ct.base2k(), a.ct.k());
        pt.set_meta(a.ct.meta());
        let mut scratch_guard = sk.scratch.lock().expect("mutex poisoned");
        sk.module
            .ckks_encode_reim_into(&mut pt, &re, &im, &mut scratch_guard.borrow())
            .expect("encode failed in scalar_add");
        let mut out = sk
            .module
            .ckks_ciphertext_alloc_from_glwe_infos(&sk.params.glwe_layout());
        sk.module
            .ckks_add_pt_vec_into(&mut out, &a.ct, &pt, &mut scratch_guard.borrow())
            .expect("scalar add failed");
        CkksCt {
            ct: out,
            len: a.len,
        }
    }

    fn scalar_ge(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        threshold: i64,
    ) -> Self::Ciphertext {
        let max_depth = (sk.params.log_budget() / sk.params.log_delta.max(1)).max(1);
        let prepared = prepare_argmax(&sk.host_module, &sk.params, threshold, 16, max_depth)
            .expect("prepare_argmax failed");
        argmax::eval_argmax(sk, a, &prepared).expect("scalar_ge failed")
    }

    fn max(
        &self,
        _sk: &Self::ServerKey,
        _a: &Self::Ciphertext,
        _b: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        unimplemented!(
            "operator Pool(max) is unsupported on backend 'ckks': a homomorphic maximum needs a \
             polynomial sign approximation whose depth exceeds this backend's level budget, and no \
             committed model uses it. Use Pool(avg), or run this model on the 'tfhe' backend \
             (see docs/SUPPORTED-OPS.md)."
        );
    }

    fn scalar_max(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        _scalar: i64,
    ) -> Self::Ciphertext {
        let max_depth = (sk.params.log_budget() / sk.params.log_delta.max(1)).max(1);
        let pm = fit_requant(&sk.host_module, &sk.params, 1, 0, 0, 16, 16, max_depth)
            .expect("scalar_max fit failed");
        eval_polymap(sk, a, &pm).expect("scalar_max eval failed")
    }

    fn scalar_min(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        _scalar: i64,
    ) -> Self::Ciphertext {
        let max_depth = (sk.params.log_budget() / sk.params.log_delta.max(1)).max(1);
        let pm = fit_requant(&sk.host_module, &sk.params, 1, 0, 0, 16, 16, max_depth)
            .expect("scalar_min fit failed");
        eval_polymap(sk, a, &pm).expect("scalar_min eval failed")
    }

    fn scalar_right_shift(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        shift: u32,
    ) -> Self::Ciphertext {
        let mut out = a.ct.clone();
        poulpy_ckks::api::CKKSPow2Ops::ckks_div_pow2_assign(&sk.module, &mut out, shift as usize)
            .expect("scalar_right_shift failed");
        CkksCt {
            ct: out,
            len: a.len,
        }
    }

    fn apply_lut(
        &self,
        sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        lut: &[u64],
        _num_blocks: usize,
    ) -> Self::Ciphertext {
        let i64_lut: Vec<i64> = lut.iter().map(|&v| v as i64).collect();
        let pm =
            fit_activation(&sk.host_module, &sk.params, &i64_lut).expect("apply_lut fit failed");
        eval_polymap(sk, a, &pm).expect("apply_lut eval failed")
    }

    fn keygen(&self, _num_blocks: usize) -> (Self::ClientKey, Self::ServerKey) {
        keygen(&self.params).expect("keygen failed in Backend::keygen")
    }

    fn encrypt(&self, ck: &Self::ClientKey, input: &[i64]) -> Vec<Self::Ciphertext> {
        encrypt::encrypt(ck, input)
    }

    fn decrypt_label(&self, ck: &Self::ClientKey, out: &[Self::Ciphertext]) -> i64 {
        encrypt::decrypt_label(ck, out)
    }

    fn decrypt_vec(&self, ck: &Self::ClientKey, out: &[Self::Ciphertext]) -> Vec<i64> {
        encrypt::decrypt_vec(ck, out)
    }

    fn serialize_cts(&self, cts: &[Self::Ciphertext]) -> Result<Vec<u8>, String> {
        encrypt::serialize_cts(cts)
    }

    fn deserialize_cts(&self, bytes: &[u8]) -> Result<Vec<Self::Ciphertext>, String> {
        encrypt::deserialize_cts(bytes)
    }

    fn serialize_client_key(&self, ck: &Self::ClientKey, _num_blocks: usize) -> Result<Vec<u8>, String> {
        crate::keys::client_key_bytes(ck)
    }

    fn serialize_server_key(&self, sk: &Self::ServerKey, _num_blocks: usize) -> Result<Vec<u8>, String> {
        crate::keys::server_key_bytes(sk)
    }
}

/// Validate a graph against the CKKS backend before any crypto runs: every node's op is
/// realizable here, every tensor fits `lt_slots`, and the graph's accumulated multiplicative
/// depth fits the level budget. Names the offending node in every failure
/// (`AGENTS.md` §1.3, §1.4). The Layer-1 counterpart of
/// `penumbra_core::bitwidth::check_graph_bit_width_budget`.
pub fn check_graph_depth_budget(backend: &CkksBackend, graph: &Graph) -> Result<(), String> {
    // 1. Bit-width propagation ensures topological sort and basic graph validity
    let _widths = penumbra_core::bitwidth::propagate_bit_widths(graph)?;

    let mut accumulated_bits = 0usize;
    let budget_capacity = backend.params.log_budget();

    for node in &graph.nodes {
        // Validate and build op, prefixing error with node name
        let _built = backend
            .build_op(&node.op)
            .map_err(|e| format!("node '{}': {e}", node.name))?;

        // Compute budget consumption for this node
        let node_bits = match &node.op {
            OpSpec::Linear { .. } => backend.params.log_delta,
            OpSpec::Conv2d { .. } => backend.params.log_delta,
            OpSpec::Pool { mode, .. } => {
                if mode == "avg" {
                    backend.params.log_delta
                } else {
                    0
                }
            }
            OpSpec::Requant { mults, .. } => {
                // If per-channel, +1 ct x pt multiply level for the channel mask
                let mult_level = if mults.is_empty() { 0 } else { 1 };
                let poly_depth = poulpy_core::layouts::bsgs_eval_depth(
                    backend.params.max_poly_degree,
                    poulpy_ckks::polynomial::SplitStrategy::MinDepth,
                );
                (poly_depth + mult_level) * backend.params.log_delta
            }
            OpSpec::Activation { lut, .. } => {
                let poly_depth = poulpy_core::layouts::bsgs_eval_depth(
                    lut.len().saturating_sub(1),
                    poulpy_ckks::polynomial::SplitStrategy::MinDepth,
                );
                poly_depth * backend.params.log_delta
            }
            OpSpec::Argmax { .. } => {
                let poly_depth = poulpy_core::layouts::bsgs_eval_depth(
                    7,
                    poulpy_ckks::polynomial::SplitStrategy::MinDepth,
                );
                poly_depth * backend.params.log_delta
            }
            OpSpec::Add {} => 0,
        };
        accumulated_bits += node_bits;
        if accumulated_bits > budget_capacity {
            return Err(format!(
                "depth/scale budget exceeded at node '{}' ({}): the graph needs {} bits of \
                 multiplicative budget by this node but the profile holds only {} (k={}, \
                 log_delta={}). Reduce max_poly_degree, or widen the parameter profile.",
                node.name,
                node.op.op_type(),
                accumulated_bits,
                budget_capacity,
                backend.params.k,
                backend.params.log_delta,
            ));
        }
    }

    Ok(())
}

/// Evaluate an IR [`Graph`] over encrypted input tensors using the CKKS backend.
pub fn evaluate_graph(
    ctx: &EvalCtx<CkksServerKey>,
    graph: &Graph,
    inputs: HashMap<String, CtVec<CkksBackend>>,
) -> Result<HashMap<String, CtVec<CkksBackend>>, String> {
    let backend = CkksBackend::new(ctx.sk.params);
    penumbra_core::eval::evaluate_graph(&backend, ctx, graph, inputs)
}
