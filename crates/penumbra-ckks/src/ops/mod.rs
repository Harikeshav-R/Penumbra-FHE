//! Op implementations (Layer 1) for the CKKS backend.

pub mod add;
pub mod argmax;
pub mod matvec;
pub mod polymap;

pub use add::eval_add;
pub use argmax::{prepare_argmax, PreparedArgmax};
pub use matvec::{
    avg_pool_matrix, conv2d_matrix, eval_matvec, linear_matrix, prepare_linear_map, PlainMatrix,
    PreparedLinearMap,
};
pub use polymap::{
    eval_per_channel_requant, eval_polymap, fit_activation, fit_per_channel_requant, fit_requant,
    PerChannelRequantMap, PolyMap,
};

use penumbra_core::backend::{CtVec, EvalCtx};
use penumbra_core::ops::Op as CoreOp;

use crate::backend::CkksBackend;

/// The stable op-eval interface specialized for the CKKS backend.
pub trait Op: CoreOp<CkksBackend> + Send + Sync {
    fn eval(
        &self,
        ctx: &EvalCtx<crate::keys::CkksServerKey>,
        inputs: &CtVec<CkksBackend>,
    ) -> CtVec<CkksBackend> {
        CoreOp::eval(self, ctx, inputs)
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        CoreOp::output_bits(self, input_bits)
    }

    fn eval_n(
        &self,
        ctx: &EvalCtx<crate::keys::CkksServerKey>,
        inputs: &[&CtVec<CkksBackend>],
    ) -> CtVec<CkksBackend> {
        CoreOp::eval_n(self, ctx, inputs)
    }

    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        CoreOp::output_bits_n(self, input_bits)
    }

    fn internal_bits_n(&self, input_bits: &[usize]) -> usize {
        CoreOp::internal_bits_n(self, input_bits)
    }
}

impl<T: CoreOp<CkksBackend> + Send + Sync + ?Sized> Op for T {}

// ─── Linear Op ───────────────────────────────────────────────────────────────

pub struct Linear {
    pub(crate) prepared: PreparedLinearMap,
    pub(crate) weight_bits: usize,
}

impl Linear {
    pub fn rotation_count(&self) -> usize {
        self.prepared.rotation_count()
    }
}

impl CoreOp<CkksBackend> for Linear {
    fn eval(
        &self,
        ctx: &EvalCtx<crate::keys::CkksServerKey>,
        inputs: &CtVec<CkksBackend>,
    ) -> CtVec<CkksBackend> {
        let out = eval_matvec(ctx.sk, &inputs[0], &self.prepared)
            .expect("eval_matvec failed in Linear::eval");
        vec![out]
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        let fan_in = self.prepared.cols.max(1);
        let fan_in_bits = (fan_in as f64).log2().ceil() as usize;
        input_bits + self.weight_bits + fan_in_bits
    }
}

// ─── Conv2d Op ──────────────────────────────────────────────────────────────

pub struct Conv2d {
    pub(crate) prepared: PreparedLinearMap,
    pub(crate) weight_bits: usize,
    pub(crate) in_channels: usize,
    pub(crate) kernel_h: usize,
    pub(crate) kernel_w: usize,
}

impl Conv2d {
    pub fn rotation_count(&self) -> usize {
        self.prepared.rotation_count()
    }
}

impl CoreOp<CkksBackend> for Conv2d {
    fn eval(
        &self,
        ctx: &EvalCtx<crate::keys::CkksServerKey>,
        inputs: &CtVec<CkksBackend>,
    ) -> CtVec<CkksBackend> {
        let out = eval_matvec(ctx.sk, &inputs[0], &self.prepared)
            .expect("eval_matvec failed in Conv2d::eval");
        vec![out]
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        let fan_in = (self.in_channels * self.kernel_h * self.kernel_w).max(1);
        let fan_in_bits = (fan_in as f64).log2().ceil() as usize;
        input_bits + self.weight_bits + fan_in_bits
    }
}

// ─── Pool(avg) Op ───────────────────────────────────────────────────────────

pub struct PoolAvg {
    pub(crate) prepared: PreparedLinearMap,
    pub(crate) pool_h: usize,
    pub(crate) pool_w: usize,
}

impl PoolAvg {
    pub fn rotation_count(&self) -> usize {
        self.prepared.rotation_count()
    }
}

impl CoreOp<CkksBackend> for PoolAvg {
    fn eval(
        &self,
        ctx: &EvalCtx<crate::keys::CkksServerKey>,
        inputs: &CtVec<CkksBackend>,
    ) -> CtVec<CkksBackend> {
        let out = eval_matvec(ctx.sk, &inputs[0], &self.prepared)
            .expect("eval_matvec failed in PoolAvg::eval");
        vec![out]
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        let fan_in = (self.pool_h * self.pool_w).max(1);
        let fan_in_bits = (fan_in as f64).log2().ceil() as usize;
        input_bits + fan_in_bits
    }
}

// ─── Requant Op ─────────────────────────────────────────────────────────────

pub enum RequantKind {
    PerTensor(PolyMap),
    PerChannel(PerChannelRequantMap),
}

pub struct Requant {
    pub(crate) kind: RequantKind,
    pub(crate) out_bits: usize,
}

impl CoreOp<CkksBackend> for Requant {
    fn eval(
        &self,
        ctx: &EvalCtx<crate::keys::CkksServerKey>,
        inputs: &CtVec<CkksBackend>,
    ) -> CtVec<CkksBackend> {
        let out = match &self.kind {
            RequantKind::PerTensor(pm) => {
                eval_polymap(ctx.sk, &inputs[0], pm).expect("eval_polymap failed in Requant::eval")
            }
            RequantKind::PerChannel(map) => eval_per_channel_requant(ctx.sk, &inputs[0], map)
                .expect("eval_per_channel_requant failed in Requant::eval"),
        };
        vec![out]
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        self.out_bits
    }
}

// ─── Activation Op ──────────────────────────────────────────────────────────

pub struct Activation {
    pub(crate) pm: PolyMap,
    pub(crate) output_bits: usize,
}

impl CoreOp<CkksBackend> for Activation {
    fn eval(
        &self,
        ctx: &EvalCtx<crate::keys::CkksServerKey>,
        inputs: &CtVec<CkksBackend>,
    ) -> CtVec<CkksBackend> {
        let out = eval_polymap(ctx.sk, &inputs[0], &self.pm)
            .expect("eval_polymap failed in Activation::eval");
        vec![out]
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        self.output_bits
    }
}

// ─── Argmax Op ──────────────────────────────────────────────────────────────

pub struct Argmax {
    pub(crate) prepared: PreparedArgmax,
}

impl CoreOp<CkksBackend> for Argmax {
    fn eval(
        &self,
        ctx: &EvalCtx<crate::keys::CkksServerKey>,
        inputs: &CtVec<CkksBackend>,
    ) -> CtVec<CkksBackend> {
        let out = argmax::eval_argmax(ctx.sk, &inputs[0], &self.prepared)
            .expect("eval_argmax failed in Argmax::eval");
        vec![out]
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        1
    }
}

// ─── Add Op ─────────────────────────────────────────────────────────────────

pub struct Add;

impl CoreOp<CkksBackend> for Add {
    fn eval(
        &self,
        _ctx: &EvalCtx<crate::keys::CkksServerKey>,
        _inputs: &CtVec<CkksBackend>,
    ) -> CtVec<CkksBackend> {
        unreachable!("Add is a 2-input op; use eval_n")
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        unreachable!("Add is a 2-input op; use output_bits_n")
    }

    fn eval_n(
        &self,
        ctx: &EvalCtx<crate::keys::CkksServerKey>,
        inputs: &[&CtVec<CkksBackend>],
    ) -> CtVec<CkksBackend> {
        assert_eq!(inputs.len(), 2, "Add expects exactly 2 inputs");
        let a = &inputs[0][0];
        let b = &inputs[1][0];
        let out = eval_add(ctx.sk, a, b).expect("eval_add failed in Add::eval_n");
        vec![out]
    }

    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        assert_eq!(input_bits.len(), 2, "Add expects exactly 2 inputs");
        input_bits.iter().max().copied().unwrap_or(0) + 1
    }
}
