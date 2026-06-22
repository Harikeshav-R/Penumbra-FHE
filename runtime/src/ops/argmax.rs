//! `Argmax` — pick the predicted class from an encrypted logit/score vector.
//!
//! Covers the classification head (`PROJECT.md` §6). Phase 2 implements the **2-class**
//! special case as a threshold on a single logit: the predicted class is `1` iff
//! `z >= threshold`, else `0`. This is the roadmap-sanctioned first cut (ROADMAP Phase 2)
//! and exploits that a 2-class softmax/sigmoid is monotone — so the label is a comparison,
//! needing no wide-domain LUT and no `Requant` (deferred to Phase 4).
//!
//! The comparison is the only nonlinearity here; it is internally a small number of
//! block-level PBSs (`scalar_ge_parallelized`). A true `> 2`-class argmax generalizes via
//! pairwise `max`/`gt` over the score vector — a later phase, against this same trait.

use super::{CtVec, EvalCtx, Op};

/// 2-class argmax: threshold a single encrypted logit into an encrypted `0`/`1` label.
pub struct Argmax {
    /// Decision threshold in the quantized accumulator domain. Class `1` iff `z >= threshold`.
    pub threshold: i64,
}

impl Op for Argmax {
    fn eval(&self, ctx: &EvalCtx, inputs: &CtVec) -> CtVec {
        assert_eq!(
            inputs.len(),
            1,
            "Phase-2 Argmax handles the 2-class single-logit case; got {} inputs",
            inputs.len()
        );
        let sk = ctx.sk;

        // z >= threshold -> encrypted boolean, then widen to a full-width label radix so
        // the eval loop and decrypt path stay homogeneous (a CtVec of one radix integer).
        let ge = sk.scalar_ge_parallelized(&inputs[0], self.threshold);
        vec![ge.into_radix(ctx.num_blocks, sk)]
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        // The output is a single class bit.
        1
    }
}
