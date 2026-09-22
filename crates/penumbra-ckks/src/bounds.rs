//! Committed per-model error bounds: max |decrypted − quantized-cleartext| in integer units.
//! Exceeding one is a bug — scale, level, or polynomial degree — before it is noise
//! (`docs/BACKENDS.md`, "one invariant, two comparators"). The constants are set at 1.5x
//! the max error measured over all committed samples of each fixture by
//! `cargo run --example calibrate`, not from a single sample.

pub const PHASE2_LOGREG: f64 = 0.5;
pub const PHASE4_CNN: f64 = 10.0;
pub const PHASE5_DIGITS: f64 = 60.0;
pub const PHASE5_QAT: f64 = 50.0;
pub const PHASE6_ONNX: f64 = 60.0;
pub const PHASE6_SKLEARN: f64 = 1e-3;
pub const PHASE7_FACES: f64 = 120.0;
