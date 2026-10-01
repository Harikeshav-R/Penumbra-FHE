//! Committed per-model error bounds: max |decrypted − quantized-cleartext| in integer units.
//! Exceeding one is a bug — scale, level, or polynomial degree — before it is noise
//! (`docs/BACKENDS.md`, "one invariant, two comparators").
//!
//! Calibration protocol (Phase 15, D7): derived as exactly 2.0 * p99 error over
//! the calibration split (never the test split), committed before test evaluation.
//! Source artifact: `docs/results/phase15-ckks-calibration.json`.

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=128, p99=0.497604, margin=2.0
pub const PHASE2_LOGREG: f64 = 0.9952086625708774;

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=128, p99=8.419279, margin=2.0
pub const PHASE4_CNN: f64 = 16.838558349372605;

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=128, p99=75.086494, margin=2.0
pub const PHASE5_DIGITS: f64 = 150.17298863027258;

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=128, p99=15.151181, margin=2.0
pub const PHASE5_QAT: f64 = 30.302361276093663;

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=128, p99=75.519944, margin=2.0
pub const PHASE6_ONNX: f64 = 151.03988828779268;

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=128, p99=0.000222, margin=2.0
pub const PHASE6_SKLEARN: f64 = 0.0004435212437314817;

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=60, p99=144.368803, margin=2.0
pub const PHASE7_FACES: f64 = 288.73760588410875;

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=128, p99=68.675869, margin=2.0
pub const PHASE8_BRANCH: f64 = 137.35173852570975;

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=128, p99=30.480172, margin=2.0
pub const PHASE8_BN_CNN: f64 = 60.96034317865079;

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=128, p99=190.748842, margin=2.0
pub const PHASE8_GAP_CNN: f64 = 381.4976843681731;

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=128, p99=35.459806, margin=2.0
pub const PHASE8_TANH: f64 = 70.91961172186478;

/// Derived from docs/results/phase15-ckks-calibration.json (commit 1aaa8eb), n=128, p99=28.843788, margin=2.0
pub const PHASE11_TABULAR_MLP: f64 = 57.687575156856255;

/// Property corpus (tests/fixtures/property_corpus.json): max over every accepted model and sample.
pub const PROPERTY_CORPUS: f64 = 19.0;
