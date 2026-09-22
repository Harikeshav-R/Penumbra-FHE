//! Model fixture registry and loaders.

use std::path::PathBuf;

use penumbra_core::ir::Graph;
use serde_json::Value;

/// Definition of a committed model fixture in the repository.
#[derive(Debug, Clone, Copy)]
pub struct ModelFixture {
    /// Registry key used on the command line and in `PENUMBRA_BENCH_MODELS`.
    pub key: &'static str,
    /// Fixture path relative to this crate's manifest directory.
    pub path: &'static str,
    /// Human label for the report.
    pub label: &'static str,
    /// Included in the default `cargo bench` set (cheap enough to sample 10x).
    pub default_bench: bool,
}

pub const MODELS: &[ModelFixture] = &[
    ModelFixture {
        key: "phase2_logreg",
        path: "../../examples/mnist/phase2_fixture.json",
        label: "Phase-2 logreg",
        default_bench: true,
    },
    ModelFixture {
        key: "phase4_cnn",
        path: "../../examples/mnist/phase4_cnn_fixture.json",
        label: "Phase-4 CNN",
        default_bench: false,
    },
    ModelFixture {
        key: "phase5_digits",
        path: "../../examples/mnist/phase5_digits_fixture.json",
        label: "Phase-5 digits (PTQ)",
        default_bench: false,
    },
    ModelFixture {
        key: "phase5_qat",
        path: "../../examples/mnist/phase5_qat_fixture.json",
        label: "Phase-5 digits (QAT)",
        default_bench: false,
    },
    ModelFixture {
        key: "phase6_onnx",
        path: "../../examples/mnist/phase6_onnx_fixture.json",
        label: "Phase-6 ONNX",
        default_bench: false,
    },
    ModelFixture {
        key: "phase6_sklearn",
        path: "../../examples/mnist/phase6_sklearn_fixture.json",
        label: "Phase-6 sklearn",
        default_bench: false,
    },
    ModelFixture {
        key: "phase7_faces",
        path: "../../examples/faces/phase7_faces_fixture.json",
        label: "Phase-7 faces",
        default_bench: false,
    },
];

/// A loaded model fixture ready for session evaluation.
pub struct LoadedModel {
    pub fixture: &'static ModelFixture,
    pub graph: Graph,
    pub inputs: Vec<Vec<i64>>,
    pub expected_logits: Option<Vec<Vec<i64>>>,
    pub expected_labels: Option<Vec<i64>>,
}

/// Look up a model fixture by its unique key.
pub fn find(key: &str) -> Result<&'static ModelFixture, String> {
    MODELS.iter().find(|m| m.key == key).ok_or_else(|| {
        let valid = MODELS.iter().map(|m| m.key).collect::<Vec<_>>().join(", ");
        format!("unknown model '{key}'; valid models are: {valid}")
    })
}

/// Read and deserialize the fixture file from disk.
pub fn load(fixture: &'static ModelFixture) -> Result<LoadedModel, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join(fixture.path);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read fixture file {}: {e}", path.display()))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|e| format!("cannot parse JSON fixture {}: {e}", path.display()))?;

    let graph_json = value
        .get("graph")
        .ok_or_else(|| format!("fixture {} missing 'graph' field", path.display()))?;
    let graph = Graph::from_json(&graph_json.to_string())
        .map_err(|e| format!("failed to load graph from {}: {e}", path.display()))?;

    let inputs_val = value.get("test_inputs").ok_or_else(|| {
        format!("fixture {} missing 'test_inputs' field", path.display())
    })?;
    let inputs: Vec<Vec<i64>> = serde_json::from_value(inputs_val.clone())
        .map_err(|e| format!("failed to parse test_inputs in {}: {e}", path.display()))?;

    let expected_logits: Option<Vec<Vec<i64>>> = value
        .get("expected_logits")
        .and_then(|v| serde_json::from_value(v.clone()).ok());

    let expected_labels: Option<Vec<i64>> = value
        .get("expected_labels")
        .and_then(|v| serde_json::from_value(v.clone()).ok());

    Ok(LoadedModel {
        fixture,
        graph,
        inputs,
        expected_logits,
        expected_labels,
    })
}

/// Parse `PENUMBRA_BENCH_MODELS`: unset -> `default_bench` entries; `all` -> every entry;
/// otherwise a comma-separated list of keys, erroring on an unknown key.
pub fn selection_from_env() -> Result<Vec<&'static ModelFixture>, String> {
    match std::env::var("PENUMBRA_BENCH_MODELS") {
        Ok(val) => {
            let trimmed = val.trim();
            if trimmed.is_empty() {
                Ok(MODELS.iter().filter(|m| m.default_bench).collect())
            } else if trimmed.eq_ignore_ascii_case("all") {
                Ok(MODELS.iter().collect())
            } else {
                trimmed
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(find)
                    .collect()
            }
        }
        Err(std::env::VarError::NotPresent) => {
            Ok(MODELS.iter().filter(|m| m.default_bench).collect())
        }
        Err(e) => Err(format!("failed to read PENUMBRA_BENCH_MODELS: {e}")),
    }
}
