//! `penumbra-mnist-scale-probe` — Scale feasibility probe for raw 28x28 MNIST under TFHE.
//!
//! Evaluates the 784-input probe CNN under TFHE (ROADMAP Phase 16, Task 4).
//! Times only the server `Session::eval` call, asserts exactness against the integer reference,
//! records CKKS capacity exclusion evidence, and enforces provenance guards.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use penumbra_bench::protocol::{canonical_graph_hash, PaperReportMeta};
use penumbra_bench::report::NodeReport;
use penumbra_bench::session::Session;
use penumbra_bench::tfhe_backend;
use penumbra_core::ir::Graph;
use penumbra_core::profile::GraphProfile;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug)]
struct CliArgs {
    fixture_path: PathBuf,
    threads: Option<usize>,
    out_path: PathBuf,
}

fn parse_args() -> Result<CliArgs, String> {
    let mut args = std::env::args().skip(1);
    let mut fixture_path: Option<PathBuf> = None;
    let mut threads: Option<usize> = None;
    let mut out_path: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fixture" => {
                let val = args.next().ok_or("--fixture requires a path argument")?;
                fixture_path = Some(PathBuf::from(val));
            }
            "--threads" => {
                let val = args
                    .next()
                    .ok_or("--threads requires an integer argument")?;
                let t: usize = val
                    .parse()
                    .map_err(|e| format!("invalid --threads value '{val}': {e}"))?;
                if t == 0 {
                    return Err("--threads must be positive (got 0)".to_string());
                }
                threads = Some(t);
            }
            "--out" => {
                let val = args.next().ok_or("--out requires a path argument")?;
                out_path = Some(PathBuf::from(val));
            }
            "-h" | "--help" => {
                println!(
                    "Usage: penumbra-mnist-scale-probe --fixture <PATH> [--threads <N>] --out <PATH>"
                );
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown argument '{other}'"));
            }
        }
    }

    Ok(CliArgs {
        fixture_path: fixture_path.ok_or("missing required argument --fixture")?,
        threads,
        out_path: out_path.ok_or("missing required argument --out")?,
    })
}

#[derive(Debug, Deserialize)]
struct MinimalProbeFixture {
    graph: Graph,
    test_inputs: Vec<Vec<i64>>,
    expected_logits: Vec<Vec<i64>>,
    #[serde(default)]
    sample_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProbeProfileSummary {
    pub total_secs: f64,
    pub counter_totals: BTreeMap<String, u64>,
    pub measured_totals: BTreeMap<String, u64>,
    pub nodes: Vec<NodeReport>,
}

impl From<&GraphProfile> for ProbeProfileSummary {
    fn from(p: &GraphProfile) -> Self {
        let counter_totals: BTreeMap<String, u64> = p
            .counter_totals()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let measured_totals: BTreeMap<String, u64> = p
            .measured_totals()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        let nodes = p.nodes.iter().map(NodeReport::from).collect();

        Self {
            total_secs: p.total.as_secs_f64(),
            counter_totals,
            measured_totals,
            nodes,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MnistScaleProbeResult {
    pub schema_version: u32,
    pub probe: String,
    pub backend: String,
    pub machine_model: String,
    pub os_product_version: String,
    pub kernel_version: String,
    pub build_commit: String,
    pub runtime_commit: String,
    pub dirty: bool,
    pub rustc_version: String,
    pub requested_threads: usize,
    pub actual_threads: usize,
    pub fixture_path: String,
    pub fixture_sha256: String,
    pub graph_sha256: String,
    pub sample_id: String,
    pub input_length: usize,
    pub ckks_linear_capacity: usize,
    pub ckks_capacity_error: String,
    pub ckks_capacity_source: String,
    pub controlled_suite_exclusion_reason: String,
    pub elapsed_secs: f64,
    pub within_600_secs: bool,
    pub expected_logits: Vec<i64>,
    pub actual_logits: Vec<i64>,
    pub logits_exact_match: bool,
    pub profile: ProbeProfileSummary,
}

fn compute_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(feature = "ckks")]
fn check_ckks_capacity() -> Result<(String, usize, String), String> {
    let params = &penumbra_ckks::DEFAULT_PARAMS;
    let slot_capacity = params.lt_slots;
    let (ck, _sk) = penumbra_ckks::keygen(params)
        .map_err(|e| format!("CKKS keygen failed during capacity probe: {e}"))?;
    let dummy_floats = vec![0.0f64; 784];
    match penumbra_ckks::keys::encrypt_raw(&ck, &dummy_floats) {
        Ok(_) => Err(
            "unexpected success: 784 floats succeeded in encrypt_raw despite slot capacity limit"
                .to_string(),
        ),
        Err(e) => Ok((
            e,
            slot_capacity,
            "crates/penumbra-ckks/src/keys.rs:202-209".to_string(),
        )),
    }
}

#[cfg(not(feature = "ckks"))]
fn check_ckks_capacity() -> Result<(String, usize, String), String> {
    Err(
        "penumbra-mnist-scale-probe must be built with --features penumbra-bench/ckks to observe real CKKS capacity rejection"
            .to_string(),
    )
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    // Rayon configuration
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .map_err(|e| format!("failed to initialize Rayon global thread pool: {e}"))?;
    }
    let requested_threads = args.threads.unwrap_or(0);
    let actual_threads = rayon::current_num_threads();

    if let Some(t) = args.threads {
        if actual_threads != t {
            return Err(format!(
                "requested threads ({t}) != actual Rayon threads ({actual_threads})"
            ));
        }
    }

    // Provenance verification
    let meta = PaperReportMeta::capture("probe", requested_threads)?;

    // Early CKKS capacity preflight: fails before any expensive TFHE keygen/eval
    let (ckks_capacity_error, ckks_linear_capacity, ckks_capacity_source) = check_ckks_capacity()?;

    // Read fixture file and compute raw file SHA-256
    let fixture_raw = fs::read(&args.fixture_path).map_err(|e| {
        format!(
            "failed to read fixture {}: {e}",
            args.fixture_path.display()
        )
    })?;
    let fixture_sha256 = compute_sha256(&fixture_raw);

    // Strongly-typed minimal fixture deserialization (no serde_json::Value clones)
    let fixture: MinimalProbeFixture = serde_json::from_slice(&fixture_raw)
        .map_err(|e| format!("failed to deserialize fixture JSON: {e}"))?;

    // Canonical graph SHA-256 using the canonical protocol serializer
    let graph_sha256 = canonical_graph_hash(&fixture.graph)?;

    if fixture.test_inputs.is_empty() {
        return Err("fixture test_inputs is empty".to_string());
    }
    if fixture.expected_logits.is_empty() {
        return Err("fixture expected_logits is empty".to_string());
    }

    let sample_input = &fixture.test_inputs[0];
    if sample_input.len() != 784 {
        return Err(format!(
            "expected sample input length 784, got {}",
            sample_input.len()
        ));
    }

    let expected_logits = &fixture.expected_logits[0];
    if expected_logits.len() != 10 {
        return Err(format!(
            "expected 10 reference logits, got {}",
            expected_logits.len()
        ));
    }

    let sample_id = fixture
        .sample_ids
        .first()
        .cloned()
        .unwrap_or_else(|| "test_0".to_string());

    // Set up TFHE Session
    println!("Initializing TFHE session for 28x28 MNIST scale probe...");
    let backend = tfhe_backend();
    let session = Session::new(backend, &fixture.graph)?;

    // Encrypt sample 0 outside timed eval
    println!("Encrypting 784-element sample '{sample_id}'...");
    let ct_input = session.encrypt(sample_input);

    // Timed server eval ONLY
    println!("Evaluating graph encrypted under TFHE (timing server eval)...");
    let started = Instant::now();
    let (ct_output, profile) = session.eval(&fixture.graph, &ct_input)?;
    let elapsed_secs = started.elapsed().as_secs_f64();
    println!("Server evaluation completed in {elapsed_secs:.3} s");

    // Decrypt output logits
    let actual_logits = session.decrypt(&ct_output);

    // Verify bit-for-bit exactness against reference
    if &actual_logits != expected_logits {
        return Err(format!(
            "MNIST28 exactness violation: actual {:?} != expected {:?}",
            actual_logits, expected_logits
        ));
    }
    println!("Bit-for-bit exactness verified: actual logits match expected reference.");

    let result = MnistScaleProbeResult {
        schema_version: 1,
        probe: "mnist28_scale_probe".to_string(),
        backend: "tfhe".to_string(),
        machine_model: meta.machine_model,
        os_product_version: meta.os_product_version,
        kernel_version: meta.kernel_version,
        build_commit: meta.build_commit,
        runtime_commit: meta.runtime_commit,
        dirty: meta.dirty,
        rustc_version: meta.rustc_version,
        requested_threads: meta.requested_threads,
        actual_threads: meta.actual_threads,
        fixture_path: args.fixture_path.display().to_string(),
        fixture_sha256,
        graph_sha256,
        sample_id,
        input_length: 784,
        ckks_linear_capacity,
        ckks_capacity_error,
        ckks_capacity_source,
        controlled_suite_exclusion_reason: concat!(
            "Excluded from controlled comparison suite under approved D18 capacity amendment: ",
            "raw input dimension 784 exceeds CKKS linear-transform slot capacity 256; ",
            "retained as TFHE scale feasibility probe."
        )
        .to_string(),
        elapsed_secs,
        within_600_secs: elapsed_secs <= 600.0,
        expected_logits: expected_logits.clone(),
        actual_logits,
        logits_exact_match: true,
        profile: ProbeProfileSummary::from(&profile),
    };

    if let Some(parent) = args.out_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create output directory {}: {e}",
                parent.display()
            )
        })?;
    }
    let json_bytes = serde_json::to_vec_pretty(&result)
        .map_err(|e| format!("failed to serialize probe result: {e}"))?;
    fs::write(&args.out_path, json_bytes)
        .map_err(|e| format!("failed to write output {}: {e}", args.out_path.display()))?;

    println!("Wrote scale probe result to {}", args.out_path.display());
    Ok(())
}
