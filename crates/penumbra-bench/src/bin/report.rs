//! `penumbra-bench-report` — CLI tool emitting benchmark and comparison reports.
//!
//! Generates latency, op breakdown, and resource overhead tables across models and backends.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use penumbra_bench::models::{find, load, ModelFixture, MODELS};
use penumbra_bench::report::{run_model, to_json, to_markdown, ModelRun};
use penumbra_bench::{available_backends, tfhe_backend};

#[cfg(feature = "ckks")]
use penumbra_bench::ckks_backend;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

enum OutputFormat {
    Markdown,
    Json,
}

struct CliArgs {
    models: Vec<&'static ModelFixture>,
    backends: Vec<String>,
    samples: usize,
    format: OutputFormat,
    out_path: Option<PathBuf>,
}

fn parse_args() -> Result<CliArgs, String> {
    let mut args = std::env::args().skip(1);
    let mut models_arg: Option<String> = None;
    let mut backends_arg: Option<String> = None;
    let mut samples: usize = 1;
    let mut format = OutputFormat::Markdown;
    let mut out_path: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--models" => {
                models_arg = Some(args.next().ok_or_else(|| {
                    "--models requires an argument (e.g. 'all' or 'phase2_logreg')".to_string()
                })?);
            }
            "--backends" => {
                backends_arg = Some(args.next().ok_or_else(|| {
                    "--backends requires an argument (e.g. 'tfhe' or 'tfhe,ckks')".to_string()
                })?);
            }
            "--samples" => {
                let s = args
                    .next()
                    .ok_or_else(|| "--samples requires an integer argument".to_string())?;
                samples = s
                    .parse()
                    .map_err(|e| format!("invalid --samples value '{s}': {e}"))?;
                if samples == 0 {
                    return Err("--samples must be >= 1".to_string());
                }
            }
            "--format" => {
                let f = args.next().ok_or_else(|| {
                    "--format requires an argument ('markdown' or 'json')".to_string()
                })?;
                match f.to_lowercase().as_str() {
                    "markdown" | "md" => format = OutputFormat::Markdown,
                    "json" => format = OutputFormat::Json,
                    other => {
                        return Err(format!(
                            "unknown format '{other}'; valid formats are 'markdown' or 'json'"
                        ));
                    }
                }
            }
            "--out" => {
                out_path =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--out requires a file path argument".to_string()
                    })?));
            }
            "-h" | "--help" => {
                let avail = available_backends().join(", ");
                let valid_models = MODELS.iter().map(|m| m.key).collect::<Vec<_>>().join(", ");
                println!(
                    "usage: penumbra-bench-report [OPTIONS]\n\n\
                     Options:\n  \
                       --models <all|key,...>   Models to run (default: all)\n                           \
                                                Valid keys: {valid_models}\n  \
                       --backends <name,...>    Backends to run (default: {avail})\n                           \
                                                Available: {avail}\n  \
                       --samples <N>            Number of samples to evaluate per model (default: 1)\n  \
                       --format <markdown|json> Output format (default: markdown)\n  \
                       --out <PATH>             Write output to PATH instead of stdout\n  \
                       -h, --help               Show this help message"
                );
                std::process::exit(0);
            }
            unknown => {
                return Err(format!(
                    "unknown option '{unknown}'. Run with --help to see valid options."
                ));
            }
        }
    }

    let models = match models_arg {
        None => MODELS.iter().collect(),
        Some(m) if m.eq_ignore_ascii_case("all") => MODELS.iter().collect(),
        Some(m) => {
            let mut list = Vec::new();
            for key in m.split(',') {
                let k = key.trim();
                if !k.is_empty() {
                    list.push(find(k)?);
                }
            }
            if list.is_empty() {
                return Err("no valid models specified in --models".to_string());
            }
            list
        }
    };

    let avail = available_backends();
    let backends = match backends_arg {
        None => avail.iter().map(|&s| s.to_string()).collect(),
        Some(b) => {
            let mut list = Vec::new();
            for raw in b.split(',') {
                let name = raw.trim().to_lowercase();
                if name.is_empty() {
                    continue;
                }
                if name == "ckks" && !avail.contains(&"ckks") {
                    return Err(
                        "backend 'ckks' is not compiled into this binary; rebuild with --features ckks on a nightly toolchain".to_string(),
                    );
                }
                if !avail.contains(&name.as_str()) {
                    let avail_str = avail.join(", ");
                    return Err(format!(
                        "unknown backend '{name}'; available backends in this build: {avail_str}"
                    ));
                }
                list.push(name);
            }
            if list.is_empty() {
                return Err("no valid backends specified in --backends".to_string());
            }
            list
        }
    };

    Ok(CliArgs {
        models,
        backends,
        samples,
        format,
        out_path,
    })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let mut runs: Vec<ModelRun> = Vec::new();

    for fixture in &args.models {
        let loaded = load(fixture)?;
        for backend_name in &args.backends {
            eprintln!(
                "Running model '{}' ({}) on backend '{}' ({} sample(s))...",
                fixture.key, fixture.label, backend_name, args.samples
            );
            match backend_name.as_str() {
                "tfhe" => {
                    let run = run_model(tfhe_backend(), &loaded, args.samples)?;
                    runs.push(run);
                }
                #[cfg(feature = "ckks")]
                "ckks" => {
                    let run = run_model(ckks_backend(), &loaded, args.samples)?;
                    runs.push(run);
                }
                _ => unreachable!(),
            }
        }
    }

    let output = match args.format {
        OutputFormat::Markdown => to_markdown(&runs),
        OutputFormat::Json => to_json(&runs)?,
    };

    if let Some(path) = args.out_path {
        fs::write(&path, output)
            .map_err(|e| format!("cannot write report to {}: {e}", path.display()))?;
        eprintln!("Report written to {}", path.display());
    } else {
        println!("{output}");
    }

    Ok(())
}
