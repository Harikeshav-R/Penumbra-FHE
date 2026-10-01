//! Criterion latency benchmarks across backend x model (ROADMAP Phase 12.3).

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
use penumbra_bench::latency::{backend_selection_from_env, bench_session};
use penumbra_bench::models::{load, selection_from_env};
use penumbra_bench::session::Session;

fn bench_inference(c: &mut Criterion) {
    let fixtures = selection_from_env().unwrap_or_else(|e| panic!("invalid model selection: {e}"));
    let backends = backend_selection_from_env();

    for fixture in fixtures {
        let model = load(fixture)
            .unwrap_or_else(|e| panic!("failed to load fixture '{}': {e}", fixture.key));
        if model.inputs.is_empty() {
            panic!("fixture '{}' has no test inputs", fixture.key);
        }
        let input = &model.inputs[0];

        let mut group = c.benchmark_group(format!("inference/{}", fixture.key));
        group.sample_size(10);
        group.sampling_mode(SamplingMode::Flat);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(60));

        for backend_name in &backends {
            match backend_name.as_str() {
                "tfhe" => {
                    let backend = penumbra_bench::tfhe_backend();
                    let session = Session::new(backend, &model.graph).unwrap_or_else(|e| {
                        panic!("TFHE session creation failed for '{}': {e}", fixture.key)
                    });
                    let input_cts = session.encrypt(input);
                    bench_session(
                        &mut group,
                        "tfhe",
                        fixture.key,
                        &session,
                        &model.graph,
                        &input_cts,
                    );
                }
                #[cfg(feature = "ckks")]
                "ckks" => {
                    if fixture.key == "phase8_trees" || fixture.key == "phase8_xgb" {
                        continue;
                    }
                    let backend = if fixture.key == "phase8_branch" {
                        let p = penumbra_ckks::params::DEFAULT_PARAMS
                            .with_max_poly_degree(3)
                            .unwrap();
                        penumbra_ckks::CkksBackend::new(p)
                    } else {
                        penumbra_bench::ckks_backend()
                    };
                    let session = Session::new(backend, &model.graph).unwrap_or_else(|e| {
                        panic!("CKKS session creation failed for '{}': {e}", fixture.key)
                    });
                    let input_cts = session.encrypt(input);
                    bench_session(
                        &mut group,
                        "ckks",
                        fixture.key,
                        &session,
                        &model.graph,
                        &input_cts,
                    );
                }
                other => panic!("unknown backend: {other}"),
            }
        }

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(60));
    targets = bench_inference
}
criterion_main!(benches);
