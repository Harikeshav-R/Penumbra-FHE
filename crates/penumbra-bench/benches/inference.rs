//! Criterion latency benchmarks across backend x model (ROADMAP Phase 12.3).

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use penumbra_bench::models::{load, selection_from_env};
use penumbra_bench::session::Session;

fn bench_inference(c: &mut Criterion) {
    let fixtures = selection_from_env().unwrap_or_else(|e| panic!("invalid model selection: {e}"));

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

        // TFHE benchmark arm
        {
            let backend = penumbra_bench::tfhe_backend();
            let session = Session::new(backend, &model.graph).unwrap_or_else(|e| {
                panic!("TFHE session creation failed for '{}': {e}", fixture.key)
            });
            let input_cts = session.encrypt(input);
            let graph = &model.graph;
            group.bench_function(BenchmarkId::new("tfhe", fixture.key), |b| {
                b.iter(|| {
                    let res = session.eval(graph, &input_cts).expect("TFHE eval failed");
                    black_box(res);
                });
            });
        }

        // CKKS benchmark arm
        #[cfg(feature = "ckks")]
        {
            let backend = penumbra_bench::ckks_backend();
            let session = Session::new(backend, &model.graph).unwrap_or_else(|e| {
                panic!("CKKS session creation failed for '{}': {e}", fixture.key)
            });
            let input_cts = session.encrypt(input);
            let graph = &model.graph;
            group.bench_function(BenchmarkId::new("ckks", fixture.key), |b| {
                b.iter(|| {
                    let res = session.eval(graph, &input_cts).expect("CKKS eval failed");
                    black_box(res);
                });
            });
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
