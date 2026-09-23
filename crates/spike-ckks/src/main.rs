use anyhow::Result;
use poulpy_ckks::approximation::minimax;
use poulpy_ckks::polynomial::Parity;
use spike_ckks::{SpikeContext, compute_error_breakdown};
use std::time::Instant;

fn main() -> Result<()> {
    println!("===============================================================================");
    println!("  Penumbra-FHE: Phase 12.0 CKKS Standalone Benchmark Spike");
    println!("  poulpy-ckks 0.8.3, Hardware Acceleration: FFT64 (Release Profile)");
    println!("===============================================================================\n");

    let mut ctx = SpikeContext::new()?;
    let slots = ctx.slots;
    println!("Configuration:");
    println!("  Total Slots per Ciphertext: {slots}");
    println!("  Ring Dimension: {}", slots * 2);

    // 1. Key generation benchmark
    println!("\n[1/5] Benchmarking Key Generation (Secret Key + Tensor/Relinearization Key)...");
    let t0 = Instant::now();
    let (sk, tsk) = ctx.keygen([42u8; 32]);
    let keygen_dur = t0.elapsed();
    println!("  -> Keygen Duration: {:?}", keygen_dur);

    // 2. Encryption benchmark
    println!("\n[2/5] Benchmarking Encryption (Vector of {slots} values in [-1, 1])...");
    let input_vec: Vec<f64> = (0..slots)
        .map(|i| 2.0 * (i as f64) / ((slots - 1) as f64) - 1.0)
        .collect();
    let t0 = Instant::now();
    let ct = ctx.encrypt(&input_vec, &sk);
    let encrypt_dur = t0.elapsed();
    println!("  -> Encryption Duration: {:?}", encrypt_dur);

    // 3. Decryption benchmark
    println!("\n[3/5] Benchmarking Decryption & Decoding...");
    let t0 = Instant::now();
    let dec_re = ctx.decrypt(&ct, &sk);
    let decrypt_dur = t0.elapsed();
    println!("  -> Decryption Duration: {:?}", decrypt_dur);
    println!(
        "  -> Total Roundtrip (Encrypt + Decrypt): {:?}",
        encrypt_dur + decrypt_dur
    );

    let max_roundtrip_err = input_vec
        .iter()
        .zip(dec_re.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!(
        "  -> Decryption Roundtrip Max Error: {:e}",
        max_roundtrip_err
    );

    // 4. Linear layer benchmark (mul_pt + add_pt)
    println!("\n[4/5] Benchmarking Packed Linear Layer (y = 0.5 * x + 0.1 on {slots} slots)...");
    let weight = 0.5f64;
    let bias = 0.1f64;
    let iters = 100;
    let t0 = Instant::now();
    let mut linear_ct = ct.clone();
    for _ in 0..iters {
        linear_ct = ctx.eval_linear(&ct, weight, bias)?;
    }
    let linear_dur = t0.elapsed() / iters;
    println!("  -> Linear Layer Eval Duration: {:?}", linear_dur);

    let linear_re = ctx.decrypt(&linear_ct, &sk);
    let mut max_linear_err = 0.0f64;
    for i in 0..slots {
        let expected = input_vec[i] * weight + bias;
        let err = (linear_re[i] - expected).abs();
        if err > max_linear_err {
            max_linear_err = err;
        }
    }
    println!(
        "  -> Linear Layer Max Error vs Cleartext Float: {:e}",
        max_linear_err
    );

    // 5. Polynomial non-linearity (ReLU) benchmarks across degrees
    println!("\n[5/5] Benchmarking Polynomial Non-Linearity (ReLU on [-1, 1])...");
    let relu = |x: f64| if x > 0.0 { x } else { 0.0 };

    for degree in [3, 7, 15] {
        let minimax_fit = minimax(relu, -1.0, 1.0, degree, Parity::Full)?;
        let t0 = Instant::now();
        let (relu_ct, _pred_err) = ctx.eval_relu(&ct, &tsk, degree)?;
        let eval_dur = t0.elapsed();

        let relu_re = ctx.decrypt(&relu_ct, &sk);
        let breakdown = compute_error_breakdown(&input_vec, &relu_re, relu, |x| {
            minimax_fit.poly.evaluate_on_interval(x)
        });

        println!("  Degree {:2}:", degree);
        println!("    Latency:                       {:?}", eval_dur);
        println!(
            "    Math Approx Error (poly vs relu): {:e}",
            breakdown.poly_approx_error
        );
        println!(
            "    Total Error (decrypted vs relu):  {:e}",
            breakdown.total_homomorphic_error
        );
        println!(
            "    Crypto Noise Contribution:     {:e}",
            breakdown.fhe_noise_contribution
        );
    }

    println!("\n===============================================================================");
    println!("  Phase 12.0 Standalone Spike Completed Successfully");
    println!("===============================================================================\n");
    Ok(())
}
