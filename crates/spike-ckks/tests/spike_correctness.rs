use poulpy_ckks::approximation::minimax;
use poulpy_ckks::polynomial::Parity;
use spike_ckks::{SpikeContext, compute_error_breakdown};

#[test]
fn test_coexistence_with_tfhe() {
    // 1. TFHE shortint path
    use tfhe::shortint::gen_keys;
    use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
    let (tfhe_ck, tfhe_sk) = gen_keys(PARAM_MESSAGE_2_CARRY_2_KS_PBS);
    let x = 2u64;
    let ct_tfhe = tfhe_ck.encrypt(x);
    let prod_tfhe = tfhe_sk.scalar_mul(&ct_tfhe, 1);
    let dec_tfhe = tfhe_ck.decrypt(&prod_tfhe);
    assert_eq!(dec_tfhe, 2, "TFHE path failed");

    // 2. CKKS path in the same binary
    let mut ctx = SpikeContext::new().expect("SpikeContext init");
    let (sk, _) = ctx.keygen([1u8; 32]);
    let input = vec![0.5; ctx.slots];
    let ct_ckks = ctx.encrypt(&input, &sk);
    let dec_ckks = ctx.decrypt(&ct_ckks, &sk);
    let err = (dec_ckks[0] - 0.5).abs();
    assert!(err < 1e-5, "CKKS path error too high: {err}");
}

#[test]
fn test_roundtrip_precision() {
    let mut ctx = SpikeContext::new().expect("SpikeContext init");
    let (sk, _) = ctx.keygen([2u8; 32]);
    let input: Vec<f64> = (0..ctx.slots)
        .map(|i| 2.0 * (i as f64) / ((ctx.slots - 1) as f64) - 1.0)
        .collect();

    let ct = ctx.encrypt(&input, &sk);
    let dec = ctx.decrypt(&ct, &sk);

    let max_err = input
        .iter()
        .zip(dec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    assert!(
        max_err < 1e-6,
        "CKKS roundtrip error exceeds tolerance: {max_err:e}"
    );
}

#[test]
fn test_linear_layer_correctness() {
    let mut ctx = SpikeContext::new().expect("SpikeContext init");
    let (sk, _) = ctx.keygen([3u8; 32]);
    let input: Vec<f64> = (0..ctx.slots)
        .map(|i| (i as f64) / (ctx.slots as f64) - 0.5)
        .collect();

    let weight = 0.75f64;
    let bias = -0.25f64;

    let ct = ctx.encrypt(&input, &sk);
    let out_ct = ctx
        .eval_linear(&ct, weight, bias)
        .expect("eval_linear failed");
    let dec = ctx.decrypt(&out_ct, &sk);

    for (i, (&x, &d)) in input.iter().zip(dec.iter()).enumerate() {
        let expected = weight * x + bias;
        let err = (d - expected).abs();
        assert!(
            err < 1e-5,
            "Slot {i} linear evaluation error exceeds tolerance: expected {expected}, got {d}, err={err:e}"
        );
    }
}

#[test]
fn test_relu_polynomial_error_bounds() {
    let mut ctx = SpikeContext::new().expect("SpikeContext init");
    let (sk, tsk) = ctx.keygen([4u8; 32]);
    let input: Vec<f64> = (0..ctx.slots)
        .map(|i| 2.0 * (i as f64) / ((ctx.slots - 1) as f64) - 1.0)
        .collect();
    let relu = |x: f64| if x > 0.0 { x } else { 0.0 };

    let ct = ctx.encrypt(&input, &sk);

    // Degree 3: bound ~ 0.07
    let (relu_ct3, pred_err3) = ctx.eval_relu(&ct, &tsk, 3).expect("eval_relu degree 3");
    let dec3 = ctx.decrypt(&relu_ct3, &sk);
    let breakdown3 = compute_error_breakdown(&input, &dec3, relu, |x| {
        minimax(relu, -1.0, 1.0, 3, Parity::Full)
            .unwrap()
            .poly
            .evaluate_on_interval(x)
    });
    assert!(
        breakdown3.total_homomorphic_error < 0.08,
        "Degree 3 error exceeds bound: {:e}",
        breakdown3.total_homomorphic_error
    );
    assert!((breakdown3.poly_approx_error - pred_err3).abs() < 1e-4);

    // Degree 7: bound ~ 0.03
    let (relu_ct7, pred_err7) = ctx.eval_relu(&ct, &tsk, 7).expect("eval_relu degree 7");
    let dec7 = ctx.decrypt(&relu_ct7, &sk);
    let breakdown7 = compute_error_breakdown(&input, &dec7, relu, |x| {
        minimax(relu, -1.0, 1.0, 7, Parity::Full)
            .unwrap()
            .poly
            .evaluate_on_interval(x)
    });
    assert!(
        breakdown7.total_homomorphic_error < 0.03,
        "Degree 7 error exceeds bound: {:e}",
        breakdown7.total_homomorphic_error
    );
    assert!((breakdown7.poly_approx_error - pred_err7).abs() < 1e-4);

    // Degree 15: bound ~ 0.015
    let (relu_ct15, pred_err15) = ctx.eval_relu(&ct, &tsk, 15).expect("eval_relu degree 15");
    let dec15 = ctx.decrypt(&relu_ct15, &sk);
    let breakdown15 = compute_error_breakdown(&input, &dec15, relu, |x| {
        minimax(relu, -1.0, 1.0, 15, Parity::Full)
            .unwrap()
            .poly
            .evaluate_on_interval(x)
    });
    assert!(
        breakdown15.total_homomorphic_error < 0.015,
        "Degree 15 error exceeds bound: {:e}",
        breakdown15.total_homomorphic_error
    );
    assert!((breakdown15.poly_approx_error - pred_err15).abs() < 1e-4);
}

#[test]
fn test_fhe_noise_bound() {
    let mut ctx = SpikeContext::new().expect("SpikeContext init");
    let (sk, tsk) = ctx.keygen([5u8; 32]);
    let input: Vec<f64> = (0..ctx.slots)
        .map(|i| 2.0 * (i as f64) / ((ctx.slots - 1) as f64) - 1.0)
        .collect();
    let relu = |x: f64| if x > 0.0 { x } else { 0.0 };

    let ct = ctx.encrypt(&input, &sk);
    let (relu_ct, _) = ctx.eval_relu(&ct, &tsk, 7).expect("eval_relu");
    let dec = ctx.decrypt(&relu_ct, &sk);

    let fit = minimax(relu, -1.0, 1.0, 7, Parity::Full).expect("minimax");
    let breakdown =
        compute_error_breakdown(&input, &dec, relu, |x| fit.poly.evaluate_on_interval(x));

    // Cryptographic noise should be strictly below 1e-5
    assert!(
        breakdown.fhe_noise_contribution < 1e-5,
        "Cryptographic noise is too high: {:e}",
        breakdown.fhe_noise_contribution
    );
}
