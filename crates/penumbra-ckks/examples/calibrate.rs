#![cfg(feature = "ckks")]

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use penumbra_ckks::backend::{check_graph_depth_budget, evaluate_graph, CkksBackend};
use penumbra_ckks::bounds;
use penumbra_ckks::encrypt::{decrypt_raw_vec, decrypt_vec, encrypt};
use penumbra_ckks::keys::{keygen, schedule_rotations};
use penumbra_ckks::params::{log_sparsity, slots, DEFAULT_PARAMS};
use penumbra_core::backend::EvalCtx;
use penumbra_core::ir::Graph;
use serde_json::Value;

fn load_fixture(rel_path: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(rel_path);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture at {}: {e}", path.display()));
    serde_json::from_str(&text).expect("valid JSON")
}

fn as_i64_vec(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_i64().expect("int"))
        .collect()
}

fn main() {
    println!("============================================================");
    println!(" Penumbra-FHE CKKS Backend Profile Calibration (Phase 12.2) ");
    println!("============================================================");

    let params = DEFAULT_PARAMS;

    println!("\nCalibrated Profile:");
    println!("  Ring dimension N:        {}", params.n);
    println!("  Total CKKS slots (N/2):  {}", slots(&params));
    println!("  Torus width k:           {}", params.k);
    println!("  Scaling factor log_delta:{}", params.log_delta);
    println!("  Multiplicative budget:   {} bits", params.log_budget());
    println!("  Gadget base2k:           {}", params.base2k);
    println!("  Gadget dsize:            {}", params.dsize);
    println!("  Transform dim lt_slots:  {}", params.lt_slots);
    println!("  BSGS giant_step:         {}", params.giant_step);
    println!("  Log sparsity:            {}", log_sparsity(&params));
    println!("  Max polynomial degree:   {}", params.max_poly_degree);

    let rotations = schedule_rotations(&params);
    println!("  Automorphism key count:  {} rotations", rotations.len());

    let backend = CkksBackend::new(params);

    println!("\nGenerating key material...");
    let t0 = Instant::now();
    let (ck, sk) = keygen(&params).expect("keygen failed");
    let keygen_dur = t0.elapsed();
    println!("Keygen completed in {:.2?}", keygen_dur);

    let dummy_in = vec![1i64; 256];
    let dummy_ct = encrypt(&ck, &dummy_in);
    let ct_bytes = penumbra_ckks::encrypt::serialize_cts(&dummy_ct)
        .unwrap()
        .len();
    println!(
        "Single packed ciphertext size: {} bytes ({:.2} MB)",
        ct_bytes,
        ct_bytes as f64 / (1024.0 * 1024.0)
    );

    let fixtures = [
        (
            "Phase 6 sklearn",
            "examples/mnist/phase6_sklearn_fixture.json",
            "logits",
            bounds::PHASE6_SKLEARN,
        ),
        (
            "Phase 2 logreg",
            "examples/mnist/phase2_fixture.json",
            "labels",
            bounds::PHASE2_LOGREG,
        ),
        (
            "Phase 4 cnn",
            "examples/mnist/phase4_cnn_fixture.json",
            "logits",
            bounds::PHASE4_CNN,
        ),
        (
            "Phase 5 digits",
            "examples/mnist/phase5_digits_fixture.json",
            "logits",
            bounds::PHASE5_DIGITS,
        ),
        (
            "Phase 5 qat",
            "examples/mnist/phase5_qat_fixture.json",
            "logits",
            bounds::PHASE5_QAT,
        ),
        (
            "Phase 6 onnx",
            "examples/mnist/phase6_onnx_fixture.json",
            "logits",
            bounds::PHASE6_ONNX,
        ),
        (
            "Phase 7 faces",
            "examples/faces/phase7_faces_fixture.json",
            "logits",
            bounds::PHASE7_FACES,
        ),
    ];

    let ctx = EvalCtx::new(&sk, 8);

    println!("\nEvaluating committed fixtures against declared bounds...");
    println!(
        "{:<18} | {:<12} | {:<12} | {:<14} | {:<10} | {:<10}",
        "Fixture", "Depth Check", "Max |Err|", "Declared Bound", "Within Bound", "Match Label"
    );
    println!(
        "{:-<18}-+-{:-<12}-+-{:-<12}-+-{:-<14}-+-{:-<10}-+-{:-<10}",
        "", "", "", "", "", ""
    );

    for (name, path, mode, bound) in fixtures {
        let val = load_fixture(path);
        let graph: Graph = serde_json::from_value(val["graph"].clone()).expect("valid graph");

        let budget_check = check_graph_depth_budget(&backend, &graph);
        let budget_status = if budget_check.is_ok() { "PASS" } else { "FAIL" };

        let input_0 = as_i64_vec(&val["test_inputs"][0]);
        let mut inputs_map = HashMap::new();
        inputs_map.insert(graph.inputs[0].clone(), encrypt(&ck, &input_0));

        let eval_res = evaluate_graph(&ctx, &graph, inputs_map);

        if let Ok(outputs) = eval_res {
            let out_cts = &outputs[&graph.outputs[0]];
            let raw_floats = decrypt_raw_vec(&ck, out_cts);
            let rounded_ints = decrypt_vec(&ck, out_cts);

            let want_label = val["expected_labels"][0].as_i64().unwrap();
            let (want_ints, pred_label) = if mode == "labels" {
                (vec![want_label], rounded_ints[0])
            } else {
                let want = as_i64_vec(&val["expected_logits"][0]);
                let pred = raw_floats
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as i64)
                    .unwrap();
                (want, pred)
            };

            let mut max_err = 0.0f64;
            for (i, &want) in want_ints.iter().enumerate() {
                let got_f = raw_floats[i];
                let err = (got_f - want as f64).abs();
                if err > max_err {
                    max_err = err;
                }
            }

            let within_bound = if max_err <= bound { "PASS" } else { "FAIL" };
            let match_label = if pred_label == want_label {
                "YES"
            } else {
                "NO"
            };

            println!(
                "{:<18} | {:<12} | {:<12.6e} | {:<14.1e} | {:<10} | {:<10}",
                name, budget_status, max_err, bound, within_bound, match_label
            );
        } else {
            let err_msg = eval_res.err().unwrap();
            println!("{:<18} | {:<12} | ERROR: {}", name, budget_status, err_msg);
        }
    }
}
