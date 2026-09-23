//! The CLI client/server split under a non-default crypto profile.
//!
//! Two regressions are gated here: `serve` used to hard-code `TfheProfile::default()` and
//! reject a `"gaussian"` server key (`PROJECT.md` §12's single TFHE override knob), and the
//! CLI ciphertext files bypassed the scheme-tagged wire envelope (`docs/BACKENDS.md`). The
//! assertion is the golden invariant: the decrypted output equals the quantized-cleartext
//! value bit-for-bit (`AGENTS.md` §1.1) — TFHE is exact under either noise distribution.

use std::io::Write;
use std::process::{Command, Stdio};

use penumbra_fhe_runtime::{
    keygen_with_profile, save_client_key, save_server_key, Graph, Node, OpSpec, TfheProfile,
    SCHEMA_VERSION,
};

#[test]
fn serve_cli_round_trips_a_gaussian_profile_key() {
    let dir = std::env::temp_dir().join(format!("penumbra_serve_cli_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // 2*2 + 3*(-1) + 1 = 2 — the same tiny model the profile unit test uses.
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 2,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "lin".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Linear {
                weights: vec![vec![2, 3]],
                bias: vec![1],
                weight_bits: 3,
            },
        }],
    };
    let model_path = dir.join("model.fhe");
    std::fs::write(&model_path, graph.to_json()).unwrap();

    // The non-default profile — the case `serve` used to reject outright.
    let (ck, sk) = keygen_with_profile(graph.num_blocks, TfheProfile::Gaussian);
    let client_key = dir.join("client.key");
    let server_key = dir.join("server.key");
    save_client_key(&ck, graph.num_blocks, TfheProfile::Gaussian, &client_key).unwrap();
    save_server_key(&sk, graph.num_blocks, TfheProfile::Gaussian, &server_key).unwrap();

    let in_cts = dir.join("in.cts");
    let mut enc = Command::new(env!("CARGO_BIN_EXE_encrypt"))
        .arg(&client_key)
        .arg(&in_cts)
        .stdin(Stdio::piped())
        .spawn()
        .expect("encrypt binary spawns");
    enc.stdin.as_mut().unwrap().write_all(b"[[2,-1]]").unwrap();
    assert!(enc.wait().unwrap().success(), "encrypt must succeed");

    let out_cts = dir.join("out.cts");
    let served = Command::new(env!("CARGO_BIN_EXE_serve"))
        .args([&model_path, &server_key, &in_cts, &out_cts])
        .output()
        .expect("serve binary runs");
    assert!(
        served.status.success(),
        "serve must accept a gaussian-profile server key: {}",
        String::from_utf8_lossy(&served.stderr)
    );

    let decrypted = Command::new(env!("CARGO_BIN_EXE_decrypt"))
        .args([&client_key, &out_cts])
        .output()
        .expect("decrypt binary runs");
    assert!(
        decrypted.status.success(),
        "decrypt must succeed: {}",
        String::from_utf8_lossy(&decrypted.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&decrypted.stdout).unwrap();
    assert_eq!(
        parsed["outputs"],
        serde_json::json!([[2]]),
        "decrypted output must equal the cleartext value bit-for-bit"
    );

    std::fs::remove_dir_all(&dir).ok();
}
