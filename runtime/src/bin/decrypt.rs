//! `penumbra decrypt <client.key> <out.cts>` — client-side result decryption.
//!
//! The final step of the Phase-9 client/server split (`PROJECT.md` §11): the client takes the
//! encrypted outputs the server returned and decrypts them with its **secret** key, printing
//! `{"outputs": [[i64, …], …]}` (the same shape `predict` emits) on stdout. The client then
//! interprets the raw values (argmax / label) — the runtime never decides that (`PROJECT.md` §11).
//!
//! A thin CLI over the public API (`decrypt_vec`), touching neither `ops/` nor `eval.rs`.

use std::path::PathBuf;
use std::process::ExitCode;

use penumbra_fhe_runtime::{decrypt_vec, load_client_key, CtVec};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: decrypt <client.key> <out.cts>\n  decrypts the server's output ciphertext \
                 and prints {\"outputs\": [[i64, …], …]}";
    let client_path = PathBuf::from(args.next().ok_or_else(|| usage.to_string())?);
    let cts_path = PathBuf::from(args.next().ok_or_else(|| usage.to_string())?);

    let (ck, _num_blocks) = load_client_key(&client_path)?;

    let bytes = std::fs::read(&cts_path)
        .map_err(|e| format!("cannot read ciphertext from {}: {e}", cts_path.display()))?;
    let outputs: Vec<CtVec> = bincode::deserialize(&bytes).map_err(|e| {
        format!(
            "cannot deserialize ciphertext batch from {} (is it a Penumbra .cts file?): {e}",
            cts_path.display()
        )
    })?;

    let decrypted: Vec<Vec<i64>> = outputs.iter().map(|ct| decrypt_vec(&ck, ct)).collect();
    let result = serde_json::json!({ "outputs": decrypted });
    println!("{result}");
    Ok(())
}
