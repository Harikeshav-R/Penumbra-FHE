//! `penumbra encrypt <client.key> <out.cts>` — client-side input encryption.
//!
//! Part of the Phase-9 client/server split (`PROJECT.md` §11). Reads the **secret** client key
//! and a JSON batch of quantized integer input rows on stdin (`[[i64, …], …]`, the same shape
//! `predict` accepts), encrypts each row into a ciphertext tensor, and writes the batch of
//! ciphertexts to `out.cts` (bincode). Only ciphertext leaves this step — it is then handed to
//! the server, which never sees the plaintext or the secret key.
//!
//! A thin CLI over the public API (`encrypt`), touching neither `ops/` nor `eval.rs`.

use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

use penumbra_fhe_runtime::{encrypt, load_client_key, CtVec};

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
    let usage = "usage: encrypt <client.key> <out.cts>\n  reads a JSON batch of integer input \
                 rows on stdin, writes encrypted ciphertext to <out.cts>";
    let client_path = PathBuf::from(args.next().ok_or_else(|| usage.to_string())?);
    let out_path = PathBuf::from(args.next().ok_or_else(|| usage.to_string())?);

    let (ck, _num_blocks, _profile) = load_client_key(&client_path)?;

    let mut stdin_text = String::new();
    std::io::stdin()
        .read_to_string(&mut stdin_text)
        .map_err(|e| format!("cannot read input batch from stdin: {e}"))?;
    let inputs: Vec<Vec<i64>> = serde_json::from_str(&stdin_text).map_err(|e| {
        format!("stdin is not a JSON array of integer input rows ([[i64, …], …]): {e}")
    })?;

    // One ciphertext tensor per input row; the batch is the wire payload for the server.
    let cts: Vec<CtVec> = inputs.iter().map(|row| encrypt(&ck, row)).collect();
    let bytes =
        bincode::serialize(&cts).map_err(|e| format!("cannot serialize ciphertext batch: {e}"))?;
    std::fs::write(&out_path, bytes)
        .map_err(|e| format!("cannot write ciphertext to {}: {e}", out_path.display()))?;

    eprintln!(
        "encrypted {} input row(s) -> {}",
        cts.len(),
        out_path.display()
    );
    Ok(())
}
