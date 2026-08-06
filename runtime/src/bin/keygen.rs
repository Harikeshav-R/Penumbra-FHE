//! `penumbra keygen <num_blocks> <client.key> <server.key>` — the client-side key ceremony.
//!
//! Part of the Phase-9 client/server split (`PROJECT.md` §11, `ROADMAP.md` Phase 9). The
//! **client** generates a key pair once and persists it, then reuses it across inferences
//! (`penumbra.client.KeySet`). This binary writes:
//!   - `client.key` — the **secret** key (encrypt + decrypt). Never give it to the server.
//!   - `server.key` — the **public** evaluation key (enables bootstrapping, cannot decrypt).
//!
//! Both are tagged with `num_blocks` (a key is only compatible with a model of the same radix
//! width; see `keys::save_client_key`). This is a thin CLI over the crate's public API — no
//! crypto here beyond `keygen`, and it touches neither `ops/` nor `eval.rs` (`AGENTS.md` §1.2).

use std::path::PathBuf;
use std::process::ExitCode;

use penumbra_fhe_runtime::{keygen, save_client_key, save_server_key};

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
    let usage = "usage: keygen <num_blocks> <client.key> <server.key>\n  generates a key pair for \
                 a model of the given radix width (num_blocks)";
    let num_blocks: usize = args
        .next()
        .ok_or_else(|| usage.to_string())?
        .parse()
        .map_err(|e| format!("num_blocks must be a positive integer: {e}\n{usage}"))?;
    let client_path = PathBuf::from(args.next().ok_or_else(|| usage.to_string())?);
    let server_path = PathBuf::from(args.next().ok_or_else(|| usage.to_string())?);

    if num_blocks == 0 {
        return Err(format!(
            "num_blocks must be > 0 (radix integers need at least one block)\n{usage}"
        ));
    }

    // Keygen is the expensive per-`num_blocks` cost; doing it once here is the whole point of
    // persisting keys for reuse (ROADMAP Phase 9).
    let (ck, sk) = keygen(num_blocks);
    save_client_key(&ck, num_blocks, &client_path)?;
    save_server_key(&sk, num_blocks, &server_path)?;

    eprintln!(
        "wrote client key {} and server key {} for num_blocks={num_blocks}",
        client_path.display(),
        server_path.display()
    );
    Ok(())
}
