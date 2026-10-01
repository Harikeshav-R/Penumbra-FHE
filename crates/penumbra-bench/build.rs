//! Build script for penumbra-bench capturing provenance metadata at build time.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_PKG_VERSION");

    // Watch git references for rebuild freshness
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--git-path", "HEAD"])
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--git-path", "packed-refs"])
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--symbolic-full-name", "HEAD"])
        .output()
    {
        if output.status.success() {
            let ref_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !ref_name.is_empty() && ref_name != "HEAD" {
                if let Ok(ref_path_out) = Command::new("git")
                    .args(["rev-parse", "--git-path", &ref_name])
                    .output()
                {
                    if ref_path_out.status.success() {
                        let path = String::from_utf8_lossy(&ref_path_out.stdout)
                            .trim()
                            .to_string();
                        println!("cargo:rerun-if-changed={path}");
                    }
                }
            }
        }
    }

    // Capture build commit SHA
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=PENUMBRA_BUILD_COMMIT={commit}");

    // Capture rustc -Vv
    let rustc_v = Command::new("rustc")
        .args(["-Vv"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=PENUMBRA_BUILD_RUSTC={rustc_v}");
}
