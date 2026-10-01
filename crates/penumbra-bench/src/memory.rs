//! Peak server RSS capture via `getrusage`.

use std::mem::MaybeUninit;

use crate::protocol::ServerMemoryMetrics;

/// Convert raw `ru_maxrss` from `libc::rusage` to bytes across operating systems.
/// On macOS, `ru_maxrss` is already bytes.
/// On Linux, `ru_maxrss` is in KiB and must be multiplied by 1024.
pub fn ru_maxrss_to_bytes(raw_maxrss: libc::c_long, os: &str) -> Result<u64, String> {
    if raw_maxrss < 0 {
        return Err(format!("negative ru_maxrss value: {raw_maxrss}"));
    }
    let val = raw_maxrss as u64;
    match os {
        "macos" | "ios" => Ok(val),
        "linux" | "android" => val
            .checked_mul(1024)
            .ok_or_else(|| format!("overflow converting ru_maxrss {val} KiB to bytes")),
        other => {
            // Default to Linux-like KiB for unknown Unix platforms, but warn
            val.checked_mul(1024)
                .ok_or_else(|| format!("overflow converting ru_maxrss {val} on {other} to bytes"))
        }
    }
}

/// Capture current process peak RSS.
pub fn capture_peak_server_rss(sample_id: &str) -> Result<ServerMemoryMetrics, String> {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if ret != 0 {
        return Err(format!(
            "getrusage failed with errno: {}",
            std::io::Error::last_os_error()
        ));
    }
    let usage = unsafe { usage.assume_init() };
    let os = std::env::consts::OS;
    let peak_bytes = ru_maxrss_to_bytes(usage.ru_maxrss, os)?;
    let worker_pid = std::process::id();

    Ok(ServerMemoryMetrics {
        peak_server_rss_bytes: peak_bytes,
        method: "getrusage".to_string(),
        scope: "fresh_process_server_key_load_and_eval".to_string(),
        worker_pid,
        sample_id: sample_id.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ru_maxrss_conversion() {
        // macOS: 123 -> 123 bytes
        let mac = ru_maxrss_to_bytes(123, "macos").unwrap();
        assert_eq!(mac, 123);

        // Linux: 123 KiB -> 125952 bytes (123 * 1024)
        let lin = ru_maxrss_to_bytes(123, "linux").unwrap();
        assert_eq!(lin, 125952);

        // Negative value error
        assert!(ru_maxrss_to_bytes(-1, "linux").is_err());

        // Overflow on Linux
        assert!(ru_maxrss_to_bytes(libc::c_long::MAX, "linux").is_err());
    }

    #[test]
    fn test_capture_peak_server_rss() {
        let mem = capture_peak_server_rss("sample_0").unwrap();
        assert!(mem.peak_server_rss_bytes > 0);
        assert_eq!(mem.method, "getrusage");
        assert_eq!(mem.scope, "fresh_process_server_key_load_and_eval");
        assert_eq!(mem.sample_id, "sample_0");
    }
}
