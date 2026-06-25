//! Cross-language bit-width conformance — the Rust half (`AGENTS.md` §1.3, §5).
//!
//! The bit-width growth rules exist in both Python (`penumbra.bitwidth`) and Rust
//! (`Op::output_bits`). A committed table of `(op, input_bits) -> expected` cases
//! (`tests/fixtures/bitwidth_cases.json`) pins them together: both languages must reproduce
//! every `expected`, so the two implementations cannot drift and silently mis-size a radix.
//! This half checks the Rust side; `tests/test_bitwidth_conformance.py` checks Python.
//!
//! No keygen, no FHE — each case deserializes its op via the IR `OpSpec`, builds the runtime
//! `Op`, and calls `output_bits_n`, so this runs instantly even in debug.

use std::path::PathBuf;

use penumbra_fhe_runtime::OpSpec;
use serde_json::Value;

fn load_cases() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/bitwidth_cases.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read bit-width table {}: {e}", path.display()));
    serde_json::from_str(&text).expect("bit-width table is valid JSON")
}

/// Every committed case's `expected` is reproduced by the Rust `Op::output_bits_n`.
#[test]
fn rust_output_bits_matches_committed_table() {
    let table = load_cases();
    let cases = table["cases"].as_array().expect("`cases` is an array");
    assert!(!cases.is_empty(), "the conformance table must not be empty");

    for case in cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");

        // The op payload is the same JSON the IR uses; deserialize it through OpSpec so this
        // test also exercises the wire format the table shares with the IR.
        let spec: OpSpec = serde_json::from_value(case["op"].clone())
            .unwrap_or_else(|e| panic!("case {name}: op does not deserialize: {e}"));
        let op = spec
            .build()
            .unwrap_or_else(|e| panic!("case {name}: op fails to build: {e}"));

        let input_bits: Vec<usize> = case["input_bits"]
            .as_array()
            .expect("input_bits is an array")
            .iter()
            .map(|v| v.as_u64().expect("input_bits entries are integers") as usize)
            .collect();
        let expected = case["expected"].as_u64().expect("expected is an integer") as usize;

        let got = op.output_bits_n(&input_bits);
        assert_eq!(
            got, expected,
            "bit-width drift in case {name:?}: Rust output_bits_n gave {got}, committed table \
             expects {expected} — Python and Rust rules disagree"
        );
    }
}
