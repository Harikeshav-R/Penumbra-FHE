#![cfg(feature = "ckks")]

use serde_json::Value;

#[allow(dead_code)]
pub fn as_i64_vec(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("must be array")
        .iter()
        .map(|x| x.as_i64().expect("must be integer"))
        .collect()
}
