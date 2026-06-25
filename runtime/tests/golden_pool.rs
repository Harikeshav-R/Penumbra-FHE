//! Golden exactness for the `Pool` op (`AGENTS.md` §1.1, ROADMAP Phase 4).
//!
//! Pooling is spatial, but the inter-op currency is a flat `CtVec`. This test pins the
//! channel-major / row-major layout contract and asserts both modes match cleartext:
//! - `avg` emits the window **sum** (the `/k` is deferred to `Requant`),
//! - `max` emits the window maximum.
//!
//! A 2-channel 4x4 → 2x2 (2x2 window, stride 2) case exercises the per-channel indexing.
//!
//! Run with `cargo test --release` — debug FHE is impractically slow.

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    Node, OpSpec, SCHEMA_VERSION,
};

/// A pooling shape (kept as one struct so the helpers stay readable, not 9-argument calls).
struct PoolCfg {
    in_h: usize,
    in_w: usize,
    channels: usize,
    pool_h: usize,
    pool_w: usize,
    stride: usize,
}

fn pool_graph(mode: &str, num_blocks: usize, input_bits: usize, cfg: &PoolCfg) -> Graph {
    Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks,
        input_bits,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "pool".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Pool {
                mode: mode.to_string(),
                in_h: cfg.in_h,
                in_w: cfg.in_w,
                channels: cfg.channels,
                pool_h: cfg.pool_h,
                pool_w: cfg.pool_w,
                stride: cfg.stride,
            },
        }],
    }
}

/// Cleartext pooling over a channel-major, row-major `[channels][in_h][in_w]` tensor.
fn pool_cleartext(x: &[i64], avg: bool, cfg: &PoolCfg) -> Vec<i64> {
    let out_h = (cfg.in_h - cfg.pool_h) / cfg.stride + 1;
    let out_w = (cfg.in_w - cfg.pool_w) / cfg.stride + 1;
    let mut out = Vec::with_capacity(cfg.channels * out_h * out_w);
    for c in 0..cfg.channels {
        let base = c * cfg.in_h * cfg.in_w;
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut vals = Vec::new();
                for ky in 0..cfg.pool_h {
                    for kx in 0..cfg.pool_w {
                        let y = oy * cfg.stride + ky;
                        let xx = ox * cfg.stride + kx;
                        vals.push(x[base + y * cfg.in_w + xx]);
                    }
                }
                out.push(if avg {
                    vals.iter().sum() // sum-pool (the /k is deferred to Requant)
                } else {
                    *vals.iter().max().unwrap()
                });
            }
        }
    }
    out
}

fn run_pool(mode: &str) -> (Vec<i64>, Vec<i64>) {
    // 2 channels, 4x4 each; 2x2 window, stride 2 -> 2x2 output per channel.
    let cfg = PoolCfg {
        in_h: 4,
        in_w: 4,
        channels: 2,
        pool_h: 2,
        pool_w: 2,
        stride: 2,
    };
    let num_blocks = 6; // 12-bit signed: holds sums of small values with headroom
    let input_bits = 5; // values in roughly [-16, 15]

    // 32 elements: channel 0 then channel 1, each row-major; a mix of negatives and positives.
    let x: Vec<i64> = (0..(cfg.channels * cfg.in_h * cfg.in_w) as i64)
        .map(|i| i - 5)
        .collect();

    let graph = pool_graph(mode, num_blocks, input_bits, &cfg);
    check_graph_bit_width_budget(&graph).expect("Pool budget must fit");

    let expected = pool_cleartext(&x, mode == "avg", &cfg);

    let (ck, sk) = keygen(num_blocks);
    let ctx = EvalCtx {
        sk: &sk,
        num_blocks,
    };
    let mut env = HashMap::new();
    env.insert("x".to_string(), encrypt(&ck, &x));
    let out = evaluate_graph(&ctx, &graph, env).expect("graph evaluates");
    let got = decrypt_vec(&ck, &out["y"]);
    (got, expected)
}

#[test]
fn fhe_avg_pool_matches_cleartext_sum() {
    let (got, expected) = run_pool("avg");
    assert_eq!(
        got, expected,
        "GOLDEN VIOLATION: FHE avg(sum) pool {got:?} != cleartext {expected:?}"
    );
}

#[test]
fn fhe_max_pool_matches_cleartext() {
    let (got, expected) = run_pool("max");
    assert_eq!(
        got, expected,
        "GOLDEN VIOLATION: FHE max pool {got:?} != cleartext {expected:?}"
    );
}
