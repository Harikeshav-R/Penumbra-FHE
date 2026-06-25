//! Golden exactness for the `Conv2d` op (`AGENTS.md` §1.1, ROADMAP Phase 4).
//!
//! Conv is `Linear` applied at every spatial position against a shared plaintext kernel, so
//! like `Linear` it is PBS-free and must equal the plain-integer convolution bit-for-bit.
//! This test checks a multi-input-channel convolution (stride 1, no padding) and a padded
//! case against a cleartext im2col reference, pinning both the MAC arithmetic and the
//! channel-major / row-major layout (and virtual zero padding).
//!
//! Run with `cargo test --release` — debug FHE is impractically slow.

use std::collections::HashMap;

use penumbra_fhe_runtime::{
    check_graph_bit_width_budget, decrypt_vec, encrypt, evaluate_graph, keygen, EvalCtx, Graph,
    Node, OpSpec, SCHEMA_VERSION,
};

/// A convolution shape (one struct keeps helpers readable rather than 10-argument calls).
struct ConvCfg {
    in_h: usize,
    in_w: usize,
    in_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
}

fn conv_graph(
    num_blocks: usize,
    input_bits: usize,
    weight_bits: usize,
    weights: Vec<Vec<i64>>,
    bias: Vec<i64>,
    cfg: &ConvCfg,
) -> Graph {
    Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks,
        input_bits,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "conv".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Conv2d {
                weights,
                bias,
                weight_bits,
                in_h: cfg.in_h,
                in_w: cfg.in_w,
                in_channels: cfg.in_channels,
                kernel_h: cfg.kernel_h,
                kernel_w: cfg.kernel_w,
                stride: cfg.stride,
                padding: cfg.padding,
            },
        }],
    }
}

/// Cleartext convolution over a channel-major, row-major input, with virtual zero padding.
fn conv_cleartext(x: &[i64], weights: &[Vec<i64>], bias: &[i64], cfg: &ConvCfg) -> Vec<i64> {
    let out_h = (cfg.in_h + 2 * cfg.padding - cfg.kernel_h) / cfg.stride + 1;
    let out_w = (cfg.in_w + 2 * cfg.padding - cfg.kernel_w) / cfg.stride + 1;
    let in_hw = cfg.in_h * cfg.in_w;
    let mut out = Vec::with_capacity(weights.len() * out_h * out_w);
    for (kernel, &b) in weights.iter().zip(bias) {
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = 0i64;
                for ic in 0..cfg.in_channels {
                    for ky in 0..cfg.kernel_h {
                        let iy = (oy * cfg.stride + ky) as isize - cfg.padding as isize;
                        for kx in 0..cfg.kernel_w {
                            let ix = (ox * cfg.stride + kx) as isize - cfg.padding as isize;
                            if iy < 0
                                || ix < 0
                                || iy as usize >= cfg.in_h
                                || ix as usize >= cfg.in_w
                            {
                                continue;
                            }
                            let w = kernel[(ic * cfg.kernel_h + ky) * cfg.kernel_w + kx];
                            let idx = ic * in_hw + iy as usize * cfg.in_w + ix as usize;
                            acc += w * x[idx];
                        }
                    }
                }
                out.push(acc + b);
            }
        }
    }
    out
}

fn run_conv(weights: Vec<Vec<i64>>, bias: Vec<i64>, cfg: &ConvCfg) -> (Vec<i64>, Vec<i64>) {
    let num_blocks = 8; // 16-bit signed: ample for these small accumulators
    let input_bits = 4;
    let weight_bits = 4;

    let n_in = cfg.in_channels * cfg.in_h * cfg.in_w;
    // Small mixed-sign inputs in roughly [-4, 4].
    let x: Vec<i64> = (0..n_in as i64).map(|i| (i % 9) - 4).collect();

    let graph = conv_graph(
        num_blocks,
        input_bits,
        weight_bits,
        weights.clone(),
        bias.clone(),
        cfg,
    );
    check_graph_bit_width_budget(&graph).expect("Conv2d budget must fit");

    let expected = conv_cleartext(&x, &weights, &bias, cfg);

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

/// 2 input channels → 2 output channels, 3x3 kernel, stride 1, no padding, over a 4x4 map.
#[test]
fn fhe_conv2d_matches_cleartext_no_padding() {
    let cfg = ConvCfg {
        in_h: 4,
        in_w: 4,
        in_channels: 2,
        kernel_h: 3,
        kernel_w: 3,
        stride: 1,
        padding: 0,
    };
    // Two output channels; each kernel is in_channels*kh*kw = 18 weights.
    let weights = vec![
        (0..18).map(|i| (i % 5) - 2).collect::<Vec<i64>>(),
        (0..18).map(|i| 1 - (i % 3)).collect::<Vec<i64>>(),
    ];
    let bias = vec![1i64, -3];
    let (got, expected) = run_conv(weights, bias, &cfg);
    assert_eq!(
        got, expected,
        "GOLDEN VIOLATION: FHE Conv2d (no pad) {got:?} != cleartext {expected:?}"
    );
}

/// 1 channel, 3x3 kernel, stride 1, padding 1 (same-size output) — exercises virtual padding.
#[test]
fn fhe_conv2d_matches_cleartext_with_padding() {
    let cfg = ConvCfg {
        in_h: 4,
        in_w: 4,
        in_channels: 1,
        kernel_h: 3,
        kernel_w: 3,
        stride: 1,
        padding: 1,
    };
    let weights = vec![(0..9).map(|i| (i % 3) - 1).collect::<Vec<i64>>()];
    let bias = vec![2i64];
    let (got, expected) = run_conv(weights, bias, &cfg);
    assert_eq!(
        got, expected,
        "GOLDEN VIOLATION: FHE Conv2d (pad=1) {got:?} != cleartext {expected:?}"
    );
}
