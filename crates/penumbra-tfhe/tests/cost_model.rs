use std::collections::BTreeMap;
use std::path::PathBuf;

use penumbra_core::backend::Backend;
use penumbra_core::ir::{Graph, OpSpec};
use penumbra_tfhe::ops::Conv2d;
use penumbra_tfhe::TfheBackend;

fn load_fixture_graph(rel_path: &str) -> Graph {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel_path);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
    Graph::from_json(&v["graph"].to_string()).expect("valid Graph")
}

#[test]
fn test_tfhe_cost_model_fixtures() {
    let backend = TfheBackend::default();

    // 1. Phase-5 digits fixture: conv0__requant should report 108 bootstraps (12 channels x 3x3)
    let g5 = load_fixture_graph("../../examples/mnist/phase5_digits_fixture.json");
    let node5 = g5
        .nodes
        .iter()
        .find(|n| n.name == "conv0__requant")
        .expect("conv0__requant node");
    let op5 = backend.build_op(&node5.op).expect("build op5");
    let cost5: BTreeMap<&'static str, u64> = op5.cost(&[108]).into_iter().collect();
    assert_eq!(cost5.get("bootstraps"), Some(&108));
    assert_eq!(cost5.get("cmp_pbs_ops"), Some(&(3 * 108)));

    // 2. Phase-7 faces fixture: conv0__requant should report 128 bootstraps
    let g7 = load_fixture_graph("../../examples/faces/phase7_faces_fixture.json");
    let node7 = g7
        .nodes
        .iter()
        .find(|n| n.name == "conv0__requant")
        .expect("conv0__requant node");
    let op7 = backend.build_op(&node7.op).expect("build op7");
    let cost7: BTreeMap<&'static str, u64> = op7.cost(&[128]).into_iter().collect();
    assert_eq!(cost7.get("bootstraps"), Some(&128));
    assert_eq!(cost7.get("cmp_pbs_ops"), Some(&(3 * 128)));

    // 3. Phase-6 sklearn fixture: linear0 should report 140 scalar_mul (distinct non-zero weights per row)
    let g6 = load_fixture_graph("../../examples/mnist/phase6_sklearn_fixture.json");
    let node6 = g6
        .nodes
        .iter()
        .find(|n| n.name == "linear0")
        .expect("linear0 node");
    let op6 = backend.build_op(&node6.op).expect("build op6");
    let cost6: BTreeMap<&'static str, u64> = op6.cost(&[64]).into_iter().collect();
    assert_eq!(cost6.get("scalar_mul"), Some(&140));
}

#[test]
fn test_tfhe_conv2d_mac_count() {
    // Hand-built Conv2d:
    // 1 in_channel, 3x3 input (in_h=3, in_w=3), 1 out_channel, 2x2 kernel, stride=1, padding=1.
    // Weights: [1, 0, 2, 3] (one zero weight at ky=0, kx=1).
    let conv = Conv2d {
        weights: vec![vec![1, 0, 2, 3]],
        bias: vec![0],
        weight_bits: 2,
        in_h: 3,
        in_w: 3,
        in_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride: 1,
        padding: 1,
    };

    // By hand:
    // out_h = (3 + 2 - 2) + 1 = 4, out_w = 4.
    // Non-zero taps across 4x4 output:
    // (ky=0, kx=0, w=1): in-bounds for oy in 1..=3, ox in 1..=3 => 9 taps.
    // (ky=0, kx=1, w=0): skipped because w == 0 => 0 taps.
    // (ky=1, kx=0, w=2): in-bounds for oy in 0..=2, ox in 1..=3 => 9 taps.
    // (ky=1, kx=1, w=3): in-bounds for oy in 0..=2, ox in 0..=2 => 9 taps.
    // Total = 27.
    assert_eq!(conv.mac_count(), 27);

    let backend = TfheBackend::default();
    let spec = OpSpec::Conv2d {
        weights: vec![vec![1, 0, 2, 3]],
        bias: vec![0],
        weight_bits: 2,
        in_h: 3,
        in_w: 3,
        in_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride: 1,
        padding: 1,
    };
    let op = backend.build_op(&spec).expect("build conv2d");
    let cost: BTreeMap<&'static str, u64> = op.cost(&[9]).into_iter().collect();
    assert_eq!(cost.get("scalar_mul"), Some(&27));
    assert_eq!(cost.get("ct_add"), Some(&12)); // 27 taps across 15 active pixels => 27 - 15 = 12 additions
    assert_eq!(cost.get("scalar_add"), Some(&16)); // 1 out_channel * 4 * 4
}

#[test]
fn test_tfhe_requant_zero_multipliers_and_biases() {
    let backend = TfheBackend::default();
    let spec = OpSpec::Requant {
        shift: 4,
        mult: 1,
        round_bias: 0,
        out_bits: 2,
        clamp_lut: vec![0, 1, 2, 3],
        mults: vec![],
        shifts: vec![],
        round_biases: vec![],
        channel_size: None,
    };
    let op = backend.build_op(&spec).expect("build requant");
    let cost: BTreeMap<&'static str, u64> = op.cost(&[10]).into_iter().collect();
    assert_eq!(cost.get("bootstraps"), Some(&10));
    assert_eq!(cost.get("cmp_pbs_ops"), Some(&30));
    assert_eq!(cost.get("scalar_mul"), None);
    assert_eq!(cost.get("scalar_add"), None);
}
