//! Parity test asserting that both backends consume the same IR, run the same models,
//! and walk identical node lists through the shared graph walker (ROADMAP Phase 12.3).

use penumbra_bench::models::{find, load, MODELS};
use penumbra_bench::session::Session;
use penumbra_bench::tfhe_backend;
use penumbra_core::backend::Backend;

#[test]
fn test_op_set_parity_every_committed_fixture_no_crypto() {
    let tfhe = tfhe_backend();
    #[cfg(feature = "ckks")]
    let ckks = penumbra_bench::ckks_backend();

    for fixture in MODELS {
        let loaded = load(fixture)
            .unwrap_or_else(|e| panic!("failed to load fixture '{}': {e}", fixture.key));

        // 1. TFHE budget check and build_op for every node
        tfhe.check_graph_budget(&loaded.graph)
            .unwrap_or_else(|e| panic!("TFHE budget check failed for '{}': {e}", fixture.key));
        for node in &loaded.graph.nodes {
            tfhe.build_op(&node.op).unwrap_or_else(|e| {
                panic!(
                    "TFHE build_op failed for node '{}' in '{}': {e}",
                    node.name, fixture.key
                )
            });
        }

        // 2. CKKS budget check and build_op for every node
        #[cfg(feature = "ckks")]
        {
            ckks.check_graph_budget(&loaded.graph)
                .unwrap_or_else(|e| panic!("CKKS budget check failed for '{}': {e}", fixture.key));
            for node in &loaded.graph.nodes {
                ckks.build_op(&node.op).unwrap_or_else(|e| {
                    panic!(
                        "CKKS build_op failed for node '{}' in '{}': {e}",
                        node.name, fixture.key
                    )
                });
            }
        }
    }
}

#[test]
fn test_same_entry_point_same_node_list() {
    let fixture = find("phase2_logreg").expect("phase2_logreg fixture");
    let loaded = load(fixture).expect("load phase2_logreg");
    assert!(!loaded.inputs.is_empty(), "must have test inputs");
    let input = &loaded.inputs[0];

    // 1. Run through TFHE session
    let tfhe = tfhe_backend();
    let tfhe_session = Session::new(tfhe, &loaded.graph).expect("TFHE session");
    let tfhe_in_cts = tfhe_session.encrypt(input);
    let (tfhe_out_cts, tfhe_profile) = tfhe_session
        .eval(&loaded.graph, &tfhe_in_cts)
        .expect("TFHE eval");

    assert_eq!(tfhe_profile.backend, "tfhe");
    assert_eq!(tfhe_profile.op_types(), vec!["Linear", "Argmax"]);
    assert_eq!(tfhe_profile.nodes.len(), loaded.graph.nodes.len());
    for (prof_node, graph_node) in tfhe_profile.nodes.iter().zip(&loaded.graph.nodes) {
        assert_eq!(prof_node.name, graph_node.name);
    }

    let tfhe_counters = tfhe_profile.counter_totals();
    assert!(
        tfhe_counters.contains_key("cmp_pbs_ops"),
        "TFHE Argmax should emit cmp_pbs_ops"
    );

    // Verify output correctness
    let tfhe_pred = tfhe_session.decrypt_label(&tfhe_out_cts);
    if let Some(labels) = &loaded.expected_labels {
        assert_eq!(tfhe_pred, labels[0], "TFHE prediction should match oracle");
    }

    // 2. Run through CKKS session when compiled with ckks
    #[cfg(feature = "ckks")]
    {
        let ckks = penumbra_bench::ckks_backend();
        let ckks_session = Session::new(ckks, &loaded.graph).expect("CKKS session");
        let ckks_in_cts = ckks_session.encrypt(input);
        let (ckks_out_cts, ckks_profile) = ckks_session
            .eval(&loaded.graph, &ckks_in_cts)
            .expect("CKKS eval");

        assert_eq!(ckks_profile.backend, "ckks");
        assert_eq!(ckks_profile.op_types(), tfhe_profile.op_types());
        assert_eq!(ckks_profile.nodes.len(), tfhe_profile.nodes.len());
        for (ckks_node, tfhe_node) in ckks_profile.nodes.iter().zip(&tfhe_profile.nodes) {
            assert_eq!(ckks_node.name, tfhe_node.name);
            assert_eq!(ckks_node.op_type, tfhe_node.op_type);
        }

        let ckks_counters = ckks_profile.counter_totals();
        assert!(
            ckks_counters.contains_key("rotations"),
            "CKKS should emit rotations"
        );
        assert!(
            ckks_counters.contains_key("rescales"),
            "CKKS should emit rescales"
        );
        assert!(
            ckks_counters.contains_key("depth_levels"),
            "CKKS should emit depth_levels"
        );

        // Verify prediction matches
        let ckks_pred = ckks_session.decrypt_label(&ckks_out_cts);
        if let Some(labels) = &loaded.expected_labels {
            assert_eq!(ckks_pred, labels[0], "CKKS prediction should match oracle");
        }
    }
}
