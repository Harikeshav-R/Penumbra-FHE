//! The graph walker must name the offending node when a backend rejects an op
//! (`AGENTS.md` §1.4, `ROADMAP.md` Phase 12.2).

use std::collections::HashMap;

use penumbra_core::backend::{Backend, EvalCtx};
use penumbra_core::eval::evaluate_graph;
use penumbra_core::ir::{Graph, Node, OpSpec, SCHEMA_VERSION};
use penumbra_core::ops::Op;

/// Rejects every op, so `build_op` is the only reachable failure.
struct RejectingBackend;

impl Backend for RejectingBackend {
    type Ciphertext = i64;
    type ServerKey = ();
    type ClientKey = ();

    fn name(&self) -> &'static str {
        "stub"
    }

    fn build_op(&self, spec: &OpSpec) -> Result<Box<dyn Op<Self>>, String> {
        Err(format!(
            "operator {} is unsupported on backend 'stub'",
            spec.op_type()
        ))
    }

    fn check_graph_budget(&self, _graph: &Graph) -> Result<(), String> {
        Ok(())
    }

    fn create_trivial_zero(&self, _sk: &Self::ServerKey, _num_blocks: usize) -> Self::Ciphertext {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn add(
        &self,
        _sk: &Self::ServerKey,
        _a: &Self::Ciphertext,
        _b: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn scalar_mul(
        &self,
        _sk: &Self::ServerKey,
        _a: &Self::Ciphertext,
        _scalar: i64,
    ) -> Self::Ciphertext {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn scalar_add(
        &self,
        _sk: &Self::ServerKey,
        _a: &Self::Ciphertext,
        _scalar: i64,
    ) -> Self::Ciphertext {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn scalar_ge(
        &self,
        _sk: &Self::ServerKey,
        _a: &Self::Ciphertext,
        _threshold: i64,
    ) -> Self::Ciphertext {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn max(
        &self,
        _sk: &Self::ServerKey,
        _a: &Self::Ciphertext,
        _b: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn scalar_max(
        &self,
        _sk: &Self::ServerKey,
        _a: &Self::Ciphertext,
        _scalar: i64,
    ) -> Self::Ciphertext {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn scalar_min(
        &self,
        _sk: &Self::ServerKey,
        _a: &Self::Ciphertext,
        _scalar: i64,
    ) -> Self::Ciphertext {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn scalar_right_shift(
        &self,
        _sk: &Self::ServerKey,
        _a: &Self::Ciphertext,
        _shift: u32,
    ) -> Self::Ciphertext {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn apply_lut(
        &self,
        _sk: &Self::ServerKey,
        _a: &Self::Ciphertext,
        _lut: &[u64],
        _num_blocks: usize,
    ) -> Self::Ciphertext {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn keygen(&self, _num_blocks: usize) -> (Self::ClientKey, Self::ServerKey) {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn encrypt(&self, _ck: &Self::ClientKey, _input: &[i64]) -> Vec<Self::Ciphertext> {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn decrypt_label(&self, _ck: &Self::ClientKey, _out: &[Self::Ciphertext]) -> i64 {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn decrypt_vec(&self, _ck: &Self::ClientKey, _out: &[Self::Ciphertext]) -> Vec<i64> {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn serialize_cts(&self, _cts: &[Self::Ciphertext]) -> Result<Vec<u8>, String> {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn deserialize_cts(&self, _bytes: &[u8]) -> Result<Vec<Self::Ciphertext>, String> {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn serialize_client_key(
        &self,
        _ck: &Self::ClientKey,
        _num_blocks: usize,
    ) -> Result<Vec<u8>, String> {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }

    fn serialize_server_key(
        &self,
        _sk: &Self::ServerKey,
        _num_blocks: usize,
    ) -> Result<Vec<u8>, String> {
        unimplemented!("evaluation is unreachable: build_op rejects first")
    }
}

#[test]
fn build_op_rejection_names_the_node() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![Node {
            name: "layer0".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            op: OpSpec::Add {},
        }],
    };

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![1i64, 2, 3]);

    let err = evaluate_graph(&RejectingBackend, &EvalCtx::new(&(), 4), &graph, inputs)
        .expect_err("a rejecting backend must fail the walk");

    assert!(
        err.contains("node 'layer0'"),
        "error must name the offending node: {err}"
    );
    assert!(
        err.contains("unsupported on backend 'stub'"),
        "error must preserve the backend's own message: {err}"
    );
}
