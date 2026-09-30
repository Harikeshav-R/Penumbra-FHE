//! Public API tests for penumbra-core eval loop, Op trait defaults, and profiling.

use std::collections::HashMap;

use penumbra_core::backend::{Backend, CtVec, EvalCtx};
use penumbra_core::bitwidth::check_bit_width_budget;
use penumbra_core::eval::{evaluate, evaluate_graph, evaluate_graph_profiled};
use penumbra_core::ir::{Graph, Node, OpSpec, SCHEMA_VERSION};
use penumbra_core::ops::Op;
use penumbra_core::profile::GraphProfile;

struct TestBackend;

impl Backend for TestBackend {
    type Ciphertext = i64;
    type ServerKey = ();
    type ClientKey = ();

    fn name(&self) -> &'static str {
        "test_backend"
    }

    fn build_op(&self, spec: &OpSpec) -> Result<Box<dyn Op<Self>>, String> {
        match spec {
            OpSpec::Add {} => Ok(Box::new(TestAdd)),
            _ => Err("unsupported".into()),
        }
    }

    fn check_graph_budget(&self, _graph: &Graph) -> Result<(), String> {
        Ok(())
    }

    fn create_trivial_zero(&self, _sk: &Self::ServerKey, _num_blocks: usize) -> Self::Ciphertext {
        0
    }

    fn add(
        &self,
        _sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        b: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        a + b
    }

    fn scalar_mul(
        &self,
        _sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext {
        a * scalar
    }

    fn scalar_add(
        &self,
        _sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext {
        a + scalar
    }

    fn scalar_ge(
        &self,
        _sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        threshold: i64,
    ) -> Self::Ciphertext {
        if *a >= threshold {
            1
        } else {
            0
        }
    }

    fn max(
        &self,
        _sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        b: &Self::Ciphertext,
    ) -> Self::Ciphertext {
        (*a).max(*b)
    }

    fn scalar_max(
        &self,
        _sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext {
        (*a).max(scalar)
    }

    fn scalar_min(
        &self,
        _sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        scalar: i64,
    ) -> Self::Ciphertext {
        (*a).min(scalar)
    }

    fn scalar_right_shift(
        &self,
        _sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        shift: u32,
    ) -> Self::Ciphertext {
        *a >> shift
    }

    fn apply_lut(
        &self,
        _sk: &Self::ServerKey,
        a: &Self::Ciphertext,
        lut: &[u64],
        _nb: usize,
    ) -> Self::Ciphertext {
        lut[*a as usize] as i64
    }

    fn keygen(&self, _nb: usize) -> (Self::ClientKey, Self::ServerKey) {
        ((), ())
    }

    fn encrypt(&self, _ck: &Self::ClientKey, input: &[i64]) -> Vec<Self::Ciphertext> {
        input.to_vec()
    }

    fn decrypt_label(&self, _ck: &Self::ClientKey, out: &[Self::Ciphertext]) -> i64 {
        out[0]
    }

    fn decrypt_vec(&self, _ck: &Self::ClientKey, out: &[Self::Ciphertext]) -> Vec<i64> {
        out.to_vec()
    }

    fn serialize_cts(&self, _cts: &[Self::Ciphertext]) -> Result<Vec<u8>, String> {
        Ok(vec![])
    }

    fn deserialize_cts(&self, _bytes: &[u8]) -> Result<Vec<Self::Ciphertext>, String> {
        Ok(vec![])
    }

    fn serialize_client_key(&self, _ck: &Self::ClientKey, _nb: usize) -> Result<Vec<u8>, String> {
        Ok(vec![])
    }

    fn serialize_server_key(&self, _sk: &Self::ServerKey, _nb: usize) -> Result<Vec<u8>, String> {
        Ok(vec![])
    }

    fn measured_counters(&self) -> Vec<(&'static str, u64)> {
        vec![("test_counter", 10)]
    }
}

struct TestAdd;

impl Op<TestBackend> for TestAdd {
    fn eval(&self, _ctx: &EvalCtx<()>, _inputs: &CtVec<TestBackend>) -> CtVec<TestBackend> {
        unreachable!()
    }

    fn output_bits(&self, bits: usize) -> usize {
        bits
    }

    fn eval_n(&self, _ctx: &EvalCtx<()>, inputs: &[&CtVec<TestBackend>]) -> CtVec<TestBackend> {
        let a = &inputs[0];
        let b = &inputs[1];
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        input_bits[0].max(input_bits[1]) + 1
    }
}

struct SingleInputOp;

impl Op<TestBackend> for SingleInputOp {
    fn eval(&self, _ctx: &EvalCtx<()>, inputs: &CtVec<TestBackend>) -> CtVec<TestBackend> {
        inputs.iter().map(|x| x * 2).collect()
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        input_bits + 1
    }
}

#[test]
fn test_op_trait_defaults() {
    let op = SingleInputOp;
    let ctx = EvalCtx {
        sk: &(),
        num_blocks: 4,
    };
    let input = vec![5i64, 10];
    let slice = [&input];

    assert_eq!(op.eval_n(&ctx, &slice), vec![10, 20]);
    assert_eq!(op.output_bits_n(&[4]), 5);
    assert_eq!(op.eval_multi(&ctx, &slice), vec![vec![10, 20]]);
    assert_eq!(op.output_bits_multi(&[4]), vec![5]);
    assert_eq!(op.internal_bits_n(&[4]), 5);
    assert!(op.cost(&[2]).is_empty());
    assert_eq!(TestBackend.measured_counters(), vec![("test_counter", 10)]);

    let ops: Vec<Box<dyn Op<TestBackend>>> = vec![Box::new(SingleInputOp), Box::new(SingleInputOp)];
    assert!(check_bit_width_budget(&ops, 2, 4).is_ok());
    assert!(check_bit_width_budget(&ops, 7, 4).is_err());
}

#[test]
fn test_evaluate_linear_chain() {
    let ctx = EvalCtx {
        sk: &(),
        num_blocks: 4,
    };
    let ops: Vec<Box<dyn Op<TestBackend>>> = vec![Box::new(SingleInputOp), Box::new(SingleInputOp)];
    let input = vec![3i64];
    let res = evaluate(&ctx, &ops, &input);
    assert_eq!(res, vec![12]);
}

#[test]
fn test_evaluate_graph_profiled_and_error_paths() {
    let backend = TestBackend;
    let ctx = EvalCtx {
        sk: &(),
        num_blocks: 4,
    };

    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["sum".to_string()],
        nodes: vec![Node {
            name: "add".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["sum".to_string()],
            op: OpSpec::Add {},
        }],
    };

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vec![10i64, 20]);
    inputs.insert("b".to_string(), vec![3i64, 4]);

    let mut prof = GraphProfile::default();
    let out = evaluate_graph_profiled(&backend, &ctx, &graph, inputs.clone(), &mut prof).unwrap();
    assert_eq!(out["sum"], vec![13, 24]);
    assert_eq!(prof.backend, "test_backend");
    assert_eq!(prof.nodes.len(), 1);

    // Mismatched declared vs provided inputs
    let mut bad_inputs = HashMap::new();
    bad_inputs.insert("a".to_string(), vec![10]);
    let err = evaluate_graph(&backend, &ctx, &graph, bad_inputs).unwrap_err();
    assert!(err.contains("do not match the provided input tensors"));

    // Node with empty inputs
    let empty_node_graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["a".to_string()],
        outputs: vec!["out".to_string()],
        nodes: vec![Node {
            name: "empty_node".to_string(),
            inputs: vec![],
            outputs: vec!["out".to_string()],
            op: OpSpec::Add {},
        }],
    };
    let mut inputs2 = HashMap::new();
    inputs2.insert("a".to_string(), vec![1]);
    let err = evaluate_graph(&backend, &ctx, &empty_node_graph, inputs2).unwrap_err();
    assert!(err.contains("has no inputs"));

    // Missing output from graph
    let missing_out_graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["nonexistent".to_string()],
        nodes: vec![Node {
            name: "add".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["sum".to_string()],
            op: OpSpec::Add {},
        }],
    };
    let err = evaluate_graph(&backend, &ctx, &missing_out_graph, inputs).unwrap_err();
    assert!(err.contains("no node produced it"));

    // Node reading a tensor which no earlier node produced
    let unlinked_node_graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["a".to_string()],
        outputs: vec!["out".to_string()],
        nodes: vec![Node {
            name: "unlinked".to_string(),
            inputs: vec!["missing".to_string()],
            outputs: vec!["out".to_string()],
            op: OpSpec::Activation {
                lut: vec![0, 1],
                output_bits: 1,
            },
        }],
    };
    let mut inputs_unlinked = HashMap::new();
    inputs_unlinked.insert("a".to_string(), vec![1]);
    let err = evaluate_graph(&backend, &ctx, &unlinked_node_graph, inputs_unlinked).unwrap_err();
    assert!(err.contains("which no node produces and is not a graph input"));
}

#[test]
fn test_evaluate_graph_profiled_counter_contract_violations() {
    struct UnstableCountersBackend {
        inner: TestBackend,
        mode: std::sync::atomic::AtomicUsize,
    }

    impl Backend for UnstableCountersBackend {
        type Ciphertext = i64;
        type ServerKey = ();
        type ClientKey = ();

        fn name(&self) -> &'static str {
            "unstable"
        }

        fn build_op(&self, spec: &OpSpec) -> Result<Box<dyn Op<Self>>, String> {
            let op = self.inner.build_op(spec)?;
            struct WrappedOp(Box<dyn Op<TestBackend>>);
            impl Op<UnstableCountersBackend> for WrappedOp {
                fn eval(
                    &self,
                    _ctx: &EvalCtx<()>,
                    inputs: &CtVec<UnstableCountersBackend>,
                ) -> CtVec<UnstableCountersBackend> {
                    let dummy_ctx = EvalCtx {
                        sk: &(),
                        num_blocks: 4,
                    };
                    self.0.eval(&dummy_ctx, inputs)
                }
                fn output_bits(&self, bits: usize) -> usize {
                    self.0.output_bits(bits)
                }
                fn eval_n(
                    &self,
                    _ctx: &EvalCtx<()>,
                    inputs: &[&CtVec<UnstableCountersBackend>],
                ) -> CtVec<UnstableCountersBackend> {
                    let dummy_ctx = EvalCtx {
                        sk: &(),
                        num_blocks: 4,
                    };
                    self.0.eval_n(&dummy_ctx, inputs)
                }
                fn output_bits_n(&self, input_bits: &[usize]) -> usize {
                    self.0.output_bits_n(input_bits)
                }
            }
            Ok(Box::new(WrappedOp(op)))
        }

        fn check_graph_budget(&self, graph: &Graph) -> Result<(), String> {
            self.inner.check_graph_budget(graph)
        }

        fn measured_counters(&self) -> Vec<(&'static str, u64)> {
            let m = self.mode.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            match m {
                0 => vec![("counter_a", 1)],
                1 => vec![("counter_a", 1), ("counter_b", 2)],
                2 => vec![("counter_x", 1)],
                _ => vec![("counter_y", 1)],
            }
        }

        fn create_trivial_zero(&self, sk: &Self::ServerKey, num_blocks: usize) -> Self::Ciphertext {
            self.inner.create_trivial_zero(sk, num_blocks)
        }
        fn add(
            &self,
            sk: &Self::ServerKey,
            a: &Self::Ciphertext,
            b: &Self::Ciphertext,
        ) -> Self::Ciphertext {
            self.inner.add(sk, a, b)
        }
        fn scalar_mul(
            &self,
            sk: &Self::ServerKey,
            a: &Self::Ciphertext,
            scalar: i64,
        ) -> Self::Ciphertext {
            self.inner.scalar_mul(sk, a, scalar)
        }
        fn scalar_add(
            &self,
            sk: &Self::ServerKey,
            a: &Self::Ciphertext,
            scalar: i64,
        ) -> Self::Ciphertext {
            self.inner.scalar_add(sk, a, scalar)
        }
        fn scalar_ge(
            &self,
            sk: &Self::ServerKey,
            a: &Self::Ciphertext,
            threshold: i64,
        ) -> Self::Ciphertext {
            self.inner.scalar_ge(sk, a, threshold)
        }
        fn max(
            &self,
            sk: &Self::ServerKey,
            a: &Self::Ciphertext,
            b: &Self::Ciphertext,
        ) -> Self::Ciphertext {
            self.inner.max(sk, a, b)
        }
        fn scalar_max(
            &self,
            sk: &Self::ServerKey,
            a: &Self::Ciphertext,
            scalar: i64,
        ) -> Self::Ciphertext {
            self.inner.scalar_max(sk, a, scalar)
        }
        fn scalar_min(
            &self,
            sk: &Self::ServerKey,
            a: &Self::Ciphertext,
            scalar: i64,
        ) -> Self::Ciphertext {
            self.inner.scalar_min(sk, a, scalar)
        }
        fn scalar_right_shift(
            &self,
            sk: &Self::ServerKey,
            a: &Self::Ciphertext,
            shift: u32,
        ) -> Self::Ciphertext {
            self.inner.scalar_right_shift(sk, a, shift)
        }
        fn apply_lut(
            &self,
            sk: &Self::ServerKey,
            a: &Self::Ciphertext,
            lut: &[u64],
            num_blocks: usize,
        ) -> Self::Ciphertext {
            self.inner.apply_lut(sk, a, lut, num_blocks)
        }
        fn keygen(&self, num_blocks: usize) -> (Self::ClientKey, Self::ServerKey) {
            self.inner.keygen(num_blocks)
        }
        fn encrypt(&self, ck: &Self::ClientKey, values: &[i64]) -> Vec<Self::Ciphertext> {
            self.inner.encrypt(ck, values)
        }
        fn decrypt_label(&self, ck: &Self::ClientKey, cts: &[Self::Ciphertext]) -> i64 {
            self.inner.decrypt_label(ck, cts)
        }
        fn decrypt_vec(&self, ck: &Self::ClientKey, cts: &[Self::Ciphertext]) -> Vec<i64> {
            self.inner.decrypt_vec(ck, cts)
        }
        fn serialize_cts(&self, cts: &[Self::Ciphertext]) -> Result<Vec<u8>, String> {
            self.inner.serialize_cts(cts)
        }
        fn deserialize_cts(&self, bytes: &[u8]) -> Result<Vec<Self::Ciphertext>, String> {
            self.inner.deserialize_cts(bytes)
        }
        fn serialize_client_key(
            &self,
            ck: &Self::ClientKey,
            num_blocks: usize,
        ) -> Result<Vec<u8>, String> {
            self.inner.serialize_client_key(ck, num_blocks)
        }
        fn serialize_server_key(
            &self,
            sk: &Self::ServerKey,
            num_blocks: usize,
        ) -> Result<Vec<u8>, String> {
            self.inner.serialize_server_key(sk, num_blocks)
        }
    }

    let backend = UnstableCountersBackend {
        inner: TestBackend,
        mode: std::sync::atomic::AtomicUsize::new(0),
    };
    let ctx = EvalCtx {
        sk: &(),
        num_blocks: 4,
    };
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 4,
        input_bits: 4,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["out".to_string()],
        nodes: vec![Node {
            name: "add".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["out".to_string()],
            op: OpSpec::Add {},
        }],
    };

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vec![1i64]);
    inputs.insert("b".to_string(), vec![2i64]);

    // First run: mode 0 -> 1 (length mismatch)
    let mut prof = GraphProfile::default();
    let err =
        evaluate_graph_profiled(&backend, &ctx, &graph, inputs.clone(), &mut prof).unwrap_err();
    assert!(err.contains(
        "violated the stable-order contract for measured_counters: before had 1 counters, after had 2"
    ));

    // Second run: mode 2 -> 3 (name mismatch)
    let mut prof2 = GraphProfile::default();
    let err2 = evaluate_graph_profiled(&backend, &ctx, &graph, inputs, &mut prof2).unwrap_err();
    assert!(err2.contains("counter name mismatch ('counter_x' vs 'counter_y')"));
}
