//! Tests for topological evaluation, Kahn ordering, and multi-output ops in penumbra-core.

use std::cell::RefCell;
use std::collections::HashMap;

type SeenBitsRecord = (&'static str, Vec<usize>, Vec<usize>);

thread_local! {
    static SEEN_BITS: RefCell<Vec<SeenBitsRecord>> = const { RefCell::new(Vec::new()) };
}

use penumbra_core::backend::{Backend, CtVec, EvalCtx};
use penumbra_core::eval::evaluate_graph;
use penumbra_core::ir::{Graph, Node, OpSpec, SCHEMA_VERSION};
use penumbra_core::ops::Op;

struct StubBackend;

impl Backend for StubBackend {
    type Ciphertext = i64;
    type ServerKey = ();
    type ClientKey = ();

    fn name(&self) -> &'static str {
        "stub"
    }

    fn build_op(&self, spec: &OpSpec) -> Result<Box<dyn Op<Self>>, String> {
        match spec {
            OpSpec::Add {} => Ok(Box::new(StubAdd)),
            OpSpec::Concat { sizes } => Ok(Box::new(StubConcat {
                sizes: sizes.clone(),
            })),
            OpSpec::Split { sizes } => Ok(Box::new(StubSplit {
                sizes: sizes.clone(),
            })),
            _ => Err(format!("op {} unsupported in StubBackend", spec.op_type())),
        }
    }

    fn build_op_with_bits(
        &self,
        spec: &OpSpec,
        input_bits: &[usize],
        output_bits: &[usize],
    ) -> Result<Box<dyn Op<Self>>, String> {
        SEEN_BITS.with(|b| {
            b.borrow_mut()
                .push((spec.op_type(), input_bits.to_vec(), output_bits.to_vec()));
        });
        self.build_op(spec)
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
        _num_blocks: usize,
    ) -> Self::Ciphertext {
        lut[*a as usize] as i64
    }

    fn keygen(&self, _num_blocks: usize) -> (Self::ClientKey, Self::ServerKey) {
        ((), ())
    }

    fn encrypt(&self, _ck: &Self::ClientKey, input: &[i64]) -> Vec<Self::Ciphertext> {
        input.to_vec()
    }

    fn decrypt_label(&self, _ck: &Self::ClientKey, out: &[Self::Ciphertext]) -> i64 {
        out.first().copied().unwrap_or(0)
    }

    fn decrypt_vec(&self, _ck: &Self::ClientKey, out: &[Self::Ciphertext]) -> Vec<i64> {
        out.to_vec()
    }

    fn serialize_cts(&self, _cts: &[Self::Ciphertext]) -> Result<Vec<u8>, String> {
        Ok(Vec::new())
    }

    fn deserialize_cts(&self, _bytes: &[u8]) -> Result<Vec<Self::Ciphertext>, String> {
        Ok(Vec::new())
    }

    fn serialize_client_key(
        &self,
        _ck: &Self::ClientKey,
        _num_blocks: usize,
    ) -> Result<Vec<u8>, String> {
        Ok(Vec::new())
    }

    fn serialize_server_key(
        &self,
        _sk: &Self::ServerKey,
        _num_blocks: usize,
    ) -> Result<Vec<u8>, String> {
        Ok(Vec::new())
    }
}

struct StubAdd;

impl Op<StubBackend> for StubAdd {
    fn eval(&self, _ctx: &EvalCtx<()>, _input: &CtVec<StubBackend>) -> CtVec<StubBackend> {
        panic!("Add is a two-input op; call eval_n")
    }

    fn eval_n(&self, _ctx: &EvalCtx<()>, inputs: &[&CtVec<StubBackend>]) -> CtVec<StubBackend> {
        assert_eq!(inputs.len(), 2);
        inputs[0]
            .iter()
            .zip(inputs[1])
            .map(|(a, b)| a + b)
            .collect()
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        panic!("Add is a two-input op; call output_bits_n")
    }

    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        input_bits[0].max(input_bits[1]) + 1
    }
}

struct StubConcat {
    sizes: Vec<usize>,
}

impl Op<StubBackend> for StubConcat {
    fn eval(&self, _ctx: &EvalCtx<()>, _input: &CtVec<StubBackend>) -> CtVec<StubBackend> {
        panic!("Concat is a multi-input op; call eval_n")
    }

    fn eval_n(&self, _ctx: &EvalCtx<()>, inputs: &[&CtVec<StubBackend>]) -> CtVec<StubBackend> {
        assert_eq!(inputs.len(), self.sizes.len());
        inputs.iter().flat_map(|t| t.iter().copied()).collect()
    }

    fn output_bits(&self, _input_bits: usize) -> usize {
        panic!("Concat is a multi-input op; call output_bits_n")
    }

    fn output_bits_n(&self, input_bits: &[usize]) -> usize {
        input_bits.iter().copied().max().unwrap_or(0)
    }
}

struct StubSplit {
    sizes: Vec<usize>,
}

impl Op<StubBackend> for StubSplit {
    fn eval(&self, _ctx: &EvalCtx<()>, _input: &CtVec<StubBackend>) -> CtVec<StubBackend> {
        panic!("Split is a multi-output op; call eval_multi")
    }

    fn eval_multi(
        &self,
        _ctx: &EvalCtx<()>,
        inputs: &[&CtVec<StubBackend>],
    ) -> Vec<CtVec<StubBackend>> {
        assert_eq!(inputs.len(), 1);
        let mut results = Vec::with_capacity(self.sizes.len());
        let mut offset = 0;
        for &sz in &self.sizes {
            results.push(inputs[0][offset..offset + sz].to_vec());
            offset += sz;
        }
        results
    }

    fn output_bits(&self, input_bits: usize) -> usize {
        input_bits
    }

    fn output_bits_multi(&self, input_bits: &[usize]) -> Vec<usize> {
        vec![input_bits[0]; self.sizes.len()]
    }
}

#[test]
fn branching_dag_evaluates_correctly() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 8,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
        nodes: vec![
            Node {
                name: "split".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["s0".to_string(), "s1".to_string()],
                op: OpSpec::Split { sizes: vec![2, 2] },
            },
            Node {
                name: "add_half".to_string(),
                inputs: vec!["s0".to_string(), "s1".to_string()],
                outputs: vec!["sum".to_string()],
                op: OpSpec::Add {},
            },
            Node {
                name: "concat".to_string(),
                inputs: vec!["s0".to_string(), "sum".to_string()],
                outputs: vec!["c".to_string()],
                op: OpSpec::Concat { sizes: vec![2, 2] },
            },
            Node {
                name: "final_add".to_string(),
                inputs: vec!["x".to_string(), "c".to_string()],
                outputs: vec!["out".to_string()],
                op: OpSpec::Add {},
            },
        ],
    };

    let backend = StubBackend;
    let ctx = EvalCtx::new(&(), 4);
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![10, 20, 30, 40]);

    let outputs = evaluate_graph(&backend, &ctx, &graph, inputs).expect("evaluation succeeds");
    // s0 = [10, 20], s1 = [30, 40]
    // sum = [40, 60]
    // c = [10, 20, 40, 60]
    // out = x + c = [10+10, 20+20, 30+40, 40+60] = [20, 40, 70, 100]
    assert_eq!(outputs.get("out"), Some(&vec![20, 40, 70, 100]));
}

#[test]
fn non_topological_node_order_evaluates_correctly() {
    // Exact same graph as above, but with nodes in reverse order:
    // final_add, concat, add_half, split
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 8,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
        nodes: vec![
            Node {
                name: "final_add".to_string(),
                inputs: vec!["x".to_string(), "c".to_string()],
                outputs: vec!["out".to_string()],
                op: OpSpec::Add {},
            },
            Node {
                name: "concat".to_string(),
                inputs: vec!["s0".to_string(), "sum".to_string()],
                outputs: vec!["c".to_string()],
                op: OpSpec::Concat { sizes: vec![2, 2] },
            },
            Node {
                name: "add_half".to_string(),
                inputs: vec!["s0".to_string(), "s1".to_string()],
                outputs: vec!["sum".to_string()],
                op: OpSpec::Add {},
            },
            Node {
                name: "split".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["s0".to_string(), "s1".to_string()],
                op: OpSpec::Split { sizes: vec![2, 2] },
            },
        ],
    };

    let backend = StubBackend;
    let ctx = EvalCtx::new(&(), 4);
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![10, 20, 30, 40]);

    let outputs =
        evaluate_graph(&backend, &ctx, &graph, inputs).expect("stable Kahn reorders and succeeds");
    assert_eq!(outputs.get("out"), Some(&vec![20, 40, 70, 100]));
}

#[test]
fn graph_with_cycle_fails_with_cycle_message() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 8,
        input_bits: 4,
        inputs: vec!["x".to_string()],
        outputs: vec!["a".to_string()],
        nodes: vec![
            Node {
                name: "nodeA".to_string(),
                inputs: vec!["b".to_string()],
                outputs: vec!["a".to_string()],
                op: OpSpec::Add {},
            },
            Node {
                name: "nodeB".to_string(),
                inputs: vec!["a".to_string()],
                outputs: vec!["b".to_string()],
                op: OpSpec::Add {},
            },
        ],
    };

    let backend = StubBackend;
    let ctx = EvalCtx::new(&(), 4);
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![1, 2]);

    let err = evaluate_graph(&backend, &ctx, &graph, inputs).expect_err("cycle must fail");
    assert!(err.contains("graph has a cycle: node(s)"), "got: {err}");
    assert!(
        err.contains("are never ready — their inputs depend on their own outputs"),
        "got: {err}"
    );
}

#[test]
fn node_declaring_wrong_number_of_outputs_fails() {
    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 8,
        input_bits: 4,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["o1".to_string(), "o2".to_string()],
        nodes: vec![Node {
            name: "bad_add".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["o1".to_string(), "o2".to_string()],
            op: OpSpec::Add {},
        }],
    };

    let backend = StubBackend;
    let ctx = EvalCtx::new(&(), 4);
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vec![1, 2]);
    inputs.insert("b".to_string(), vec![3, 4]);

    let err = evaluate_graph(&backend, &ctx, &graph, inputs).expect_err("arity mismatch must fail");
    assert!(
        err.contains("node 'bad_add' (Add) declares 2 output tensor(s) but its op produced 1"),
        "got: {err}"
    );
}

#[test]
fn eval_loop_passes_derived_bit_widths_to_backend() {
    SEEN_BITS.with(|b| b.borrow_mut().clear());

    let graph = Graph {
        schema_version: SCHEMA_VERSION.to_string(),
        num_blocks: 8,
        input_bits: 4,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["out".to_string()],
        nodes: vec![
            Node {
                name: "add1".to_string(),
                op: OpSpec::Add {},
                inputs: vec!["a".to_string(), "b".to_string()],
                outputs: vec!["add1_out".to_string()],
            },
            Node {
                name: "split".to_string(),
                op: OpSpec::Split { sizes: vec![1, 1] },
                inputs: vec!["add1_out".to_string()],
                outputs: vec!["s0".to_string(), "s1".to_string()],
            },
            Node {
                name: "out_node".to_string(),
                op: OpSpec::Add {},
                inputs: vec!["s0".to_string(), "s1".to_string()],
                outputs: vec!["out".to_string()],
            },
        ],
    };

    let backend = StubBackend;
    let ctx = EvalCtx::new(&(), 8);
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vec![1, 2]);
    inputs.insert("b".to_string(), vec![3, 4]);

    let res =
        evaluate_graph(&backend, &ctx, &graph, inputs).expect("evaluate_graph should succeed");
    assert_eq!(res["out"], vec![4 + 6]);

    let recorded = SEEN_BITS.with(|b| b.borrow().clone());
    assert_eq!(
        recorded,
        vec![
            ("Add", vec![4, 4], vec![5]),
            ("Split", vec![5], vec![5, 5]),
            ("Add", vec![5, 5], vec![6]),
        ]
    );
}
