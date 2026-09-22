//! Generic evaluation session driving benchmarks and reports.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use penumbra_core::backend::{Backend, CtVec, EvalCtx};
use penumbra_core::eval::evaluate_graph_profiled;
use penumbra_core::ir::Graph;
use penumbra_core::profile::GraphProfile;

/// A configured evaluation session binding backend, keys, and graph capacity.
pub struct Session<B: Backend> {
    pub backend: B,
    pub ck: B::ClientKey,
    pub sk: B::ServerKey,
    pub num_blocks: usize,
    pub keygen: Duration,
}

impl<B: Backend> Session<B> {
    /// Run the backend's own budget preflight, then generate keys for this graph.
    pub fn new(backend: B, graph: &Graph) -> Result<Self, String> {
        backend.check_graph_budget(graph)?;
        let t = Instant::now();
        let (ck, sk) = backend.keygen(graph.num_blocks);
        Ok(Self {
            backend,
            ck,
            sk,
            num_blocks: graph.num_blocks,
            keygen: t.elapsed(),
        })
    }

    /// Encrypt an input vector into ciphertexts using the session client key.
    pub fn encrypt(&self, input: &[i64]) -> CtVec<B> {
        self.backend.encrypt(&self.ck, input)
    }

    /// The measured forward pass: one call into the shared walker, nothing else.
    pub fn eval(
        &self,
        graph: &Graph,
        input: &CtVec<B>,
    ) -> Result<(CtVec<B>, GraphProfile), String> {
        if graph.inputs.len() != 1 {
            return Err(format!(
                "Session::eval expects single-input graph; got {} inputs: {:?}",
                graph.inputs.len(),
                graph.inputs
            ));
        }
        let input_name = graph.inputs[0].clone();
        let mut env = HashMap::with_capacity(1);
        env.insert(input_name, input.clone());

        let ctx = EvalCtx::new(&self.sk, self.num_blocks);
        let mut profile = GraphProfile::default();
        let mut outputs = evaluate_graph_profiled(&self.backend, &ctx, graph, env, &mut profile)?;

        if graph.outputs.len() != 1 {
            return Err(format!(
                "Session::eval expects single-output graph; got {} outputs: {:?}",
                graph.outputs.len(),
                graph.outputs
            ));
        }
        let output_name = &graph.outputs[0];
        let out = outputs
            .remove(output_name)
            .ok_or_else(|| format!("missing output tensor '{output_name}' in result"))?;
        Ok((out, profile))
    }

    /// Decrypt output ciphertexts to a vector of signed integers.
    pub fn decrypt(&self, cts: &[B::Ciphertext]) -> Vec<i64> {
        self.backend.decrypt_vec(&self.ck, cts)
    }

    /// Decrypt single-output ciphertext to a scalar label.
    pub fn decrypt_label(&self, cts: &[B::Ciphertext]) -> i64 {
        self.backend.decrypt_label(&self.ck, cts)
    }

    /// Return wire serialized sizes: (client_key_bytes, server_key_bytes).
    pub fn key_bytes(&self) -> Result<(usize, usize), String> {
        let ck_bytes = self
            .backend
            .serialize_client_key(&self.ck, self.num_blocks)?;
        let sk_bytes = self
            .backend
            .serialize_server_key(&self.sk, self.num_blocks)?;
        Ok((ck_bytes.len(), sk_bytes.len()))
    }

    /// Return serialized size in bytes of a ciphertext slice.
    pub fn ct_bytes(&self, cts: &[B::Ciphertext]) -> Result<usize, String> {
        let bytes = self.backend.serialize_cts(cts)?;
        Ok(bytes.len())
    }
}
