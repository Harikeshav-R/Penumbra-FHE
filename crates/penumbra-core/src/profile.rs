//! Per-node timing and op-counting for the graph walker (Layer 2).

use std::collections::BTreeMap;
use std::time::Duration;

/// One node's measured execution: how long it took to build and to evaluate, the tensor
/// shapes it saw, and the backend's own cost counters for it.
#[derive(Debug, Clone)]
pub struct NodeProfile {
    pub name: String,
    pub op_type: &'static str,
    pub build: Duration,
    pub eval: Duration,
    pub input_lens: Vec<usize>,
    pub output_len: usize,
    /// Backend-declared cost counters (`Op::cost`). Names are opaque to Layer 2.
    pub counters: Vec<(&'static str, u64)>,
}

/// A whole graph walk, node by node, in evaluation order.
#[derive(Debug, Clone, Default)]
pub struct GraphProfile {
    pub backend: &'static str,
    pub total: Duration,
    pub nodes: Vec<NodeProfile>,
}

/// Aggregated build+eval time and call count for one op type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpTypeStats {
    pub calls: u32,
    pub build: Duration,
    pub eval: Duration,
}

impl GraphProfile {
    /// Aggregate stats grouped by op type.
    pub fn by_op_type(&self) -> BTreeMap<&'static str, OpTypeStats> {
        let mut stats: BTreeMap<&'static str, OpTypeStats> = BTreeMap::new();
        for node in &self.nodes {
            let entry = stats.entry(node.op_type).or_default();
            entry.calls += 1;
            entry.build += node.build;
            entry.eval += node.eval;
        }
        stats
    }

    /// Every counter summed across nodes — the graph's cost proxy.
    pub fn counter_totals(&self) -> BTreeMap<&'static str, u64> {
        let mut totals: BTreeMap<&'static str, u64> = BTreeMap::new();
        for node in &self.nodes {
            for &(name, count) in &node.counters {
                *totals.entry(name).or_default() += count;
            }
        }
        totals
    }

    /// The op-type sequence in evaluation order — used to assert backend parity.
    pub fn op_types(&self) -> Vec<&'static str> {
        self.nodes.iter().map(|n| n.op_type).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_profile_aggregations() {
        let mut profile = GraphProfile {
            backend: "mock",
            total: Duration::from_millis(50),
            nodes: Vec::new(),
        };

        profile.nodes.push(NodeProfile {
            name: "node1".into(),
            op_type: "Linear",
            build: Duration::from_millis(1),
            eval: Duration::from_millis(10),
            input_lens: vec![64],
            output_len: 10,
            counters: vec![("rotations", 2), ("rescales", 1)],
        });

        profile.nodes.push(NodeProfile {
            name: "node2".into(),
            op_type: "Requant",
            build: Duration::from_millis(2),
            eval: Duration::from_millis(20),
            input_lens: vec![10],
            output_len: 10,
            counters: vec![("rescales", 3), ("poly_evals", 1)],
        });

        profile.nodes.push(NodeProfile {
            name: "node3".into(),
            op_type: "Linear",
            build: Duration::from_millis(3),
            eval: Duration::from_millis(15),
            input_lens: vec![10],
            output_len: 2,
            counters: vec![("rotations", 5)],
        });

        assert_eq!(profile.op_types(), vec!["Linear", "Requant", "Linear"]);

        let stats = profile.by_op_type();
        assert_eq!(stats.len(), 2);
        assert_eq!(stats["Linear"].calls, 2);
        assert_eq!(stats["Linear"].build, Duration::from_millis(4));
        assert_eq!(stats["Linear"].eval, Duration::from_millis(25));
        assert_eq!(stats["Requant"].calls, 1);

        let totals = profile.counter_totals();
        assert_eq!(totals.get("rotations"), Some(&7));
        assert_eq!(totals.get("rescales"), Some(&4));
        assert_eq!(totals.get("poly_evals"), Some(&1));
        assert_eq!(totals.get("unknown"), None);
    }
}
