"""Phase-4 CNN fixture guard, Python side (``AGENTS.md`` §1.1, §5).

The bit-for-bit FHE gate lives in Rust (``runtime/tests/golden_cnn.rs``). This Python test
guards the *other* half: that the committed ``phase4_cnn_fixture.json`` is self-consistent —
its ``expected_logits``/``expected_labels`` are exactly what the quantized-integer CNN produces,
the graph round-trips and fits its radix budget, and the automatic ``Requant`` is present (so a
re-run of the compile pass would be a no-op). If the fixture drifts from the reference
arithmetic, this fails here (fast, no FHE) rather than as a confusing Rust golden violation.

No heavy ML deps and no network: it reads the committed JSON and recomputes the integer CNN
with NumPy (a core dependency).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from penumbra.bitwidth import check_bit_width_budget
from penumbra.compile import insert_requants
from penumbra.ir import Conv2dSpec, Graph, LinearSpec, PoolSpec, RequantSpec

FIXTURE = Path(__file__).resolve().parent.parent / "examples" / "mnist" / "phase4_cnn_fixture.json"


def _cleartext_logits(graph: Graph, x: list[int]) -> list[int]:
    """Integer CNN oracle: walk the graph in plain ints, mirroring each op's arithmetic.

    Identical in spirit to the Rust oracle in ``golden_cnn.rs`` — the two share the fixture as
    the meeting point, so if they ever diverge one of them fails against the committed values.
    """
    env: dict[str, list[int]] = {graph.inputs[0]: list(x)}
    for node in graph.nodes:
        v = env[node.inputs[0]]
        op = node.op
        if isinstance(op, Conv2dSpec):
            out_h = (op.in_h + 2 * op.padding - op.kernel_h) // op.stride + 1
            out_w = (op.in_w + 2 * op.padding - op.kernel_w) // op.stride + 1
            in_hw = op.in_h * op.in_w
            out: list[int] = []
            for kernel, b in zip(op.weights, op.bias, strict=True):
                for oy in range(out_h):
                    for ox in range(out_w):
                        acc = 0
                        for ic in range(op.in_channels):
                            for ky in range(op.kernel_h):
                                iy = oy * op.stride + ky - op.padding
                                for kx in range(op.kernel_w):
                                    ix = ox * op.stride + kx - op.padding
                                    if not (0 <= iy < op.in_h and 0 <= ix < op.in_w):
                                        continue
                                    w = kernel[(ic * op.kernel_h + ky) * op.kernel_w + kx]
                                    acc += w * v[ic * in_hw + iy * op.in_w + ix]
                        out.append(acc + b)
            env[node.outputs[0]] = out
        elif isinstance(op, RequantSpec):
            ceil = (1 << op.out_bits) - 1
            env[node.outputs[0]] = [min(max(val >> op.shift, 0), ceil) for val in v]
        elif isinstance(op, PoolSpec):
            out_h = (op.in_h - op.pool_h) // op.stride + 1
            out_w = (op.in_w - op.pool_w) // op.stride + 1
            out = []
            for c in range(op.channels):
                base = c * op.in_h * op.in_w
                for oy in range(out_h):
                    for ox in range(out_w):
                        vals = [
                            v[base + (oy * op.stride + ky) * op.in_w + (ox * op.stride + kx)]
                            for ky in range(op.pool_h)
                            for kx in range(op.pool_w)
                        ]
                        out.append(sum(vals) if op.mode == "avg" else max(vals))
            env[node.outputs[0]] = out
        elif isinstance(op, LinearSpec):
            env[node.outputs[0]] = [
                sum(w * val for w, val in zip(row, v, strict=True)) + b
                for row, b in zip(op.weights, op.bias, strict=True)
            ]
        else:  # pragma: no cover - the fixture only uses the ops above
            raise AssertionError(f"unexpected op {op.op_type}")
    return env[graph.outputs[0]]


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_cnn_graph_round_trips_and_fits_budget():
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    assert Graph.from_json(g.to_json()) == g, "CNN IR round-trip must be exact"
    check_bit_width_budget(g)  # raises if any tensor exceeds the radix


def test_cnn_graph_has_inserted_requant_and_is_idempotent():
    """The committed graph already carries its auto-inserted Requant (compile pass ran)."""
    g = Graph.from_dict(_fixture()["graph"])
    kinds = [n.op.op_type for n in g.nodes]
    assert kinds == ["Conv2d", "Requant", "Pool", "Linear"], kinds
    # Re-running the pass must be a no-op on the committed (already-requantized) graph.
    assert insert_requants(g) == g, "insert_requants must be idempotent on the committed graph"


def test_cnn_fixture_logits_and_labels_match_oracle():
    """Committed logits/labels are exactly what the integer CNN produces (drift guard)."""
    fx = _fixture()
    g = Graph.from_dict(fx["graph"])
    inputs = fx["test_inputs"]
    expected_logits = fx["expected_logits"]
    expected_labels = fx["expected_labels"]

    for i, x in enumerate(inputs):
        logits = _cleartext_logits(g, x)
        assert logits == expected_logits[i], f"sample {i}: logits drifted from the oracle"
        assert int(np.argmax(logits)) == expected_labels[i], f"sample {i}: label drifted"
