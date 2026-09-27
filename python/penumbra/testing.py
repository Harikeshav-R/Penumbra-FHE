"""Random small backend-neutral models and property test corpus (ROADMAP Phase 11).

Provides random model generation across four model families (mlp, cnn, branch, tree)
with strict bit-width tracking and CKKS domain sizing, and a committed property corpus
consumed by Rust TFHE and CKKS replays.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from penumbra.bitwidth import check_bit_width_budget, minimal_num_blocks, propagate_bit_widths
from penumbra.ir import (
    SCHEMA_VERSION,
    ActivationSpec,
    AddSpec,
    ArgmaxSpec,
    CompareSpec,
    ConcatSpec,
    Conv2dSpec,
    Graph,
    LinearSpec,
    Node,
    PoolSpec,
    RequantSpec,
    SplitSpec,
)
from penumbra.quantization.lut import identity_clamp_lut
from penumbra.reference import evaluate_graph_int

FAMILIES: tuple[str, ...] = ("mlp", "cnn", "branch", "tree")
MAX_NUM_BLOCKS = 8
CORPUS_SCHEMA = "penumbra-property-corpus/1"
CORPUS_BASE_SEED = 20260926
CORPUS_MODELS_PER_FAMILY = 6
CORPUS_INPUTS_PER_MODEL = 2

# Input widths hard-coded in the CKKS polynomial fits (crates/penumbra-ckks/src/backend.rs
# build_op: Requant input_bits = 14, Argmax 16, Compare 8). Generated graphs must stay inside
# them so CKKS evaluates within its approximation domain.
CKKS_REQUANT_INPUT_BITS = 14
CKKS_ARGMAX_INPUT_BITS = 16
CKKS_COMPARE_INPUT_BITS = 8


@dataclass(frozen=True)
class RandomModel:
    """A randomly generated backend-neutral model with test inputs and oracle outputs."""

    name: str
    family: str
    seed: int
    graph: Graph
    test_inputs: list[list[int]]

    def expected_outputs(self) -> list[list[int]]:
        """Evaluate the model in integer cleartext for each test input."""
        out_name = self.graph.outputs[0]
        return [evaluate_graph_int(self.graph, {"x": row})[out_name] for row in self.test_inputs]


def random_model(
    seed: int,
    family: str,
    *,
    n_inputs: int = CORPUS_INPUTS_PER_MODEL,
) -> RandomModel:
    """Generate a random small backend-neutral model within resource budgets."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")

    rng = random.Random(seed)
    name = f"prop_{family}_{seed}"

    if family == "mlp":
        n_in = rng.randint(2, 6)
        input_bits = rng.randint(2, 4)
        h = rng.randint(2, 4)
        fc1_w = [[rng.randint(-3, 3) for _ in range(n_in)] for _ in range(h)]
        fc1_b = [rng.randint(-8, 8) for _ in range(h)]
        nodes = [
            Node(
                name="fc1",
                inputs=["x"],
                outputs=["h1"],
                op=LinearSpec(weights=fc1_w, bias=fc1_b, weight_bits=3),
            )
        ]
        out_bits = rng.choice([1, 2])
        shift = rng.randint(1, 5)
        mult = rng.randint(1, 3)
        round_bias = rng.choice([0, 1 << (shift - 1)])
        if rng.random() < 0.25:
            clamp_lo = rng.randint(-8, -1)
            u_min = (clamp_lo * mult + round_bias) >> shift
            zero_point = max(0, -u_min)
        else:
            clamp_lo = 0
            zero_point = 0
        nodes.append(
            Node(
                name="rq1",
                inputs=["h1"],
                outputs=["h1q"],
                op=RequantSpec(
                    shift=shift,
                    mult=mult,
                    round_bias=round_bias,
                    clamp_lo=clamp_lo,
                    zero_point=zero_point,
                    out_bits=out_bits,
                    clamp_lut=identity_clamp_lut(out_bits),
                ),
            )
        )
        curr_out = "h1q"
        if rng.random() < 0.5:
            act_lut = [rng.randint(0, 3) for _ in range(4)]
            nodes.append(
                Node(
                    name="act1",
                    inputs=["h1q"],
                    outputs=["h1a"],
                    op=ActivationSpec(lut=act_lut, output_bits=2),
                )
            )
            curr_out = "h1a"
        k = rng.randint(1, 3)
        fc2_w = [[rng.randint(-3, 3) for _ in range(h)] for _ in range(k)]
        fc2_b = [rng.randint(-8, 8) for _ in range(k)]
        nodes.append(
            Node(
                name="fc2",
                inputs=[curr_out],
                outputs=["logits"],
                op=LinearSpec(weights=fc2_w, bias=fc2_b, weight_bits=3),
            )
        )
        final_out = "logits"
        if k == 1 and rng.random() < 0.5:
            thresh = rng.randint(-4, 4)
            nodes.append(
                Node(
                    name="argmax",
                    inputs=["logits"],
                    outputs=["label"],
                    op=ArgmaxSpec(threshold=thresh),
                )
            )
            final_out = "label"

        test_inputs = [
            [rng.randint(0, (1 << input_bits) - 1) for _ in range(n_in)] for _ in range(n_inputs)
        ]

    elif family == "cnn":
        c_in = rng.randint(1, 2)
        side = rng.randint(3, 4)
        input_bits = rng.randint(2, 3)
        out_c = rng.randint(1, 2)
        stride = rng.choice([1, 2])
        padding = rng.choice([0, 1])
        conv_w = [[rng.randint(-2, 2) for _ in range(c_in * 4)] for _ in range(out_c)]
        conv_b = [rng.randint(-4, 4) for _ in range(out_c)]
        nodes = [
            Node(
                name="conv",
                inputs=["x"],
                outputs=["c1"],
                op=Conv2dSpec(
                    weights=conv_w,
                    bias=conv_b,
                    weight_bits=3,
                    in_h=side,
                    in_w=side,
                    in_channels=c_in,
                    kernel_h=2,
                    kernel_w=2,
                    stride=stride,
                    padding=padding,
                ),
            )
        ]
        out_h = (side + 2 * padding - 2) // stride + 1
        out_w = (side + 2 * padding - 2) // stride + 1
        rq_out_bits = rng.choice([1, 2])
        if rng.random() < 0.5:
            mults = [rng.randint(1, 3) for _ in range(out_c)]
            shifts = [rng.randint(1, 4) for _ in range(out_c)]
            round_biases = [rng.choice([0, 1 << (s - 1)]) for s in shifts]
            channel_size = out_h * out_w
            nodes.append(
                Node(
                    name="rq",
                    inputs=["c1"],
                    outputs=["c1q"],
                    op=RequantSpec(
                        shift=0,
                        mult=1,
                        round_bias=0,
                        clamp_lo=0,
                        zero_point=0,
                        out_bits=rq_out_bits,
                        clamp_lut=identity_clamp_lut(rq_out_bits),
                        mults=mults,
                        shifts=shifts,
                        round_biases=round_biases,
                        channel_size=channel_size,
                    ),
                )
            )
        else:
            shift = rng.randint(1, 5)
            mult = rng.randint(1, 3)
            round_bias = rng.choice([0, 1 << (shift - 1)])
            nodes.append(
                Node(
                    name="rq",
                    inputs=["c1"],
                    outputs=["c1q"],
                    op=RequantSpec(
                        shift=shift,
                        mult=mult,
                        round_bias=round_bias,
                        clamp_lo=0,
                        zero_point=0,
                        out_bits=rq_out_bits,
                        clamp_lut=identity_clamp_lut(rq_out_bits),
                    ),
                )
            )
        curr_out = "c1q"
        curr_h = out_h
        curr_w = out_w
        if rng.random() < 0.5:
            p_mode = rng.choice(["avg", "max"])
            p_stride = rng.choice([1, 2])
            p_pad = rng.choice([0, 1])
            if curr_h < 2:
                p_pad = 1
            nodes.append(
                Node(
                    name="pool",
                    inputs=["c1q"],
                    outputs=["pool"],
                    op=PoolSpec(
                        mode=p_mode,
                        in_h=curr_h,
                        in_w=curr_w,
                        channels=out_c,
                        pool_h=2,
                        pool_w=2,
                        stride=p_stride,
                        padding=p_pad,
                    ),
                )
            )
            curr_out = "pool"
            curr_h = (curr_h + 2 * p_pad - 2) // p_stride + 1
            curr_w = (curr_w + 2 * p_pad - 2) // p_stride + 1
        flat_dim = out_c * curr_h * curr_w
        k = rng.randint(2, 3)
        fc_w = [[rng.randint(-2, 2) for _ in range(flat_dim)] for _ in range(k)]
        fc_b = [rng.randint(-4, 4) for _ in range(k)]
        nodes.append(
            Node(
                name="fc",
                inputs=[curr_out],
                outputs=["y"],
                op=LinearSpec(weights=fc_w, bias=fc_b, weight_bits=3),
            )
        )
        final_out = "y"
        test_inputs = [
            [rng.randint(0, (1 << input_bits) - 1) for _ in range(c_in * side * side)]
            for _ in range(n_inputs)
        ]

    elif family == "branch":
        n = rng.choice([4, 6])
        input_bits = rng.randint(2, 4)
        _ = rng.randint(2, 3)
        # Clamped to 2: m=3 gives 18 bits (due to +2 carry/sign guard bits) ->
        # nb=9 > MAX_NUM_BLOCKS=8
        m = 2
        fc1_w = [[rng.randint(-2, 2) for _ in range(n)] for _ in range(2 * m)]
        fc1_b = [rng.randint(-4, 4) for _ in range(2 * m)]
        shift = rng.randint(1, 5)
        mult = rng.randint(1, 3)
        round_bias = rng.choice([0, 1 << (shift - 1)])
        fc_a_w = [[rng.randint(-2, 2) for _ in range(m)] for _ in range(m)]
        fc_a_b = [rng.randint(-4, 4) for _ in range(m)]
        k = rng.randint(2, 3)
        head_w = [[rng.randint(-2, 2) for _ in range(2 * m)] for _ in range(k)]
        head_b = [rng.randint(-4, 4) for _ in range(k)]
        nodes = [
            Node(
                name="fc1",
                inputs=["x"],
                outputs=["h1"],
                op=LinearSpec(weights=fc1_w, bias=fc1_b, weight_bits=3),
            ),
            Node(
                name="rq1",
                inputs=["h1"],
                outputs=["h1q"],
                op=RequantSpec(
                    shift=shift,
                    mult=mult,
                    round_bias=round_bias,
                    clamp_lo=0,
                    zero_point=0,
                    out_bits=2,
                    clamp_lut=identity_clamp_lut(2),
                ),
            ),
            Node(name="split", inputs=["h1q"], outputs=["a", "b"], op=SplitSpec(sizes=[m, m])),
            Node(
                name="fc_a",
                inputs=["a"],
                outputs=["a2"],
                op=LinearSpec(weights=fc_a_w, bias=fc_a_b, weight_bits=3),
            ),
            Node(name="add", inputs=["a2", "b"], outputs=["s"], op=AddSpec()),
            Node(name="concat", inputs=["s", "b"], outputs=["sb"], op=ConcatSpec(sizes=[m, m])),
            Node(
                name="head",
                inputs=["sb"],
                outputs=["y"],
                op=LinearSpec(weights=head_w, bias=head_b, weight_bits=3),
            ),
        ]
        final_out = "y"
        test_inputs = [
            [rng.randint(0, (1 << input_bits) - 1) for _ in range(n)] for _ in range(n_inputs)
        ]

    else:  # family == "tree"
        n = rng.randint(3, 6)
        input_bits = 4
        t = rng.randint(2, 4)
        indices = [rng.randrange(n) for _ in range(t)]
        thresholds = [rng.randint(0, 15) for _ in range(t)]
        k = rng.randint(2, 3)
        head_w = [[rng.randint(-2, 2) for _ in range(t)] for _ in range(k)]
        head_b = [rng.randint(-2, 2) for _ in range(k)]
        nodes = [
            Node(
                name="cmp",
                inputs=["x"],
                outputs=["bits"],
                op=CompareSpec(indices=indices, thresholds=thresholds),
            ),
            Node(
                name="head",
                inputs=["bits"],
                outputs=["y"],
                op=LinearSpec(weights=head_w, bias=head_b, weight_bits=3),
            ),
        ]
        final_out = "y"
        test_inputs = [
            [rng.randint(0, (1 << input_bits) - 1) for _ in range(n)] for _ in range(n_inputs)
        ]

    raw_graph = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=MAX_NUM_BLOCKS,
        input_bits=input_bits,
        inputs=["x"],
        outputs=[final_out],
        nodes=nodes,
    )
    nb = minimal_num_blocks(raw_graph)
    if nb > MAX_NUM_BLOCKS:
        raise RuntimeError(
            f"Model {family}_{seed} exceeds MAX_NUM_BLOCKS: nb={nb} > {MAX_NUM_BLOCKS}"
        )

    graph = Graph(
        schema_version=SCHEMA_VERSION,
        num_blocks=nb,
        input_bits=input_bits,
        inputs=["x"],
        outputs=[final_out],
        nodes=nodes,
    )

    # Self-checks
    check_bit_width_budget(graph)
    widths = propagate_bit_widths(graph)
    for node in graph.nodes:
        in_w = widths[node.inputs[0]]
        if node.op.op_type == "Requant" and in_w > CKKS_REQUANT_INPUT_BITS:
            raise RuntimeError(
                f"Model {family}_{seed} node {node.name} "
                f"Requant input {in_w} > {CKKS_REQUANT_INPUT_BITS}"
            )
        if node.op.op_type == "Argmax" and in_w > CKKS_ARGMAX_INPUT_BITS:
            raise RuntimeError(
                f"Model {family}_{seed} node {node.name} "
                f"Argmax input {in_w} > {CKKS_ARGMAX_INPUT_BITS}"
            )
        if node.op.op_type == "Compare" and in_w > CKKS_COMPARE_INPUT_BITS:
            raise RuntimeError(
                f"Model {family}_{seed} node {node.name} "
                f"Compare input {in_w} > {CKKS_COMPARE_INPUT_BITS}"
            )

    return RandomModel(
        name=name,
        family=family,
        seed=seed,
        graph=graph,
        test_inputs=test_inputs,
    )


def build_corpus() -> dict[str, Any]:
    """Build the seeded, 24-model property test corpus across all four families."""
    models: list[dict[str, Any]] = []
    for fi, fam in enumerate(FAMILIES):
        for j in range(CORPUS_MODELS_PER_FAMILY):
            seed = CORPUS_BASE_SEED + 1000 * fi + j
            m = random_model(seed, fam)
            models.append(
                {
                    "name": m.name,
                    "family": m.family,
                    "seed": m.seed,
                    "graph": m.graph.to_dict(),
                    "test_inputs": m.test_inputs,
                    "expected_outputs": m.expected_outputs(),
                }
            )
    return {
        "schema": CORPUS_SCHEMA,
        "base_seed": CORPUS_BASE_SEED,
        "models": models,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: generate and write the property test corpus to disk."""
    parser = argparse.ArgumentParser(description="Generate Penumbra property test corpus.")
    _ = parser.add_argument("--emit", required=True, type=Path, help="Target JSON path")
    args = parser.parse_args(argv)

    corpus = build_corpus()
    emit_path = Path(args.emit)
    emit_path.parent.mkdir(parents=True, exist_ok=True)
    emit_path.write_text(json.dumps(corpus, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
