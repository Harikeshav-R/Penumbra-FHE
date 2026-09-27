"""Shared one-command runner: replay a committed example fixture under encryption.

Loads the fixture's IR graph and its already-quantized ``test_inputs`` (graph input domain), runs
keygen -> encrypt -> evaluate -> decrypt in-process (``penumbra.client.run_encrypted``), and checks
the decrypted outputs against the quantized-cleartext oracle ``evaluate_graph_int``
(``AGENTS.md`` §1.1): bit-for-bit under TFHE (exit 1 on any mismatch); report-only under CKKS
(the CKKS gate is the Rust golden against the declared bound in
``crates/penumbra-ckks/src/bounds.rs``, measured on unrounded outputs).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import penumbra as fhe
from penumbra.client import CryptoProfile, available_backends, run_encrypted
from penumbra.ir import ArgmaxSpec, Graph
from penumbra.reference import evaluate_graph_int

CKKS_BUILD_HINT = (
    'RUSTUP_TOOLCHAIN=nightly uv run --with "maturin>=1.9,<2.0" '
    "maturin develop --release --features ckks"
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Example:
    key: str
    fixture: Path
    # CKKS max_poly_degree override (None = backend default); mirrors the Rust goldens,
    # e.g. phase8_branch needs 3 (crates/penumbra-bench/tests/backend_parity.rs:41-44).
    ckks_max_poly_degree: int | None = None


def _label(graph: Graph, out: list[int]) -> int:
    # Same rule as Model.predict_encrypted (python/penumbra/model.py:831-834).
    if graph.nodes and isinstance(graph.nodes[-1].op, ArgmaxSpec):
        return int(out[0])
    return int(np.argmax(out))


def main(
    examples: Sequence[Example],
    *,
    default: str,
    description: str,
    argv: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    keys = [ex.key for ex in examples]
    parser.add_argument(
        "--model",
        choices=keys,
        default=default,
        help=f"Model fixture to replay (default: {default})",
    )
    parser.add_argument(
        "--backend",
        choices=("tfhe", "ckks"),
        default="tfhe",
        help="FHE backend to evaluate against (default: tfhe)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to run from test_inputs (default: all)",
    )

    args = parser.parse_args(argv)

    ex = next(e for e in examples if e.key == args.model)
    fx = json.loads(ex.fixture.read_text(encoding="utf-8"))
    test_inputs: list[list[int]] = fx["test_inputs"]
    n_avail = len(test_inputs)

    if args.samples is not None:
        if args.samples < 1 or args.samples > n_avail:
            parser.error(f"--samples must be in [1, {n_avail}] for '{ex.key}'")
        n = args.samples
    else:
        n = n_avail

    graph = Graph.from_dict(fx["graph"])
    rows = test_inputs[:n]

    in_name = graph.inputs[0]
    out_name = graph.outputs[0]

    refs = [evaluate_graph_int(graph, {in_name: row})[out_name] for row in rows]
    for i, ref in enumerate(refs):
        ref_label = _label(graph, ref)
        expected_label = fx["expected_labels"][i]
        if ref_label != expected_label:
            raise SystemExit(
                f"fixture drift: sample {i} reference label {ref_label} != "
                f"committed expected_labels {expected_label}; regenerate the fixture"
            )

    b = args.backend
    if b not in available_backends():
        avail = ", ".join(available_backends())
        raise SystemExit(
            f"backend '{b}' is not compiled into this build (available: {avail}). "
            f"Build the CKKS backend from source with:\n  {CKKS_BUILD_HINT}\n"
            "then run with `uv run --no-sync python ...` so uv does not "
            "reinstall the default build."
        )

    profile = (
        CryptoProfile.ckks(ex.ckks_max_poly_degree)
        if b == "ckks" and ex.ckks_max_poly_degree is not None
        else None
    )

    rel_fixture = ex.fixture.resolve().relative_to(REPO_ROOT)
    ops_str = " → ".join(n.op.op_type for n in graph.nodes)
    cap_bits = fhe.radix_capacity_bits(graph.num_blocks)

    print(f"Model:       {ex.key}")
    print(f"Fixture:     {rel_fixture}")
    print(f"Backend:     {b}")
    print(f"Samples:     {n} of {n_avail}")
    print(f"Radix:       {graph.num_blocks} blocks ({cap_bits}-bit radix)")
    print(f"IR Ops:      {ops_str}")
    print("-" * 60)

    t0 = time.perf_counter()
    try:
        outs = run_encrypted(graph, rows, backend=b, profile=profile)
    except ValueError as e:
        raise SystemExit(f"backend '{b}' cannot run '{ex.key}': {e}") from e
    elapsed = time.perf_counter() - t0

    mismatches = 0
    max_err_all = 0.0
    label_matches = 0

    if b == "tfhe":
        for i, (out, ref) in enumerate(zip(outs, refs, strict=True)):
            out_label = _label(graph, out)
            ref_label = _label(graph, ref)
            ok = out == ref
            if not ok:
                mismatches += 1
            status = "OK" if ok else "MISMATCH"
            print(f"sample {i}: label={out_label} logits={out} reference={ref} {status}")
    else:
        for i, (out, ref) in enumerate(zip(outs, refs, strict=True)):
            out_label = _label(graph, out)
            ref_label = _label(graph, ref)
            if out_label == ref_label:
                label_matches += 1
            max_err = max(abs(o - r) for o, r in zip(out, ref, strict=True))
            if max_err > max_err_all:
                max_err_all = max_err
            print(
                f"sample {i}: label={out_label} reference_label={ref_label} " f"max|err|={max_err}"
            )

    print("-" * 60)
    print(
        f"{n} sample(s) in {elapsed:.1f} s wall-clock on '{b}' "
        "(includes one keygen; not a benchmark — see docs/BENCHMARKS.md)"
    )

    if b == "tfhe":
        if mismatches > 0:
            raise SystemExit(
                "GOLDEN VIOLATION: encrypted output != quantized-cleartext oracle "
                f"({mismatches} mismatch(es))"
            )
        print("All samples matched bit-for-bit against quantized-cleartext oracle.")
    else:
        print(f"CKKS max |err|: {max_err_all:.2f}")
        print(f"Labels agree: {label_matches}/{n}")
        print(
            "Note: CKKS is report-only; decrypted values were rounded to integers by the bridge. "
            "The gate is the Rust CKKS golden against declared bounds."
        )
