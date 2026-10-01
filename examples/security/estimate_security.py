# pyright: reportMissingImports=false
"""Lattice security estimator execution script run inside SageMath.

Computes concrete security estimates against the pinned lattice-estimator commit
for Penumbra TFHE and CKKS parameter sets (Phase 15).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from estimator import LWE, ND, RC
from estimator.simulator import GSA
from sage.all import RR, ZZ, log, oo


def run_estimate_for_tuple(param_info: dict[str, Any]) -> dict[str, Any]:
    n = int(param_info["n"])
    q_bits = int(param_info["q_bits"])
    q = ZZ(2) ** q_bits

    sec_dist = param_info["secret_distribution"]
    if sec_dist == "Binary":
        xs = ND.Binary
    elif sec_dist == "Ternary":
        xs = ND.Ternary
    else:
        raise ValueError(f"unsupported secret distribution: {sec_dist}")

    err_dist = param_info["error_distribution"]
    err_param = param_info["error_param"]

    if err_dist.startswith("TUniform"):
        xe = ND.TUniform(int(err_param))
    elif "Gaussian" in err_dist:
        xe = ND.DiscreteGaussian(float(err_param))
    else:
        raise ValueError(f"unsupported error distribution: {err_dist}")

    params = LWE.Parameters(n=n, q=q, Xs=xs, Xe=xe, m=oo)

    # Algorithm selection per dimension:
    # - n <= 1000: full attack suite including BKW
    # - 1000 < n <= 4000: all lattice primal and dual attacks, omitting BKW (non-competitive)
    # - n > 4000: all primal and dual lattice reduction attacks (usvp, bdd, dual, dual_hybrid),
    #   omitting hybrid guessing over 16384 dimensions for dense ternary secrets
    if n <= 1000:
        deny_list = ("arora-gb",)
    elif n <= 4000:
        deny_list = ("arora-gb", "bkw")
    else:
        deny_list = ("arora-gb", "bkw", "bdd_hybrid", "bdd_mitm_hybrid")

    results = LWE.estimate(
        params,
        red_cost_model=RC.MATZOV,
        red_shape_model=GSA,
        deny_list=deny_list,
        jobs=1,
        catch_exceptions=True,
        quiet=True,
    )

    attack_details: dict[str, Any] = {}
    finite_rop_bits: list[float] = []

    for attack_name, cost in results.items():
        raw_rop = cost.get("rop")
        is_infinite = raw_rop == oo

        if is_infinite or raw_rop is None:
            rop_str = "Infinity"
            rop_bits = None
        else:
            rop_str = str(raw_rop)
            # Compute log2 in Sage to avoid Python float overflow on large integers
            log2_rop = RR(log(raw_rop, 2))
            rop_bits = float(log2_rop)
            finite_rop_bits.append(rop_bits)

        attack_entry: dict[str, Any] = {
            "attack": attack_name,
            "rop": rop_str,
            "rop_bits": rop_bits,
            "beta": int(cost["beta"]) if "beta" in cost and cost["beta"] is not None else None,
            "d": int(cost["d"]) if "d" in cost and cost["d"] is not None else None,
            "raw": str(cost),
        }
        attack_details[attack_name] = attack_entry

    # Record skipped attacks with explicit reasoning
    for skipped in deny_list:
        attack_details[skipped] = {
            "attack": skipped,
            "rop": "Inapplicable / non-competitive for large dimension or dense secret",
            "rop_bits": None,
            "beta": None,
            "d": None,
            "raw": "skipped via deny_list",
        }

    if not finite_rop_bits:
        raise RuntimeError(f"no finite attack results for {param_info['name']}")

    min_rop_bits = min(finite_rop_bits)

    return {
        "name": param_info["name"],
        "parameters": param_info,
        "min_rop_bits": min_rop_bits,
        "attacks": attack_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to inputs.json")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    with open(args.input) as f:
        inputs = json.load(f)

    estimates: dict[str, Any] = {}
    for key in ["tfhe_classic_lwe", "tfhe_classic_glwe", "ckks_ciphertext", "ckks_evaluation_key"]:
        if key not in inputs:
            raise KeyError(f"missing required tuple {key} in inputs")
        print(f"\n=== Estimating {key} ===", file=sys.stderr)
        estimates[key] = run_estimate_for_tuple(inputs[key])

    tfhe_min = min(
        estimates["tfhe_classic_lwe"]["min_rop_bits"],
        estimates["tfhe_classic_glwe"]["min_rop_bits"],
    )
    ckks_min = min(
        estimates["ckks_ciphertext"]["min_rop_bits"],
        estimates["ckks_evaluation_key"]["min_rop_bits"],
    )

    output = {
        "cost_model": "MATZOV",
        "shape_model": "GSA",
        "sample_bound": "unbounded (m=oo)",
        "assumptions": {
            "glwe_unstructured_lwe": (
                "GLWE/RLWE modeled as unstructured LWE at dimension rank * ring_degree"
            ),
            "ckks_noise_approximation": (
                "DiscreteGaussian(3.2) nominal approximation to rounded 6-sigma truncated sampler"
            ),
        },
        "tfhe_summary": {
            "lwe_min_rop_bits": estimates["tfhe_classic_lwe"]["min_rop_bits"],
            "glwe_min_rop_bits": estimates["tfhe_classic_glwe"]["min_rop_bits"],
            "overall_min_rop_bits": tfhe_min,
        },
        "ckks_summary": {
            "ciphertext_min_rop_bits": estimates["ckks_ciphertext"]["min_rop_bits"],
            "evaluation_key_min_rop_bits": estimates["ckks_evaluation_key"]["min_rop_bits"],
            "overall_min_rop_bits": ckks_min,
        },
        "tuples": estimates,
    }

    out_json = json.dumps(output, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out_json + "\n")
    else:
        print(out_json)


if __name__ == "__main__":
    main()
