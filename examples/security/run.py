"""Host runner for containerized lattice-estimator security execution (Phase 15)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PINNED_ESTIMATOR_COMMIT = "53da5982597709ba0fdf94ea37a84d822310fd84"
PINNED_SAGE_DIGEST = (
    "sagemath/sagemath:10.6"
    "@sha256:19995db6194f4a4bab18ce9a88556fd15b9ed5e916b4504fefe618a7796ddbdb"
)
PINNED_UV_DIGEST = (
    "ghcr.io/astral-sh/uv:0.12.21"
    "@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711"
)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def ensure_estimator_checkout(estimator_dir: Path) -> str:
    """Ensure lattice-estimator is checked out at the pinned commit."""
    if not estimator_dir.exists():
        estimator_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning lattice-estimator to {estimator_dir}...", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "https://github.com/malb/lattice-estimator.git", str(estimator_dir)],
            check=True,
        )

    subprocess.run(
        ["git", "-C", str(estimator_dir), "fetch", "origin"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(estimator_dir), "checkout", PINNED_ESTIMATOR_COMMIT],
        check=True,
    )
    res = subprocess.run(
        ["git", "-C", str(estimator_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    commit = res.stdout.strip()
    if commit != PINNED_ESTIMATOR_COMMIT:
        raise RuntimeError(
            f"estimator commit mismatch: got {commit}, expected {PINNED_ESTIMATOR_COMMIT}"
        )
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="target/security/inputs.json",
        help="Path to inputs.json (default: target/security/inputs.json)",
    )
    parser.add_argument(
        "--out",
        default="docs/results/phase15-security-estimates.json",
        help="Path to output estimates json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    # Verify Colima / Docker environment
    colima_sock = Path.home() / ".colima/penumbra-security/docker.sock"
    env = os.environ.copy()
    if colima_sock.exists():
        env["DOCKER_HOST"] = f"unix://{colima_sock}"

    # Get docker and colima versions
    docker_ver_res = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        env=env,
        capture_output=True,
        text=True,
    )
    if docker_ver_res.returncode != 0:
        raise RuntimeError(f"failed to connect to docker daemon: {docker_ver_res.stderr}")
    docker_version = docker_ver_res.stdout.strip()

    colima_ver_res = subprocess.run(
        ["colima", "version"],
        capture_output=True,
        text=True,
    )
    colima_version = colima_ver_res.stdout.strip() if colima_ver_res.returncode == 0 else "unknown"

    estimator_dir = REPO_ROOT / "target/security/lattice-estimator"
    estimator_commit = ensure_estimator_checkout(estimator_dir)

    dockerfile_path = REPO_ROOT / "examples/security/Dockerfile"
    image_tag = "penumbra-security-estimator:phase15"

    print("Building container image...", file=sys.stderr)
    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "-t",
            image_tag,
            "-f",
            str(dockerfile_path),
            str(REPO_ROOT),
        ],
        env=env,
        check=True,
    )

    # Relative path from REPO_ROOT inside container mount /workspace
    container_input_path = f"/workspace/{input_path.relative_to(REPO_ROOT)}"

    print("Running lattice security estimation in container...", file=sys.stderr)
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-v",
        f"{REPO_ROOT}:/workspace:ro",
        "-v",
        f"{estimator_dir}:/estimator:ro",
        "-e",
        "PYTHONPATH=/estimator",
        "-e",
        "UV_CACHE_DIR=/tmp/uv-cache",
        image_tag,
        "run",
        "--no-project",
        "--no-config",
        "--",
        "sage",
        "-python",
        "/workspace/examples/security/estimate_security.py",
        "--input",
        container_input_path,
    ]

    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"estimator container failed with return code {res.returncode}")

    # Isolate JSON payload from stdout
    stdout_text = res.stdout.strip()
    if "{" in stdout_text and "}" in stdout_text:
        start_idx = stdout_text.find("{")
        end_idx = stdout_text.rfind("}") + 1
        raw_output = json.loads(stdout_text[start_idx:end_idx])
    else:
        raw_output = json.loads(stdout_text)
    # Enrich with provenance
    runtime_commit_res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    runtime_commit = runtime_commit_res.stdout.strip()

    final_artifact = {
        "schema_version": 1,
        "meta": {
            "runtime_commit": runtime_commit,
            "input_file": str(input_path.relative_to(REPO_ROOT)),
            "input_sha256": compute_sha256(input_path),
            "estimator_commit": estimator_commit,
            "sage_image": PINNED_SAGE_DIGEST,
            "uv_image": PINNED_UV_DIGEST,
            "docker_version": docker_version,
            "colima_version": colima_version,
        },
        "results": raw_output,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_artifact, f, indent=2)
        f.write("\n")

    print(f"Security estimates successfully written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
