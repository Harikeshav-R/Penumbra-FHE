"""Tests for security parameter extraction, Dockerfile pins, and estimator runner."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PINNED_ESTIMATOR_COMMIT = "53da5982597709ba0fdf94ea37a84d822310fd84"
PINNED_SAGE_DIGEST = "sha256:19995db6194f4a4bab18ce9a88556fd15b9ed5e916b4504fefe618a7796ddbdb"
PINNED_UV_DIGEST = "sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711"


def test_dockerfile_pins():
    dockerfile_path = REPO_ROOT / "examples/security/Dockerfile"
    assert dockerfile_path.exists()
    content = dockerfile_path.read_text()
    assert PINNED_SAGE_DIGEST in content
    assert PINNED_UV_DIGEST in content
    assert "--platform=linux/amd64" in content


def test_pinned_estimator_commit_in_runner():
    runner_path = REPO_ROOT / "examples/security/run.py"
    assert runner_path.exists()
    content = runner_path.read_text()
    assert PINNED_ESTIMATOR_COMMIT in content


def test_security_input_schema_validation(tmp_path):
    # Verify synthetic security inputs structure
    sample_inputs = {
        "tfhe_classic_lwe": {
            "name": "tfhe_classic_lwe",
            "n": 918,
            "q": "2^64",
            "q_bits": 64,
            "secret_distribution": "Binary",
            "error_distribution": "TUniform(45)",
            "error_param": 45.0,
        },
        "tfhe_classic_glwe": {
            "name": "tfhe_classic_glwe",
            "n": 2048,
            "q": "2^64",
            "q_bits": 64,
            "secret_distribution": "Binary",
            "error_distribution": "TUniform(17)",
            "error_param": 17.0,
        },
        "ckks_ciphertext": {
            "name": "ckks_ciphertext",
            "n": 16384,
            "q": "2^360",
            "q_bits": 360,
            "secret_distribution": "Ternary",
            "error_distribution": "DiscreteGaussian(3.2)",
            "error_param": 3.2,
        },
        "ckks_evaluation_key": {
            "name": "ckks_evaluation_key",
            "n": 16384,
            "q": "2^432",
            "q_bits": 432,
            "secret_distribution": "Ternary",
            "error_distribution": "DiscreteGaussian(3.2)",
            "error_param": 3.2,
        },
    }
    p = tmp_path / "inputs.json"
    p.write_text(json.dumps(sample_inputs))

    loaded = json.loads(p.read_text())
    assert loaded["tfhe_classic_lwe"]["n"] == 918
    assert loaded["tfhe_classic_lwe"]["q_bits"] == 64
    assert loaded["tfhe_classic_glwe"]["n"] == 2048
    assert loaded["ckks_ciphertext"]["q_bits"] == 360
    assert loaded["ckks_evaluation_key"]["q_bits"] == 432
