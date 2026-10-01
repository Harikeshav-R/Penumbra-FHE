# Security Policy

## Status & threat model

Penumbra-FHE is **research/prototype-grade software**. It is **not** audited production
cryptography. Do not use it to protect real secrets without an independent security review.

The privacy promise (`PROJECT.md` §11):

- The **client** holds the secret key and performs encryption and decryption.
- The **server** holds only the public evaluation/server key and the plaintext model
  weights. It runs the entire forward pass on ciphertext and **never sees the plaintext
  input or output.**

This holds for **every backend** — the server sees only ciphertext regardless of the scheme.
What differs is the cryptographic assumption underneath it.

What this means in practice:

- **Confidentiality of input/output:** Rests on the mathematical security of the underlying
  FHE scheme as implemented by the selected backend's library, and on the chosen parameter
  profile. Penumbra-FHE ships a secure default profile per backend and does not let users
  hand-roll insecure parameters.
- **Model weights and graph structure are public to the server:** The server learns the model
  graph topology, node operator types, plaintext weights and biases, and input/output tensor
  shapes, but learns nothing about the ciphertext values themselves. Penumbra-FHE protects the
  *data*, not the *model*.
- **Key protection:** Key files are written unencrypted to disk by `KeySet.generate`
  (`python/penumbra/client.py`). The secret key (`client.key`) must be protected with appropriate
  filesystem permissions and must never be transmitted or exposed to the evaluation server.
- **Out-of-scope threats:** This project does not defend against side channels, malicious-server
  result tampering, or traffic analysis. Integrity and verifiable computation are out of scope.

## Parameter security level

All supported backend parameter profiles target $\ge 128$-bit classical security under standard
lattice-based assumptions. In Phase 15, concrete security estimates were executed using the
pinned [lattice-estimator](https://github.com/malb/lattice-estimator) (commit `53da598`) under SageMath 10.6,
evaluating the MATZOV reduction cost model (`RC.MATZOV`) and GSA shape model (`GSA`) under unbounded samples ($m=\infty$).
Full results and provenance are recorded in [`docs/results/phase15-security-estimates.json`](./results/phase15-security-estimates.json),
reproducible via `python examples/security/run.py`.

### Concrete Lattice Estimates Summary

| Component | Ring Dim ($n$) | Modulus ($q$) | Secret Dist | Error Dist | Min Lattice ROP | Primary Bottleneck Attack |
|---|---|---|---|---|---:|---|
| **TFHE Classic LWE** | 918 | $2^{64}$ | Binary | TUniform(45) | 134.9 bits | Dual hybrid ($\beta=366$) |
| **TFHE Classic GLWE** | 2048 | $2^{64}$ | Binary | TUniform(17) | 134.8 bits | Dual hybrid ($\beta=360$) |
| **CKKS Ciphertext** | 16384 | $2^{360}$ | Ternary | DiscreteGaussian(3.2) | 155.3 bits | Primal BDD ($\beta=426, \eta=468$) |
| **CKKS Evaluation Key** | 16384 | $2^{432}$ | Ternary | DiscreteGaussian(3.2) | 127.6 bits | Primal BDD ($\beta=327, \eta=355$) |

### TFHE (`tfhe-rs` 1.8.1)

TFHE operates over discrete torus integers. All five supported profiles use vetted `tfhe-rs`
parameter sets at `message_bits = 2, carry_bits = 2`. The classical security of the default `classic` profile
measures at **134.8 bits** overall (LWE 134.9 bits, GLWE 134.8 bits), exceeding the 128-bit security floor:

| Profile | `tfhe-rs` Constant | Probability of Decryption Failure ($p_{\text{fail}}$) | Notes |
|---|---|---|---|
| `classic` (default) | `PARAM_MESSAGE_2_CARRY_2_KS_PBS` | $2^{-129.581}$ | Lowest serial latency; recommended default (`crates/penumbra-tfhe/src/keys.rs`) |
| `gaussian` | `PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128` | $\le 2^{-128}$ | Discrete Gaussian noise distribution |
| `multibit2` | `PARAM_MESSAGE_2_CARRY_2_GROUP_2_KS_PBS` | $2^{-140.341}$ | 2-bit PBS grouping; larger evaluation key |
| `multibit3` | `PARAM_MESSAGE_2_CARRY_2_GROUP_3_KS_PBS` | $2^{-128.235}$ | 3-bit PBS grouping |
| `multibit4` | `PARAM_MESSAGE_2_CARRY_2_GROUP_4_KS_PBS` | $2^{-134.345}$ | 4-bit PBS grouping |

Note that decryption failure probability ($p_{\text{fail}} \le 2^{-128}$) represents statistical correctness
(the chance that carry noise spills into the message space), which is distinct from the computational cost of
lattice reduction attacks ($134.8$ bits).

### CKKS (`poulpy-ckks` 0.8.3)

Because `poulpy-ckks` does not ship general parameter presets, Penumbra-FHE defines its own
calibrated parameter profile (`crates/penumbra-ckks/src/params.rs`):

- **Ring dimension:** $N = 16384$
- **Ciphertext modulus:** $k = \log_2(q) = 360$ bits
- **Evaluation key modulus:** $\log_2(q) = 432$ bits (gadget decomposition precision)
- **Scaling factor:** $\log_2(\Delta) = 30$ bits (multiplicative depth budget: 330 bits)
- **Secret key distribution:** Uniform ternary secret ($\{-1, 0, 1\}$)

Lattice security estimation yields **155.3 bits** for ciphertexts and **127.6 bits** for evaluation keys
(the public tensor and automorphism keys require higher auxiliary precision $q=2^{432}$ during gadget decomposition).
The overall CKKS security level is bounded by the evaluation key at **127.6 bits** (at the 128-bit target).
GLWE/RLWE is modeled as unstructured LWE at $n = \text{rank} \times N = 16384$, and discrete Gaussian noise
($\sigma=3.2$) is modeled as a nominal approximation to Poulpy's rounded, 6-sigma truncated sampler.
## CKKS-specific caveats

Two things are worth stating plainly, because they are easy to overlook when a second scheme
is added for benchmarking purposes:

**CKKS is not IND-CPA^D secure.** Because CKKS is *approximate*, a decrypted result carries
residual noise that depends on the secret key. An adversary who can submit chosen ciphertexts
and observe the corresponding decryptions can recover the secret key (Li–Micciancio, 2021).
Penumbra-FHE applies **no noise flooding**; the Python runtime bridge rounds decrypted floating-point
values to integers (`crates/penumbra-ckks/src/encrypt.rs`), which is an engineering conversion rather
than a formal IND-CPA^D countermeasure. In Penumbra's client/server deployment model, the client both
encrypts inputs and decrypts outputs, and decrypted values are not returned to untrusted third parties,
so the attack is out of the model as written. However, **never return a CKKS decryption to a party
that could have chosen or influenced the ciphertext.** This has no TFHE analogue: TFHE is exact,
so its decryptions carry no key-dependent residual noise.

**`poulpy` is young and unaudited.** The CKKS backend depends on `poulpy-ckks` (0.8.x), whose own
documentation notes that its public API is subject to change. It is actively maintained and
authored by an established cryptographic engineer, but it carries less deployment history
than `tfhe-rs`. Treat the CKKS backend as experimental, and do not read "the TFHE backend is
research-grade" as implying both backends share the same level of maturity.

## Reporting a vulnerability

If you discover a security issue, please report it **privately** rather than opening a
public issue:

- Use [GitHub private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  on this repository, or
- Email the maintainer directly.

Please include a description, reproduction steps, and the potential impact. We will
acknowledge receipt and work with you on a coordinated disclosure.

> Note: cryptographic weaknesses in an underlying scheme or library should also be reported
> upstream — to the [`tfhe-rs` project](https://github.com/zama-ai/tfhe-rs) for the TFHE
> backend, or to [`poulpy`](https://github.com/phantomzone-org/poulpy) for the CKKS backend.
