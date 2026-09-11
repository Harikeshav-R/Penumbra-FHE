# Security Policy

## Status & threat model

Penumbra-FHE is **research/prototype-grade software**. It is **not** audited production
cryptography. Do not use it to protect real secrets without an independent security review.

The privacy promise (`PROJECT.md` §11):

- The **client** holds the secret key and performs encryption/decryption.
- The **server** holds only the public evaluation/server key and the plaintext model
  weights. It runs the entire forward pass on ciphertext and **never sees the plaintext
  input or output.**

This holds for **every backend** — the server sees only ciphertext regardless of the scheme.
What differs is the cryptographic assumption underneath it.

What this means in practice:

- **Confidentiality of the input/output** rests on the security of the underlying FHE scheme
  as implemented by the selected backend's library, and on the chosen parameter profile.
  Penumbra-FHE ships a secure default profile per backend and does not let users hand-roll
  insecure parameters.
- **The model weights are not secret** from the server — they are plaintext. Penumbra-FHE
  protects the *data*, not the *model*.
- This project does not (yet) defend against side channels, malicious-server result
  tampering, or traffic analysis. Integrity/verifiability is out of scope for now.

## Parameter security level

| Backend | Library | Default profile | Notes |
|---|---|---|---|
| `tfhe` | [`tfhe-rs`](https://github.com/zama-ai/tfhe-rs) | the library's vetted parameter sets | the reference backend |
| `ckks` | [`poulpy-ckks`](https://github.com/phantomzone-org/poulpy) | the crate's `presets` | see the caveats below |

Parameter tuning (Phase 10) optimizes speed **within a fixed security level** — security is
never traded for performance silently. The backends' security levels are kept **matched**,
both so neither is weakened and so the scheme comparison (`docs/COMPARISON.md`) is fair.

## CKKS-specific caveats

Two things are worth stating plainly, because they are easy to overlook when a second scheme
is added for benchmarking purposes.

**CKKS is not IND-CPA^D secure.** Because CKKS is *approximate*, a decrypted result carries
residual noise that depends on the secret key. An adversary who can submit chosen ciphertexts
and observe the corresponding decryptions can recover the key (Li–Micciancio, 2021). The
standard mitigations are to add noise-flooding before releasing a decryption, or to never
release decryptions of adversarially-chosen ciphertexts at all. In Penumbra's deployment
model the client both encrypts and decrypts, and decrypted results are not published, so the
attack is out of the model as written — but it becomes live the moment a decryption is shared
with the party that supplied the ciphertext. This has **no TFHE analogue**: TFHE is exact, so
its decryptions carry no such residue.

**`poulpy` is young and unaudited.** The CKKS backend depends on a 0.8.x crate whose own
documentation describes its public API as subject to change. It is actively maintained and
authored by an established cryptographic engineer, but it carries less deployment history
than `tfhe-rs`. Treat the CKKS backend as the more experimental of the two, and do not read
"the TFHE backend is research-grade" as implying they are at the same level of maturity.

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
