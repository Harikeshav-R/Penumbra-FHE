# Security Policy

## Status & threat model

Penumbra-FHE is **research/prototype-grade software**. It is **not** audited production
cryptography. Do not use it to protect real secrets without an independent security review.

The privacy promise (`PROJECT.md` §11):

- The **client** holds the secret key and performs encryption/decryption.
- The **server** holds only the public evaluation/server key and the plaintext model
  weights. It runs the entire forward pass on ciphertext and **never sees the plaintext
  input or output.**

What this means in practice:

- **Confidentiality of the input/output** rests on the security of the underlying TFHE
  scheme as implemented by [`tfhe-rs`](https://github.com/zama-ai/tfhe-rs) and the chosen
  parameter profile. Penumbra-FHE ships a secure default profile and does not let users
  hand-roll insecure parameters.
- **The model weights are not secret** from the server — they are plaintext. Penumbra-FHE
  protects the *data*, not the *model*.
- This project does not (yet) defend against side channels, malicious-server result
  tampering, or traffic analysis. Integrity/verifiability is out of scope for now.

## Parameter security level

The default parameter profile is taken from `tfhe-rs`'s vetted parameter sets. Parameter
tuning (Phase 10) optimizes speed **within a fixed security level** — security is never
traded for performance silently.

## Reporting a vulnerability

If you discover a security issue, please report it **privately** rather than opening a
public issue:

- Use [GitHub private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  on this repository, or
- Email the maintainer directly.

Please include a description, reproduction steps, and the potential impact. We will
acknowledge receipt and work with you on a coordinated disclosure.

> Note: cryptographic weaknesses in the underlying TFHE scheme or in `tfhe-rs` itself should
> also be reported upstream to the [`tfhe-rs` project](https://github.com/zama-ai/tfhe-rs).
