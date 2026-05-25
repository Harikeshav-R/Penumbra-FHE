# Security Policy

Penumbra is pre-alpha cryptography software. Do not use it in production. It has not been audited.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| v0.x    | Latest minor only  |

## Threat Model Summary

Penumbra assumes a semi-honest server model where the server executing homomorphic operations does not deviate from the protocol, but may attempt to learn information from the transcript. The shape and size of the computation graph (e.g., number of layers, tensor dimensions) are considered public and leak to the server. Penumbra does *not* provide execution integrity; a malicious server can return arbitrary or incorrect results without detection. We rely on TFHE-rs for cryptographic hardness and do not provide additional mitigations against hardware-level side channels.

For the comprehensive threat model, refer to [ARCHITECTURE.md §7](ARCHITECTURE.md#7-security-model).

## Scope

**In Scope for Security Reports:**
* Incorrect usage of cryptographic primitives
* Depth-cost miscalculations that could break correctness in a privacy-relevant way (e.g., causing a silent decryption failure)
* Errors in polynomial coefficients affecting bounds or precision beyond accepted tolerances
* Key handling and storage bugs within Penumbra's implementation
* Non-constant-time operations introduced by Penumbra (not those inherited from TFHE-rs)
* Supply-chain risks in our official build or release processes

**Out of Scope:**
* Denial of Service (DoS) attacks against a user's own infrastructure or deployment
* Weaknesses or bugs in TFHE-rs itself (these should be reported to Zama)
* Model extraction attacks (this is an active research area outside the current scope of Penumbra)
* Side channels at the hardware or micro-architectural level

## Reporting a Vulnerability

**Primary Method:** Email `r.harikeshav@gmail.com` with `[SECURITY]` in the subject line.
(PGP key fingerprint will be published in this section once generated)

**Alternative Method:** Create a private vulnerability report via GitHub Security Advisories.

**Do not open a public issue for a security vulnerability.**

## Response Timeline

* **Acknowledgment:** Within 72 hours of receiving the report.
* **Initial Assessment:** Within 7 days.
* **Patch Timeline:** Dependent on the severity of the issue. We aim to address critical cryptographic flaws as an immediate priority.

## Disclosure Policy

We practice coordinated vulnerability disclosure. We will request up to 90 days to develop and release a fix before public disclosure, and less time for low-severity issues. Reporters will be credited in the security advisory unless they request anonymity.

## Hall of Fame

(none yet)
