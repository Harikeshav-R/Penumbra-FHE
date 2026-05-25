# Penumbra Philosophy

> **Why this document exists.** Architecture documents tell you *what* the system is. Roadmaps tell you *when* things ship. This document tells you *why* — the principles that govern decisions when the docs are silent, and the temperament we want the codebase to have. If you disagree with something here, open an issue. If you ignore something here, expect pushback in review.

## 1. The thing we are trying to do

Make it possible for an ordinary ML engineer — someone competent in PyTorch and curious about cryptography but not an expert in either FHE or compiler design — to take a trained model, encrypt their inputs, send them to a server they do not trust, and get back a result the server could not have inspected.

That is the user story. Every design decision should be checked against it.

We are **not** trying to:
- Invent new cryptography.
- Beat TFHE-rs at the low level.
- Be the fastest FHE library.
- Support every model architecture.
- Replace cloud ML platforms.

We **are** trying to:
- Make existing FHE techniques usable for ML practitioners.
- Document the costs honestly.
- Build the missing layer between TFHE-rs and ONNX.
- Be a credible, well-engineered, well-documented open source project.

## 2. The values that resolve disputes

When a design decision is contested, walk this list top-to-bottom. The earlier value wins.

### 2.1 Correctness over performance

A correct slow inference is a feature. An incorrect fast inference is a bug pretending to be a feature.

If a performance optimization changes behavior — even in edge cases, even by 0.001 in floating-point output — it is rejected unless the change is documented and the test suite is updated to accept the new behavior. We do not chase benchmarks by lying about what the system computes.

### 2.2 Honest framing over impressive framing

We will say "this MNIST inference takes 90 seconds on an M3 Pro." We will not say "fast" when we mean "less slow than the alternatives."

We will say "polynomial activation introduces 3% accuracy degradation on this model." We will not say "minor accuracy difference."

We will say "we have not been audited; this is a research project; do not deploy against an adversary you would not trust your savings account to."

If the honest version of a claim sounds bad, the answer is to make the thing better, not to make the claim vaguer.

### 2.3 Boring code over clever code

Cryptography is hard enough without clever code. Prefer:

- Explicit over implicit.
- Long names over short.
- Verbose error handling over `?` chains that obscure what failed.
- One operation per line over four chained transformations.
- Comments explaining *why* over comments explaining *what*.

If a reviewer needs to read a function three times to understand it, the function is wrong, regardless of how elegant it is.

### 2.4 Separable components over monolithic convenience

Every component is in its own crate. Every crate has a `lib.rs` that documents its purpose in the first 20 lines. Every cross-crate dependency is justified in `Cargo.toml` with a comment.

If you are tempted to add a dependency on `penumbra-runtime` from inside `penumbra-ir`, stop. Open an issue. Explain why the IR needs to know about the runtime. The likely answer is that something belongs in a third crate, or that you are smuggling logic across the architectural seam.

### 2.5 Determinism over speed

Same input → same output, byte-for-byte. This is non-negotiable.

If we need a random number, we accept an explicit RNG parameter. We do not read from `thread_rng()` implicitly. We do not let parallelism reorder operations in a way that changes results. We do not depend on hashmap iteration order. We do not use `f32`'s non-associativity as a free pass to reorder arithmetic.

The cost of determinism is sometimes performance. We pay it.

### 2.6 Documentation as a first-class artifact

A function without documentation is incomplete. A PR that adds a public API without docstrings or rustdoc is incomplete. A change that affects user-visible behavior without a `CHANGELOG.md` entry is incomplete.

We are not writing documentation as a chore at the end of a feature. We are writing it as we go. The docs are the contract.

### 2.7 Crypto code is special

Code that touches a `Ciphertext`, a key, or a TFHE-rs primitive is held to a higher standard:

- Two reviewers required.
- Test coverage must be present **before** the code lands, not as a follow-up.
- Refactoring this code for stylistic reasons alone is not permitted. Changes must have an intent.
- Polynomial coefficients in `penumbra-compiler` are not edited without re-deriving them from source and updating the derivation reference.

This is not because the rest of the code is unimportant. It is because the rest of the code's bugs are recoverable. Cryptographic bugs are not.

## 3. The framing for the world

Penumbra is a research project that ships software. Both halves matter.

As a **research project**, we are allowed to:
- Document negative results. ("Polynomial activations of degree 7 do not improve accuracy enough to justify the depth cost.")
- Publish honest benchmarks. ("FHE is slow. Here is exactly how slow, and where the time goes.")
- Say "we don't know yet" in public.

As **software**, we are obligated to:
- Ship working artifacts on PyPI and crates.io.
- Maintain a working main branch.
- Respond to issues in a timeline documented in `CONTRIBUTING.md`.
- Not break users without warning, deprecation, and migration notes.

The two halves cooperate. The research framing prevents us from overpromising. The software framing prevents us from disappearing into a TODO swamp.

## 4. The ethical posture

### 4.1 Honest benchmarks are a feature

The handoff document said it well: "publishing honest benchmarks ('FHE is slow, here is exactly why') is more credible than overselling." We preserve this framing in everything we publish.

When the blog post lands, the headline is not "Penumbra makes FHE fast." It is "Penumbra makes FHE usable for ML, and here is what 'usable' costs."

### 4.2 We do not pretend to be a cryptographic audit

Penumbra wraps TFHE-rs. We inherit its security properties. We do not extend them. We do not weaken them (within our power). We do not claim them as our own.

A user who reads `SECURITY.md` should leave knowing exactly what we guarantee, what we don't, and where to look for higher assurance.

### 4.3 We do not weaponize privacy

Privacy-preserving ML can be used for medical diagnosis without exposing patient records. It can also be used to run inference on people who have no opportunity to opt out, while making oversight harder.

We acknowledge the second possibility exists. We do not refuse to build the tool — privacy is a primitive, and primitives are dual-use — but we will not market it in ways that emphasize evasion of legitimate oversight.

### 4.4 We credit our dependencies clearly

TFHE-rs is the project that makes Penumbra possible. Zama's contribution is acknowledged in the README, in `NOTICE`, in every release post. We do not let our compiler-and-runtime layer obscure the fact that the cryptography is theirs.

## 5. The contributor temperament we want

These are the qualities we look for in contributors — including AI agents (see [`AGENTS.md`](AGENTS.md)).

### Patient

FHE is slow. Compilation is slow. Debugging encrypted computation is slow. People who get frustrated and start cutting corners will introduce bugs that take a year to find. Patience is a technical skill here.

### Skeptical of their own code

The bar for "this works" is not "the test passes once." It is "the test passes repeatedly, the property tests don't find counterexamples, the benchmarks haven't regressed, and a reviewer agrees."

### Willing to write things down

The next person who reads your code will not be you. They will not have your context. The job is not done until the context is in a comment, a docstring, a commit message, a `CHANGELOG.md` entry, or a doc page.

### Comfortable saying "I don't know"

In a field where the math is unforgiving, the most dangerous contributor is the one who fakes confidence. We would rather a PR be marked "I don't understand why this works; please double-check" than have it merged silently.

### Generous with attribution

If a reviewer suggests a substantive improvement, credit them in the commit message. If a community member files an issue that leads to a fix, credit them in `CHANGELOG.md`. The project is not a single-author achievement; the credit should reflect that.

## 6. The non-goals

These are things we have **explicitly chosen not to do**. If you want them, you want a different project.

| Non-goal | Why |
|---|---|
| Transformer / attention support in v0.x | The depth requirements push beyond what is currently practical under FHE. We may revisit at v1.x. |
| GPU acceleration | TFHE-rs does not support GPU FHE. Building this ourselves is out of scope. |
| Training under FHE | This is a research area unto itself. Penumbra is inference-only. |
| Multi-party computation | MPC is a different cryptographic regime with different tradeoffs. Out of scope. |
| Encrypted model weights (model privacy) | We protect input/output privacy. Model weights are plaintext on the server. Both privacy is hard and rare. |
| A custom FHE scheme | We are not cryptographers. We use TFHE via TFHE-rs. |
| Beating SEAL / OpenFHE on raw operation throughput | We are not competing with those projects. We are layering on top of TFHE-rs. |
| Browser / WASM deployment | TFHE-rs WASM support is immature. Not in v0.x. |
| Real-time inference (sub-second) | The math forbids this today. Setting this as a goal would force us to lie about it. |

## 7. What success looks like

By the end of Month 3, we judge ourselves against:

1. **Does a competent ML engineer who has never seen FHE before manage to run encrypted MNIST in under an hour from a fresh clone?** This is the usability test.
2. **Are the benchmarks honest, reproducible, and documented?** This is the credibility test.
3. **Does the codebase invite contribution?** Open it up. Read AGENTS.md. Read CONTRIBUTING.md. Can a new contributor make a meaningful PR within a week of starting? This is the project-health test.
4. **Has the project produced one durable artifact — a blog post, a paper, a talk — that others will cite?** This is the contribution test.

If all four are "yes," v0.1 is a success regardless of how many GitHub stars it has.

If the first one is "no," nothing else matters.

## 8. The thing we will resist

The temptation to expand scope.

Every interesting FHE/ML project we see will tempt us. New schemes, new architectures, new optimizations. Each of these will look like a 1-week side project and consume a month.

The scope is in [`ROADMAP.md`](ROADMAP.md). The components are in [`ARCHITECTURE.md`](ARCHITECTURE.md). The non-goals are in this document. We do not expand scope without first removing scope.

If a feature does not appear in `ROADMAP.md`, it is not on the path to v0.1.

This is the discipline that gets us to a shipped artifact in 3 months instead of an abandoned repo in 6.

---

> *"Penumbra: light is present, but obscured."* The motif is also the discipline. We compute in shadow because we choose to. We move slowly because we have to. We do not pretend the constraint isn't there. We make a virtue of it.
