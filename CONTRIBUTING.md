# Contributing to Penumbra

First — thank you for considering a contribution. Penumbra is an open source project and welcomes contributions from anyone who shares its goals (see [`PHILOSOPHY.md`](PHILOSOPHY.md)).

This guide covers the practical mechanics of contributing. Before reading it, make sure you've also read:

- **[`PHILOSOPHY.md`](PHILOSOPHY.md)** — what this project values
- **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — how the code is organized
- **[`AGENTS.md`](AGENTS.md)** — coding standards (binds humans and AI agents equally)
- **[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)** — community expectations

## Table of contents

1. [Ways to contribute](#1-ways-to-contribute)
2. [Development setup](#2-development-setup)
3. [The contribution lifecycle](#3-the-contribution-lifecycle)
4. [Code review](#4-code-review)
5. [Release process](#5-release-process)
6. [Getting help](#6-getting-help)

---

## 1. Ways to contribute

Not all contributions are code. All of these are welcome:

- **Bug reports.** Use the bug report issue template. Include a minimal reproducer.
- **Feature requests.** Check `ROADMAP.md` first; if it's in scope but not yet implemented, you can take it. If it's not in scope, open an issue to discuss.
- **Documentation.** Typos, clarifications, new tutorials — all welcome. Documentation PRs are some of the most valuable contributions a project can receive.
- **Examples and tutorials.** A working end-to-end example for a use case we haven't covered (medical imaging, fraud detection, etc.) is high-value.
- **Benchmarks.** Reproducing our benchmark numbers on different hardware, or proposing new benchmarks, helps build credibility.
- **Code.** Bug fixes, new features that are on the roadmap, performance improvements. Open an issue first to coordinate.
- **Security review.** If you have cryptographic expertise and want to audit the depth-cost table, the polynomial coefficients, or the bootstrapping placement, please do.

If you're not sure where to start, look for issues tagged [`good first issue`](https://github.com/Harikeshav-R/penumbra-fhe/labels/good%20first%20issue).

---

## 2. Development setup

### 2.1 Prerequisites

| Tool | Version | Install |
|---|---|---|
| Rust | stable, latest | <https://rustup.rs> |
| Python | 3.12+ | <https://www.python.org/downloads/> |
| `maturin` | 1.7+ | `pipx install maturin` |
| `pre-commit` | 3.0+ | `pipx install pre-commit` |
| `ruff` | 0.5+ | `pipx install ruff` |
| `pyrefly` | latest | `pipx install pyrefly` |

A C compiler is needed (Xcode CLI tools on macOS; build-essential on Linux; Visual Studio Build Tools on Windows x86_64). TFHE-rs has C dependencies for SIMD acceleration.

### 2.2 Cloning and setting up

```bash
git clone https://github.com/Harikeshav-R/penumbra-fhe.git
cd penumbra-fhe

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Create a venv and install in editable mode
python3.12 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
maturin develop --release

# Verify
cargo test --workspace
pytest tests/
```

If `maturin develop` fails:

- Check that you are inside an activated venv. `maturin develop` installs into the active interpreter.
- Check that the Rust version matches `rust-toolchain.toml`.
- See the troubleshooting section in `docs/tutorials/development.rst`.

### 2.3 Building the docs

```bash
cd docs
make html
open _build/html/index.html
```

### 2.4 Running benchmarks

```bash
cargo bench --workspace
```

Results go to `target/criterion/`. Reference benchmarks are in `benchmarks/results/` for comparison.

---

## 3. The contribution lifecycle

### 3.1 Before you start

For any non-trivial change:

1. **Search existing issues.** Don't duplicate.
2. **Open an issue** describing what you want to change and why.
3. **Wait for maintainer feedback** before starting significant work. We don't want you to write 500 lines of code only to find the design is wrong for the project.
4. **Get the issue assigned to you** if you're going to implement it. This prevents two people working on the same thing.

For trivial changes (typos, comment fixes, etc.) you can skip straight to a PR.

### 3.2 Branching

Branch from `main`. Use the naming convention from [`AGENTS.md`](AGENTS.md#62-branch-naming):

```
<type>/<short-kebab-description>
```

Examples:

```bash
git checkout -b feat/greedy-bootstrap-placement
git checkout -b fix/avx512-bootstrap-refresh
git checkout -b docs/clarify-quantization-scale
```

### 3.3 Committing

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). The commit-msg pre-commit hook enforces the format. See [`AGENTS.md`](AGENTS.md#61-conventional-commits) for the full type/scope list.

Sign your commits when possible (`git commit -S`). Branch protection on `main` will eventually require signed commits.

### 3.4 Before you push

Run the full local CI loop:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
ruff check .
ruff format --check .
pyrefly check python/
maturin develop --release
pytest tests/
```

If any of these fail, CI will fail. Fix locally first.

### 3.5 Opening a PR

- Push your branch.
- Open a PR against `main`.
- Fill in the PR template completely. Yes, every section.
- Link the issue your PR addresses.
- If the PR is not ready for review, open it as a draft.

### 3.6 PR size

Aim for **under 400 lines of diff**, excluding generated files, fixture data, and documentation. Larger PRs are not categorically rejected, but require justification and are slower to review.

### 3.7 Definition of done

A PR is done when:

- All CI checks pass on Ubuntu, macOS, and Windows x86_64.
- At least one human reviewer has approved (two for crypto changes).
- All review comments are addressed.
- The branch is up to date with `main`.
- The CHANGELOG is updated if the change is user-visible.

See [`AGENTS.md`](AGENTS.md#part-4--definition-of-done) for the exhaustive list.

---

## 4. Code review

### 4.1 What reviewers look for

- Correctness (does it do what it claims?)
- Test coverage (is the new behavior verified?)
- Documentation (rustdoc/docstrings/CHANGELOG?)
- Adherence to `ARCHITECTURE.md` (right component, right boundaries?)
- Style (does it match the existing code?)
- Compliance with `AGENTS.md` hard rules

Reviewers may ask for changes for reasons not in this list. Be open to feedback.

### 4.2 What reviewers do not police

- Personal style preferences (e.g., "I would have named this variable differently"). These are suggestions, not blocks.
- Optimizations that are not on the critical path.
- Whether the issue should have been filed in the first place (that's an upstream decision).

### 4.3 Response time expectations

- **Maintainer first response:** within 7 days for issues, within 7 days for PRs from existing contributors, within 14 days for PRs from new contributors (we want to give a thorough first review).
- **Contributor response to review feedback:** within 14 days, or the PR may be marked stale.

If you need more time, say so in a comment. We don't close PRs on people; we close PRs on silence.

### 4.4 Handling disagreement

If you disagree with a reviewer:

1. Discuss the substance. State what you think and why.
2. If you remain disagreed after discussion, the maintainer's decision stands for this PR.
3. If you think the maintainer's decision sets a problematic precedent, open a separate issue to discuss the policy, not the specific PR.

We're collaborating in good faith. Respectful disagreement makes the project better.

---

## 5. Release process

Releases are cut by the maintainer. Contributors do not need to know the full process, but if you're curious:

1. The `release` workflow is triggered from a release branch.
2. The workflow bumps `Cargo.toml` and `pyproject.toml` versions.
3. CI builds wheels for all supported platforms.
4. Maintainer reviews artifacts.
5. Maintainer pushes a signed tag (`vX.Y.Z`).
6. The workflow publishes to PyPI and crates.io.
7. A GitHub Release is created with notes derived from `CHANGELOG.md`.

The full process is documented in `docs/development/release.rst`.

---

## 6. Getting help

- **Questions about using Penumbra:** open a [discussion](https://github.com/Harikeshav-R/penumbra-fhe/discussions).
- **Bugs:** open an [issue](https://github.com/Harikeshav-R/penumbra-fhe/issues) using the bug report template.
- **Security issues:** see [`SECURITY.md`](SECURITY.md). Do not file public issues for security problems.
- **Maintainer contact:** r.harikeshav@gmail.com (please prefer issues/discussions for non-private matters).

---

## A note on AI-assisted contributions

Many contributors will use AI coding agents (Claude Code, Cursor, Aider, etc.) to help with PRs. **This is welcome.** We use AI agents ourselves. The rules:

1. **The human submitter is responsible for the contribution.** If a reviewer asks "why does this do X?", "the agent wrote it" is not an answer.
2. **The agent must follow [`AGENTS.md`](AGENTS.md).** If your agent doesn't, the PR is rejected — not because we object to AI, but because the result will not meet the project's standards.
3. **Be transparent.** If a PR was substantially AI-authored, mentioning it in the PR description is appreciated but not required.

We don't have a separate process for AI-assisted PRs. Same rules, same review, same bar.

---

Thank you again. Every contribution — bug report, typo fix, benchmark run — makes the project better.
