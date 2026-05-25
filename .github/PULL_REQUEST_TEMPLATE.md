## Summary

[Write a one-paragraph summary of what this PR does.]

## Type of change

- [ ] `feat` (New feature, user-visible)
- [ ] `fix` (Bug fix)
- [ ] `docs` (Documentation only)
- [ ] `refactor` (Code change that neither fixes a bug nor adds a feature)
- [ ] `perf` (Performance improvement)
- [ ] `test` (Adding or modifying tests)
- [ ] `build` (Build system or dependency changes)
- [ ] `ci` (CI configuration)
- [ ] `chore` (Miscellaneous)
- [ ] `crypto` (Changes to cryptographic primitives or polynomial coefficients; requires two reviewers)

## Linked issues

Closes #
Refs #

## Changes

- [List substantive changes here]
- ...

## Testing

- [Explain what tests were added]
- [Explain how the reviewer can verify locally]

## Definition of Done checklist

- [ ] All CI checks pass on Ubuntu, macOS, and Windows x86_64.
- [ ] `cargo fmt --all -- --check` passes.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes.
- [ ] `cargo test --workspace` passes.
- [ ] `ruff check .` passes.
- [ ] `pyrefly check python/` passes.
- [ ] `pytest tests/` passes.
- [ ] User-facing changes documented in `CHANGELOG.md` under `## [Unreleased]` (if applicable).
- [ ] Docstrings/rustdoc added for new public API (if applicable).

## Crypto checklist (only if applicable)

- [ ] Correctness test using `proptest` (e.g. `decrypt(op(encrypt(x), encrypt(y))) == op(x, y)` for >=100 cases).
- [ ] Depth-cost entry added to `crates/penumbra-analyzer/src/depth_costs.rs` with empirical justification.
- [ ] A benchmark in `benchmarks/runtime/` measuring the operation's latency.
- [ ] No `unwrap()`, `panic!`, `expect()` in the implementation path.
- [ ] Explicit RNG handling. No implicit `thread_rng()`.

## Screenshots / output (if applicable)

## Reviewer notes

[Anything the reviewer should know upfront]
