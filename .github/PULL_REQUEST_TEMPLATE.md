<!-- Thanks for contributing! Keep PRs scoped to one logical change. -->

## What & why

<!-- What does this change do, and why? Link any relevant ROADMAP phase or issue. -->

## Type of change

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `docs` — documentation only
- [ ] `test` — tests only
- [ ] `refactor` / `perf` — no behavior change / performance
- [ ] `build` / `ci` / `chore` — tooling, deps, scaffolding

## Ground-rule checklist (see CONTRIBUTING.md)

- [ ] **Golden invariant holds:** FHE output == quantized-cleartext output, bit-for-bit
      (golden test added/updated and passing for anything touching eval).
- [ ] **No crypto edits for a new use case:** this did not require editing
      `runtime/src/ops/` or `eval.rs` to support a new model. *(If it did, explain below.)*
- [ ] **IR in lockstep (if IR changed):** both `python/penumbra/ir.py` and
      `runtime/src/ir.rs` updated, schema-version bumped, conformance test + `docs/IR-SPEC.md`
      updated.
- [ ] **New op (if applicable):** registry entry + Rust impl + bit-width rule + golden test
      + `docs/SUPPORTED-OPS.md` updated.
- [ ] **Fails loudly, early:** unsupported ops / over-budget bit-widths error at
      compile/load time with actionable messages.

## Quality

- [ ] `cargo fmt --check` and `cargo clippy -D warnings` pass.
- [ ] `ruff` and `black --check` pass.
- [ ] `cargo test --release` and `pytest` pass locally.

## Notes for reviewers

<!-- Anything that needs explanation: tradeoffs, follow-ups, or an unavoidable backend edit. -->
