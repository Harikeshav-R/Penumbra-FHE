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

- [ ] **Golden invariant holds:** encrypted output matches the quantized-cleartext reference
      at the right comparator — bit-for-bit under TFHE, within the declared bound (and with
      the measured error reported) under CKKS. Golden test added/updated and passing for
      anything touching eval.
- [ ] **No crypto edits for a new use case:** this did not require editing a backend's `ops/`
      or the eval loop to support a new model. *(If it did, explain below.)*
- [ ] **No waist edits for a backend change:** this did not require changing the IR, the op
      vocabulary, or adding a scheme branch to `penumbra-core`. *(If it did, explain below.)*
- [ ] **Backend parity:** both backends still consume the same IR and run under the same
      harness. Any cross-backend numbers come from that harness on one pinned machine.
- [ ] **IR in lockstep (if IR changed):** both `python/penumbra/ir.py` and the Rust `ir.rs`
      updated, schema-version bumped, conformance test + `docs/IR-SPEC.md` updated — and the
      change is **backend-neutral**.
- [ ] **New op (if applicable):** registry entry + impl (or a loud "unsupported") in *every*
      backend + bit-width rule + golden test per comparator + `docs/SUPPORTED-OPS.md` updated.
- [ ] **Fails loudly, early:** unsupported ops, over-budget bit-widths, ops a backend cannot
      realize, and key/ciphertext handed to the wrong backend all error at compile/load time
      with actionable messages.
- [ ] **Upstream breakage logged (if applicable):** anything `poulpy-ckks`'s changing API
      broke is recorded in `docs/NOTES-ckks.md`, not quietly worked around.

## Quality

- [ ] `cargo fmt --check` and `cargo clippy -D warnings` pass.
- [ ] `ruff` and `black --check` pass.
- [ ] `cargo test --release` and `pytest` pass locally, for every backend touched.

## Notes for reviewers

<!-- Anything that needs explanation: tradeoffs, follow-ups, an unavoidable backend edit, or
     an unavoidable change to penumbra-core. -->
