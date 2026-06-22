# Contributing to Penumbra-FHE

Thanks for your interest! This document covers how to set up, the rules that keep the
architecture sound, and the canonical way to extend the library.

Before anything else, read:
- [`PROJECT.md`](PROJECT.md) — the architecture and why it's shaped this way.
- [`ROADMAP.md`](ROADMAP.md) — the phased build plan.
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — toolchain, build, and test commands.

## Ground rules (non-negotiable)

These are the invariants that keep Penumbra-FHE correct and general. PRs that violate them
will not be merged.

1. **The golden exactness invariant.** FHE output must equal the quantized-cleartext output
   **bit-for-bit**. TFHE is exact — any discrepancy is a quantization or implementation bug,
   never crypto noise. Every change touching eval ships with a passing golden test.
2. **New use case ⇒ new graph, never new crypto.** Adding a model/use case must not require
   editing the Rust backend (`runtime/src/ops/`, `eval.rs`). If it does, the abstraction
   leaked — the fix is a more general op or a missing registry entry, not a use-case hack.
3. **Keep the IR in lockstep.** Any change to the IR updates **both** `python/penumbra/ir.py`
   and `runtime/src/ir.rs`, bumps the schema-version field, and updates the cross-language
   conformance test + `docs/IR-SPEC.md` — in the **same change**.
4. **Fail loudly, early.** Unsupported ops and over-budget bit-widths are caught at
   compile/load time with actionable messages, never silently at runtime.

## Development setup

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md). In short:

```bash
cd runtime && cargo test --release        # Rust runtime (use --release for FHE)
cd python  && uv sync --all-extras && uv run pytest   # Python front end (uv, not poetry)
```

## The canonical "add an op" path

Adding a new operator is the most common extension. Do all five, in one PR:

1. **Registry entry** — map the ONNX op → internal op in `python/penumbra/op_registry.py`.
2. **Rust implementation** — implement it against `tfhe-rs` primitives in `runtime/src/ops/`.
3. **Bit-width rule** — declare how the op grows the bit-width budget (`PROJECT.md` §9), so
   automatic `Requant` insertion stays correct.
4. **Golden test** — assert FHE == quantized-cleartext for the new op.
5. **Docs** — add it to `docs/SUPPORTED-OPS.md` (the documented list must match what the
   validator accepts).

Remember: **runtime ≈ number of bootstraps.** Prefer realizations that avoid unnecessary
PBS. Linear/conv with plaintext weights are cheap; activations/requant/compare are not.

## Commit & PR conventions

- **Branch names** follow Conventional Branch format: `<type>/<short-kebab-description>`
  (e.g. `feat/conv2d-op`, `fix/accumulator-overflow`, `docs/ir-spec`).
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org):
  `<type>(<scope>): <imperative description>`.
  - Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`.
  - Scopes: `runtime`, `python`, `ir`, `ops`, `quant`, `onnx`, `ci`, `docs`, `examples`.
  - An IR schema-version bump is a **breaking change** (`feat(ir)!:` + `BREAKING CHANGE:`).
- Keep each commit scoped to one logical change.
- **Run formatters & linters before pushing** — they are enforced in CI and warnings are
  treated as errors (`docs/DEVELOPMENT.md`).

## Pull requests

- Fill out the PR template.
- Ensure CI is green: `cargo test`, `pytest`, `clippy`, `ruff`, `black`, and the golden test.
- Describe how your change preserves the four ground rules above.

## License

By contributing, you agree that your contributions are licensed under the
[Apache 2.0 License](LICENSE).
