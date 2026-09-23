# Contributing to Penumbra-FHE

Thanks for your interest! This document covers how to set up, the rules that keep the
architecture sound, and the canonical way to extend the library.

Before anything else, read:
- [`PROJECT.md`](PROJECT.md) — the architecture and why it's shaped this way.
- [`ROADMAP.md`](ROADMAP.md) — the phased build plan.
- [`docs/BACKENDS.md`](docs/BACKENDS.md) — the backend boundary, if you're touching crypto.
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — toolchain, build, and test commands.

## Ground rules (non-negotiable)

These are the invariants that keep Penumbra-FHE correct and general. PRs that violate them
will not be merged.

1. **The golden invariant.** Encrypted output must match the quantized-cleartext output — the
   reference is always `python/penumbra/reference.py`. The comparator is per-backend: **TFHE
   bit-for-bit** (TFHE is exact, so any discrepancy is a quantization or implementation bug,
   never crypto noise), **CKKS within the model's declared error bound**, with the measured
   error reported. Every change touching eval ships with a passing golden test at the right
   comparator. Never compare a backend against a friendlier reference to flatter its numbers.
2. **New use case ⇒ new graph, never new crypto.** Adding a model/use case must not require
   editing a backend (`runtime/src/ops/`, `eval.rs`). If it does, the abstraction leaked —
   the fix is a more general op or a missing registry entry, not a use-case hack.
3. **New backend ⇒ new crate, never new IR.** Adding a scheme must not require changing the
   IR, the op vocabulary, or the eval loop. Both backends read the same IR file and run under
   the same harness — **backend parity**, which is what makes the scheme comparison valid.
4. **Keep the IR in lockstep, and backend-neutral.** Any change to the IR updates **both**
   `python/penumbra/ir.py` and the Rust `ir.rs`, bumps the schema-version field, and updates
   the cross-language conformance test + `docs/IR-SPEC.md` — in the **same change**. A change
   motivated by one scheme is an architectural fork, not a routine bump.
5. **Fail loudly, early.** Unsupported ops, over-budget bit-widths, ops a particular backend
   cannot realize, and key/ciphertext material handed to the wrong backend are all caught at
   compile/load time with actionable messages, never silently at runtime.

## Development setup

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md). In short:

```bash
cargo test --workspace --release                     # Rust runtime (use --release for FHE)
uv sync --all-extras && uv run pytest               # Python front end (uv, not poetry)
```

## The canonical "add an op" path

Adding a new operator is the most common extension. Do all five, in one PR:

1. **Registry entry** — map the ONNX op → internal op in `python/penumbra/op_registry.py`.
2. **Implementation in every backend** — implement it against each backend's primitives, or
   **reject it loudly** on backends that cannot realize it (naming the op, the node, and the
   backend). Never approximate silently, and never let the op vocabulary fork per backend.
3. **Bit-width rule** — declare how the op grows the bit-width budget (`PROJECT.md` §9), so
   automatic `Requant` insertion stays correct, and note its consequence for each backend's
   resource budget (radix capacity for TFHE, depth/scale for CKKS).
4. **Golden test** — assert the invariant **at each backend's comparator** for the new op.
5. **Docs** — add it to `docs/SUPPORTED-OPS.md` with its per-backend support status (the
   documented list must match what the validator accepts — this is itself tested).

Remember that the cost models differ: **TFHE runtime ≈ number of bootstraps**, so prefer
realizations that avoid unnecessary PBS; **CKKS runtime ≈ depth × (rotations + rescales)**, so
prefer realizations that stay shallow. Linear/conv with plaintext weights are cheap under
both; activations/requant/compare are the expensive half under both, for different reasons.

## The canonical "add a backend" path

Much rarer, and deliberately bounded. The full path is in
[`docs/BACKENDS.md`](docs/BACKENDS.md#adding-a-backend-the-canonical-path); in short: a new
crate implementing the `Backend` trait, a declared op matrix with loud rejections, a declared
comparator plus correctness tests at it, registration with the shared harness, and docs.

If you cannot implement the trait without editing `penumbra-core`, stop — that is an
architectural fork, not a routine addition.

## Commit & PR conventions

- **Branch names** follow Conventional Branch format: `<type>/<short-kebab-description>`
  (e.g. `feat/conv2d-op`, `fix/accumulator-overflow`, `docs/ir-spec`).
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org):
  `<type>(<scope>): <imperative description>`.
  - Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`.
  - Scopes: `runtime`, `core`, `tfhe`, `ckks`, `bench`, `python`, `ir`, `ops`, `quant`,
    `onnx`, `ci`, `docs`, `examples`.
  - An IR schema-version bump is a **breaking change** (`feat(ir)!:` + `BREAKING CHANGE:`).
- Keep each commit scoped to one logical change.
- **Run formatters & linters before pushing** — they are enforced in CI and warnings are
  treated as errors (`docs/DEVELOPMENT.md`).

## Pull requests

- Fill out the PR template.
- Ensure CI is green: `cargo test`, `pytest`, `clippy`, `ruff`, `black`, and the golden test
  for every backend your change touches.
- Describe how your change preserves the five ground rules above.

## License

By contributing, you agree that your contributions are licensed under the
[Apache 2.0 License](LICENSE).
