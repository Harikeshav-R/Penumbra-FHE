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
   editing a backend (`crates/penumbra-tfhe/`, `crates/penumbra-ckks/`, or the eval loop in `crates/penumbra-core/src/eval.rs`).
   If it does, the abstraction leaked — the fix is a more general op or a missing registry entry, not a use-case hack.
3. **New backend ⇒ new crate, never new IR.** Adding a scheme must not require changing the
   IR, the op vocabulary, or the eval loop. Both backends read the same IR file and run under
   the same harness — **backend parity**, which is what makes the scheme comparison valid.
4. **Keep the IR in lockstep, and backend-neutral.** Any change to the IR updates **both**
   `python/penumbra/ir.py` and the Rust `crates/penumbra-core/src/ir.rs`, bumps the schema-version field, and updates
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
cargo +nightly test -p penumbra-ckks --features ckks --release  # CKKS backend (x86-64: RUSTFLAGS="-C target-feature=+avx2,+fma")
```

## Repository layout

| Path | Layer | Role |
|---|---|---|
| `python/penumbra/` | Layer 3 | ONNX loader, registry, quantization, IR emitter, reference oracle |
| `crates/penumbra-core/` | Layer 2 | Backend-neutral IR, eval loop, bit-width, `Backend`/`Op` traits (no crypto) |
| `crates/penumbra-tfhe/` | Layer 1 | Reference TFHE/CGGI backend over tfhe-rs |
| `crates/penumbra-ckks/` | Layer 1 | CKKS backend over poulpy-ckks |
| `crates/penumbra-bench/` | Harness | Shared comparison harness (unpublished) |
| `crates/penumbra-py/` | Bindings | PyO3 bindings exposing the runtime to Python in-process |
| `runtime/` | Facade / CLI | `penumbra-fhe-runtime`: facade re-exports, CLIs in `runtime/src/bin/`, TFHE goldens in `runtime/tests/` |
| `tests/` | Tests | Python tests + shared fixtures |
| `examples/` | Examples | Use cases (`mnist`, `faces`, `tabular`, `trees`, `client_server`) — graphs only |

## The canonical "add an op" path

Adding a new operator is the most common extension. Follow these steps, naming exact files. The worked example in git history is the Compare op (commits `2539378`, `2a0d392`, `51d91ef`, `f12d63d`):

1. **Map onto an existing op?** If the ONNX op can be lowered directly to an existing internal op, only steps 2, 5 (tests), and 6 apply.
2. **Front door:**
   - Add an `OnnxOpRule` to `REGISTRY` in `python/penumbra/op_registry.py`.
   - Add lowering logic in `python/penumbra/onnx_loader.py`.
   - Add a float layer class in `python/penumbra/layers.py` if users should be able to assemble the layer by hand in `fhe.Model([...])`.
3. **New internal op = IR change** (AGENTS.md §5):
   - Add an `OpSpec` enum variant, its `op_type()` string, and its `validate()` arm in `crates/penumbra-core/src/ir.rs`.
   - Add a `*Spec` dataclass, its `from_dict` dispatch, and the unknown-op error message in `python/penumbra/ir.py`.
   - Re-export the new spec in `python/penumbra/__init__.py`.
   - Bump `SCHEMA_VERSION` in both files with a history comment.
   - Regenerate `schema_version` in committed fixtures (`tests/fixtures/`).
   - Extend `tests/test_ir_conformance.py` to cover the new op payload.
   - Document the spec in `docs/IR-SPEC.md`.
   - Commit as `feat(ir)!: add <Op> op and bump schema to <version>` with a `BREAKING CHANGE:` footer.
4. **Reference + bit-width rule:**
   - Define exact integer semantics in `python/penumbra/reference.py` (`evaluate_graph_int`).
   - Implement the growth rule in `crates/penumbra-core/src/bitwidth.rs` and mirror it in `python/penumbra/bitwidth.py`.
   - Add test cases to `tests/fixtures/bitwidth_cases.json`, read by both `tests/test_bitwidth_conformance.py` and `runtime/tests/bitwidth_conformance.rs`.
   - Implement backend budget checks in `check_graph_budget`: radix capacity for TFHE, multiplicative depth/scale for CKKS.
5. **Every backend:**
   - **TFHE:** Add `crates/penumbra-tfhe/src/ops/<op>.rs` implementing `penumbra_core::ops::Op<TfheBackend>` (`eval` or `eval_n`/`eval_multi`, `output_bits*`, `cost`), register in `crates/penumbra-tfhe/src/ops/mod.rs`, add an arm to `TfheBackend::build_op` in `crates/penumbra-tfhe/src/backend.rs`, and re-export in `runtime/src/lib.rs`.
   - **CKKS:** Implement in `crates/penumbra-ckks/src/ops/` and add an arm to `CkksBackend::build_op` in `crates/penumbra-ckks/src/backend.rs`. If the op cannot be realized on CKKS, implement a load-time rejection using the `Pool(max)` pattern (`backend.rs:96-101`: `"operator X is unsupported on backend 'ckks': <why>. Use …, or run this model on the 'tfhe' backend"`) and test it in `crates/penumbra-ckks/tests/ckks_unsupported_ops.rs`.
6. **Tests at each comparator:**
   - TFHE bit-for-bit goldens in `runtime/tests/golden_*.rs` (slow ones get `#[ignore]` and are added to the `.github/workflows/slow-goldens.yml` matrix).
   - CKKS within declared error bound in `crates/penumbra-ckks/tests/ckks_golden_ops.rs` (model bounds in `crates/penumbra-ckks/src/bounds.rs`).
   - Cost model tests in `crates/penumbra-tfhe/tests/cost_model.rs`.
   - Op-set parity tests in `crates/penumbra-bench/tests/backend_parity.rs`.
7. **Docs:**
   - Update `docs/SUPPORTED-OPS.md` op table and backend-support matrix (the ONNX front-door table is pinned by `tests/test_supported_ops_doc.py`).
   - Update `docs/IR-SPEC.md`.

Remember that the cost models differ: **TFHE runtime ≈ number of bootstraps**, so prefer
realizations that avoid unnecessary PBS; **CKKS runtime ≈ depth × (rotations + rescales)**, so
prefer realizations that stay shallow. Linear/conv with plaintext weights are cheap under
both; activations/requant/compare are the expensive half under both, for different reasons.

## The canonical "add a backend" path

Much rarer, and deliberately bounded. The full path is in
[`docs/BACKENDS.md`](docs/BACKENDS.md#adding-a-backend-the-canonical-path); in short:

1. Create a new crate `crates/penumbra-<scheme>/`, add it to `members` and `[workspace.dependencies]` in the root `Cargo.toml`, and make it depend on `penumbra-core` plus exactly one FHE library.
2. Implement `penumbra_core::backend::Backend` (`crates/penumbra-core/src/backend.rs:35`):
   - Associated types `Ciphertext`, `ServerKey`, `ClientKey`.
   - `name`, `build_op`, `check_graph_budget`, optional `measured_counters`.
   - Server primitives (`create_trivial_zero`, `add`, `scalar_mul`, `scalar_add`, `scalar_ge`, `max`, `scalar_max`, `scalar_min`, `scalar_right_shift`, `apply_lut`).
   - Client boundary (`keygen`, `encrypt`, `decrypt_label`, `decrypt_vec`, `serialize_cts`, `deserialize_cts`, `serialize_client_key`, `serialize_server_key`).
   - Scheme tag constant, like `SCHEME_TFHE` (`crates/penumbra-tfhe/src/keys.rs:28`), wrapped in the `penumbra_core::wire` envelopes.
3. Declare the op matrix: `build_op` for every `OpSpec`, or a loud load-time rejection.
4. Declare the comparator and add golden tests against `python/penumbra/reference.py`.
5. Register the backend in the harness: a `<scheme>_backend()` constructor and an `available_backends()` entry in `crates/penumbra-bench/src/lib.rs`, backend dispatch in `crates/penumbra-bench/src/bin/report.rs` and `benches/inference.rs`, and `tests/backend_parity.rs`.
6. Expose it to Python behind a cargo feature in `crates/penumbra-py/src/lib.rs` (`available_backends` and the backend `match` arms).
7. Docs: `docs/BACKENDS.md` tables, `docs/SUPPORTED-OPS.md` columns, and `docs/NOTES-<scheme>.md`.

If you cannot implement the trait without editing `penumbra-core`, stop — that is an
architectural fork, not a routine addition. See [`docs/BACKENDS.md#adding-a-backend-the-canonical-path`](docs/BACKENDS.md#adding-a-backend-the-canonical-path) for the rationale.

## Versioning & releases

- **Policy:** Semantic Versioning ([SemVer](https://semver.org/)). `penumbra-fhe` (PyPI) and the four published crates (`penumbra-core`, `penumbra-tfhe`, `penumbra-ckks`, `penumbra-fhe-runtime`) share one version, single-sourced in root `Cargo.toml` `[workspace.package].version` (Python reads it through maturin; `penumbra.__version__` reads the installed metadata). The IR `schema_version` is independent, and bumping it is a breaking change (AGENTS.md §5).

- **One-time setup (repository owner):**
  1. On PyPI, add a pending trusted publisher: project `penumbra-fhe`, owner `Harikeshav-R`, repository `Penumbra-FHE`, workflow `release.yml`, environment `pypi`.
  2. In GitHub → Settings → Environments, create `pypi` and `crates-io`.
  3. Create a crates.io API token with scopes `publish-new` and `publish-update`, and store it as secret `CARGO_REGISTRY_TOKEN` in the `crates-io` environment.

- **Cutting a release `X.Y.Z`:**
  1. On a `chore/release-X.Y.Z` branch, set `[workspace.package].version` and the three `[workspace.dependencies]` `version` values to `X.Y.Z`, then run `cargo update --workspace && uv lock`.
  2. Rename `## [Unreleased]` entries in `CHANGELOG.md` into `## [X.Y.Z] - YYYY-MM-DD` using the tag date, and update the footer links.
  3. Merge to `main` after CI is green.
  4. Create and push the annotated tag:
     ```bash
     git switch main && git pull
     git tag -a vX.Y.Z -m "Penumbra-FHE X.Y.Z"
     git push origin vX.Y.Z
     ```
  5. `.github/workflows/release.yml` checks that tag matches `[workspace.package].version` and that the changelog section exists, builds wheels + sdist, and publishes to PyPI and crates.io.

- **For 1.0.0 specifically:** the version and changelog are already prepared, so steps 1–2 reduce to correcting the `[1.0.0]` date to the tag date.

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
