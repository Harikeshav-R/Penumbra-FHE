# Tree ensembles — random forests & XGBoost

Tabular inference via tree ensembles lowered to standard Penumbra IR ops: **train → quantize → export IR → encrypted inference**.

- **Random Forest (scikit-learn):** `RandomForestClassifier` (5 trees, max depth 3) trained on Wisconsin Breast Cancer.
- **Gradient Boosted Trees (XGBoost):** `XGBClassifier` (5 trees, max depth 3) trained on Wisconsin Breast Cancer.

For a neural-network approach on the same tabular dataset, see [`examples/tabular/`](../tabular/).

This example contains **no cryptography** (`PROJECT.md` §4). The crypto lives entirely in the backend crates, and these fixtures are backend-agnostic: the same committed IR graph is what *every* backend evaluates ([`docs/BACKENDS.md`](../../docs/BACKENDS.md)).

## Run it (one command)

```bash
uv run python examples/trees/run.py            # [--model KEY] [--samples N] [--backend tfhe|ckks]
```

Prerequisite is `uv sync` only: it builds the PyO3 extension from source and requires no network
access and no `ml` extra. The command replays the committed fixture's pre-quantized `test_inputs`
in-process through keygen → encrypt → evaluate → decrypt via `penumbra.client.run_encrypted`.
Under TFHE (default), the decrypted outputs are checked bit-for-bit against the quantized-cleartext
reference `evaluate_graph_int`, exiting with code 1 on any mismatch (`AGENTS.md` §1.1).

**CKKS rejection note:** Chained sharp comparison steps exceed the CKKS multiplicative level budget
(360 > 330 bits, `docs/SUPPORTED-OPS.md:234`). Running with `--backend ckks` fails loudly at load time
with an actionable depth-budget error message rather than computing degraded noise (`AGENTS.md` §1.4).

### Models

| `--model` Key | Fixture | Generator | IR Graph | TFHE s/sample |
|---|---|---|---|---:|
| `phase8_trees` (default) | `phase8_trees_fixture.json` | `tree_export.py` | `Compare → Linear → Compare → Linear` | ~10.4 s |
| `phase8_xgb` | `phase8_xgb_fixture.json` | `xgb_export.py` | `Compare → Linear → Compare → Linear` | — |

*Latency is wall-clock per sample under TFHE `classic` profile (`docs/BENCHMARKS.md:200`).*

---

## The 4-Stage Lowering (Sum of Comparisons)

`crates/penumbra-core/src/backend.rs` exposes no ciphertext × ciphertext multiply, so tree paths cannot be computed via indicator products (muxes). Instead, trees are lowered using the sum-of-comparisons formulation:

| Stage | Node | Op | Semantics |
|---|---|---|---|
| 1 | `split_cmp` | `Compare` | $b_g = [x[\text{feature}_g] \ge T_g]$ — one comparison PBS per internal split across all trees |
| 2 | `leaf_score` | `Linear` | $\text{score}_l = \sum_{g \in \text{path}(l)} (\pm 1) \cdot b_g + |\text{left}(l)|$ — attains max $\text{depth}_l$ iff every condition on path holds |
| 3 | `leaf_sel` | `Compare` | $[\text{score}_l \ge \text{depth}_l]$ — one-hot active leaf indicator ($score_l \ge depth_l \iff score_l = depth_l$) |
| 4 | `logits` | `Linear` | $\sum_l V[c][l] \cdot \text{leaf\_sel}[l] + \text{bias}_c$ — class logits, summed across trees |

The client argmaxes the decrypted logits, matching the standard multi-class convention (`PROJECT.md` §11).

---

## Reproducing

```bash
# Generate scikit-learn Random Forest fixture (examples/trees/phase8_trees_fixture.json):
uv run --extra ml python examples/trees/tree_export.py

# Generate XGBoost fixture (examples/trees/phase8_xgb_fixture.json):
uv run --extra ml python examples/trees/xgb_export.py

# Run golden test (exact bit-for-bit check under TFHE):
cargo test -p penumbra-fhe-runtime --release --test golden_trees -- --include-ignored --nocapture
```
