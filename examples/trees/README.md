# Tabular Tree Ensembles — Decision Trees, Random Forests & XGBoost

Tabular inference via tree ensembles lowered to standard Penumbra IR ops: **train → quantize → export IR → encrypted inference**.

- **Random Forest (scikit-learn):** `RandomForestClassifier` (5 trees, max depth 3) trained on Wisconsin Breast Cancer.
- **Gradient Boosted Trees (XGBoost):** `XGBClassifier` (5 trees, max depth 3) trained on Wisconsin Breast Cancer.

This example contains **no cryptography** (`PROJECT.md` §4). The crypto lives entirely in the backend crates, and these fixtures are backend-agnostic: the same committed IR graph is what *every* backend evaluates (`docs/BACKENDS.md`).

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
cargo test --release --test golden_trees -- --nocapture
```
