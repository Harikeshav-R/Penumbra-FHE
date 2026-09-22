# Investigation: CKKS Error Bound Violation on `phase7_faces` (Option C Blueprint)

This document records the complete root-cause analysis, mathematical derivation, and remediation blueprint for the CKKS bound violation discovered during the Phase 12.4 cross-backend comparison sweep on `phase7_faces` (Sample 1).

---

## 1. Summary of Defect

- **Model:** `phase7_faces` (Phase-7 Olivetti face recognition, `examples/faces/phase7_faces_fixture.json`).
- **Topology:** `Conv2d(1→8, 3×3, stride 4) → Requant(per-ch, 8ch) → Linear(128→8)`.
- **Backend:** CKKS (`poulpy-ckks 0.8.3`, Apple Silicon `FFT64Neon` HAL, Option B SIMD packing over $N=16384$, `lt_slots = 256`).
- **Declared Bound:** `bounds::PHASE7_FACES = 150.0` (`crates/penumbra-ckks/src/bounds.rs:11`).
- **Observed Behavior:**
  - **Sample 0 (`test_inputs[0]`):** Max absolute error = **`105.0`** ($\le 150.0$). Passes bound.
  - **Sample 1 (`test_inputs[1]`):** Max absolute error = **`192.0`** ($> 150.0$). **Violates bound by 42.0.**
  - **Winning logit flipped:** On Sample 1, the expected class is 2 with expected logit `+185`. CKKS decrypted `-7` (error of `192.0`). Because the winning logit collapsed below zero, the predicted argmax flipped from class 2 to class 5.

### Per-Logit Comparison on Sample 1
| Logit Index | Expected Cleartext | Decrypted CKKS | Error ($\text{CKKS} - \text{Expected}$) | Absolute Error |
|:---:|:---:|:---:|:---:|:---:|
| 0 | -102 | -41 | +61.0 | 61.0 |
| 1 | -262 | -264 | -2.0 | 2.0 |
| **2 (Target)** | **+185** | **-7** | **-192.0** | **192.0 (Bound Violated)** |
| 3 | +24 | +11 | -13.0 | 13.0 |
| 4 | +7 | -27 | -34.0 | 34.0 |
| 5 | -40 | +21 | +61.0 | 61.0 |
| 6 | +51 | -33 | -84.0 | 84.0 |
| 7 | -96 | -64 | +32.0 | 32.0 |

---

## 2. Why Existing Golden Tests Missed It

In Phase 12.2:
- `crates/penumbra-ckks/tests/ckks_golden_faces.rs:42-44`:
  ```rust
  let input_0 = as_i64_vec(&fx["test_inputs"][0]);
  let want_logits = as_i64_vec(&fx["expected_logits"][0]);
  ```
- `crates/penumbra-ckks/examples/calibrate.rs:139`:
  ```rust
  let input_0 = as_i64_vec(&val["test_inputs"][0]);
  ```

Both the calibration example and the golden test evaluated **only Sample 0** (`test_inputs[0]`), where the max error happened to be 105.0. The declared bound of 150.0 was set based solely on this single-sample calibration. Multi-sample evaluation in Phase 12.4 exposed the deficiency on Sample 1.

---

## 3. Mathematical Root Cause Analysis

The error is not cryptographic noise (which is $< 10^{-7}$), but an accumulation of **continuous target function distortion** inside `fit_requant` amplified across 128 weights in `linear2`.

### 3.1 The Continuous Target Distortion in `fit_requant`
In `crates/penumbra-ckks/src/ops/polymap.rs:54-58`:
```rust
let f = move |t: f64| -> f64 {
    let relu = t.max(0.0);
    let scaled = (relu * (mult as f64) + (round_bias as f64)) / divisor;
    scaled.clamp(0.0, max_val)
};
```

1. **Integer vs Float Semantics:**
   In quantized integer arithmetic (Layer 2 / TFHE):
   $$\text{out} = \left\lfloor \frac{\max(t, 0) \cdot m + \text{round\_bias}}{2^s} \right\rfloor$$
   Here, $\text{round\_bias} = 2^{s-1}$. In integer arithmetic, adding $2^{s-1}$ before right-shifting by $s$ bits implements **rounding to the nearest integer** ($\lfloor x + 0.5 \rfloor = \text{round}(x)$).
   Crucially, when $t \le 0$, $\max(t, 0) = 0$, so:
   $$\left\lfloor \frac{0 + 2^{s-1}}{2^s} \right\rfloor = \lfloor 0.5 \rfloor = 0$$
   The output is strictly **0**.

2. **The Float Trap in `fit_requant`:**
   In floating-point arithmetic without integer truncation:
   $$\frac{0 \cdot m + 2^{s-1}}{2^s} = \mathbf{0.5}$$
   The continuous target function $f(t)$ evaluates to **`0.5`** for all negative inputs $t \le 0$!
   Instead of fitting a function that is flat zero for negative values, Chebyshev polynomial fitting is instructed to approximate a function that is flat **`+0.5`** for all negative $t$, and adds an un-truncated $+0.5$ offset for all positive $t$.

3. **Per-Channel Systematic Offset:**
   On Sample 1, the post-convolution activations for channels 1, 3, and 4 are completely negative:
   - Channel 1: $t \in [-488, -184]$ (cleartext target: $0$, polynomial evaluates to $\approx 0.515$)
   - Channel 3: $t \in [-1475, -739]$ (cleartext target: $0$, polynomial evaluates to $\approx 0.480$)
   - Channel 4: $t \in [-789, -341]$ (cleartext target: $0$, polynomial evaluates to $\approx 0.496$)
   Every negative activation receives an unintended $+0.5$ bias from the continuous target definition.

### 3.2 Error Amplification Through `linear2`
`linear2` takes the 128 post-Requant activations ($8 \text{ channels} \times 16 \text{ pixels}$) and computes 8 logits:
$$\text{logit}_k = \sum_{j=0}^{127} W_{k, j} \cdot y_j + b_k$$

The weights in `linear2` have large magnitudes:
- Row 2 (Logit 2) has an $L_1$ norm of **$\sum_{j=0}^{127} |W_{2, j}| = 1351$** (weights range from $-29$ to $+31$).
- When an average approximation bias $\epsilon_j \approx +0.4 \text{ to } +0.5$ across the 128 elements is multiplied by $W_{2, j}$, the errors accumulate constructively:
  $$\Delta \text{logit}_2 = \sum_{j=0}^{127} W_{2, j} \cdot \epsilon_j \approx \mathbf{-200.25}$$

Simulating the degree-15 Chebyshev polynomial fit with the current target function predicts:
$$\Delta \text{logit}_2 = -200.25 \quad (\text{Actual Measured Error} = \mathbf{-192.00})$$
The mathematical prediction matches the empirical decryption error almost exactly.

### 3.3 Secondary Factor: Overly Wide Chebyshev Domain $x_{\max}$
In `crates/penumbra-ckks/src/ops/polymap.rs:45-51`:
- $x_{\max}$ is calculated from $t_{\text{sat}}$ and clamped to $2^{\text{input\_bits} - 1} = 2^{13} = 8192$.
- The Chebyshev approximation is forced to fit over $[-8192, 8192]$ (a span of 16,384 units).
- However, the actual post-conv activations on Sample 1 only span $[-1475, 2423]$. Fitting degree-15 over $[-8192, 8192]$ wastes polynomial capacity on ranges that are never visited, increasing oscillation (Runge-like ripple) in the active $[-1500, 2500]$ window.

---

## 4. Verification of the Proposed Fix

If $f(t)$ in `fit_requant` is corrected to represent the ideal continuous scaling without the un-truncated $+0.5$ bias:
$$f_{\text{ideal}}(t) = \text{clamp}\left(\frac{\max(t, 0) \cdot m}{2^s}, 0, \text{max\_val}\right)$$

Simulated error on Sample 1:
- Logit 2 predicted error drops from **`-200.25`** down to **`-42.31`**!
- All 8 logit errors fall within $[-42.3, +31.5]$, well below the declared bound of $150.0$.
- Logit 2 decrypted value becomes $\approx 185 - 42.3 = 142.7$, retaining its argmax victory ($142.7 > 21.0$). Label match on Sample 1 is **restored to 100% (2/2)**.

---

## 5. Blueprint for Future Execution (Option C)

When executing Option C in the future, follow this implementation checklist:

### Step 1: Update `fit_requant` in `crates/penumbra-ckks/src/ops/polymap.rs`
In `crates/penumbra-ckks/src/ops/polymap.rs:54-58`, change:
```rust
// FROM:
let f = move |t: f64| -> f64 {
    let relu = t.max(0.0);
    let scaled = (relu * (mult as f64) + (round_bias as f64)) / divisor;
    scaled.clamp(0.0, max_val)
};

// TO:
let f = move |t: f64| -> f64 {
    let relu = t.max(0.0);
    let scaled = (relu * (mult as f64)) / divisor;
    scaled.clamp(0.0, max_val)
};
```
*(Optional refinement)*: If tighter bounds are desired, compute $x_{\max}$ from the dynamic range or clip $x_{\max} = \min(x_{\max}, 4096)$.

### Step 2: Update Golden Tests to Cover Multi-Sample
In `crates/penumbra-ckks/tests/ckks_golden_faces.rs`:
Add Sample 1 to the test loop so regressions on either sample are caught immediately:
```rust
for (i, (input, want_logits)) in fx["test_inputs"].as_array().unwrap().iter().zip(fx["expected_logits"].as_array().unwrap()).enumerate() {
    // evaluate both sample 0 and sample 1
}
```

### Step 3: Verify All 7 Fixtures Under the New Requant
Run:
```bash
cargo +nightly test -p penumbra-ckks --features ckks --release
cargo +nightly run -p penumbra-ckks --features ckks --release --example calibrate
```
Ensure all 7 models remain within their declared bounds (e.g. `phase4_cnn`, `phase5_digits`, `phase5_qat`, `phase6_onnx` also use `Requant` and must be confirmed green).

### Step 4: Re-run Sweep and Update Published Artifacts
Re-run the comparison sweep on `phase7_faces`:
```bash
./target/release/penumbra-bench-report --models phase7_faces --backends tfhe,ckks --samples 2 --format json --out target/bench-results/phase7_faces.json
```
Re-merge via `/tmp/merge_comparison.py`, update `docs/results/phase12-4-comparison.json`, and update the tables in `docs/BENCHMARKS.md` and `docs/COMPARISON.md`.
