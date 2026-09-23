# Research Investigation: Cryptographic Tradeoffs in Homomorphic Machine Learning Inference

**Harikeshav Rameshkumar** | B.S. Computer Science, The Ohio State University (Expected May 2027)  
*Prospective Ph.D. Applicant, UC Berkeley EECS (Fall 2027)* | Contact: [email / portfolio / GitHub]

---

### Motivation & Core Inquiry

In privacy-preserving machine learning (PPML), standard literature relies on a familiar rule of thumb: **TFHE** is best suited for small, discrete models with non-linear lookup tables and exact comparisons, whereas **CKKS** is preferred for large-scale, batched linear algebra that tolerates floating-point approximation. 

However, this comparison is rarely subjected to strictly controlled empirical testing under identical conditions—the same quantized model graphs, identical 128-bit quantum security margins, and a unified benchmarking harness. Moreover, practitioners commonly rely on complex, general-purpose FHE compilers whose automated noise scheduling and circuit lowering often obscure where cryptographic overhead actually originates.

This project began as an independent research investigation into three questions:
1. **The Exactness Question:** *Can we strictly separate quantization error from cryptographic noise, and what does it cost to enforce bit-for-bit exactness in homomorphic evaluation?*
2. **The Scheme Boundary:** *Where is the concrete empirical crossover point where CKKS SIMD slot-packing overtakes TFHE exact scalar execution for realistic tensor workloads?*
3. **The Oblivious Access Wall:** *At what point does pure non-interactive homomorphic inference become asymptotically or concretely impractical, necessitating hybrid protocols or ORAM-backed indexing?*

---

### Research Approach & Architecture: Penumbra & Veil-FHE

To investigate these questions empirically without compiler-induced confounding variables, I designed **Penumbra-FHE** as a controlled experimental testbed, later folding in **Veil-FHE** (a CKKS exploration) as a pluggable comparative backend.

* **Isolating the Minimal ML Abstraction ("The Two Waists"):**
  Rather than building a full compiler for arbitrary programs, I formalized a minimal, backend-neutral Intermediate Representation ([`IR Schema 0.6.0`](file:///Users/hari/Developer/Penumbra-FHE/docs/IR-SPEC.md)) over a constrained vocabulary of ~8 tensor operations. This sits above a scheme-agnostic `Backend` trait ([`docs/BACKENDS.md`](file:///Users/hari/Developer/Penumbra-FHE/docs/BACKENDS.md)), ensuring that both cryptographic backends evaluate the exact same graph byte-for-byte with zero scheme-specific optimizations leaking into the model definition.
* **The "Golden Exactness" Invariant as an Oracle:**
  To study noise behavior scientifically, I formulated an integer reference oracle ([`python/penumbra/reference.py`](file:///Users/hari/Developer/Penumbra-FHE/python/penumbra/reference.py)) that computes the quantized model in unbounded integer cleartext arithmetic. Under TFHE, encrypted execution must match this oracle **bit-for-bit**; under CKKS, error must remain within a declared bound ($\epsilon$). This decoupled quantization distortion (addressed via Brevitas QAT and MSE calibration) from cryptographic noise growth.
* **Studying Non-Linearity & Accumulator Dynamics:**
  Multi-layer networks induce accumulator growth ($\sim \log_2(\text{fan\_in})$ bits per layer). Because TFHE Programmable Bootstrapping (PBS) is concretely feasible only over narrow radix blocks ($\le 2$ bits), intermediate accumulators must be compressed. I modeled the precision-latency tradeoff of fixed-point rescaling (`Requant`) coupled with single-block PBS noise refreshes versus CKKS Chebyshev polynomial approximations over budgeted multiplicative levels.

---

### Empirical Observations & Findings

1. **Quantization vs. Crypto Noise Decoupling:**
   Across small CNNs (MNIST) and representation classifiers (Olivetti faces), evaluating low-bit models (4-bit inputs/weights, 2-bit activations) under TFHE proved that **zero cryptographic noise** leaks into outputs. When combined with Quantization-Aware Training (QAT), the quantized integer model fully recovered the float baseline accuracy (0.94 float $\to$ 0.94 FHE), confirming that accuracy loss in discrete FHE is purely an artifact of quantization geometry rather than lattice noise accumulation.
2. **The Bottleneck Discrepancy:**
   In TFHE, linear layers (matrix-vector multiplies and convolutions against plaintext weights) are practically free of bootstrapping cost, executing via cheap scalar additions and multiplications. The entire latency profile ($\sim$minutes per sample) is concentrated in the non-linear boundaries (the number of PBS operations required to rescale accumulators and evaluate activations). Conversely, in CKKS, polynomial approximations consume multiplicative depth rapidly, necessitating parameter sizes ($N \ge 2^{14}$) that inflate ciphertext expansion and memory footprint.
3. **The Limits of Pure FHE in Biometric Inference:**
   While closed-set face classification (evaluating a fixed dense head under FHE) completed successfully with zero modifications to the backend, attempting to extend this to **open-set verification** (comparing an encrypted embedding against a gallery of enrolled templates) revealed a fundamental limit: homomorphic distance computation across large galleries without access-pattern leakage requires linear scans over all templates.

---

### Research Questions for Graduate Work

This investigation highlighted key tensions that I hope to explore in doctoral research:

* **ORAM & Oblivious Indexing for Encrypted Search:** In private database queries and open-set biometric verification, computing dot-products or Euclidean distances homomorphically is insufficient if the subsequent top-$k$ selection leaks memory access patterns. I am interested in exploring how Oblivious RAM (ORAM) and oblivious permutation/selection primitives can be co-designed with FHE representations to enable sublinear confidential querying.
* **Hybrid Protocol Partitioning (FHE vs. 2PC/MPC):** Non-linear layers (ReLU, MaxPool, Argmax) represent the worst-case workload for both TFHE (expensive PBS) and CKKS (high-degree polynomial depth). Comparing pure homomorphic evaluation against client-aided interactive protocols (e.g., Garbled Circuits or Secret Sharing, as in Delphi/Muse) raises rich questions about how to dynamically partition a computation across trust boundaries based on client network constraints.
* **IND-CPA^D Decryption Leakage in Iterative Serving:** Practical deployment of approximate schemes (CKKS) in client-server settings faces key-recovery vulnerabilities through residual decryption noise (Li–Micciancio 2021). I am eager to study how noise-flooding techniques or hybrid discrete-continuous representations can provide provable multi-query security without prohibitive overhead.

***

### Codebase & Reference Artifacts
* **Repository:** [github.com/Harikeshav-R/Penumbra-FHE](https://github.com/Harikeshav-R/Penumbra-FHE)
* **Architecture & Experimental Rationale:** [`PROJECT.md`](file:///Users/hari/Developer/Penumbra-FHE/PROJECT.md)
* **Backend Formalization:** [`docs/BACKENDS.md`](file:///Users/hari/Developer/Penumbra-FHE/docs/BACKENDS.md)
* **Comparative Study Protocol:** [`docs/COMPARISON.md`](file:///Users/hari/Developer/Penumbra-FHE/docs/COMPARISON.md)
* **Quantization & Accumulator Tracking:** [`docs/QUANTIZATION.md`](file:///Users/hari/Developer/Penumbra-FHE/docs/QUANTIZATION.md)
