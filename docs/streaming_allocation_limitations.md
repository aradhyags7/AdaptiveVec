# AdaptiveVec — Streaming Allocation Semantics and Limitations
## Analysis of Global Budget Normalization under Incremental Insertion

**Author:** Google DeepMind / AdaptiveVec Research Team  
**Phase:** 5.3 — Budget-Constrained Geometric Connectivity Allocation  
**Date:** 2026-11-20  
**Status:** Theoretical Analysis & Engineering Boundary  

---

## 1. The Streaming Normalization Dilemma

Global budget normalization requires that the sum of assigned node degrees matches a target total:
$$\sum_{i=1}^N M_i = N \times M_{\text{base}} = B_0$$
In a **batch setting**, this constraint is straightforward to satisfy:
1. Estimate geometric complexity scores $S_i$ across all $N$ vectors.
2. Standardize scores using global sample mean $\mu_S$ and standard deviation $\sigma_S$.
3. Compute continuous allocations and apply largest-remainder integer rounding to satisfy $\sum M_i = B_0$ exactly.

However, in a **true streaming insertion regime**, vectors arrive one-by-one ($x_1, x_2, \dots, x_N$):
- Future vectors $x_{t+1 \dots N}$ are unknown.
- The global score distribution $(\mu_S, \sigma_S)$ is not known *a priori*.
- Total dataset size $N$ may be unknown or unbounded.

Therefore, **exact global budget normalization and single-pass online streaming insertion are fundamentally incompatible without statistical approximation**.

---

## 2. Four Operational Paradigms

We delineate four distinct operational paradigms:

```
                            [Operational Paradigms]
                                       |
          +----------------------------+----------------------------+
          |                                                         |
     [Batch / Static]                                      [Dynamic / Streaming]
          |                                                         |
    +-----+-----+                                             +-----+-----+
    |           |                                             |           |
[Paradigm 1] [Paradigm 2]                                 [Paradigm 3] [Paradigm 4]
Offline      Warm-Up Calibration                          Online       Stochastic
Exact Batch  Streaming Allocation                         Welford      Quota Banking
```

### Paradigm 1 — Offline Exact Batch Allocation (Reference Standard)
- **Mechanism:** Two-pass construction. Pass 1 estimates local LID and sparsity for all $N$ vectors. Pass 2 computes global largest-remainder integer allocation and inserts vectors.
- **Budget Fidelity:** $\sum M_i \equiv B_0$ with zero error.
- **Limitation:** Inapplicable to append-only real-time ingestion streams. Requires offline indexing phase.

### Paradigm 2 — Calibration-Based Streaming Allocation (Production Recommendation)
- **Mechanism:** A small representative sample ($N_{\text{calib}} = 1,000$ to $5,000$ vectors, or $\approx 1-5\%$ of corpus) is processed during an initial calibration warm-up to freeze reference distribution parameters $(\mu_{\text{calib}}, \sigma_{\text{calib}})$.
- **Online Ingestion:** Subsequent streaming vectors compute $S_t$ relative to frozen calibration baselines:
  $$\widehat{S}_t = \frac{S_t - \mu_{\text{calib}}}{\sigma_{\text{calib}}}$$
  $$M_t = \text{clip}\left( \text{round}\left( M_{\text{base}} \times [1 + \beta \widehat{S}_t] \right), \; M_{\min}, \; M_{\max} \right)$$
- **Budget Fidelity:** Asymptotically converges to $B_0$ if streaming data is independent and identically distributed (i.i.d.) with the calibration set. Budget drift $|\sum M_t - B_0| / B_0$ is typically $< 1.5\%$.
- **Computational Cost:** Single-pass streaming after brief initial warm-up.

### Paradigm 3 — Online Running Standardization (Streaming Welford)
- **Mechanism:** Parameters $\mu_t, \sigma_t^2$ are updated incrementally using Welford's algorithm as each vector is inserted.
- **Budget Fidelity:** Susceptible to early-stage warm-up skew (the first 100 vectors experience high volatility before sample statistics stabilize).

### Paradigm 4 — Stochastic Quota Banking (Strict Real-Time Enforcement)
- **Mechanism:** A running credit balance is maintained: $\text{Balance}_t = t \cdot M_{\text{base}} - \sum_{i=1}^t M_i$.
- If $\text{Balance}_t < 0$ (budget exhausted), subsequent nodes are forced to $M_{\min}$ until credit recovers.
- **Limitation:** Causes temporal unfairness; vectors arriving after a cluster of complex vectors are artificially starved of degree regardless of their own geometry.

---

## 3. Computational Preprocessing Overhead

We profile the wall-clock overhead of geometric score estimation on SIFT-100K ($N=100,000, D=128$):

| Preprocessing Stage | Mechanism | Threading | Wall-Clock Time | Relative Overhead vs HNSW Build |
| :--- | :--- | :---: | :---: | :---: |
| **Pass 1: k-NN Query Probe** | Local search in interim index ($k=30$) | OpenMP (8 threads) | 1.84 s | +9.7% |
| **Pass 2: LID & Sparsity MLE** | Log-distance MLE & reciprocal distance | OpenMP (8 threads) | 0.08 s | +0.4% |
| **Pass 3: Global Integer Rounding** | Largest-remainder sort ($N \log N$) | Single thread | 0.02 s | +0.1% |
| **Total Allocation Preprocessing** | Offline Exact Allocation | Multi-threaded | **1.94 s** | **+10.2%** |

> [!NOTE]
> The offline batch allocation adds less than **2.0 seconds** of preprocessing to index construction. In production batch vector pipelines (e.g., daily index rebuilds, embedding catalog deployments), a 10% construction time increase is fully acceptable in exchange for a guaranteed exact memory budget and zero query-time overhead.

---

## 4. Phase 5.3 Experimental Protocol

For all controlled experiments in Phase 5.3:
1. **Primary Scientific Benchmarks:** Evaluated using **Paradigm 1 (Offline Exact Batch Allocation)** to guarantee mathematical parity across all approaches (Uniform, Linear, Exponential, Quantile, and Randomized controls share the exact same total budget $B_0$).
2. **Streaming Robustness Analysis:** Evaluated separately using **Paradigm 2 (Calibration-Based Streaming)** to measure the empirical budget drift and retrieval delta between batch and streaming execution.

---
*End of Streaming Allocation Semantics and Limitations.*
