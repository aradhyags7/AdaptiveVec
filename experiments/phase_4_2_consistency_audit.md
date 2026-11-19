# AdaptiveVec — Phase 4.2: Experimental Consistency and Numerical Audit Report

**Author**: Senior Systems Research Engineer  
**Date**: November 20, 2026  
**Commit Baseline**: `66a0190`  
**Active Branch**: `phase-4.2-lightweight-adaptation`  
**Datasets Investigated**: Transformer-384D (384D, Unit Cosine), SIFT-100K (128D, Euclidean)  
**Compiler & Toolchain**: MSYS2 UCRT64 GCC 14.2.0 (`-O3 -mavx2 -mfma -fopenmp -std=c++17`)  
**Hardware Environment**: AMD Ryzen 5 5600H (6 cores / 12 threads @ 3.30 GHz base / 4.25 GHz boost), 16.0 GB DDR4, Windows 11

---

## 1. Executive Summary: The Transformer-384D Recall Discrepancy

### 1.1 The Anomaly
Between Phase 3 and Phase 4.1, reported Recall@10 on Transformer-384D appeared to decline substantially:
- **Phase 3 Reported (Calibrated SQ8)**: **92.90% [91.25%, 94.47%]** at $efSearch = 64$.
- **Phase 4.1 Reported (Workload Benchmark)**: **70.25%** (Static $ef=64$) and **72.49%** (Adaptive Controller).

This prompted a mandatory Stage 1 consistency audit before proceeding with lightweight controller development.

### 1.2 Root Cause Identification
Through forensic reproduction (`scratch/audit_transformer_recall.cpp`), the discrepancy was isolated and confirmed:

1. **Query Perturbation without Ground-Truth Recalculation**:
   In `cpp/phase4_1_workload_comparisons.cpp` (lines 173–193), the 8-phase non-stationary benchmark introduced synthetic query perturbations to simulate spatial drift and difficulty changes:
   ```cpp
   // Phase 2: Abrupt difficulty change
   for (size_t d = 0; d < dim; ++d) q_vec[d] += 0.30f * std::sin((float)(d + i));
   // Phase 3: Gradual spatial drift
   float drift_mag = (i - 600) * 0.001f;
   for (size_t d = 0; d < dim; ++d) q_vec[d] += drift_mag;
   // Phase 4: Recurring oscillation
   float osc = (i % 50 < 25) ? 0.35f : 0.0f;
   for (size_t d = 0; d < dim; ++d) q_vec[d] += osc;
   ```
   However, the evaluation code assigned ground truth using the unperturbed source index:
   ```cpp
   std::vector<tableint> topk_gt(gt[base_idx_q].begin(), gt[base_idx_q].begin() + 10);
   schedule.push_back({q_vec, topk_gt, phase});
   ```
   In a 384-dimensional unit hypersphere, adding $+0.30 \sin(d)$ rotates the vector substantially away from its original orientation. Because the true nearest neighbors of the perturbed vector $q'$ are completely different from those of $q$, evaluating the nearest neighbors of $q'$ against the unperturbed ground truth of $q$ artificially compressed the measured recall to $\sim 68\% - 72\%$.

2. **Index Configuration Differences**:
   - Phase 3 Calibrated Index: Constructed with $M = 24$, $efConstruction = 80$.
   - Phase 4.1 Comparison Index: Constructed with base $M = 16$, $efConstruction = 64$.

3. **Empirical Verification of Numerical Integrity**:
   When evaluated on clean, unperturbed queries against exact brute-force ground truth:
   - **FP32 Clean Index ($M=16, efC=64$)**:
     - $efSearch = 32$: **84.45%**
     - $efSearch = 48$: **87.75%**
     - $efSearch = 64$: **90.20%**
     - $efSearch = 96$: **92.13%**
   - **Uniform SQ8 Clean Index ($M=16, efC=64$)**:
     - $efSearch = 32$: **84.35%**
     - $efSearch = 48$: **87.55%**
     - $efSearch = 64$: **89.57%**
     - $efSearch = 96$: **91.95%**
   - **Phase 3 Calibrated SQ8 Index ($M=24, efC=80$)**:
     - $efSearch = 64$: **92.90%**

### 1.3 Conclusion
There is **no numerical correctness regression** in AdaptiveVec's search engine, SIMD distance kernels, or SQ8 quantization routines. The measured drop in Phase 4.1 was an artifact of evaluating perturbed vectors against stale unperturbed ground truth.

---

## 2. Dataset and Environment Provenance

### 2.1 Cryptographic Dataset Fingerprints
To guarantee experimental reproducibility and data integrity across all phases:

| Dataset | Filename | Record Count | Dimensions | Format | SHA-256 Checksum |
|---|---|---:|---:|:---:|---|
| **Transformer-384D Base** | `transformer_384d_base.fvecs` | 57,893 | 384 | FP32 fvecs | `c732c1cc4007bf2d8cd30704fe39767f86e71fb93c1f59a86ffa7089f86116b7` |
| **Transformer-384D Query** | `transformer_384d_query.fvecs` | 1,000 | 384 | FP32 fvecs | `9d908e5660658c5578e9af66a5b038548d88c092f207607782e1829f243d4a22` |
| **Transformer-384D Groundtruth** | `transformer_384d_groundtruth.ivecs` | 1,000 | 100 | Int32 ivecs | `3aa018db8c32058d9a3b588dad273757b3532bb3afd7084410ab5f312cec4aa5` |
| **SIFT-100K Base** | `sift_subset_100k.fvecs` | 100,000 | 128 | FP32 fvecs | `7145bedfe09fe8d1afcc9ad743eb6ba6218dc766b77c4b0bbe82d22b78e265ba` |
| **SIFT-100K Query** | `sift_query.fvecs` | 10,000 | 128 | FP32 fvecs | `f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc` |
| **SIFT-100K Groundtruth** | `sift_subset_groundtruth.ivecs` | 10,000 | 100 | Int32 ivecs | `1da6f1693773497fa42c73ba09615e6172b3c86490acc6c209e1714eeca0c339` |

### 2.2 Metric and Normalization Semantics
- **Transformer-384D**: All vectors are $L_2$-normalized unit vectors ($\|x\|_2 = 1.0 \pm 10^{-6}$). The distance metric is `SpaceType::Cosine`, defined as:
  $$D_{cos}(u, v) = 1.0 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
  Because vectors are pre-normalized, $D_{cos}(u, v) = 1.0 - u \cdot v = \frac{1}{2} \|u - v\|_2^2$. Monotonicity holds exactly with squared Euclidean distance.
- **SIFT-100K**: Standard 128-dimensional integer SIFT descriptors cast to float32. The distance metric is squared Euclidean distance (`SpaceType::L2`).

### 2.3 Recall@10 Evaluation Protocol
Recall@k is computed strictly as:
$$\text{Recall}@k(q) = \frac{|\mathcal{R}_k(q) \cap \mathcal{G}_k(q)|}{k}$$
where $\mathcal{R}_k(q)$ is the set of $k$ retrieved IDs and $\mathcal{G}_k(q)$ is the set of $k$ true nearest neighbors from the exact brute-force ground truth. Ties are resolved deterministically by ID order.

---

## 3. Telemetry and Accounting Reconciliation

### 3.1 Telemetry Overhead (< 3% Engineering Target)
In Phase 4.1, telemetry overhead was measured across 10 repeated isolated trials:
- **Baseline QPS**: $26,202.8 \pm 926.9$ [95% CI: 25,018.7, 27,260.4]
- **Telemetry QPS**: $26,388.2 \pm 901.8$ [95% CI: 24,889.0, 27,762.3]
- **Relative Overhead**: Median $-0.00\%$, Mean $-0.64\% \pm 4.51\%$ [95% CI: $-3.42\%, +2.01\%$]
Because lockless circular buffer logging is non-blocking and operates on pre-allocated cache-line-aligned arrays, logging latency is negligible compared to graph distance calculations.

### 3.2 Reference Monitoring Overhead Root Cause
In Phase 4.1, the runtime controller executed non-invasive reference query evaluations every $N_{eval} = 50$ production queries using a batch of $M_{ref} = 100$ reference queries.
- On SIFT-100K (128D), evaluating 100 queries against the index every 50 queries represents:
  $$\text{Extra Query Work} = \frac{100 \text{ ref queries}}{50 \text{ prod queries}} = 200\% \text{ additional search work}$$
- This heavy sampling frequency caused SIFT throughput to drop from $\sim 8,400$ QPS to $\sim 3,980$ QPS ($\sim 52\%$ throughput loss), and Transformer throughput to drop from $\sim 3,100$ QPS to $\sim 579$ QPS ($\sim 81\%$ throughput loss).
- **Phase 4.2 Remedy**: Decouple monitoring into incremental amortized sampling (1 query per $K$ production requests) or low-frequency periodic evaluation ($N_{eval} \ge 1,000$), strictly bounding CPU overhead to $< 5\%$.

### 3.3 Latency, Throughput, and Violation Accounting
- **p50, p95, p99 Latency**: Computed across all live production queries using high-resolution timers (`std::chrono::steady_clock`).
- **Throughput (QPS)**: Computed end-to-end as $\text{Total Live Queries} / \text{Total Elapsed Time}$, strictly including all controller logic, telemetry writes, and reference evaluations.
- **Constraint Violations**: Formally computed as:
  $$V_{total} = \sum_{q} \mathbb{I}(\text{Recall}(q) < R_{target}) + \mathbb{I}(\text{Latency}(q) > L_{target})$$
  Ensures that individual query failures are not hidden by aggregate averages.

---

## 4. Phase 4.2 Non-Stationary Workload Definition

To avoid the artificial ground-truth misalignment observed in Phase 4.1, Phase 4.2 uses **authentic partitioned query workloads**:
1. **Cluster Shift**: Queries are partitioned by intrinsic cluster membership (e.g. Scifact vs Arguana vs NFCorpus for Transformer; k-means cluster partitions for SIFT).
2. **Density Variance**: Queries ordered by local neighbor density (dense cores vs sparse outliers).
3. **True Drift**: Any continuous spatial perturbation is accompanied by exact, verified brute-force ground-truth recalculation for the perturbed coordinates.

This guarantees that all measured recall figures reflect genuine retrieval fidelity.
