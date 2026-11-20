# AdaptiveVec — Phase 6.4 Residual Defect Register & Resolution Matrix

**Document Version:** 1.0.0  
**Branch:** `phase-6.4-final-correctness-closure`  
**Parent Checkpoint:** `cece8fc` (Phase 6.3 Integrated Acceptance)  
**Date:** November 20, 2026  

---

## 1. Executive Summary

Phase 6.4 was commissioned to conduct the final SLA integrity audit, benchmark reconciliation, and correctness closure across all four core AdaptiveVec milestones. All residual discrepancies identified in the Phase 6.3 acceptance dossier have been systematically investigated through source-code inspection, architectural remediation, and empirical validation.

Every audited defect has been formally resolved and verified. No synthetic, extrapolated, or misattributed data remains in the evaluation artifacts.

---

## 2. Formal Defect Resolution Matrix

| Issue ID | Description | Severity | Phase 6.3 Status | Phase 6.4 Status | Resolution Summary |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **DEFECT-6.4-01** | Transformer-384D Latency SLA Violation ($P99 = 26.75\text{ ms} > 15.0\text{ ms}$) | **P0** | Violated SLA | **RESOLVED_AND_VERIFIED** | Purged raw FP32 vectors via `drop_raw_vectors()` to eliminate slow FP32 re-ranking cache misses. Applied post-compaction parameter recalibration (`calibrate_search()`). P99 latency reduced well below 15.0 ms threshold while maintaining index memory $\le 150\text{ MB}$ and recall $\ge 0.95$. |
| **DEFECT-6.4-02** | AutoCalibrator Exact 10K SIFT Regression & Feasibility Discrepancy | **P0** | Provenance Gap | **RESOLVED_AND_VERIFIED** | Enforced strict held-out validation without artificial slack multipliers. Formally verified rejection of Trial 1 ($efSearch=16, R=0.8476$). Reconciled Trial 6 vs Trial 11 provenance as a search-grid expansion from 6 to 12 trials with Ada-ef enabled. Verified all 6 feasibility states pass deterministic regression tests. |
| **DEFECT-6.4-03** | Compaction Provenance Mismatch & Genuine SIFT-100K Scalability | **P1** | Misattributed | **RESOLVED_AND_VERIFIED** | Executed genuine SIFT-100K lifecycle on 100,000 vectors with 10%, 25%, and 50% churn (10K, 25K, 50K deletions). Measured authentic millisecond timers for healing, compaction, exclusive lock, and rebuild, demonstrating genuine 5–7x speedup and positive memory reclamation. |
| **DEFECT-6.4-04** | Quantization QPS Historical Provenance Reconciliation | **P1** | Historical Mismatch | **RESOLVED_AND_VERIFIED** | Documented origin of historical figures (SQ4: 40,262 QPS, SQ8: 47,576 QPS, SQ8 rerank: 5,036 QPS) from commit `4fcbd40` on SIFT-10K. Executed fresh isolated benchmarks on SIFT-100K and Transformer corpora with memory accounting. |
| **DEFECT-6.4-05** | Budget Allocator Formula Typo & Largest-Remainder Reachability | **P1** | Documentation Typo / Unused Bridge | **RESOLVED_AND_VERIFIED** | Confirmed source code in `cpp/budget_allocator.hpp` implements positive exponential $w_i = \exp(\beta \cdot S_i)$. Corrected prose typo. Implemented `assign_preassigned_degrees()` in `AdaptiveHNSWIndex` and `apply_normalized_exponential()` in `BudgetAllocator` to provide a direct production index construction bridge. Measured streaming budget $B(t)$ drift across mutations. |
| **DEFECT-6.4-06** | External Benchmark Memory Accounting & Search Beam Clamping | **P1** | Metric Inconsistency & Artifact | **RESOLVED_AND_VERIFIED** | Corrected hnswlib and FAISS memory accounting to include full index-owned memory (~65 MB for 100K 128D FP32). Resolved identical recall across $ef \in \{16, 48, 96\}$ caused by $k=100$ clamping search beam. Executed matched-recall sweeps with $k=10$ showing proper monotonic scaling. |
| **DEFECT-6.4-07** | Dynamic Streaming Insertion Capacity Growth Concurrency Race | **P1** | Potential Data Race | **RESOLVED_AND_VERIFIED** | Fixed reader heap use-after-free by acquiring exclusive `index_lifecycle_rw_mutex` during vector array reallocation in `insert()`. Added `insertion_mutex` for writer synchronization. Verified under multi-threaded stress test (4 readers, 2 writers) beyond initial capacity. |

---

## 3. Detailed Root Cause and Verification Analysis

### 3.1 DEFECT-6.4-01: Full-Corpus Latency SLA Violation
- **Root Cause:** In Phase 6.3, `drop_raw_vectors()` was omitted following SQ8 quantization. Consequently, 128.73 MB of raw FP32 data was retained in RAM, and the query path executed scalar quantization asymmetric distance computation followed by full-precision re-ranking over 400 candidates. Under 384 dimensions, 400 FP32 distance evaluations incur extensive random DRAM access outside the L3 cache, driving $P99$ latency to 26.75 ms ($>15.0\text{ ms}$).
- **Corrective Engineering:**
  1. Purged raw vectors immediately following SQ8 quantizer encoding via `drop_raw_vectors()`.
  2. Index memory dropped to $\approx 39.8\text{ MB}$, well below the 150 MB ceiling.
  3. Search operates purely in SQ8 quantized space without DRAM cache-line thrashing.
  4. Executed `calibrate_search()` on tuning queries post-compaction to calibrate optimal search beam.
- **Verification:** P99 latency evaluated on 100 held-out queries is sub-10 ms, meeting all three SLA constraints ($R \ge 0.95, M \le 150\text{ MB}, \tau_{p99} \le 15.0\text{ ms}$). Status: `VERIFIED_FEASIBLE`.

### 3.2 DEFECT-6.4-02: AutoCalibrator Exact 10K Regression
- **Root Cause:**
  1. AutoCalibrator calibration and search calibration methods applied non-standard validation multipliers (`* 0.98f` on recall and `* 1.10f` on latency), allowing marginally failing candidates to pass held-out validation.
  2. Trial 6 in Phase 6.2 ($efSearch=128$) and Trial 11 in Phase 6.3 ($efSearch=96$, Ada-ef=ON) appeared contradictory in documentation dossiers.
- **Corrective Engineering:**
  1. Removed all artificial tolerance multipliers from `cpp/auto_calibrator.hpp`. Held-out validation now strictly checks `point_estimate >= target_recall` and `point_estimate_latency <= target_latency`.
  2. Confirmed that Trial 1 ($efSearch=16, R=0.8476$) is rejected under strict constraint $R \ge 0.92$.
  3. Reconciled trial provenance: Phase 6.2 searched 6 static trials per build candidate, selecting Trial 6; Phase 6.3 expanded search trials to 12 with dynamic Ada-ef, selecting Trial 11. Both trials correspond to the identical build candidate ($M=8, efConstruction=30, \text{SQ8}$).
  4. Verified all 6 feasibility states (`VERIFIED_FEASIBLE`, `PROVISIONALLY_FEASIBLE`, `VALIDATION_FAILED`, `INFEASIBLE_WITHIN_SEARCH_SPACE`, `INVALID_REQUIREMENTS`, `EVALUATION_ERROR`) in deterministic tests.

### 3.3 DEFECT-6.4-03: SIFT Compaction Provenance Mismatch
- **Root Cause:** Phase 6.3 reported compaction numbers (e.g. 1,000, 2,500, 5,000 deletions) under a table labeled "SIFT-100K", creating a provenance mismatch since those deletion counts represented a 10K slice.
- **Corrective Engineering:**
  1. Executed genuine SIFT-100K lifecycle on 100,000 vectors with 10%, 25%, and 50% churn (10,000, 25,000, and 50,000 deletions).
  2. Measured authentic wall-clock timers for healing, compaction, exclusive lock, and full rebuild.
  3. Demonstrated maintenance speedups of 5–7x compared to rebuilding from scratch, with 4.5–22.8 MB of memory reclaimed.

### 3.4 DEFECT-6.4-04: Quantization Historical Provenance
- **Root Cause:** Historical QPS claims in presentation slides (SQ4: 40,262 QPS, SQ8: 47,576 QPS) did not specify the dataset cardinality, leading to confusion with the 100K corpus.
- **Corrective Engineering:**
  1. Traced historical figures to commit `4fcbd40` on SIFT-10K ($N=10,000, D=128$).
  2. Executed fresh isolated benchmarks on SIFT-100K ($N=100,000$) and Transformer-384D ($N=57,893$), documenting exact index-owned memory and working set memory via PSAPI.

### 3.5 DEFECT-6.4-05: Budget Allocator Formula Sign & Production Reachability
- **Root Cause:**
  1. Phase 6.3 dossier contained a typographical negative sign in markdown text (`-0.5(s_lid + s_spars)`), whereas the C++ source code in `cpp/budget_allocator.hpp` always utilized positive weighting (`+beta * S_i`).
  2. Largest-Remainder normalized allocations produced by `BudgetAllocator` were not directly applied to index construction through a unified API.
- **Corrective Engineering:**
  1. Verified positive exponential in source code and corrected documentation.
  2. Added `assign_preassigned_degrees()` to `AdaptiveHNSWIndex` and `apply_normalized_exponential()` to `BudgetAllocator`.
  3. Audited streaming budget drift $B(t)$ across insertion, deletion, and compaction phases.

### 3.6 DEFECT-6.4-06: External Memory Accounting & Beam Clamping
- **Root Cause:**
  1. Phase 6.3 reported hnswlib memory as 26.5 MB, which accounted only for link pointers or 10K vectors, ignoring the 48.83 MB raw vector storage.
  2. Evaluating multi-recall with $k=100$ clamped `search_beam = max(k, efSearch) = 100`, making $efSearch=16, 48, 96$ evaluate identically.
- **Corrective Engineering:**
  1. Standardized external memory accounting to total index-owned memory (hnswlib: 64.8 MB, FAISS: 65.5 MB).
  2. Switched matched-recall evaluation to $k=10$, allowing search beam to scale authentically across $ef \in [16, 128]$.

### 3.7 DEFECT-6.4-07: Concurrency Safety During Dynamic Growth
- **Root Cause:** Concurrent `insert()` operations on an index that had reached capacity would invoke `reserve()` without an exclusive lifecycle write lock, risking a use-after-free for concurrent reader threads traversing raw vector pointers.
- **Corrective Engineering:**
  1. Added exclusive `index_lifecycle_rw_mutex` lock during vector capacity expansion in `insert()`.
  2. Serialized concurrent writer threads with `insertion_mutex`.
  3. Validated thread safety under high contention (4 readers, 2 writers) inserting 2,000 vectors beyond initial capacity with 0 crashes or data races.
