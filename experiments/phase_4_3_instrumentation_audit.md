# AdaptiveVec — Phase 4.3: Mandatory Instrumentation and Counter Audit Report

**Author**: Senior Systems Research Engineer  
**Date**: November 20, 2026  
**Commit Baseline**: `2962679`  
**Active Branch**: `phase-4.3-uncertainty-aware-search`  
**Dataset Audited**: SIFT-100K (128D, Euclidean), Transformer-384D (384D, Unit Cosine)  
**Compiler & Environment**: MSYS2 UCRT64 GCC 14.2.0, AVX2, FMA, OpenMP, AMD Ryzen 5 5600H

---

## 1. Executive Summary: The $6 \times efSearch$ Synthetic Placeholder Anomaly

### 1.1 Discovery of Synthetic Placeholder Formula
During the Stage 1 audit of distance evaluation metrics, forensic examination of `cpp/phase4_2_matched_recall_benchmarks.cpp` and `cpp/phase4_1_workload_comparisons.cpp` revealed that fixed-budget distance evaluation counts were **never actually measured from hardware or traversal state**.

In `cpp/phase4_2_matched_recall_benchmarks.cpp` (lines 214 & 224):
```cpp
if (m.is_oracle) {
    int o_ef = oracle_budgets[q];
    res = test_idx->search(q_ptr, 10, o_ef);
    d_comps = o_ef * 6; // <-- Hardcoded synthetic formula
} else {
    res = test_idx->search(q_ptr, 10, m.ef_fixed);
    d_comps = m.ef_fixed * 6; // <-- Hardcoded synthetic formula
}
```
Similarly, in `cpp/phase4_1_workload_comparisons.cpp` (line 305):
```cpp
double mean_dist_comps = (method == "Fixed efSearch=32") ? 280.0 : 420.0;
```
Because `AdaptiveHNSWIndex::search()` historically returned only `std::vector<Candidate>` without a per-query instrumentation record, earlier benchmark runners substituted synthetic linear approximations ($6 \times efSearch$).

### 1.2 Multi-Threaded Atomic Contention and Double-Counting
Inspection of `cpp/adaptive_hnsw.hpp` also revealed two critical flaws in the index-level distance counters:
1. **Atomic Cache Line Bouncing**: `total_dist_computations` was declared as a global `mutable std::atomic<uint64_t>`. Calling `total_dist_computations++` inside the innermost distance evaluation loops across 12 OpenMP worker threads induced severe false sharing and CPU cache-line invalidation, bottlenecking parallel throughput and causing latency jitter.
2. **Double-Counting Defect**: In `get_node_distance_sq()`, `total_dist_computations++` was executed once, and if the vector was unquantized or raw FP32, it forwarded to `get_distance_sq()`, which incremented `total_dist_computations++` a second time.

---

## 2. Remediation: Zero-Contention Per-Query Instrumentation

To establish complete scientific accuracy and eliminate thread contention:
1. **Implemented `QuerySearchStats`**:
   A zero-allocation, thread-local structure passed directly through the traversal stack:
   ```cpp
   struct QuerySearchStats {
       uint32_t upper_layer_dist_evals = 0; // Distances computed during greedy descent (L_max down to 1)
       uint32_t layer0_dist_evals = 0;      // Distances computed in layer-0 beam search
       uint32_t rerank_dist_evals = 0;      // Distances computed during FP32 two-stage re-ranking
       uint32_t visited_nodes = 0;          // Unique nodes marked in VisitedTagPool
       uint32_t hops = 0;                   // Number of candidate expansions popped from queue
       uint32_t early_exit_stagnation_count = 0; // Early exit triggers
       
       inline uint32_t total_dist_evals() const {
           return upper_layer_dist_evals + layer0_dist_evals + rerank_dist_evals;
       }
   };
   ```
2. **Integrated Across All Traversal Paths**:
   - `search_layer(...)` instruments every node distance call without atomic locks.
   - `search(...)` cleanly separates upper-layer descent from layer-0 beam search and re-ranking.
   - `ResumableSearchEngine::search_resumable(...)` populates `QuerySearchStats` throughout both Phase 1 ($ef_{init}$) and Phase 2 ($ef_{expanded}$) exploration.

---

## 3. Empirical Measurement: Actual vs. Synthetic Distance Counts

Using the new exact instrumentation (`scratch/verify_actual_distance_counts.cpp`), we benchmarked 200 queries across the full $efSearch$ spectrum on SIFT-100K ($M=16, efC=64$, Uniform SQ8):

| $efSearch$ | Old Synthetic ($6 \times ef$) | Actual Mean Upper-Layer | Actual Mean Layer-0 | Actual Total Distance Evals | Discrepancy Ratio (Actual / Synthetic) |
|---:|---:|---:|---:|---:|---:|
| **16** | 96 | 56.9 | 340.2 | **397.2** | **4.14x** |
| **24** | 144 | 56.9 | 447.2 | **504.1** | **3.50x** |
| **32** | 192 | 56.9 | 544.7 | **601.6** | **3.13x** |
| **48** | 288 | 56.9 | 708.1 | **765.0** | **2.66x** |
| **64** | 384 | 56.9 | 852.2 | **909.1** | **2.37x** |
| **96** | 576 | 56.9 | 1074.5 | **1,131.4** | **1.96x** |
| **128** | 768 | 56.9 | 1264.6 | **1,321.5** | **1.72x** |
| **192** | 1152 | 56.9 | 1563.4 | **1,620.4** | **1.41x** |
| **256** | 1536 | 56.9 | 1802.5 | **1,859.5** | **1.21x** |

### 3.1 Forensic Analysis of the Discrepancy
- **Why Synthetic Underestimated Low $efSearch$**:
  Upper-layer greedy routing incurs an average baseline of **56.9 distance evaluations** regardless of $efSearch$ on layer 0. At $efSearch=16$, the upper layers account for $14.3\%$ of all distance calculations.
- **Why Layer-0 Work Scales Non-Linearly**:
  In HNSW with $M=16$, each node in layer 0 has an out-degree up to $2M = 32$. Exploring a beam of $efSearch = 64$ evaluates approximately 35 hops with $\sim 24$ unvisited neighbors per hop, resulting in $852$ evaluations at layer 0 alone.
- **Conclusion**: The synthetic formula severely masked the actual computational cost of graph traversal, underreporting distance calculations by up to $4.14\times$ at low budgets and $2.37\times$ at production budgets.

---

## 4. Invalidation of Prior Experimental Metrics

As mandated by Stage 1 requirements, the following previously published experimental results are formally declared **invalidated**:

1. **`experiments/phase_4_1_workload_comparisons.json`**:
   - `mean_dist_comps = 280.0` and `420.0`: Invalidation confirmed (hardcoded constant placeholders).
2. **`experiments/phase_4_2_matched_recall_results.json`**:
   - Fixed $efSearch$ distance computations (`192.0`, `288.0`, `384.0`): Invalidation confirmed (synthetic $6 \times efSearch$ calculation).
   - Oracle search budget distance computations (`152.1`, `201.1`, `208.9`): Invalidation confirmed (synthetic $6 \times efSearch$ calculation).

All subsequent Phase 4.3 benchmarks will report strictly measured `QuerySearchStats.total_dist_evals()`.
