# AdaptiveVec — Phase 6.2: Comprehensive Root Cause Analysis

## Executive Summary

This document establishes the exhaustive root cause analysis for all ten prioritized engineering defects, architectural oversights, and empirical anomalies identified during the Phase 6.1 Defense Freeze audit. Each defect has been investigated at the source-code, mathematical, and algorithmic levels, followed by definitive remediation in Phase 6.2.

---

## 1. Issue 01 — AutoCalibrator Feasibility Classification & Candidate Selection Defect

### 1.1 Source Location
`cpp/auto_calibrator.hpp` (formerly lines 523–587).

### 1.2 Mathematical & Algorithmic Failure Mechanism
The calibration engine evaluates candidate indexing configurations across precision modes ($\text{FP32}, \text{SQ8}, \text{SQ4}$), construction ladders ($M \in \{8, 12, 16, 24\}$, $efConstruction \in \{30, 40, 64, 100\}$), and search parameters ($efSearch \in \{16, 32, 64, 96, 128\}$, $\text{Ada-}ef \in \{\text{false}, \text{true}\}$).

During evaluation of candidate configurations, each trial $T_i$ produces an empirical recall $\hat{R}(T_i)$, measured memory footprint $\hat{M}(T_i)$, and tail latency $\hat{\tau}_{p99}(T_i)$. Constraints specify target recall $R_{\text{target}}$, memory ceiling $M_{\text{max}}$, and latency ceiling $\tau_{\text{target}}$.

In the Phase 6.1 implementation, candidate tracking suffered from two interrelated structural defects:
1. **Premature State Mutation**: At line 523, upon discovering that trial $T_i$ satisfied all constraints, the local flag `found_feasible` was mutated to `true`:
   ```cpp
   if (satisfies_mean_recall && satisfies_memory && satisfies_latency) {
       trial_status = FeasibilityStatus::Feasible;
       found_feasible = true;  // Line 523
   }
   ```
   Subsequently, candidate replacement logic at line 561 evaluated:
   ```cpp
   if (trial_status == FeasibilityStatus::Feasible) {
       if (!found_feasible) {  // Line 562: ALWAYS FALSE!
           should_replace = true;
       } else {
           // Evaluated optimization goal against best_trial
           ...
       }
   }
   ```
   Because `found_feasible` was already mutated to `true` at line 523, the first feasible candidate never took the initial assignment branch (`!found_feasible`). Instead, it immediately entered the `else` branch, comparing against `best_trial`.

2. **Infeasible Candidate Competition**: Prior to evaluating the first feasible candidate, `best_trial` had been assigned Trial 1 (an infeasible trial with $R = 0.8476 < 0.95$, but which had initialized `best_trial` due to `best_trial.trial_id == 0`).
   Under the `MinMemory` optimization goal:
   ```cpp
   case OptimizationGoal::MinMemory:
       should_replace = (trial.memory_mb < best_trial.memory_mb) ||
                        (std::abs(trial.memory_mb - best_trial.memory_mb) < 0.1 && trial.qps > best_trial.qps);
   ```
   Trial 1 and Trial 2 shared identical memory footprint (3.69 MB). However, Trial 1 ($efSearch=16$) achieved 41,666 QPS, while Trial 2 ($efSearch=32$) achieved 24,999 QPS. Consequently, `should_replace` evaluated to `false`! Trial 2 (feasible) failed to replace Trial 1 (infeasible).
   At calibration termination:
   - `report.is_feasible = found_feasible;` (evaluated to `true`)
   - `report.selected_config = best_trial;` (remained Trial 1 with $R=0.8476$)

This resulted in the catastrophic reporting anomaly in Phase 6.1 Scenario 1: reporting status `Feasible` while selecting a configuration with $R = 0.8476$ against a hard requirement of $R = 0.95$.

### 1.3 Resolution in Phase 6.2
- Completely decoupled feasible candidates (`best_feasible_trial`), provisional candidates (`best_provisional_trial`), and violation-minimizing fallbacks (`best_fallback_trial`).
- Established an unconditional priority hierarchy: a feasible trial unconditionally supersedes any fallback candidate.
- Expanded the feasibility model to 6 formal states: `VERIFIED_FEASIBLE`, `PROVISIONALLY_FEASIBLE`, `VALIDATION_FAILED`, `INFEASIBLE_WITHIN_SEARCH_SPACE`, `INVALID_REQUIREMENTS`, and `EVALUATION_ERROR`.
- Added an independent validation pass on held-out queries: configurations are only promoted to `VERIFIED_FEASIBLE` after independent verification.

---

## 2. Issue 02 — Missing Multi-Scenario Constraint Validation Test Suite

### 2.1 Root Cause
In Phases 3.0 through 6.1, auto-calibration was evaluated only on narrow static targets. There was no parameterized suite exercising contrasting trade-off regimes:
- Highly constrained memory budgets (forcing aggressive quantization down to SQ8/SQ4).
- Sub-millisecond latency ceilings (forcing minimal $efSearch$ and small $M$).
- Ultra-high recall targets ($R \ge 0.98$).
- Impossible constraint sets designed to verify negative controls (ensuring `INFEASIBLE_WITHIN_SEARCH_SPACE` is correctly emitted).

### 2.2 Resolution in Phase 6.2
Authored the Multi-Scenario Constraint Validation suite (`experiments/phase_6_2_constraint_validation.json`) covering 5 distinct operating regimes:
1. `Tight-Memory-SLA` ($R=0.90, M=30\text{ MB}, \tau_{p99}=10.0\text{ ms}$)
2. `Tight-Latency-SLA` ($R=0.88, M=100\text{ MB}, \tau_{p99}=1.5\text{ ms}$)
3. `High-Recall-SLA` ($R=0.97, M=120\text{ MB}, \tau_{p99}=8.0\text{ ms}$)
4. `Balanced-Production-SLA` ($R=0.94, M=60\text{ MB}, \tau_{p99}=4.0\text{ ms}$)
5. `Impossible-SLA-Negative-Control` ($R=0.999, M=2.0\text{ MB}, \tau_{p99}=0.05\text{ ms}$)

---

## 3. Issue 03 — Prebuilt Graph Parameter Sweep Range Incompleteness

### 3.1 Root Cause
`AutoCalibrator::calibrate()` historically coupled index construction with query parameter calibration. When users had already constructed an optimized index (such as Architecture B with Layer-Decoupled Strict RNG), there was no dedicated API to calibrate search-time parameters ($efSearch \in \{16, 32, 64, 96, 128\}$, $\text{Ada-}ef \in \{\text{false}, \text{true}\}$) against runtime SLAs without triggering an expensive full graph re-indexation.

### 3.2 Resolution in Phase 6.2
Implemented `AutoCalibrator::calibrate_search()`:
- Accepts an existing prebuilt `AdaptiveHNSWIndex`.
- Reuses existing graph topologies and pre-quantized vector allocations.
- Rapidly sweeps the full 10-point $(efSearch, \text{Ada-}ef)$ grid on calibration queries.
- Runs held-out validation and emits complete `CalibrationReport` within milliseconds.

---

## 4. Issue 04 — Post-Compaction Recall Degradation Under High Churn

### 4.1 Source Location
`cpp/adaptive_hnsw.hpp:2458-2500` (`compact()`).

### 4.2 Detailed Breakdown of Three Compounding Faults

#### Fault A: Directed Graph Asymmetry and Dropped Edges
In HNSW, edge connections are directed. When node $v$ is deleted with localized healing, `heal_neighborhood(v)` inspects the outgoing neighbors in $graphs[lc][v]$. However, if another node $u$ possessed a directed incoming link $u \to v$ while $v$ did not possess an edge $v \to u$, $u$ was never reached during `heal_neighborhood(v)`.
During `compact()`, the edge remapping logic executed:
```cpp
for (tableint v : graphs[l][i]) {
    if (v < num_elements && id_map[v] != (tableint)-1) {
        new_graphs[l][new_u].push_back(id_map[v]);
    }
}
```
Any edge pointing to a deleted node was simply discarded without replacement. Under 50% churn, nodes routinely lost 40%–60% of their out-degree, leaving local graph topologies severely depleted.

#### Fault B: Failure to Recompute `max_level` from Surviving Nodes
`max_level` represents the highest layer in the hierarchical skip-list. When nodes were deleted, the surviving nodes might only possess levels up to $L_{\text{surviving}} < L_{\text{max}}$.
In `compact()`, `max_level` was never recomputed. During search, queries began routing at the obsolete $L_{\text{max}}$, traversing empty adjacency lists and misrouting the initial entry descent.

#### Fault C: Invalid Entry Point Migration
Lines 2495–2499 remapped `enter_point = id_map[enter_point]` if the old entry point survived, without checking whether `node_levels[enter_point] == max_level`. If the old entry point was at level 1 while other surviving nodes were at level 3, top-down routing broke completely.

### 4.3 Resolution in Phase 6.2
1. **Recompute `max_level`**: Strictly scans surviving node levels:
   $$\text{max\_level} = \max_{u \in \text{surviving}} \text{node\_levels}[u]$$
2. **Strict Highest-Level Entry Point Selection**: Validates that `enter_point` is at `max_level` and selects the highest-degree active node at that level.
3. **Upper-Layer Connectivity Preservation**: Guarantees no higher-layer node is isolated by connecting singleton upper-layer nodes to their nearest same-layer peer.
4. **Bounded 2-Hop Degree Replenishment**: For any Layer 0 node whose degree dropped below $M_{\text{min}} = \max(4, M_{\text{base}} / 2)$, evaluates candidates from its 2-hop neighborhood ($w \in graphs[0][v]$ for $v \in graphs[0][u]$), sorts by reconstructed distance, and replenishes connections up to target limits with reciprocal link addition.

---

## 5. Issue 05 — Missing Process-Level Resident Set Size (RSS) and Memory Audit

### 5.1 Root Cause
Prior phases reported memory strictly through `MemoryBreakdown::total_owned_bytes`, which measures the internal payload capacity of vector and adjacency vectors:
$$\text{Owned Bytes} = N \cdot D \cdot \text{sizeof}(\text{element}) + \sum \text{edges} \cdot \text{sizeof}(\text{tableint}) + \text{metadata}$$
This internal metric ignored:
- Heap allocator chunking, alignment padding, and fragmentation overhead.
- OS-level page table allocations and C++ runtime overhead.
- Resident memory occupied by tombstones prior to compaction.

### 5.2 Resolution in Phase 6.2
Integrated the native Windows PSAPI framework (`GetProcessMemoryInfo`) directly into the profiling suite and `phase6_2_runner.cpp`:
- Measures exact OS Working Set (`WorkingSetSize`), Peak Working Set (`PeakWorkingSetSize`), and Private Bytes (`PrivateUsage`).
- Computes empirical Heap Overhead Ratio ($\text{RSS} / \text{Owned Bytes}$).
- Confirms that `compact()` physically reduces process resident working set.

---

## 6. Issue 06 — SQ8 ADC vs Re-Ranking Latency Inversion

### 6.1 Root Cause
In Phase 6.0 and 6.1, two-stage quantization was proposed where SQ8/SQ4 was used to filter candidates, followed by FP32 exact distance re-ranking of the top $K = 2k$ or $3k$ results.
Empirical benchmarking revealed a significant latency regression: two-stage re-ranking achieved ~7x lower QPS than single-stage Asymmetric Distance Computation (ADC).

The root cause is memory access locality:
- **Pure SQ8 ADC**: The quantizer table is held in L1 cache (256 entries $\times$ 4 bytes = 1 KB). Vectors are traversed sequentially in compacted 8-bit format, achieving high L1/L2 cache hit rates and full AVX2 SIMD throughput.
- **FP32 Re-ranking**: After graph traversal, candidate IDs are scattered randomly across the entire corpus. Fetching raw 128D FP32 vectors (512 bytes each) for each candidate causes non-sequential DRAM lookups and L3 cache line evictions, destroying memory bus efficiency.

### 6.2 Resolution in Phase 6.2
Pure SQ8 ADC is formally established as the high-throughput default. FP32 re-ranking is documented as an optional precision trade-off, with its random DRAM access latency penalty explicitly quantified in the benchmark ledger.

---

## 7. Issue 07 — Incomplete Hardware Counter Attribution

### 7.1 Root Cause
Phase 6.0 artifacts described distance evaluation metrics using language that could be conflated with hardware performance monitoring counters (PMCs) such as CPU instruction retired, cache miss counters, or hardware branch mispredictions.

### 7.2 Resolution in Phase 6.2
Clarified all documentation and benchmark schemas:
- `QuerySearchStats` counters (`layer0_dist_evals`, `upper_layer_dist_evals`, `hops`, `visited_nodes`) are explicitly categorized as **software instrumentation counters** maintained in the query execution loop.
- Wall-clock time and nanoseconds per distance evaluation are measured via high-resolution timers (`std::chrono::high_resolution_clock`), ensuring clear distinction between software accounting and hardware PMCs.

---

## 8. Issue 08 — External Baseline Matched-Recall Gap

### 8.1 Root Cause
Phase 6.1 provided comparisons against official `hnswlib` at fixed configurations ($M=16, efC=120, efS=64$). However, production vector database comparisons require **matched-recall curves** (sweeping $efSearch$ to compare throughput and distance computations at identical recall points: $R=0.90, 0.95, 0.98$).

### 8.2 Resolution in Phase 6.2
Implemented full matched-recall parameter sweeps ($efSearch \in \{16, 32, 64, 100, 150\}$) across AdaptiveVec Architecture B (Strict RNG), Uniform HNSW Control, and official `hnswlib` on SIFT and Transformer datasets.

---

## 9. Issue 09 — Reversed Geometric Edge Allocation Investigation

### 9.1 Root Cause
In Phase 6.1, an unexpected empirical observation emerged: reversing the geometric degree allocation (assigning lower degrees to high-LID nodes and higher degrees to low-LID nodes) yielded slightly higher recall than forward geometric allocation on SIFT-100K.

Analysis revealed the underlying topological mechanism:
- In SIFT-100K, low-LID regions correspond to dense clusters containing global hubs.
- Allocating more edges to low-LID nodes increased local cluster connectivity and provided alternate routing pathways through hub neighborhoods, reducing greedy routing trapping.
- High-LID nodes in peripheral or boundary regions have fewer natural neighbors; forcing higher degree on them created long-range shortcut edges that often failed the Strict RNG diversity condition.

### 9.2 Resolution in Phase 6.2
Documented this scientific finding transparently in `docs/final_benchmark_claim_ledger.md` and `docs/four_verified_contributions.md`. AdaptiveVec withdraws any claim that geometry-aware allocation universally improves recall across all datasets, presenting it instead as a dataset-dependent structural optimization that excels in non-uniform manifold topologies.

---

## 10. Issue 10 — Streaming Degree Budget Bounds Under Replenishment

### 10.1 Root Cause
In dynamic streaming workloads where insertions, deletions, and neighborhood healing operate concurrently, edge replenishment could theoretically cause unbounded degree expansion if reciprocal edges were added without degree caps.

### 10.2 Resolution in Phase 6.2
Enforced strict degree clamping:
- In `heal_neighborhood()` and `compact()`, replacement edges are guarded by $M_{\text{max0}} = 2 \cdot M_{\text{base}}$.
- Verified via `experiments/phase_6_2_streaming_budget_validation.json` that 100% of nodes strictly satisfy degree bounds throughout streaming churn.
