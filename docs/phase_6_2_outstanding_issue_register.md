# AdaptiveVec — Phase 6.2 Comprehensive Outstanding Issue Register

**Document Version:** 1.0.0  
**Branch:** `phase-6.2-comprehensive-engineering-closure`  
**Parent Freeze Commit:** `7f2a862`  
**Date:** November 20, 2026  

---

## 1. Issue Classification Taxonomy

Issues are prioritized according to strict engineering and scientific severity:
- **P0 (Critical Correctness):** Implementation defects producing incorrect mathematical decisions, false feasibility classifications, memory corruption, or data-loss risks.
- **P1 (Major Performance / Reliability):** Substantial throughput regressions, excessive maintenance latency, post-compaction recall degradation, or architectural integration gaps.
- **P2 (Engineering Improvements):** Memory accounting fidelity, streaming budget stabilization, fresh-process isolation, or local repair enhancements.
- **P3 (Documentation / Scientific Integrity):** Ambiguous terminology, unverified claims, or baseline conflation.

---

## 2. Comprehensive Issue Register

### ISSUE-01: False Feasibility Classification in AutoCalibrator
- **Identifier:** `ISSUE-01`
- **Affected Milestone:** Milestone IV (Automatic Resource-Constrained Optimization)
- **Severity:** **P0 (Critical Correctness)**
- **Source Location:** `cpp/auto_calibrator.hpp:520-590`
- **Current Behavior:** In strict-memory deployment scenarios (e.g. SIFT-10K, $R_{\text{target}}=0.92, \text{RAM}\le 4.5\text{MB}$), `AutoCalibrator` selects a trial with $efSearch=16$ achieving only 0.8476 recall (95% CI: [0.828, 0.866]), but reports `status = Feasible` and `is_feasible = true`.
- **Expected Behavior:** Any candidate that fails target recall (or any other hard constraint) must strictly be classified as `INFEASIBLE_WITHIN_SEARCH_SPACE` (or `VALIDATION_FAILED`). A feasible candidate must ALWAYS replace an infeasible candidate, and low aggregate violation must never be interpreted as feasibility.
- **Reproduction Procedure:** Run `AutoCalibrator::calibrate` on SIFT-10K with $R_{\text{target}}=0.92$, $M_{\text{max}}=4.5\text{ MB}$, and `MinMemory` goal.
- **Confirmed Root Cause:**
  1. `best_trial` is initialized with Trial 1 ($efSearch=16$, recall 0.8476), which is recorded as the best violation minimizer because no feasible trial has been encountered.
  2. When Trial 2 ($efSearch=32$, recall 0.9224) passes all constraints, `found_feasible` is set to `true` at line 523 before line 562 checks `if (!found_feasible)`.
  3. Consequently, line 562 evaluates to `false` and falls into the `else` block comparing Trial 2 against Trial 1 using the `MinMemory` objective. Because both have identical memory ($3.69\text{ MB}$) and Trial 1 had higher QPS (41,666 vs 24,999 QPS), Trial 2 does not replace Trial 1!
  4. `best_trial` remains Trial 1, but `report.is_feasible` is set to `found_feasible` (`true`), producing a false feasibility report.
- **Proposed Corrective Action:**
  1. Separate feasible candidate tracking from violation fallback tracking.
  2. Implement strict 6-state feasibility statuses (`VERIFIED_FEASIBLE`, `PROVISIONALLY_FEASIBLE`, `VALIDATION_FAILED`, `INFEASIBLE_WITHIN_SEARCH_SPACE`, `INVALID_REQUIREMENTS`, `EVALUATION_ERROR`).
  3. Feasible candidates must never compete against infeasible fallbacks.
- **Regression Tests:** Add unit test `test_calibrator_strict_feasibility_enforcement()` in `cpp/tests/test_main.cpp`.
- **Acceptance Criteria:** Zero false feasibility classifications across all 10 mandatory evaluation scenarios.
- **Status:** **UNDER RESOLUTION (Stage 1)**

---

### ISSUE-02: Missing Formal Feasibility Status Classification
- **Identifier:** `ISSUE-02`
- **Affected Milestone:** Milestone IV (Automatic Resource-Constrained Optimization)
- **Severity:** **P0 (Correctness / Contract)**
- **Source Location:** `cpp/auto_calibrator.hpp:42-56`
- **Current Behavior:** Only 3 statuses exist: `Feasible`, `ProvisionallyFeasible`, `InfeasibleWithinSearchSpace`. Invalid requirements (e.g. negative memory or recall $> 1.0$) or held-out validation failures are lumped into infeasibility.
- **Expected Behavior:** Implement all 6 formal statuses required by the specification:
  - `VERIFIED_FEASIBLE`
  - `PROVISIONALLY_FEASIBLE`
  - `VALIDATION_FAILED`
  - `INFEASIBLE_WITHIN_SEARCH_SPACE`
  - `INVALID_REQUIREMENTS`
  - `EVALUATION_ERROR`
- **Confirmed Root Cause:** Enum definition lacked status granularity.
- **Proposed Corrective Action:** Expand `enum class FeasibilityStatus` and update string serializations across C++ and Python bindings.
- **Status:** **UNDER RESOLUTION (Stage 1)**

---

### ISSUE-03: Sub-Optimal Search Parameter Exploration in AutoCalibrator
- **Identifier:** `ISSUE-03`
- **Affected Milestone:** Milestone IV (Automatic Resource-Constrained Optimization)
- **Severity:** **P1 (Major Integration)**
- **Source Location:** `cpp/auto_calibrator.hpp:454-465`
- **Current Behavior:** When exploring $efSearch \in \{16, 32, 64, 96, 128\}$, if $efSearch=16$ fails recall, the search sweep does not prioritize promoting higher $efSearch$ values to satisfy the SLA before applying the secondary optimization goal.
- **Expected Behavior:** On a fixed prebuilt graph, sweep through $efSearch$ values to identify the exact threshold where recall passes, then select the optimal candidate among all feasible trials.
- **Confirmed Root Cause:** Flawed candidate replacement logic mixed feasible and infeasible comparisons.
- **Status:** **UNDER RESOLUTION (Stage 1)**

---

### ISSUE-04: Post-Compaction Recall Degradation Under Heavy Churn
- **Identifier:** `ISSUE-04`
- **Affected Milestone:** Milestone III (Dynamic and Self-Healing Index)
- **Severity:** **P1 (Major Performance / Quality)**
- **Source Location:** `cpp/adaptive_hnsw.hpp:2458-2500`
- **Current Behavior:** On SIFT-100K under 50% deletions, recall drops from 0.9882 (after local healing) to 0.9718 (after physical compaction), a 1.64 percentage point regression. Rebuild from scratch achieves 0.9838.
- **Expected Behavior:** Post-compaction recall should remain within 0.2 percentage points of healed recall by preserving graph navigability and entry-point integrity.
- **Confirmed Root Cause:**
  1. In `compact()`, directed edges to tombstoned nodes are dropped (`if (id_map[v] != -1) new_graphs[l][new_u].push_back(id_map[v])`), but active nodes whose degrees are depleted are never replenished with replacement neighbors.
  2. `max_level` is not recalculated after nodes are deleted, leaving `max_level` pointing to levels where no active nodes survive.
  3. `enter_point` is remapped blindly without verifying that `node_levels[enter_point] == max_level`.
- **Proposed Corrective Action:**
  1. Recalculate `max_level` from surviving active nodes after compaction.
  2. Update `enter_point` to the highest-degree active node at `max_level`.
  3. Implement bounded localized repair: for any active node whose Layer 0 degree drops below $M_{\text{min}}$ due to neighbor deletions, perform a 1-hop neighbor expansion to restore minimum connectivity.
- **Acceptance Criteria:** Post-compaction recall within 0.2 percentage points of healed recall.
- **Status:** **UNDER RESOLUTION (Stage 2)**

---

### ISSUE-05: Severe Throughput Regression During FP32 Vector Re-ranking
- **Identifier:** `ISSUE-05`
- **Affected Milestone:** Milestone II (Resource-Efficient Vector Retrieval)
- **Severity:** **P1 (Major Performance Regression)**
- **Source Location:** `cpp/adaptive_hnsw.hpp:2544-2565`
- **Current Behavior:** Mode 4 (SQ8 + FP32 Re-rank 2x) suffers a ~7x throughput drop (5,036 QPS vs 47,576 QPS for Pure SQ8).
- **Expected Behavior:** Re-ranking should only incur marginal overhead proportional to $2k$ distance evaluations without severe L3 cache thrashing.
- **Confirmed Root Cause:** Graph traversal in Pure SQ8 operates entirely in L1/L2 cache using SIMD integer dot products. Re-ranking fetches uncompressed 512-byte FP32 vectors via non-contiguous random-access pointer lookups across main memory, inducing high L3 cache misses, followed by secondary candidate re-sorting.
- **Proposed Corrective Action:**
  1. Optimize re-ranking buffer access and eliminate redundant candidate allocations.
  2. Clearly document Pure SQ8 as the primary production default for high-throughput search, retaining re-ranking as an optional high-precision mode.
- **Status:** **UNDER RESOLUTION (Stage 3)**

---

### ISSUE-06: Process Working Set Contamination Across Sequential Benchmarks
- **Identifier:** `ISSUE-06`
- **Affected Milestone:** Milestone II (Resource-Efficient Vector Retrieval)
- **Severity:** **P2 (Memory Accounting Fidelity)**
- **Source Location:** `cpp/phase6_1_evidence_runner.cpp:445-645`
- **Current Behavior:** Running FP32, SQ8, and SQ4 consecutively in the same benchmark process caused `WorkingSetSize` to grow monotonically (159 MB $\to$ 163 MB $\to$ 167 MB) due to CRT heap retention, concealing the actual resident memory reduction of SQ8.
- **Expected Behavior:** Measure memory in isolated fresh processes to report genuine steady-state and peak resident working set.
- **Confirmed Root Cause:** C++ allocator heap fragmentation and page retention in sequential long-running processes.
- **Proposed Corrective Action:** Implement dedicated isolated subprocess runner for memory benchmarking.
- **Status:** **UNDER RESOLUTION (Stage 3)**

---

### ISSUE-07: Inverted/Reversed Allocation Asymmetry (Scientific Claim Correction)
- **Identifier:** `ISSUE-07`
- **Affected Milestone:** Milestone I (Geometry-Aware Graph Intelligence)
- **Severity:** **P2 (Scientific Accuracy)**
- **Source Location:** `docs/final_four_milestone_evidence.md`
- **Current Behavior:** Earlier reports claimed geometry-aware allocation universally maximizes recall. Phase 6.1 revealed that Reversed Allocation achieved higher recall (+0.22%) by over-connecting sparse boundary nodes, while Exponential allocation optimized throughput (+24%) and distance evaluations (-2.6%).
- **Expected Behavior:** Explicitly acknowledge the accuracy-efficiency trade-off: forward allocation optimizes cluster traversal, throughput, and tail latency; reversed allocation trades throughput for peripheral recall.
- **Confirmed Root Cause:** Fundamental graph routing trade-off between hub diversity and boundary cross-linking.
- **Proposed Corrective Action:** Document the anomaly, explain the mechanism, and qualify all claims in scientific publications.
- **Status:** **RESOLVED IN DOCS, RIGOROUS BENCHMARKING UNDERWAY (Stage 5)**

---

### ISSUE-08: Streaming Budget Allocation Drift
- **Identifier:** `ISSUE-08`
- **Affected Milestone:** Milestone I (Geometry-Aware Graph Intelligence)
- **Severity:** **P2 (Engineering Improvement)**
- **Source Location:** `cpp/adaptive_hnsw.hpp:1420-1460`
- **Current Behavior:** In streaming insertion mode, online Welford statistics update mean/variance, but local degree allocations $M_i$ are clamped heuristically rather than enforcing exact global Hare-Niemeyer discrete budgets.
- **Expected Behavior:** Document and verify exact offline budget enforcement vs streaming approximate budget bounds $[M_{\text{min}}, M_{\text{max}}]$.
- **Status:** **UNDER RESOLUTION (Stage 5)**

---

### ISSUE-09: Imprecise Terminology Regarding Compaction Locking
- **Identifier:** `ISSUE-09`
- **Affected Milestone:** Milestone III (Dynamic and Self-Healing Index)
- **Severity:** **P3 (Documentation Integrity)**
- **Source Location:** `docs/final_architecture.md`, `docs/december_12_defense_evidence.md`
- **Current Behavior:** Compaction was historically described as "zero downtime." In reality, an exclusive write lock is acquired that pauses concurrent search readers for 6.5–7.2 ms.
- **Expected Behavior:** Retract "zero downtime" and replace with "fast maintenance operation with millisecond-level exclusive write-lock pause."
- **Status:** **RESOLVED (Stage 0)**

---

### ISSUE-10: Imprecise Data-Race Absence Claims
- **Identifier:** `ISSUE-10`
- **Affected Milestone:** Milestone III / Safety
- **Severity:** **P3 (Scientific Integrity)**
- **Source Location:** `docs/december_12_defense_evidence.md`
- **Current Behavior:** Stating that data races are mathematically absent without ThreadSanitizer or formal verification.
- **Expected Behavior:** Accurately state: "Empirically verified thread-safe without observed data races under multi-threaded stress testing on Windows."
- **Status:** **RESOLVED (Stage 0)**
