# AdaptiveVec — Phase 6.3 Residual Issue Register

**Document Version:** 1.0.0  
**Branch:** `phase-6.3-final-integrated-acceptance`  
**Parent Checkpoint:** `ff09dff` (Phase 6.2 Engineering Closure)  
**Date:** November 20, 2026  

---

## 1. Executive Issue Audit & State Classification

Following the Phase 6.2 audit, each previously identified issue has been evaluated against rigorous empirical evidence and classified according to the six formal states:
- `RESOLVED_AND_VERIFIED`: Formally repaired, tested, and verified against quantitative acceptance criteria.
- `IMPLEMENTED_BUT_UNVERIFIED`: Code implemented, but isolated empirical validation across independent processes remains required.
- `PARTIALLY_RESOLVED`: Significant engineering progress made, but failing one or more strict quantitative acceptance criteria.
- `UNRESOLVED`: Identified defect without functional remediation.
- `INHERENT_LIMITATION`: Fundamental hardware or algorithmic property that cannot be engineered away without violating system constraints.
- `UNSUPPORTED_CAPABILITY`: Deliberate design non-goal out of project scope.

---

## 2. Issue Audit Matrix

| Issue ID | Description | Severity | Phase 6.2 Status | Phase 6.3 State Classification | Residual Scope & Action |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **ISSUE-01** | AutoCalibrator Feasibility Selection Defect | P0 | Fixed in code | `PARTIALLY_RESOLVED` | Candidate separation implemented, but exact original historical failing scenario ($R=0.92, M=4.5\text{MB}, \text{MinMemory}$) must be directly reproduced and validated. |
| **ISSUE-02** | Formal 6-State Feasibility State Machine | P0 | Implemented | `RESOLVED_AND_VERIFIED` | 6 states verified in unit tests and constraint validation suite. |
| **ISSUE-03** | Prebuilt Graph Parameter Sweeps | P1 | Implemented | `RESOLVED_AND_VERIFIED` | `calibrate_search()` successfully sweeps $efSearch \in \{16, 32, 64, 96, 128\}$. |
| **ISSUE-04** | Post-Compaction Recall Degradation Under Churn | P1 | Overhauled | `PARTIALLY_RESOLVED` | Degree replenishment and entry-point integrity implemented, but post-compaction recall drops 0.88 pp at 50% churn (exceeding 0.20 pp target). Needs diversity repair or rebuild threshold. |
| **ISSUE-05** | FP32 Re-ranking Throughput Regression | P1 | Documented | `INHERENT_LIMITATION` | ~7x latency regression is an inherent hardware L3 cache / random DRAM lookup penalty. Pure SQ8 ADC confirmed as production default. |
| **ISSUE-06** | Process Working Set Contamination Across Runs | P2 | PSAPI added | `IMPLEMENTED_BUT_UNVERIFIED` | PSAPI integrated, but requires isolated fresh-process execution to demonstrate unpolluted memory footprint per precision mode. |
| **ISSUE-07** | Reversed Geometric Allocation Anomaly | P2 | Documented | `RESOLVED_AND_VERIFIED` | Scientifically grounded in claim ledger; forward allocation optimizes traversal/QPS, reversed allocation trades throughput for fringe recall. |
| **ISSUE-08** | Streaming Budget Bounds vs Exact Discretization | P2 | Verified | `RESOLVED_AND_VERIFIED` | Verified bounded degree limits ($M_{\text{max0}} \le 32$); distinguished from exact offline Hare-Niemeyer global budget. |
| **ISSUE-09** | Compaction Exclusive-Lock Pause Terminology | P3 | Corrected | `RESOLVED_AND_VERIFIED` | Retracted "zero downtime"; documented 4–8 ms exclusive write-lock pause. |
| **ISSUE-10** | Empirical Concurrency Safety Claims | P3 | Corrected | `RESOLVED_AND_VERIFIED` | Formally designated as "empirically verified thread-safe under stress testing" rather than mathematical proof. |
| **ISSUE-11** | Disk Persistence and Index Serialization | P2 | Not in scope | `UNSUPPORTED_CAPABILITY` | Disk serialization / persistence is not implemented in memory-first prototype. Formally classified as unsupported capability. |
| **ISSUE-12** | External Throughput Gap vs Canonical C hnswlib | P2 | Documented | `INHERENT_LIMITATION` | Canonical `hnswlib` is faster due to static flat layout without locks or tombstone abstractions. AdaptiveVec trades static QPS for full dynamic lifecycle. |

---

## 3. Prioritized Phase 6.3 Residual Work Items

1. **RES-01 (Stage 1): Exact Historical AutoCalibrator Regression Validation**
   - Reproduce exact scenario: $R_{\text{target}}=0.92$, $M_{\text{max}}=4.5\text{ MB}$, $\tau_{p99}=5.0\text{ ms}$, `MinMemory`.
   - Formally verify that Trial 1 ($efSearch=16, R=0.8476$) is never selected as feasible.
   - Verify that all 6 feasibility states pass deterministic unit tests in both C++ and Python bindings.

2. **RES-02 (Stage 2): Compaction Navigability & Neighbor Quality Repair**
   - Investigate the 0.88 percentage-point degradation under 50% churn.
   - Compare replenishment strategies: surviving edges only, 2-hop nearest, 2-hop with Strict RNG diversity pruning, and full rebuild.
   - Eliminate weak random connections and restore post-compaction recall within 0.20 percentage points of healed recall, or formulate precise rebuild threshold.

3. **RES-03 (Stage 3): Fresh Isolated-Process Memory Audit & Reranking Profiling**
   - Run isolated subprocesses for FP32, Pure SQ8, Pure SQ4, SQ8+FP32, SQ4+FP32 to measure initial, peak, and steady-state working set without CRT heap retention.
   - Measure matched-recall curves vs official `hnswlib` and FAISS.

4. **RES-04 (Stage 4): Concurrency Sanitizer Documentation & Persistence Classification**
   - Document tool availability (Windows MSYS2 GCC OpenMP vs Clang TSan).
   - Explicitly classify index persistence as an unsupported capability.

5. **RES-05 (Stage 5): Full-Corpus Authentic 14-Step Lifecycle Acceptance**
   - Execute the complete end-to-end lifecycle on all 57,893 Transformer-384D vectors:
     Calibration $\to$ Build $\to$ Quantize $\to$ Baseline Search $\to$ Concurrent Updates $\to$ Deletion $\to$ Healing $\to$ Compaction $\to$ Ground Truth Re-evaluation $\to$ Verification.
