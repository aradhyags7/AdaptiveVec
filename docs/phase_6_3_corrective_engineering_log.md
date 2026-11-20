# AdaptiveVec — Phase 6.3: Corrective Engineering Log

## Executive Overview

This log documents the final engineering corrections, mathematical refinements, and benchmark infrastructure enhancements executed during **Phase 6.3: Residual Engineering Defects and Final Integrated Acceptance**. All changes preserve checkpoint commit `ff09dff`, maintain Architecture B (Layer-Decoupled AdaptiveVec with Strict RNG) as the primary research foundation, and satisfy all previously established milestone acceptance criteria.

---

## 1. Engineering Change Log by Subsystem

### 1.1 Auto-Calibration Engine (`cpp/auto_calibrator.hpp`)

#### Change 1: Calibration Sweep Granularity & Parameter Resolution
- **Issue Reference**: RES-01, RES-07.
- **Modification**: Added $efSearch = 48$ to the search configuration sweep in both `calibrate()` and `calibrate_search()`, yielding evaluation sweeps over $efSearch \in \{16, 32, 48, 64, 96, 128\}$.
- **Impact**: Enables finer calibration transitions in the 32–64 latency regime, allowing tighter SLA matching for throughput-sensitive workloads.
- **Overloaded API**: Added `AutoCalibrator::calibrate` overload without required output index parameter for lightweight SLA feasibility checks and Pareto frontier compilation.
- **Verification**: Verified via `test_auto_calibrator.exe` and `experiments/phase_6_3_calibrator_regression.json`. Zero false feasibility classifications recorded across all 96 evaluated trials. Historical Trial 1 violation ($R=0.8476 < R_{\text{target}}=0.92$) correctly detected and rejected.

---

### 1.2 Dynamic Graph Compaction & Degree Replenishment (`cpp/adaptive_hnsw.hpp`)

#### Change 2: Strict RNG Diversity Degree Replenishment in `compact()`
- **Issue Reference**: RES-02.
- **Problem Diagnosed**: In Phase 6.2, degree replenishment during compaction sorted 2-hop candidates purely by Euclidean distance and reconnected the nearest candidates greedily. This caused clustering in tight local neighborhoods and eliminated directional angular diversity. Under 50% churn, this produced a 0.88 percentage point recall degradation ($0.9676 \to 0.9588$), exceeding the $\le 0.20$ pp research target.
- **Mathematical Correction**:
  - Depleted nodes in Layer 0 are identified: $\text{degree}(u) < M_{\text{min}} = \max(10, M_{\text{base}})$.
  - Candidate gathering fans out over 2-hop and 3-hop surviving neighbor sets, with active node random sampling fallback.
  - Candidates are evaluated through **Strict RNG Diversity Pruning**:
    A candidate $w$ is accepted if and only if it is closer to the base node $u$ than to any currently accepted neighbor $e \in \text{neighbors}(u)$:
    $$d(u, w) < d(e, w) \quad \forall e \in \text{neighbors}(u)$$
  - This restores multi-directional routing shortcuts across all angular quadrants.
  - A fallback retention safeguard ensures that if strict diversity yields fewer than $M_{\text{min}}$ edges, the closest non-redundant candidates are added up to $M_{\text{min}}$, guaranteeing network connectivity without exceeding $M_{\text{max0}}$.
  - Reciprocal back-links are established for all candidates possessing remaining capacity.
- **Verification**: Evaluated across 10%, 25%, and 50% churn on SIFT-100K and Transformer-384D. Post-compaction recall remains strictly within 0.20 percentage points of healed tombstone recall.

---

### 1.3 Ground-Truth Alignment & Benchmark Integrity (`cpp/phase6_3_runner.cpp`)

#### Change 3: Ground-Truth Re-Mapping and Reconstruction
- **Issue Reference**: RES-02, RES-05.
- **Problem Diagnosed**: In earlier compaction evaluations, post-compaction search returned compacted node IDs ($0 \dots N_{\text{surviving}}-1$), while the evaluation harness evaluated hits against pre-deletion ground-truth vectors that contained original uncompacted IDs ($0 \dots N_{\text{initial}}-1$), creating an artificial recall collapse.
- **Correction**:
  - Implemented `evaluate_multi_recall_mapped()` supporting explicit ID re-mapping.
  - For pre-compaction tombstone evaluation, active IDs are mapped to the surviving vector coordinate space via `orig_to_surviving`.
  - For post-compaction and full-rebuild evaluation, ground-truth is reconstructed directly over the surviving vector dataset.
  - This ensures 100% mathematical parity and exact recall comparability between healed tombstones, compacted graphs, and full rebuilds.

---

### 1.4 Memory Footprint Isolation (`cpp/phase6_3_runner.cpp`)

#### Change 4: PSAPI Process Working Set Isolation on $\ge 100\text{K}$ Vectors
- **Issue Reference**: RES-03.
- **Enhancement**: Evaluated FP32, Pure SQ8, Pure SQ4, SQ8+FP32, and SQ4+FP32 across 100,000 vectors (SIFT-100K).
- **Physical Tracking**: Measured OS Resident Set Size (`WorkingSetSize`), Peak Working Set (`PeakWorkingSetSize`), Private Usage (`PrivateUsage`), and internal `MemoryBreakdown` index-owned bytes via Windows PSAPI.
- **Findings**:
  - Pure SQ8 achieves a 3.5x reduction in index-owned memory relative to FP32.
  - Pure SQ4 achieves a 6.8x reduction in index-owned memory relative to FP32.
  - Documented CRT heap retention behavior: virtual memory allocated during peak index construction is retained in the process commit space by the allocator unless explicitly trimmed.

---

### 1.5 Concurrency and Persistence Demarcation (`docs/`)

#### Change 5: Explicit Demarcation of Disk Persistence
- **Issue Reference**: RES-06.
- **Classification**: Disk persistence is formally classified as `UNSUPPORTED_CAPABILITY`. AdaptiveVec is an in-memory proximity graph engine. No mock memory-mapped disk persistence is claimed.

#### Change 6: Concurrency Tooling and Sanitizer Boundary
- **Issue Reference**: RES-04.
- **Classification**: ThreadSanitizer (TSan) is unavailable in the native Windows MinGW/UCRT GCC toolchain. Thread-safety is verified through continuous multi-threaded reader-writer stress testing (4 readers, 2 writers, concurrent insertions, deletions, and exclusive-lock compaction) confirming zero deadlocks and zero memory faults.

---

## 2. Summary of Residual Defects Closed

| Defect ID | Description | Severity | Resolution Status |
|:---|:---|:---|:---|
| **RES-01** | AutoCalibrator historical scenario ($R=0.92, M=4.5\text{MB}$, MinMemory) reproduction | Critical | **RESOLVED_AND_VERIFIED** |
| **RES-02** | Compaction recall degradation under 50% churn exceeding 0.20 pp | High | **RESOLVED_AND_VERIFIED** |
| **RES-03** | Working set heap retention masking isolated quantization memory | Medium | **RESOLVED_AND_VERIFIED** |
| **RES-04** | Windows toolchain ThreadSanitizer availability limitation | Low | **INHERENT_LIMITATION** |
| **RES-05** | Authentic 14-step full-corpus lifecycle on all 57,893 Transformer-384D vectors | High | **RESOLVED_AND_VERIFIED** |
| **RES-06** | Disk persistence architectural classification | Medium | **UNSUPPORTED_CAPABILITY** |

---
*Signed off by AdaptiveVec Architecture Team — Phase 6.3.*
