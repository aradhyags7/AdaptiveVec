# AdaptiveVec — Phase 6.4 Corrective Engineering Log

**Document Version:** 1.0.0  
**Branch:** `phase-6.4-final-correctness-closure`  
**Parent Checkpoint:** `cece8fc` (Phase 6.3 Integrated Acceptance)  
**Date:** November 20, 2026  

---

## 1. Overview and Scope

This document details the exact source-code modifications made across `cpp/auto_calibrator.hpp`, `cpp/adaptive_hnsw.hpp`, and `cpp/budget_allocator.hpp` to achieve complete SLA integrity, benchmark reconciliation, thread-safety closure, and deterministic validation in Phase 6.4.

---

## 2. Detailed File Modifications

### 2.1 `cpp/auto_calibrator.hpp`

#### 2.1.1 Strict Held-Out Validation Closure
- **Issue:** The AutoCalibrator previously applied artificial slack multipliers during held-out validation (`val_recall >= constraints.target_recall * 0.98f` and `val_p99_lat <= constraints.target_p99_latency_ms * 1.10f`). This allowed candidates that violated the SLA point estimate on the test split to be falsely certified as `VERIFIED_FEASIBLE`.
- **Changes Applied:**
  - Removed all `* 0.98f` and `* 1.10f` multipliers in `AutoCalibrator::calibrate()` (line 687).
  - Removed all `* 0.98f` and `* 1.10f` multipliers in `AutoCalibrator::calibrate_search()` (line 1007).
  - Validation condition now strictly enforces:
    ```cpp
    bool held_out_passed = (val_recall >= constraints.target_recall) &&
                           (val_p99_lat <= constraints.target_p99_latency_ms);
    ```
  - Added held-out validation check for `require_conservative_recall` candidates before assigning `PROVISIONALLY_FEASIBLE`.

#### 2.1.2 Reconciled Provenance
- Documented that Trial 6 in Phase 6.2 ($efSearch=128$) and Trial 11 in Phase 6.3 ($efSearch=96$, Ada-ef=ON) share the identical underlying build candidate ($M=8, efConstruction=30, \text{SQ8}$). The grid expansion from 6 to 12 search trials identified the optimal configuration with dynamic beam modulation.

---

### 2.2 `cpp/adaptive_hnsw.hpp`

#### 2.2.1 Thread-Safe Dynamic Growth in Streaming Insertion
- **Issue:** Under concurrent streaming insertions beyond initial vector capacity, `std::vector::push_back` or `reserve()` caused reallocation of the underlying buffer while reader threads were executing `search()`. Readers accessing `&data[id * dim]` suffered heap use-after-free crashes.
- **Changes Applied:**
  - Added `mutable std::recursive_mutex insertion_mutex;` to `AdaptiveHNSWIndex` to synchronize concurrent writer threads.
  - Added capacity growth check at the entry of `insert()`:
    ```cpp
    bool needs_grow = (policy.keep_raw_vectors && (data.size() + dim > data.capacity())) ||
                      (policy.enable_sq8 && (quantized_data.size() + dim > quantized_data.capacity())) ||
                      (policy.enable_sq4 && (quantized_data_sq4.size() + (dim + 1) / 2 > quantized_data_sq4.capacity())) ||
                      (cur_n + 1 > node_levels.capacity()) ||
                      (cur_n + 1 > node_params.capacity()) ||
                      (cur_n + 1 > in_degrees.capacity()) ||
                      (cur_n + 1 > deleted.capacity()) ||
                      (graphs.empty() || cur_n + 1 > graphs[0].capacity());

    if (needs_grow) {
        std::unique_lock<std::shared_mutex> write_lock(index_lifecycle_rw_mutex);
        size_t new_cap = std::max((size_t)1024, cur_n * 2 + 1024);
        reserve(new_cap);
    }
    ```
  - Replaced thread safety from optimistic read locks to strict exclusive lifecycle lock during capacity growth.

#### 2.2.2 Batch Insert Self-Deadlock Resolution
- **Issue:** `batch_insert()` acquired a shared lock on `index_lifecycle_rw_mutex` before inserting the seed node ($q=0$). The seed insertion logic called `reserve()`, which attempted to acquire an exclusive lock on the same mutex, resulting in a self-deadlock.
- **Changes Applied:**
  - Refactored `batch_insert()` to pre-reserve the entire required capacity under an exclusive lock *before* inserting seed vectors.
  - The exclusive lock is released prior to parallel insertion, allowing worker threads to safely proceed under shared read locks.

#### 2.2.3 Direct Preassigned Degree Injection API
- **Added Method:**
  ```cpp
  void assign_preassigned_degrees(const std::vector<int>& degrees, 
                                 const std::vector<float>* lids = nullptr, 
                                 const std::vector<float>* densities = nullptr) {
      preassigned_node_params.resize(degrees.size());
      for (size_t i = 0; i < degrees.size(); ++i) {
          int m_val = degrees[i];
          int m_max0_val = m_val * 2;
          float l = (lids && i < lids->size()) ? (*lids)[i] : 0.0f;
          float d = (densities && i < densities->size()) ? (*densities)[i] : 0.0f;
          preassigned_node_params[i] = NodeParams{m_val, m_val, m_max0_val, policy.ef_c_base, 0.0f, l, d};
      }
  }
  ```
  This bridges discrete Largest-Remainder allocations directly into graph construction.

---

### 2.3 `cpp/budget_allocator.hpp`

#### 2.3.1 Normalized Exponential Construction Bridge
- **Added Method:**
  ```cpp
  static void apply_normalized_exponential(
      AdaptiveHNSWIndex& index,
      const std::vector<GeometrySignal>& signals,
      int m_base = 16,
      int m_min = 8,
      int m_max = 24,
      float beta = 0.5f
  ) {
      auto degrees = allocate_exponential_normalized(signals, m_base, m_min, m_max, beta);
      std::vector<float> lids(signals.size()), densities(signals.size());
      for (size_t i = 0; i < signals.size(); ++i) {
          lids[i] = signals[i].lid;
          densities[i] = signals[i].density;
      }
      index.assign_preassigned_degrees(degrees, &lids, &densities);
  }
  ```
- **Formula Verification:** Verified that `allocate_exponential_normalized` computes positive weights $w_i = \exp(\beta \cdot S_i)$, preserving degree mass exactly via the Hare-Niemeyer / Largest-Remainder algorithm: $\sum M_i = N \cdot M_{\text{base}}$.

---

## 3. Regression Test Verification

| Test Suite | Target | Result |
| :--- | :--- | :---: |
| Native Test Harness (`cpp/tests/test_main.cpp`) | Core Data Structures & Search | **12/12 PASS** |
| AutoCalibrator Deterministic Tests (`cpp/tests/test_auto_calibrator.cpp`) | 6 Feasibility States & Calibration | **6/6 PASS** |
| Python Integration Suite (`pytest tests/`) | Bindings & Lifecycle | **34/34 PASS** |
| Concurrency Stress Test | 4 Readers, 2 Writers, Dynamic Growth | **0 Crashes / PASS** |
