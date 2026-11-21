# AdaptiveVec — Master Evidence and Experimental Claim Map (Revision 2)

**Status:** Reconciled Scientific Evidence Map (Revision 2)  
**Git Checkpoint:** `7936c61` (`phase-6.4-final-correctness-closure`)  
**Standardized Hardware Platform:** Intel Core 5 210H (8 physical cores, 12 logical threads, AVX2/FMA, 16 GB RAM), Windows 11 Build 26100, GCC 14.2.0 (`g++ -O3 -mavx2 -mfma -fopenmp -std=c++17`).

---

## 1. Milestone I: Geometry-Aware Graph Intelligence

### Claim M1-1: Normalized Exponential Throughput and Latency Advantage
- **Manuscript:** Section IV, Section IX-A, Table II, Figure 3(a).
- **Audited Empirical Finding:** On SIFT-100K ($N=100{,}000, D=128, M_{\text{base}}=16, efConstruction=120, efSearch=64, k=10$), Normalized Exponential allocation achieves **3,091.0 QPS** (+24.01% over Strict RNG Uniform at 2,492.5 QPS) and **0.486 ms P95 latency** (-37.62% from 0.780 ms). P99 latency is 1.384 ms (vs 2.062 ms). Mean distance evaluations are 1,169.1 (vs 1,200.5). Recall@10 is 0.9926 (vs 0.9929).
- **Edge Budget Conservation:** Base forward degree budget is strictly conserved at $\sum M_i = 1{,}600{,}000 = N \cdot M_{\text{base}}$ via Hare-Niemeyer discretization. Maximum Layer 0 capacity is $3{,}200{,}000$ slots ($M_{\max 0} = 2 M_i$). Realized physical graph memory is **27.50 MB** (vs 27.89 MB for uniform).
- **Raw Evidence Artifact:** `experiments/phase_6_1_geometry_reproduction.json`, `experiments/phase_6_4_geometry_implementation_audit.json`.
- **Reproducing Source Code:** `cpp/budget_allocator.hpp` lines 83–149, lines 178–204.
- **Single-Pass Feasibility Condition:** Exact budget conservation in Algorithm 1 requires $\text{deficit} \le |\mathcal{U}^+|$ (or $|\text{deficit}| \le |\mathcal{U}^-|$ for surplus). If $\text{deficit} > |\mathcal{U}^+|$, multiple increments per node would be required, which the single-pass implementation in `cpp/budget_allocator.hpp` asserts against. Across all evaluated benchmarks, $\text{deficit} \le |\mathcal{U}^+|$ strictly holds.

### Claim M1-2: Shuffled Degree Control Proves Geometric Alignment Causality
- **Manuscript:** Section IX-A, Table II.
- **Audited Empirical Finding:** Randomly permuting the degree sequence degrades throughput to **2,379.8 QPS** (-23.01% vs Normalized Exponential) and inflates P95 latency to **0.801 ms** (+64.81%), with 1,204.4 mean distance evaluations and 0.9939 recall@10.
- **Provenance Note:** The preliminary draft quoted 0.9882 recall due to a transcription error; the authentic raw benchmark artifact `phase_6_1_geometry_reproduction.json` records 0.9939 recall and 2,379.83 QPS.
- **Raw Evidence Artifact:** `experiments/phase_6_1_geometry_reproduction.json`.

### Claim M1-3: Reversed Geometry Control and Boundary Recall Trade-off
- **Manuscript:** Section IX-A, Table II.
- **Audited Empirical Finding:** Inverting the allocation policy ($M=24$ for sparse peripheral nodes, $M=8$ for dense hub nodes) achieves slightly higher recall (**0.9948 vs 0.9926, +0.22 pp**), but at the expense of **+5.60% distance evaluations** (1,234.6 vs 1,169.1) and **-5.10% throughput degradation** (2,933.4 vs 3,091.0 QPS), with P95 latency of 0.570 ms.
- **Provenance Note:** Raw values from `phase_6_1_geometry_reproduction.json` (line 56) are QPS 2,933.39, P95 0.5699 ms, distance evaluations 1,234.63.
- **Raw Evidence Artifact:** `experiments/phase_6_1_geometry_reproduction.json`.

---

## 2. Milestone II: Resource-Efficient Storage and Quantization

### Claim M2-1: Pure SQ8 Memory Reduction and Throughput Lift on SIFT-10K
- **Manuscript:** Section V-A, Section IX-B, Table III.
- **Audited Empirical Finding:** Pure SQ8 ADC slashes vector payload storage by **75.00%** (5.12 MB $\to$ 1.28 MB) and lifts throughput by **+70.13%** (47,576 vs 27,965 QPS) at 99.54% recall (vs 99.62% for FP32). Total owned index bytes drop from 7,825,132 (7.46 MB) to 4,801,320 (4.58 MB).
- **Pure SQ4 Reconciliation:** Pure SQ4 achieves 40,262.2 QPS (corrected from erroneous 48,110 QPS), 3.63 MB owned index bytes, and 0.9974 recall.
- **FP32 Re-ranking Collapse:** Storing raw FP32 alongside SQ8 bloats owned index bytes to 9,922,908 (9.46 MB) and collapses throughput to 5,036.5 QPS (~7x regression).
- **Raw Evidence Artifact:** `experiments/phase_6_1_quantization_memory_audit.json`.

### Claim M2-2: Fresh Subprocess Memory Isolation on SIFT-100K
- **Manuscript:** Section V-A, Section IX-B, Table III, Figure 3(b).
- **Audited Empirical Finding:** In independent OS subprocesses with deduplicated buffer reservation:
  - **FP32:** Vector Memory = 48.83 MB; Graph RAM = 25.70 MB; Total Index-Owned = 78.47 MB; Process WS = 93.42 MB; Recall@10 = 0.9910; QPS = 12,705.2; P99 = 0.984 ms.
  - **Pure SQ8:** Vector Memory = 12.21 MB (-75.00%); Graph RAM = 25.70 MB; Total Index-Owned = 41.85 MB (-46.66%); Process WS = 56.70 MB (-39.31%); Recall@10 = 0.9770; QPS = 12,437.2; P99 = 0.998 ms.
  - **Pure SQ4:** Vector Memory = 6.10 MB (-87.50%); Graph RAM = 25.70 MB; Total Index-Owned = 35.75 MB (-54.44%); Process WS = 49.30 MB (-47.23%); Recall@10 = 0.8810; QPS = 12,987.0; P99 = 0.952 ms.
- **Dataset Attribution Note:** The 0.8810 recall for Pure SQ4 was measured specifically on SIFT-100K ($N=100{,}000, D=128$), not Transformer-384D.
- **Raw Evidence Artifact:** `experiments/phase_6_4_fresh_precision_memory.json`.

---

## 3. Milestone III: Dynamic Self-Healing Index and Safe Replacement

### Claim M3-1: Soft Transit Bridge Routing Eliminates Pre-Compaction Degradation
- **Manuscript:** Section VI-A, Section IX-C.
- **Audited Empirical Finding:** Soft transit routing through tombstones maintains 0.9840 recall at 10% churn (vs unhealed degradation), delivering +0.04 to +0.06 pp lift over tombstone baseline.
- **Raw Evidence Artifact:** `experiments/phase_6_4_dynamic_maintenance_investigation.json`.

### Claim M3-2: Multi-Layer Replenishment and Compaction Acceptance Gates
- **Manuscript:** Section VI-B, Section IX-C, Table IV.
- **Audited Empirical Finding:**
  - **10% Churn:** Healed serving index delivers 0.9840 mean recall and 74.74 MB memory. However, its 95% bootstrap lower bound is $\text{CI}_{\text{lower}} = 0.9794 < 0.9800$, classifying it as a retained serving state (`Retained_Serving`) rather than a certified feasible deployment. Compacted candidate achieves 0.9820 recall but transient peak memory reaches **143.98 MB**, failing Gate 3 ($> 120.0$ MB), Gate 2 ($\text{CI}_{\text{lower}} = 0.9770 < 0.9800$), and Gate 6 ($\Delta R_{\text{compact}} = 0.0020$). Action: `COMPACTION_DEFERRED`.
  - **25% Churn:** Multi-Layer replenishment recovers post-compaction recall to 0.9786 (fixed $efS=64$) and 0.9865 (recalibrated $efS=96$). Transient peak memory (131.16 MB) exceeds 120 MB budget. Action: `REBUILD_RECOMMENDED` (6.11 s).
  - **50% Churn:** Compacted candidate recall drops to 0.9740 (failing Gate 1 $R \ge 0.9800$ and Gate 6 $\Delta R \le 0.0020$). Fresh Rebuild succeeds in **3.95 seconds** with **0.9878 recall** and peak memory of 113.00 MB $\le 120.0$ MB, passing all 6 gates. Action: `REBUILD_REQUIRED`.
- **Raw Evidence Artifact:** `experiments/phase_6_4_dynamic_maintenance_investigation.json`.

### Claim M3-3: Complete Exclusive Lock Synchronization Pause vs Pointer Swap
- **Manuscript:** Section VI-D, Section XII.
- **Audited Empirical Finding:** In `ServingIndexManager`, complete exclusive write-lock pause was measured at **11.68 milliseconds** across 350 streaming mutations (200 insertions, 100 deletions, 50 updates), as documented in `docs/phase_6_4_dynamic_maintenance_investigation_report.md` (line 122). This pause includes re-indexing 200 vector insertions into the candidate graph while holding the write lock. The pointer swap in isolation takes $\approx 0.014$ ms ($0.9$ to $193.9\ \mu\text{s}$).
- **Limitation Acknowledged:** Mutations replayed during the lock are not re-evaluated against the 6 acceptance gates.
- **Raw Evidence Artifact:** `docs/phase_6_4_dynamic_maintenance_investigation_report.md`, `cpp/test_concurrent_mutation_preservation.cpp`.

---

## 4. Milestone IV: Automatic Resource Optimization

### Claim M4-1: SIFT-10K Strict Memory SLA Feasibility Trajectory
- **Manuscript:** Section VII, Section IX-D, Table V.
- **Audited Empirical Finding:**
  - Constraints: $R \ge 0.92, M \le 4.5\text{ MB}, \text{P99} \le 5.0\text{ ms}$, MinMemory.
  - Selected Configuration: Trial 10 ($M=8, efC=30, efS=64$, SQ8, Ada-ef=ON).
  - Calibration: Mean recall = 0.9380 $\ge 0.92$, but 95% bootstrap lower bound is **$\text{CI}_{\text{lower}} = 0.9190 < 0.9200$**, placing it formally in `PROVISIONALLY_FEASIBLE`.
  - Deployment Validation: Across all 10,000 vectors, achieved Recall = **0.9416**, 95% CI = $[0.9334, 0.9494]$ ($\text{CI}_{\text{lower}} = 0.9334 \ge 0.9200$), Index Memory = 3.38 MB (3.3787 MB), P99 = 0.33 ms, QPS = 11,311.8, officially certifying `VERIFIED_FEASIBLE`.
- **Raw Evidence Artifact:** `experiments/phase_6_4_exact_calibrator_regression.json`.

### Claim M4-2: Full-Corpus Transformer-384D End-to-End SLA Certification
- **Manuscript:** Section VII, Section IX-D, Table V.
- **Audited Empirical Finding:** On all 57,893 Transformer-384D vectors ($D=384$):
  - Configuration: $M_{\text{base}}=16, efConstruction=100, efSearch=96$, SQ8 ADC with `drop_raw_vectors()`.
  - Target SLA: $R \ge 0.95, M \le 150.0\text{ MB}, \text{P99} \le 15.0\text{ ms}$.
  - Achieved: Recall = **0.9720**, Index-Owned Memory = **56.61 MB**, P99 Latency = **7.38 ms**, QPS = 487.2.
  - Status: `VERIFIED_FEASIBLE`.
- **Raw Evidence Artifact:** `experiments/phase_6_4_final_report.md`, `experiments/phase_6_4_exact_calibrator_regression.json`.
