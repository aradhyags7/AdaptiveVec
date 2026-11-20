# AdaptiveVec — Master Benchmark Claim Ledger

**Document Status:** Phase 6.2 Engineering Closure & Presentation Ready  
**Branch:** `phase-6.2-comprehensive-engineering-closure`  
**Parent Checkpoint:** `7f2a862` (Phase 6.1 Defense Freeze)  
**Standardized Hardware Environment:** Intel Core 5 210H (8 physical cores, 12 logical threads, AVX2/FMA, 16 GB RAM), Windows 11 Build 26100, GCC 14.2.0 (`g++ -O3 -mavx2 -mfma -fopenmp -std=c++17`).

---

## 1. Verified Core Claims Ledger

| Claim ID | Headline Finding | Dataset | Baseline Reference | AdaptiveVec Configuration | Raw Evidence File | Reproducibility Command | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLM-01** | Geometry-Aware Allocation achieves **+24.01% QPS** and **-37.62% P95 latency** at matched recall (99.26% vs 99.29%) | SIFT-100K ($N=100\text{K}, D=128$) | Strict RNG Uniform ($M=16, efC=120, efS=64$) | Normalized Exponential ($M \in [8, 24], \text{mean } 16, \beta_0=1, \beta_u=0$) | `experiments/phase_6_1_geometry_reproduction.json` | `.\cpp\phase6_1_evidence_runner.exe` | **VERIFIED** |
| **CLM-02** | Shuffled Degree Control degrades throughput to **2,379.8 QPS (-23.0%)**, proving performance requires geometric alignment | SIFT-100K ($N=100\text{K}, D=128$) | Normalized Exponential | Randomized Degree Control (Random permuted degrees) | `experiments/phase_6_1_geometry_reproduction.json` | `.\cpp\phase6_1_evidence_runner.exe` | **VERIFIED** |
| **CLM-03** | Reversed Geometry achieves higher recall (**0.9948 vs 0.9926**) at the cost of **+5.6% distance evaluations** and **-5.1% throughput** | SIFT-100K ($N=100\text{K}, D=128$) | Normalized Exponential | Reversed Geometry Control (Sparse nodes get $M=24$, hubs get $M=8$) | `experiments/phase_6_1_geometry_reproduction.json` | `.\cpp\phase6_1_evidence_runner.exe` | **VERIFIED** |
| **CLM-04** | Pure SQ8 ADC slashes vector memory by **75.0%** with **+70.13% higher QPS** (47,576 vs 27,965 QPS) at 99.54% recall | SIFT-10K ($N=10\text{K}, D=128$) | Pure FP32 Storage ($5.12\text{ MB}$ vector payload) | Pure SQ8 Storage ($1.28\text{ MB}$ vector payload, purged FP32) | `experiments/phase_6_1_quantization_memory_audit.json` | `.\cpp\phase6_1_evidence_runner.exe` | **VERIFIED** |
| **CLM-05** | Pure SQ4 ADC achieves **8.0x vector compression** ($0.64\text{ MB}$ vs $5.12\text{ MB}$) with 99.74% recall | SIFT-10K ($N=10\text{K}, D=128$) | Pure FP32 Storage | Pure SQ4 Storage (4-bit packed integer vectors) | `experiments/phase_6_1_quantization_memory_audit.json` | `.\cpp\phase6_1_evidence_runner.exe` | **VERIFIED** |
| **CLM-06** | Local healing prevents graph disconnectivity, maintaining **99.44% recall (+0.46% lift)** under 50% node churn | SIFT-100K ($N=100\text{K}, D=128$) | Unhealed Tombstones (0.9898 recall) | Active 2-Hop Local Healing (`mark_deleted(i, true)`) | `experiments/final_reproduction_manifest.json` | `.\cpp\defense_evidence_runner.exe` | **VERIFIED** |
| **CLM-07** | Physical compaction reclaims 100% of deleted memory with an exclusive-lock pause of **4.3 to 7.2 ms** | SIFT-10K / 100K | Full Index Rebuild (8,365 ms) | Dual-Buffer Compaction (`compact()`) | `experiments/phase_6_2_compaction_correctness.json` | `.\cpp\phase6_2_runner.exe` | **VERIFIED** |
| **CLM-08** | Multi-threaded search is thread-safe during concurrent streaming updates (**7,933 queries served, 0 errors**) | SIFT-100K ($N=100\text{K}, D=128$) | Sequential Execution | 4 Search Threads + 1 Ingestion Thread + 1 Delete/Heal Thread | `experiments/phase_6_1_concurrency_validation.json` | `.\cpp\phase6_1_evidence_runner.exe` | **VERIFIED** |
| **CLM-09** | `AutoCalibrator` solves multi-objective SLA constraints in **<13 seconds** with 95% bootstrap CI validation | SIFT-10K ($N=10\text{K}, D=128$) | Manual Grid Tuning | Automated 3-Stage Optimizer (Lower Bound -> Build -> Search) | `experiments/phase_6_1_calibration_constraints.json` | `.\cpp\phase6_1_evidence_runner.exe` | **VERIFIED** |
| **CLM-10** | `AutoCalibrator` strictly rejects impossible constraints as `InfeasibleWithinSearchSpace` | SIFT-10K ($N=10\text{K}, D=128$) | Feasible Execution | Constraints: Recall $\ge 99.9\%$, Memory $\le 1\text{ MB}$, P99 $\le 0.1\text{ ms}$ | `experiments/final_reproduction_manifest.json` | `.\cpp\defense_evidence_runner.exe` | **VERIFIED** |
| **CLM-11** | End-to-end integration succeeds across all four milestones on Transformer-384D | Transformer-384D ($N=57\text{K}, D=384$) | Separate Unit Tests | Integrated Pipeline: AutoCalib -> Build -> Stream -> Heal -> Compact | `experiments/phase_6_2_full_corpus_integration.json` | `.\cpp\phase6_2_runner.exe` | **VERIFIED** |
| **CLM-12** | AutoCalibrator Feasibility Integrity guarantees zero false feasibility with 6 formal states | SIFT-10K ($N=10\text{K}, D=128$) | Legacy Flawed Selection | Decoupled candidate tracking (`best_feasible_trial` vs `best_fallback_trial`) | `experiments/phase_6_2_calibrator_correctness.json` | `.\cpp\phase6_2_runner.exe` | **VERIFIED** |
| **CLM-13** | Compaction degree replenishment eliminates isolated nodes ($\min \text{deg} \ge 8$) and preserves navigability | SIFT-10K ($N=10\text{K}, D=128$) | Legacy Unreplenished Pruning | Bounded 2-hop neighbor replenishment & entry-point integrity | `experiments/phase_6_2_compaction_correctness.json` | `.\cpp\phase6_2_runner.exe` | **VERIFIED** |
| **CLM-14** | Direct OS Working Set audit confirms memory reclamation and quantifies heap allocator overhead | SIFT-10K ($N=10\text{K}, D=128$) | Internal Buffer-Only Metrics | Windows PSAPI `GetProcessMemoryInfo` runtime measurement | `experiments/phase_6_2_process_memory_audit.json` | `.\cpp\phase6_2_runner.exe` | **VERIFIED** |

---

## 2. Historical & Inherited Results Ledger

| Claim ID | Headline Finding | Dataset | Baseline Reference | AdaptiveVec Configuration | Historical Origin | Status | Note |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **HIST-01** | Geometry-Aware Allocation achieves **+15.43% QPS** and **-42.01% P95 latency** at matched memory ($178.4\text{ MB}$) | SIFT-500K ($N=500\text{K}, D=128$) | Strict RNG Uniform ($8.0\text{M}$ edges) | Layer-Decoupled Normalized Exponential ($8.0\text{M}$ edges) | Phase 5.3 Commit `965229c` (`sift500k_isolated_adaptive.json`) | **HISTORICAL** | Inherited from Phase 5.3 execution; verified via raw artifact review. |
| **HIST-02** | Official `hnswlib` delivers **37,232 QPS** at 0.9951 recall on SIFT-100K | SIFT-100K ($N=100\text{K}, D=128$) | AdaptiveVec | `hnswlib` v0.8.0 flat C-struct | Phase 5.3 Commit `965229c` (`external_baselines_comparison.json`) | **HISTORICAL** | External baseline reference. Faster due to lack of dynamic mutation primitives. |
| **HIST-03** | FAISS `IndexHNSWFlat` delivers **28,540 QPS** at 0.9942 recall on SIFT-100K | SIFT-100K ($N=100\text{K}, D=128$) | AdaptiveVec | FAISS v1.16.0 | Phase 5.3 Commit `965229c` (`faiss_baseline_500k.json`) | **HISTORICAL** | External baseline reference. Lacks compaction and local topological healing. |

---

## 3. Negative Findings & Discarded Hypotheses Ledger

| Claim ID | Discarded Hypothesis | Investigated Phase | Empirical Evidence | Final Decision & Reason |
| :---: | :--- | :---: | :--- | :--- |
| **NEG-01** | **Query-Time Adaptation (Ada-ef / Controllers):** Dynamically adjusting $efSearch$ based on entry-point distance | Phase 4.1–4.2 | Incurred reference-monitoring overhead, memory contention, and failed recall requirements | **DISCARDED:** Query hot-path adaptation rejected; moved all adaptive intelligence to index construction. |
| **NEG-02** | **Uncertainty-Aware Search Termination:** Halting search based on candidate distance convergence statistics | Phase 4.3 | Increased distance computations and tail latency; failed to outperform fixed-budget baseline | **DISCARDED:** Fixed-budget and static efSearch proven strictly superior in tail latency and throughput. |
| **NEG-03** | **Geometry-Conditioned Pruning (GCP):** Modulating RNG edge acceptance based on local cluster curvature | Phase 5.1 | -13.6% lower throughput, -0.26% recall vs Architecture B; high pruning complexity | **DISCARDED:** Strict RNG neighbor selection without heuristic over-retention established as superior control. |
| **NEG-04** | **Upper-Layer Geometric Modulation:** Modulating out-degree on upper skip layers ($l \ge 1$) based on geometry | Phase 5.2 | Distorted global long-range routing highways, reducing search navigability | **DISCARDED:** Layer-Decoupled scaling adopted ($\beta_0 = 1.0, \beta_{\text{upper}} = 0.0$). |
| **NEG-05** | **FP32 Raw Vector Re-ranking:** Re-ranking quantized candidates against stored FP32 vectors | Phase 6.1 | Caused a ~7x throughput regression (5,036 vs 47,576 QPS) due to non-sequential memory cache misses | **DISCLOSED LIMITATION:** Pure SQ8 recommended as production default; re-ranking retained only when exact FP32 precision is strictly demanded. |
| **NEG-06** | **Universal Recall Superiority Claim:** Claiming geometry-aware allocation universally increases recall | Phase 6.1 | Reversed Geometry Control achieved +0.22% higher recall by over-connecting peripheral boundaries | **CORRECTED CLAIM:** Geometry-aware allocation optimizes throughput/tail-latency efficiency per distance evaluation, not raw recall alone. |
