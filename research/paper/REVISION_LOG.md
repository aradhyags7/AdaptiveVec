# AdaptiveVec — Academic Manuscript Revision Log (Draft V2)

**Document:** AdaptiveVec Research Manuscript Revision 2  
**Date:** November 21, 2026  
**Git Checkpoint:** `7936c61` (`phase-6.4-final-correctness-closure`)  
**Standardized Evaluation Platform:** Intel Core 5 210H (8 physical cores, 12 threads, AVX2/FMA, 16 GB DDR5 RAM), Windows 11 Build 26100, GCC 14.2.0 (`g++ -O3 -mavx2 -mfma -fopenmp -std=c++17`).

---

## Summary of Draft V2 Integrity Corrections

| Issue ID | Domain | Root Cause in V1 Draft | Verified Source / Artifact | Corrected Resolution in V2 Draft |
|---|---|---|---|---|
| **ISSUE-01** | Algorithm 1 & Proposition 1 | Assumed multi-increment capacity $\sum(M_{\max} - m_i) \ge \Delta E$ in proof, whereas C++ implementation is single-pass $+1$ per node | `cpp/budget_allocator.hpp:83-149` | Reformulated Algorithm 1 and Proposition 1 to state exact single-pass condition $\text{deficit} \le |\mathcal{U}^+|$; explicitly noted that deficits exceeding $|\mathcal{U}^+|$ would require multiple increments per node, which single-pass asserts against; verified condition holds across all benchmarks. |
| **ISSUE-02** | 6 Acceptance Gates & AutoCalibrator States | Gate 5 mislabeled as "throughput non-regression" and feasibility states used ad-hoc names | `cpp/adaptive_hnsw.hpp:3216-3240`, `cpp/auto_calibrator.hpp:42-58`, `phase_6_4_dynamic_maintenance_investigation.json` | Restored Gate 5 to **Graph Invariants** (`RejectedGraphInvariantViolation`); restored 6 AutoCalibrator feasibility states to `VERIFIED_FEASIBLE`, `PROVISIONALLY_FEASIBLE`, `VALIDATION_FAILED`, `INFEASIBLE_WITHIN_SEARCH_SPACE`, `INVALID_REQUIREMENTS`, `EVALUATION_ERROR`. |
| **ISSUE-03** | Conservative SLA & 10% Churn Audit | 10% churn serving index labeled "SLA Satisfied" despite bootstrap $\text{CI}_{\text{lower}} = 0.9794 < 0.9800$ | `experiments/phase_6_4_dynamic_maintenance_investigation.json:35-43` | Reclassified 10% churn serving index as `Retained_Serving` (mean $R = 0.9840 \ge 0.9800$ satisfies point SLA, but $\text{CI}_{\text{lower}} < 0.9800$ prevents conservative certification); candidate rejected on Gate 2, Gate 3, Gate 6, triggering `COMPACTION_DEFERRED`. |
| **ISSUE-04** | AutoCalibrator & Transformer-384D Provenance | Provenance of 0.9416 recall and 3.38 MB memory not explicitly traced; Transformer-384D search beam stated as $efS=64$ | `experiments/phase_6_4_exact_calibrator_regression.json`, `experiments/phase_6_4_final_report.md:145-164` | Traced 0.9416 recall and 3.38 MB (3.3787 MB) to SIFT-10K deployment audit; documented recalibrated $efSearch=96$ for Transformer-384D (Recall 0.9720, Owned RAM 56.61 MB, P99 7.38 ms). |
| **ISSUE-05** | Mutation Replay Synchronization Pause | Manuscript claimed $\le 0.194$ ms pause for 350 mutations, conflating pointer-only swap with full re-indexing synchronization | `docs/phase_6_4_dynamic_maintenance_investigation_report.md:122`, `cpp/test_concurrent_mutation_preservation.cpp` | Corrected complete exclusive lock synchronization pause to **11.68 ms** for 350 replayed mutations (including graph re-indexing of 200 vector insertions); clarified pointer swap alone takes $\approx 0.014$ ms ($0.9$ to $193.9\ \mu\text{s}$). |
| **ISSUE-06** | SQ4 Dataset Attribution | 88.10% recall for Pure SQ4 mistakenly attributed to Transformer-384D in Table VI | `experiments/phase_6_4_fresh_precision_memory.json:84` | Attributed 88.10% recall for Pure SQ4 specifically to **SIFT-100K**; noted SQ4 was not evaluated on Transformer-384D. |
| **ISSUE-07** | Bibliographic Metadata | DATE 2025 authors and title hallucinated in draft BibTeX; FreshDiskANN authors truncated | Crossref DOI query `10.23919/DATE64628.2025.10992903`, PVLDB 2023 metadata | Updated `jin2025efficient` to true authors: Huijun Jin, Jieun Lee, Shengmin Piao, Sangmin Seo, Sein Kwon, and Sanghyun Park (DATE 2025, pp. 1-7); updated `gollapudi2023freshdiskann` to full VLDB 2023 author list. |
| **ISSUE-08** | De-escalation of Unverified Claims | Draft claimed guaranteed logarithmic navigation, guaranteed zero lost updates under arbitrary concurrency, hardware PMU cache misses | Manuscript text (§IV, §V, §VI, §XII) | Removed "guaranteed logarithmic navigation"; qualified zero lost updates to the evaluated multi-threaded test; framed cache misses as an analytical model; eliminated universal SLA safety claims. |
| **ISSUE-09** | Plotted Data & Baseline Memory Fairness | Figure 3 and Table II lacked clear separation between graph RAM, vector RAM, and process working set | `experiments/phase_6_4_external_comparisons.json:4-8`, `phase_6_4_fresh_precision_memory.json` | Explicitly separated graph-only memory (25.70 MB), vector memory, total index-owned memory (41.85 MB SQ8 vs 64.83 MB official hnswlib), and process working set (56.70 MB SQ8 vs 93.42 MB FP32). |
| **ISSUE-10** | Layout, Typography & Pagination | Draft artificially compressed layout into 6 pages, crowding equations and tables | `research/paper/build_paper_pdf.py` | Expanded to a clean, well-spaced 7-page academic preprint layout with uncrowded equations, clear booktabs tables, and legible vector SVG figures. |

---

## Detailed Section-by-Section Resolutions

### 1. Mathematical Rigor & Budget Allocation (Section IV)
- **C++ Scoring Alignment:** The scoring formula in Equation (5) is $S_i = \alpha_{\text{lid}} \tilde{\text{ID}}_i + \beta_{\text{spars}} \tilde{\text{Sparsity}}_i$, matching `cpp/budget_allocator.hpp:58`.
- **Algorithm 1:** Displays exact floor initialization $m_i \gets \max(M_{\min}, \min(M_{\max}, \lfloor \tilde{M}_i \rfloor))$, deficit calculation $\text{deficit} = B - \sum m_i$, sorting of unclipped sets $\mathcal{U}^+$ / $\mathcal{U}^-$ by fractional remainder $f_i$, and single-pass $+1$ (or $-1$) modification.
- **Proposition 1 Statement & Proof:** Proves that exact conservation holds if and only if $\text{deficit} \le |\mathcal{U}^+|$ (or $|\text{deficit}| \le |\mathcal{U}^-|$ for surplus). Explicitly notes that if $\text{deficit} > |\mathcal{U}^+|$, absorbing the deficit would require multiple increments per eligible vertex, which the single-pass code asserts against. Confirms that in all benchmark runs (SIFT-10K, SIFT-100K, SIFT-500K, Transformer-384D), $\text{deficit} \le |\mathcal{U}^+|$ strictly held.

### 2. Six Acceptance Gates & Six Feasibility States (Section VI & VII)
- **Six Gates:**
  1. $R_{\text{cand}} \ge R_{\text{target}}$ (Recall threshold)
  2. $\text{CI}_{\text{lower}} \ge R_{\text{target}}$ (Bootstrap statistical safety)
  3. $\text{PeakMemory} = M_{\text{serving}} + M_{\text{candidate}} \le M_{\text{budget}}$ (Transient dual residency)
  4. $P99_{\text{cand}} \le P99_{\text{target}}$ (Tail latency SLA)
  5. Structural Graph Invariants (valid enter-point, no isolated active nodes, consistent layers)
  6. $\Delta R_{\text{compact}} \le 0.0020$ (Compaction quality preservation)
- **Six AutoCalibrator States:**
  1. `VERIFIED_FEASIBLE`
  2. `PROVISIONALLY_FEASIBLE`
  3. `VALIDATION_FAILED`
  4. `INFEASIBLE_WITHIN_SEARCH_SPACE`
  5. `INVALID_REQUIREMENTS`
  6. `EVALUATION_ERROR`

### 3. Concurrency Synchronization Timing (Section VI & XII)
- Replaced the $0.194$ ms claim for 350 mutations with the actual measured **11.68 ms** complete exclusive lock pause (`docs/phase_6_4_dynamic_maintenance_investigation_report.md` line 122).
- Explained the difference between the complete synchronization pause (11.68 ms, which includes re-indexing 200 vector insertions into the candidate graph while holding the write lock) and the pointer swap in isolation ($\approx 0.014$ ms).

### 4. Bibliographic Authenticity (Section II & References)
- Corrected Jin et al. (DATE 2025): Huijun Jin, Jieun Lee, Shengmin Piao, Sangmin Seo, Sein Kwon, and Sanghyun Park, *"Efficient Approximate Nearest Neighbor Search via Data-Adaptive Parameter Adjustment in Hierarchical Navigable Small Graphs,"* DATE 2025, pp. 1–7 (DOI: 10.23919/DATE64628.2025.10992903).
- Corrected FreshDiskANN: Siddharth Gollapudi, Neel Karia, Varun Sivashankar, Ravishankar Krishnaswamy, Nikit Soni, and Suhas Jayaram Subramanya, *"FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Updates,"* PVLDB 2023, vol. 16, no. 11, pp. 2840–2852.
- Verified all 20 citations.
