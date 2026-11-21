# AdaptiveVec — Scientific Issues, Contradictions, and Peer-Review Defense Audit

**Document Status:** Research Consolidation & Scientific Evidence Freeze  
**Git Checkpoint:** `7936c61` (`phase-6.4-final-correctness-closure`)  
**Scope:** Exhaustive audit of all 10 critical engineering and scientific issues identified across Phase 6 development, detailing empirical contradictions, root-cause analyses, mathematical corrections, and peer-review defense strategies.

---

## Issue 1: Compaction Quality Collapse at 25% Churn

### 1. The Contradiction
In early Phase 6 benchmarks on SIFT-100K under 25% node churn, pre-compaction local healing maintained high search accuracy ($R=0.9752$), but physical compaction triggered a severe recall collapse to **$R=0.9280$** (a 4.72 pp degradation), grossly violating the target maximum compaction recall loss of $\Delta R \le 0.0020$.

### 2. Root Cause Analysis
Source code inspection of `cpp/adaptive_hnsw.hpp` revealed that the legacy compaction routine executed degree replenishment exclusively on Layer 0. On higher hierarchical skip layers ($l \ge 1$), edges referencing deleted nodes were pruned without replenishment. Over 25,000 deletions, upper-layer degrees collapsed toward zero, destroying long-range skip highways and trapping greedy routing in disconnected local subgraphs.

### 3. Engineering & Algorithmic Correction
Implemented **Multi-Layer Strict RNG Replenishment** in `compact()`:
- Applied degree replenishment across *all* levels $l \in [0, \text{max\_level}]$.
- When pruning an upper-layer edge to a deleted node, candidate replacement edges are evaluated from surviving 2-hop neighbors using Strict RNG heuristic selection.
- Enforced $\min \text{deg} \ge 8$ across active surviving nodes.

### 4. Empirical Resolution
On SIFT-100K at 25% churn, compacted candidate recall was recovered from **0.9280 to 0.9786** under fixed $efSearch=64$, and to **0.9865** under recalibrated $efSearch=96$ (`experiments/phase_6_4_dynamic_maintenance_investigation.json`), completely resolving the 0.0506 recall deficit.

---

## Issue 2: Compaction Degradation at 50% Churn and the Rebuild Threshold

### 1. The Contradiction
Under extreme 50% node churn on SIFT-100K, compacted candidates achieved $R=0.9740$, which is lower than the healed serving index ($R=0.9792$) and violates the user's hard SLA threshold ($R_{\text{target}} \ge 0.9800$), resulting in $\Delta R = 0.0052 > 0.0020$.

### 2. Root Cause Analysis
At 50% churn, half of all graph vertices are removed. Replenishing edges by stitching together survivors of mutilated 2-hop neighborhoods introduces significant path distortion compared to a graph built directly over the surviving manifold. Because the graph diameter expands and cluster bridges are fractured, greedy routing cannot achieve $\ge 0.9800$ recall without excessively expanding $efSearch$.

### 3. Engineering & Algorithmic Correction
Rather than forcing compaction replacement when it violates SLA requirements, AdaptiveVec introduced the **SLA-Safe Replacement Policy Matrix**:
- Compacted candidates are subjected to 6 hard offline acceptance gates.
- When Gate 1 ($R \ge R_{\text{target}}$) or Gate 6 ($\Delta R \le 0.0020$) fails, the candidate is **strictly rejected**.
- Under high churn, building a fresh HNSW index over 50,000 surviving vectors takes only **3.95 seconds** and yields **$R=0.9878$**, easily passing all 6 gates. The policy therefore triggers `REBUILD_REQUIRED`.

### 4. Empirical Resolution
At 50% churn, fresh background rebuild is certified as the mathematically and operationally superior maintenance path (`experiments/phase_6_4_dynamic_maintenance_investigation.json`), completing in 3.95 s vs 1.97 s for compaction while delivering +1.38 pp higher recall (0.9878 vs 0.9740).

---

## Issue 3: Transient Peak Memory Violation during Compaction Replacement

### 1. The Contradiction
Under a strict 120.0 MB memory budget on SIFT-100K at 10% churn, steady-state serving index memory is 74.74 MB, and compacted candidate memory is 69.24 MB. However, during validation prior to pointer replacement, measured peak memory reached **143.98 MB**, violating the 120.0 MB budget constraint.

### 2. Root Cause Analysis
Peak replacement memory follows the exact dual-residency formula:
$$\text{PeakMemory} = M_{\text{serving}} + M_{\text{candidate}} + M_{\text{scratch}}$$
$$\text{PeakMemory} = 74.74\text{ MB} + 69.24\text{ MB} + 0.00\text{ MB} = 143.98\text{ MB}$$
Because the serving index must remain fully operational in memory to handle concurrent user queries while the candidate index undergoes offline held-out validation, both indices coexist simultaneously in memory.

### 3. Engineering & Algorithmic Correction
- Formalized Gate 3 in the acceptance gatekeeper: $\text{PeakMemory} \le M_{\text{budget}}$.
- When transient peak memory exceeds the budget, the candidate replacement is **rejected**, and the system executes **Compaction Deferral**:
  - The serving index continues answering queries safely because it already meets the recall SLA ($R=0.9840 \ge 0.9800$).
  - Physical compaction is deferred until churn increases further (reclaiming more memory) or until an off-peak maintenance window allows external process rebuild.

### 4. Empirical Resolution
Verified in `ServingIndexManager` (`experiments/phase_6_4_dynamic_maintenance_investigation.json`). Under 10% churn, `COMPACTION_DEFERRED` is emitted, preventing out-of-memory (OOM) crashes in production while preserving SLA continuity.

---

## Issue 4: Memory Definition Ambiguity (Index-Owned vs. Process Working Set)

### 1. The Contradiction
Earlier reports quoted memory numbers varying from 41.85 MB to 93.42 MB to 311.0 MB for identical SIFT-100K datasets, creating confusion regarding actual RAM requirements.

### 2. Root Cause Analysis
Three distinct memory metrics were conflated:
1. **Index-Owned Memory:** Direct byte allocations owned by the C++ index structures (vector buffers, adjacency lists, node metadata).
2. **Process Private Committed Memory:** Virtual memory pages allocated by the CRT heap allocator. In sequential benchmark runs, freed heap blocks were retained by the C runtime pool rather than unmapped by the Windows kernel, causing artificial metric accumulation across trials (229 MB $\to$ 272 MB $\to$ 311 MB).
3. **Process Working Set (Resident Set Size):** Physical RAM pages mapped by the OS kernel, which includes dataset input arrays, evaluation queries, ground truth tables, OpenMP thread stacks, and OS DLLs.

### 3. Engineering & Algorithmic Correction
- Implemented fresh subprocess execution isolation: each benchmark runs in an independent OS process, completely eliminating CRT heap pooling artifacts.
- Fixed `reserve()` and `get_memory_breakdown()` to avoid double-counting SQ4 and SQ8 capacity buffers.
- Documented explicit two-tier memory reporting in all tables:
  $$\text{Index-Owned Memory} \quad \text{vs.} \quad \text{Process Working Set (PSAPI)}$$

### 4. Empirical Resolution
On SIFT-100K (`experiments/phase_6_4_fresh_precision_memory.json`):
- Pure FP32: Index-Owned = 78.47 MB; Process WS = 93.42 MB.
- Pure SQ8: Index-Owned = 41.85 MB; Process WS = 56.70 MB.
- Pure SQ4: Index-Owned = 35.75 MB; Process WS = 49.30 MB.

---

## Issue 5: AutoCalibrator Slack Multiplier False Feasibility

### 1. The Contradiction
In early Phase 6 versions, `AutoCalibrator` reported `VERIFIED_FEASIBLE` on Trial 1 for a strict recall requirement of $R \ge 0.92$, despite the trial achieving only $R=0.8476$.

### 2. Root Cause Analysis
The legacy decision logic incorporated artificial slack multipliers ($1.05\times$ or $1.08\times$ relaxed margins) on candidate scoring, allowing substandard configurations to pass feasibility tests. Furthermore, candidate tracking did not strictly partition feasible candidates from fallback candidates.

### 3. Engineering & Algorithmic Correction
- Eliminated all slack multipliers: strict mathematical inequalities are enforced ($R \ge R_{\text{target}}$).
- Decoupled candidate tracking into `best_feasible_trial` and `best_fallback_trial`.
- Introduced a 6-state formal feasibility model and integrated a 1,000-sample bootstrap confidence interval test requiring $\text{CI}_{\text{lower}} \ge R_{\text{target}}$.

### 4. Empirical Resolution
In `experiments/phase_6_4_exact_calibrator_regression.json`, Trial 1 is strictly rejected, and the deterministic test suite validates all 6 formal states with zero false feasibility.

---

## Issue 6: SIFT-100K 10K Slice Mismatch in Dynamic Maintenance Benchmarks

### 1. The Contradiction
Earlier Phase 6 dynamic maintenance tests reported results under the heading "SIFT-100K" but were executed on an initial 10,000-vector subset slice, misrepresenting large-scale performance.

### 2. Root Cause Analysis
A development convenience script (`run_suite2.exe`) truncated the input vector array to 10K to shorten benchmark cycle times during debugging, and this flag persisted into intermediate test scripts.

### 3. Engineering & Algorithmic Correction
- Purged all slice-shortened scripts.
- Re-executed full-corpus benchmarks strictly across all $N=100{,}000$ SIFT vectors with $k=10$ and authentic ground-truth re-computation over active surviving IDs.

### 4. Empirical Resolution
`experiments/phase_6_4_compaction_provenance.json` and `experiments/phase_6_4_dynamic_maintenance_investigation.json` confirm full $N=100{,}000$ execution, demonstrating 129–147 ms lock duration, 9.4–39.7 MB reclaimed RAM, and authentic 1.64x to 8.72x maintenance speedups over full rebuilds.

---

## Issue 7: Reversed Geometry Control and the Boundary Recall Inversion

### 1. The Contradiction
When testing the inverted control policy ($M=24$ for sparse peripheral nodes, $M=8$ for dense hub nodes), the reversed index achieved slightly higher recall (**0.9948 vs 0.9926, +0.22 pp**) than the standard Normalized Exponential index, seemingly contradicting the hypothesis that geometry awareness maximizes recall.

### 2. Root Cause Analysis
High-dimensional datasets exhibit hubness phenomena: queries frequently route through central hubs, while peripheral boundary nodes are isolated. Allocating $M=24$ to sparse nodes creates a dense peripheral boundary mesh. This improves candidate capture for boundary queries, lifting raw recall by 0.22 pp. However, traversing this boundary mesh increases distance evaluations by **+5.63%** (1,069 vs 1,012) and drops throughput by **-5.10%** (2,933.5 vs 3,091.2 QPS).

### 3. Scientific Implication and Peer-Review Defense
We explicitly state in the manuscript:
> "Geometry-aware allocation is an efficiency and throughput optimizer per distance evaluation, rather than a universal recall maximizer."

Allocating higher degree to hubs accelerates graph traversal and minimizes tail latency, which is the primary objective of modern vector search engines.

---

## Issue 8: FP32 Raw Vector Re-ranking Throughput Collapse

### 1. The Contradiction
Two-stage retrieval (quantized coarse routing followed by FP32 raw vector re-ranking) was hypothesized to provide the best of both worlds: low memory and exact FP32 recall. However, empirical benchmarking showed a severe **$\sim 7\times$ throughput collapse** (5,036 vs 47,576 QPS) on SIFT-10K.

### 2. Root Cause Analysis
During the coarse stage, compact quantized codes fit neatly into L1/L2 caches. In the re-ranking stage, candidate vectors are scattered across non-sequential memory addresses. Fetching raw 128-dimensional FP32 vectors (512 bytes per vector) for 64 or 128 candidates forces non-sequential L3 cache line fetches, saturating the CPU memory bus and stalling execution.

### 3. Engineering & Algorithmic Correction
- Pure SQ8 ADC with `drop_raw_vectors()` is established as the recommended production default.
- By purging raw FP32 vectors, SQ8 eliminates cache thrashing, achieving 12,437 QPS on SIFT-100K and 47,576 QPS on SIFT-10K with $\le 1.4\text{ pp}$ recall loss vs FP32.

---

## Issue 9: Streaming Edge Budget Drift

### 1. The Contradiction
In batch construction, Hare-Niemeyer discretization guarantees exact edge conservation ($\sum M_i = N \cdot M_{\text{base}}$). In continuous streaming ingestion, future vector distributions are unknown at insertion time, potentially causing cumulative edge budget drift.

### 2. Root Cause Analysis
Streaming nodes compute degrees based on moving-window LID estimates. Over large streaming horizons, slight asymmetries in local density estimates can lead to degree sum variance.

### 3. Empirical Quantification & Mitigation
Benchmarking in `experiments/phase_6_4_streaming_budget_audit.json` demonstrated that forward degree drift is tightly bounded within **$\le 0.43\%$** across 2,500 streaming insertions, preserving a +0.25 pp recall advantage over uniform controls under matched memory. For long-running streaming deployments, periodic background degree recalibration during scheduled compaction completely eliminates cumulative drift.

---

## Issue 10: Concurrency Safety during Dynamic Memory Growth

### 1. The Contradiction
Dynamic ingestion and vector resizing under concurrent multi-threaded read traffic can trigger race conditions and memory invalidation faults when vector buffers reallocate.

### 2. Root Cause Analysis
C++ `std::vector` reallocations invalidate raw pointers and iterators held by concurrent reader threads if resizing is unsynchronized.

### 3. Engineering Correction & Verification
- Implemented `ServingIndexManager` with reader-writer lifecycle synchronization (`index_lifecycle_rw_mutex`) and sequence-numbered append-only mutation journaling.
- Verified in `experiments/phase_6_4_concurrency_validation.json`:
  - 4 reader threads executed 19,598 queries while 2 writer threads concurrently inserted and deleted vectors.
  - Replayed 350 mutations across 12,494 queries with **zero segmentation faults**, **zero lost mutations**, and atomic pointer swap reader pause $\le 0.194$ ms.
