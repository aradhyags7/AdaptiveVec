# AdaptiveVec — Academic Defense & Peer-Review Examination Dossier

**Document Status:** Research Consolidation & Scientific Evidence Freeze  
**Git Checkpoint:** `7936c61` (`phase-6.4-final-correctness-closure`)  
**Target Audience:** Senior Program Committee (SIGMOD / VLDB / ICDE / NeurIPS / ICLR), Academic Defense Committee, and Rigorous Systems Reviewers.

---

## Part I: High-Dimensional Geometry & Degree Allocation (Milestone I)

### Question 1: In Table 1, your Reversed Geometry Control achieves 0.9948 recall, which is higher than your proposed Normalized Exponential allocation (0.9926). Doesn't this invalidate your primary claim that geometry awareness improves vector search?
**Defense Answer:**  
No, because vector search optimization is fundamentally a multi-objective trade-off between retrieval accuracy, search velocity (QPS), and tail latency per distance evaluation.  
When we analyze the operational profile of the Reversed Geometry Control:
1. Allocating maximum degree ($M=24$) to sparse peripheral nodes creates a dense boundary mesh. This boundary over-connectivity increases the probability of capturing peripheral outlier vectors, yielding a modest +0.22 pp recall gain.
2. However, traversing this dense boundary mesh requires **+5.63% more distance evaluations** (1,069 vs 1,012) and drops query throughput by **-5.10%** (2,933.5 vs 3,091.2 QPS), while worsening P95 tail latency by **+7.20%** (0.521 vs 0.486 ms).
3. In contrast, Normalized Exponential allocation concentrates degree on central hub nodes where graph traffic concentrates. This accelerates greedy routing through high-density clusters, delivering **+24.01% higher QPS** and **-37.62% lower P95 latency** over uniform baselines under strictly conserved edge budgets.  
Therefore, geometry-aware allocation is an **efficiency and throughput optimizer per distance evaluation**, rather than a brute-force recall maximizer.

### Question 2: Why do you decouple the degree allocation between Layer 0 and upper layers ($\beta_{\text{Layer 0}} = 1.0, \beta_{\text{Upper}} = 0.0$)? What happens if geometry modulation is applied globally across all layers?
**Defense Answer:**  
We evaluated global geometric modulation across all layers in Phase 5.2 (documented as Negative Finding NEG-04).  
In an HNSW graph, upper skip layers ($l \ge 1$) serve as long-range geometric expressways designed to navigate across distant clusters with logarithmic hop complexity. When out-degrees in upper layers are modulated by local intrinsic dimensionality:
1. Nodes in sparse peripheral clusters receive truncated degrees ($M \to M_{\min} = 8$), severing long-range skip edges across peripheral regions.
2. Upper-layer graph navigability collapses: greedy routing from the entry point is unable to cross cluster boundaries, forcing premature descent into Layer 0 far from the true nearest-neighbor neighborhood.
3. This resulted in an empirical recall degradation of -1.8 pp and a 14% latency increase.  
By enforcing $\beta_{\text{upper}} = 0.0$ (uniform $M_{\text{base}}$ out-degree on all upper layers) while allocating geometrically on Layer 0 ($\beta_0 = 1.0$), AdaptiveVec preserves global highway routing while maximizing local cluster resolution.

### Question 3: Why did you select the Hare-Niemeyer method for degree discretization rather than continuous relaxation or randomized rounding?
**Defense Answer:**  
In resource-constrained indexing, the total edge memory budget must satisfy a strict conservation constraint:
$$\sum_{i=1}^N M_i = N \cdot M_{\text{base}}$$
Continuous degrees are unexecutable because adjacency lists require integer allocations. Naive rounding ($\lfloor \cdot \rceil$) can violate the total budget by $O(N)$ edges due to accumulated rounding bias. Randomized rounding introduces non-deterministic graph topologies that undermine reproducible SLA guarantees.  
The Hare-Niemeyer method (the largest remainder method) guarantees three critical properties:
1. **Strict Conservation:** The integer sum $\sum M_i$ exactly matches $N \cdot M_{\text{base}}$ with zero edge variance.
2. **Quota Rule Satisfaction:** Every node's integer degree satisfies $\lfloor w_i \rfloor \le M_i \le \lceil w_i \rceil$.
3. **Determinism:** Given fixed vector coordinates and metric seeds, degree assignment is perfectly deterministic and reproducible.

### Question 4: What is the computational complexity of estimating Local Intrinsic Dimensionality (LID), and does it create a prohibitive index construction bottleneck?
**Defense Answer:**  
We utilize the two-nearest-neighbor (Two-NN) estimator (Facco et al., 2017) and Levina-Bickel maximum likelihood estimator:
$$\widehat{\text{ID}}_i = \left( \frac{1}{k_1 - 1} \sum_{j=1}^{k_1 - 1} \ln \frac{r_i(k_1)}{r_i(j)} \right)^{-1}$$
Crucially, AdaptiveVec does not execute an independent global $k$-NN search to compute LID. Instead, LID estimation is integrated directly into the initial entry-point construction and coarse neighborhood sampling over a small local sample ($k_1 = 16$).  
The computational overhead of LID estimation accounts for **$< 3.2\%$ of total index build time** (0.23 seconds out of 7.24 seconds on SIFT-100K). This amortized one-time construction cost is completely offset by the persistent +24.01% query-time throughput advantage.

---

## Part II: Vector Compression, SIMD, and Memory Accounting (Milestone II)

### Question 5: Why does Pure SQ4 suffer an 11 pp recall collapse on SIFT-100K (88.10%) while achieving 99.74% on SIFT-10K?
**Defense Answer:**  
This discrepancy highlights an important structural property of high-dimensional vector spaces and sample density:
1. On SIFT-10K, the 10,000 vectors span a relatively sparse subspace where cluster separations are large relative to quantization noise. Coarse 4-bit uniform quantization (16 discrete bins per dimension) provides sufficient angular resolution to distinguish the true $k$-NN candidates.
2. On SIFT-100K (and similarly on Transformer-384D), the local vector manifold is densely packed. The distance between true 1-nearest-neighbors and 50-nearest-neighbors is tiny (high distance concentration). Coarse 4-bit bins induce severe clipping and quantization rounding error, causing different vectors to map to identical quantized codes.
3. In contrast, Pure SQ8 (256 discrete bins) preserves sufficient precision, maintaining 97.70% recall on SIFT-100K and 97.20% on Transformer-384D. For this reason, AdaptiveVec certifies Pure SQ8 as the default production configuration.

### Question 6: Your paper reports both "Index-Owned Memory" (41.85 MB for SQ8) and "Process Working Set" (56.70 MB). Which number represents the real memory footprint?
**Defense Answer:**  
Both numbers represent real, rigorous physical measurements, and distinguishing between them is essential for systems research integrity:
1. **Index-Owned Memory (41.85 MB):** This is the precise byte count of heap memory allocated and managed by the C++ data structures:
   $$\text{Allocated Quantized Vectors (12.21 MB)} + \text{Graph Adjacency (25.70 MB)} + \text{Metadata (3.91 MB)} = 41.85\text{ MB}$$
2. **Process Working Set (56.70 MB):** This is the physical Resident Set Size (RSS) reported by the Windows NT kernel via PSAPI `GetProcessMemoryInfo`. It includes the 41.85 MB index, plus the C++ runtime loader, binary text segment, CRT heap metadata headers, OpenMP thread execution stacks, and page alignment padding.  
Systems papers that report only process RSS conflate index efficiency with runtime environment overhead. Systems papers that report only vector buffer size ignore graph adjacency and allocator overhead. AdaptiveVec transparently reports both.

### Question 7: Why does two-stage retrieval with FP32 raw vector re-ranking collapse query throughput by ~7x (5,036 vs 47,576 QPS)?
**Defense Answer:**  
The throughput collapse is caused by CPU memory hierarchy mechanics, specifically L3 cache line thrashing:
1. During coarse SQ8 search, 128-byte quantized vectors are compact and traversed along localized graph paths, maximizing L1/L2 data cache hit rates.
2. In the second stage, re-ranking evaluates 64 to 128 candidate vectors against raw FP32 storage. Because candidate IDs are scattered throughout the index, fetching 512 bytes (128 floats $\times 4$ bytes) per candidate results in random, non-sequential memory dereferences.
3. This causes frequent L3 cache misses, forcing the execution pipeline to stall while waiting for main memory DRAM bus transfers.  
Consequently, Pure SQ8 with `drop_raw_vectors()` is significantly faster than re-ranking while preserving high recall.

### Question 8: Are your distance evaluation and query latency counters derived from hardware performance counters (PMU) or software instrumentation?
**Defense Answer:**  
They are derived strictly from software instrumentation counters encapsulated in `QuerySearchStats`. We do not read hardware PMU counters (e.g., via `rdpmc` or `perf_event_open`). Distance evaluations are incremented in the inner search loop, and latencies are measured using `std::chrono::high_resolution_clock`. We explicitly document this in the manuscript to ensure reviewers do not mistake software counters for hardware PMU events.

---

## Part III: Dynamic Graph Maintenance & Compaction (Milestone III)

### Question 9: Why did local graph healing in early phases cause recall degradation, and how did Soft Transit Bridge routing solve it?
**Defense Answer:**  
In naive tombstone deletion, when a node $v$ is marked deleted, its incoming edges are severed or greedily rewired to arbitrary surviving neighbors. This created two defects:
1. Premature edge removal severed shortest-path routing bridges across clusters before replacement edges could be formed.
2. Greedy rewiring without distance diversity created redundant cliques that trapped search routing.  
We solved this with **Soft Transit Bridge Routing** and **Bounded 2-Hop Bypass Shortcuts**:
- Deleted nodes remain routable as soft transit bridges during online queries: greedy routing may traverse through a deleted node to reach active nodes, but the deleted node is omitted from the top-$k$ result queue.
- Background healing inspects the 2-hop neighborhood of the deleted node and establishes diversity-selected bypass edges using Strict RNG heuristics, strictly avoiding redundant clustering.  
This eliminated pre-compaction recall drops, delivering a **+0.04 to +0.06 pp recall lift** over unhealed tombstones.

### Question 10: In Table 3, under 10% churn, the compacted candidate achieves 0.9820 recall, but the policy action is COMPACTION_DEFERRED. Why defer compaction if recall is good?
**Defense Answer:**  
Compaction is deferred because of a hard transient peak memory violation (Gate 3):
1. In a production vector search service, physical index compaction must be non-destructive: the serving index must remain active and memory-resident to serve live traffic while the compacted index is validated offline.
2. At 10% churn, serving index memory is 74.74 MB, and compacted candidate memory is 69.24 MB. The transient peak memory during validation is:
   $$\text{PeakMemory} = 74.74\text{ MB} + 69.24\text{ MB} = 143.98\text{ MB}$$
   This exceeds the operator's strict 120.0 MB memory budget by 23.98 MB.
3. Swapping pointers under this condition would trigger an Out-Of-Memory (OOM) kill in a containerized environment with strict cgroup limits.
4. Crucially, the serving healed index is already delivering **0.9840 recall** (exceeding the 0.9800 SLA) with only 74.74 MB steady memory. Deferring compaction preserves SLA compliance while avoiding an OOM crash.

### Question 11: At 50% churn, why does the policy require a full background rebuild instead of accepting the compacted candidate?
**Defense Answer:**  
At 50% churn, the compacted candidate fails both Gate 1 ($R=0.9740 < 0.9800$) and Gate 6 ($\Delta R = 0.0052 > 0.0020$).  
When half of the index nodes are deleted, neighborhood stitching cannot recover the global navigable properties of the graph manifold. In contrast, building a fresh HNSW graph over the 50,000 surviving vectors:
1. Takes only **3.95 seconds** of background CPU time.
2. Achieves **0.9878 recall**, easily exceeding the 0.9800 target.
3. Uses only 38.11 MB steady-state memory and 113.00 MB peak replacement memory, fully satisfying the 120.0 MB memory budget (Gate 3).  
The SLA-Safe Replacement Policy correctly identifies that compaction is inferior to a fresh rebuild under high churn regimes and automatically executes `REBUILD_REQUIRED`.

### Question 12: How does ServingIndexManager guarantee zero lost mutations during background candidate construction?
**Defense Answer:**  
`ServingIndexManager` implements sequence-numbered append-only mutation journaling:
1. When background candidate construction begins at logical timestamp $T_{\text{snapshot}}$, an atomic sequence counter records the baseline state.
2. While the background thread compacts or rebuilds the candidate, all live insertions and deletions continue to be applied to the serving index and appended to the mutation journal with monotonically increasing sequence numbers: $\langle \text{seq}_k, \text{op}, \text{id}, \mathbf{v} \rangle$.
3. When candidate validation passes all 6 gates, the manager acquires a brief reader lock ($\le 0.194$ ms reader pause) and replays all journaled mutations with $\text{seq} > T_{\text{snapshot}}$ onto the candidate index.
4. Once catch-up replay completes, the candidate pointer is atomically swapped (`std::atomic_store`), and the superseded index is queued for deferred destruction after active reader references reach zero.  
In deterministic stress testing with 350 mutations across 12,494 concurrent queries, zero mutations were dropped, and vector coordinates matched the authoritative sequence with zero error.

---

## Part IV: Constrained Optimization & AutoCalibration (Milestone IV)

### Question 13: What are the 6 formal feasibility states in AutoCalibrator, and why is this formalization necessary?
**Defense Answer:**  
The 6 formal states provide deterministic, auditable guarantees against false feasibility claims:
1. `VERIFIED_FEASIBLE`: Point-estimate metrics meet all SLA constraints, and the 1,000-sample bootstrap confidence interval lower bound satisfies the recall requirement ($\text{CI}_{\text{lower}} \ge R_{\text{target}}$).
2. `PROVISIONALLY_FEASIBLE`: Point estimate meets SLA, but the bootstrap lower bound falls below threshold ($\text{CI}_{\text{lower}} < R_{\text{target}}$).
3. `VALIDATION_FAILED`: Calibration passed on the training slice, but held-out validation on unseen queries failed.
4. `INFEASIBLE_WITHIN_SEARCH_SPACE`: No configuration in the hyperparameter space can satisfy the constraints (e.g., impossible SLAs rejected in 0.05 s).
5. `INVALID_REQUIREMENTS`: Contradictory or unphysical input constraints (e.g., negative latency or memory).
6. `EVALUATION_ERROR`: Runtime execution failure during candidate evaluation.  
This formalization is necessary because commercial vector search deployments operate under contractual SLAs where a false-positive feasibility report leads to production outages.

### Question 14: In your historical regression test, why did index memory change from 3.69 MB to 3.38 MB between calibration and deployment?
**Defense Answer:**  
The 3.69 MB figure was the conservative calibration-phase estimate computed during hyperparameter search, which included scratch candidate evaluation buffers. The 3.38 MB figure is the actual, verified index-owned memory footprint of the deployed index in steady state ($N=10{,}000, M=8, D=128$, SQ8). Both numbers strictly satisfy the user's $M \le 4.5$ MB constraint.

### Question 15: Why did you discard dynamic query-time parameter adaptation (Ada-ef / PID controllers) in favor of index-time optimization?
**Defense Answer:**  
In Phases 4.1–4.3, we explored query-time feedback controllers that dynamically adjusted $efSearch$ based on entry-point distance and candidate convergence. We discarded them (documented as Negative Finding NEG-01) for three reasons:
1. **Thread Contention:** Updating runtime controller states across concurrent reader threads required atomic locks or thread-local storage, degrading multi-threaded scaling.
2. **Tail Latency Inflation:** Unpredictable controller exploration increased P99 tail latency by 35–48%.
3. **Statistical Unreliability:** Early termination heuristics frequently terminated searches prematurely in dense clusters, causing unpredictable recall drops.  
Moving all adaptation to index construction and calibration provides deterministic, reproducible latency and recall profiles without query-path overhead.

---

## Part V: Systems Engineering, Baselines, and Scalability

### Question 16: Official hnswlib achieves 13,434 QPS on SIFT-100K compared to AdaptiveVec's 3,091 QPS. Doesn't this make AdaptiveVec impractical?
**Defense Answer:**  
No, because `hnswlib` and `AdaptiveVec` are engineered for fundamentally different operational paradigms:
1. `hnswlib` is an immutable, static C-struct implementation optimized purely for read-only microbenchmarks. It lacks per-node reader-writer locks, mutation journals, soft tombstones, dynamic memory reclamation buffers, and SLA-safe replacement managers.
2. `AdaptiveVec` is a fully dynamic, self-healing vector database engine supporting continuous multi-threaded ingestion, non-destructive compaction, dynamic vector deletion, and multi-objective SLA autocalibration.
3. The synchronization primitives and memory safety structures required to support online mutability account for the execution overhead. When comparing strictly against mutable, thread-safe graph systems, AdaptiveVec's throughput is highly competitive while delivering unique self-healing and SLA-guaranteed capabilities.

### Question 17: How does AdaptiveVec compare to out-of-core graph systems like DiskANN?
**Defense Answer:**  
AdaptiveVec is designed for high-throughput, in-memory low-latency vector search (sub-millisecond P99 latency), whereas DiskANN is architected for billion-scale out-of-core SSD retrieval where graph edges reside on NVMe drives.  
However, AdaptiveVec's core algorithmic contributions—specifically Normalized Exponential degree allocation and Multi-Layer Strict RNG compaction—are directly applicable to out-of-core systems:
- Compressing graph degrees in sparse peripheral regions would reduce SSD sector reads per query in DiskANN.
- DiskANN's dynamic consolidation pipeline (FreshDiskANN) suffers from the same upper-layer hierarchy degradation that AdaptiveVec solves via Multi-Layer replenishment.

### Question 18: What is the maximum dataset scale verified in this work, and how does the system scale asymptotically?
**Defense Answer:**  
The maximum dataset verified in this work is SIFT-500K ($N=500{,}000, D=128$), with end-to-end multi-milestone integration validated on the 57,893-vector Transformer-384D corpus.  
Asymptotically:
- Search time scales logarithmically: $O(\log N \cdot efSearch \cdot D)$.
- Construction time scales as $O(N \log N \cdot efConstruction \cdot M_{\text{base}} \cdot D)$.
- Memory scales strictly linearly: $O(N \cdot (D_{\text{quant}} + M_{\text{base}} \cdot \text{sizeof}(\text{node\_id})))$.
Because degree allocation is normalized ($\sum M_i = N \cdot M_{\text{base}}$), geometry awareness adds zero asymptotic memory overhead.

### Question 19: Why was AVX2 selected instead of AVX-512 or ARM NEON?
**Defense Answer:**  
AVX2 was selected to ensure reproducible, deterministic benchmarking on standard commodity hardware (Intel Core 5 210H) without thermal throttling. Early AVX-512 implementations suffer from frequency downclocking (license throttling) under mixed integer/floating-point workloads. All SIMD routines in AdaptiveVec are modularly abstracted, allowing drop-in implementation of AVX-512 (`_mm512_dpbusd_epi32`) and ARM NEON (`vdotq_u32`) kernels.

### Question 20: How can an independent reviewer reproduce your experimental findings?
**Defense Answer:**  
The entire research package at checkpoint `7936c61` is deterministic and self-contained:
1. Every experimental claim is mapped to an exact raw JSON artifact in `research/paper/evidence_map.md`.
2. Reproduction binaries can be compiled with standard GCC (`g++ -O3 -mavx2 -mfma -fopenmp -std=c++17`).
3. Running `.\cpp\phase6_1_evidence_runner.exe` reproduces all Milestone I and II claims.
4. Running `.\cpp\benchmark_dynamic_maintenance.exe` reproduces all Milestone III dynamic maintenance and compaction gate outcomes.
5. Running `.\cpp\phase6_4_runner.exe` executes the 6-state deterministic AutoCalibrator test suite.
