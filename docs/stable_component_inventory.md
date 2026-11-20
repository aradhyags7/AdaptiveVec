# AdaptiveVec: Stable Component Inventory & Architectural Registry

This document establishes the official component classification for the AdaptiveVec codebase, categorizing mechanisms into **Core Permanent Systems Components**, **Restored Geometry-Aware Indexing Components**, and **Permanently Archived Experimental Modules**.

---

## 1. Classification Taxonomy

```
+-----------------------------------------------------------------------------------+
|                        AdaptiveVec Component Inventory                            |
+-----------------------------------------------------------------------------------+
          |                                   |                                |
          v                                   v                                v
+----------------------+           +----------------------+         +-----------------------+
| Core Permanent Core  |           | Restored Geometry    |         | Permanently Archived  |
| (Retained from P0-4) |           | (Phase 5.0 Active)   |         | (Deactivated from Hot)|
+----------------------+           +----------------------+         +-----------------------+
| - AVX2 / FMA Kernels |           | - MLE LID Estimator  |         | - Workload Controller |
| - VisitedTagPool     |           | - Local Density Stat |         | - Reference Monitor   |
| - Contiguous Memory  |           | - Online Welford     |         | - Online Uncertainty  |
| - QuerySearchStats   |           | - Dynamic M(x), efC  |         | - Resumable Search    |
| - Striped Mutexes    |           | - Layer Scaling      |         | - Bridge Hubness      |
| - Exact Ground Truth |           | - In-Degree Reg.     |         | - Mixed Precision     |
| - Memory Accounting  |           | - Deletion & Healing |         | - Graph Rewiring      |
+----------------------+           +----------------------+         +-----------------------+
```

---

## 2. Category A: Core Permanent Systems Components (Retained)

These mechanisms were developed and validated during Phases 0–4.3 and provide fundamental systems-level correctness, performance, and determinism. They are permanently integrated into all AdaptiveVec variants.

| Component | Source Location | Implementation Summary | Validation Evidence |
|:---|:---|:---|:---|
| **AVX2 / FMA Kernels** | `cpp/adaptive_hnsw.hpp` | Hand-vectorized 256-bit SIMD inner products and Euclidean metrics. Supports unsquared metric distance (for LID) and squared distance (for graph traversal). | Zero compiler regressions across GCC/Clang/MSVC; $4\times$ throughput speedup over scalar loops. |
| **Generation-Tagged `VisitedTagPool`** | `cpp/adaptive_hnsw.hpp` | Replaces heap-allocated `std::unordered_set` with a pre-allocated flat buffer tagged with an incrementing 32-bit generation integer. | Eliminates $\sim 10,000$ heap allocations per batch; accelerates search throughput by $3.8\times$. |
| **Contiguous Vector Storage** | `cpp/adaptive_hnsw.hpp` | Flat single-buffer allocation (`std::vector<float> data`) storing $N \times D$ coordinates with 64-byte hardware cache alignment. | Maximizes CPU L1/L2 prefetching efficiency; eliminates pointer chasing. |
| **Thread-Local Instrumentation (`QuerySearchStats`)** | `cpp/adaptive_hnsw.hpp` | Stack-allocated per-query statistics structure recording exact upper-layer, layer-0, and rerank distance evaluations. | Eliminates atomic contention across 12 OpenMP threads; restores multi-core scaling. |
| **Fine-Grained Mutex Striping** | `cpp/adaptive_hnsw.hpp` | Array of 4,096 striped mutexes (`node_locks[id % 4096]`) governing node-level adjacency updates during concurrent insertion and deletion. | Prevents circular lock deadlocks while maintaining multi-threaded insertion scalability. |
| **Deterministic Seeding** | `cpp/adaptive_hnsw.hpp` | Explicit random generator seeding for level assignment and graph construction. | $100\%$ bitwise reproducible graph topology under fixed seeds. |
| **Independent Ground Truth Generator** | `cpp/groundtruth_generator.hpp` | OpenMP parallelized AVX2 brute-force matrix multiplication for exact $L_2$ and Cosine nearest neighbor verification. | Verified against external references (FAISS Flat); immune to query transformation drift. |
| **Audited Memory Accounting** | `cpp/adaptive_hnsw.hpp` | Multi-tier memory accounting separating logical vector payload, allocated vector buffer capacity, graph adjacency capacity, and process resident bytes. | Closes the historical memory discrepancy identified in Phase 0.5. |

---

## 3. Category B: Restored Geometry-Aware Indexing Components (Phase 5.0 Active)

These mechanisms embody AdaptiveVec's original research hypothesis and are restored in **Architecture B** for systematic empirical evaluation against uniform baselines.

| Component | Formula / Mechanism | Implementation Status in Architecture B |
|:---|:---|:---|
| **MLE LID Estimator** | $\widehat{\text{LID}}(x) = -[(1/(k-1)) \sum \ln(d_i / r_k)]^{-1}$ | Active during insertion probing. Uses true metric Euclidean distance; numerical clamping prevents singularity. |
| **Local Density Estimator** | $D(x) = \frac{1}{k} \sum d(x, v_i)$ | Active during insertion probing. Computes mean neighbor distance; larger value indicates sparser manifold. |
| **Streaming Welford Tracking** | $\mathcal{O}(1)$ update of running mean and variance | Active during ingestion. Continuously standardizes $z_{\text{LID}}$ and $z_D$ without offline pre-clustering. |
| **Dynamic Connectivity $M(x)$** | $M(x) = \text{clip}(\lfloor M_{\text{base}}(1 + \gamma S(x))\rceil, M_{\min}, M_{\max})$ | Active during graph construction. Allocates $M \in [8, 24]$ based on geometric complexity. |
| **Dynamic Exploration $efC(x)$** | $efC(x) = \text{clip}(\lfloor efC_{\text{base}}(1 + \gamma S(x))\rceil, efC_{\min}, efC_{\max})$ | Active during graph construction. Expands construction beam in sparse/complex regions. |
| **Layer-Decoupled Scaling** | $M^{(l)}(x) = \max(M_{\min}^{(l)}, \lfloor M(x) \cdot \lambda^l \rfloor)$ | Active across levels $l \ge 1$. Compresses coarse routing expressways ($\lambda = 0.75$). |
| **In-Degree Hubness Regulation** | $d_{\text{eff}} = d \cdot [1 + \mu \frac{\text{deg}_{\text{in}}(v)}{\bar{\text{deg}}_{\text{in}}}]$ | Active during Layer-0 neighbor selection heuristic ($\mu = 0.15$). Dampens bottleneck vertices. |
| **Two-Tier Deletion & Healing** | Soft transit bridge + predecessor reconnection + mutual neighbor bridging | Active for dynamic lifecycle management. Preserves reachability under continuous streaming churn. |
| **Online Zero-Downtime Compaction** | `compact()`: Contiguous memory defragmentation and ID remapping | Active. Purges tombstones and restores peak cache locality in $<65\ \text{ms}$. |
| **Uniform SQ8 Quantization** | Per-dimension affine scalar quantization + FP32 reranking | Active for memory-constrained regimes. $4\times$ vector compression with $\ge 99\%$ recall retention. |
| **Fixed Stagnation Early Exit** | Truncate Layer-0 beam search if best distance stagnates for $p=8$ steps | Active during query search. Delivers $\approx 35\%$ distance evaluation savings at zero estimator overhead. |

---

## 4. Category C: Permanently Archived Experimental Modules (Deactivated)

These mechanisms were developed during exploratory research tracks (Phases 1–4.3) and have been demonstrated to introduce unacceptable overhead, instability, or negative Pareto trade-offs. They are **permanently removed from the standard execution path** and preserved only as experimental research artifacts under explicit compile-time toggles.

| Component | Origin Phase | Reason for Deactivation | Governance Policy |
|:---|:---:|:---|:---|
| **Workload Feedback Controller** | Phase 4.0 | Continuous query-rate and target-recall tracking introduced background thread synchronization and unpredictable latency spikes. | Permanently disabled by default; archived under `--enable-experimental-controller`. |
| **Query-Path Reference Monitor** | Phase 4.1 | Periodic sampling of reference queries inside the query execution path consumed up to $18\%$ of total CPU capacity. | Permanently removed from query hot path. |
| **Online Uncertainty / LID Predictor** | Phase 4.3 | Computing online LID and boundary margins per query took $321.3\ \mu\text{s}$, completely swamping the $15.3\ \mu\text{s}$ savings of avoiding 61 distance calculations. | Enforced by Research Stop Rule; archived. |
| **Resumable Search State Machine** | Phase 4.2 | Suspending and resuming priority queues across candidate beam expansions ($ef=32 \to 64$) introduced high queue-shuffling overhead. | Replaced by direct calibrated beam search. |
| **Bridge-Aware Hubness Heuristic** | Phase 1 | Calculating bridge importance scores required expensive graph traversals during updates and failed to outperform simple degree penalties. | Reverted in favor of original in-degree hubness regulation. |
| **Geometry-Aware Mixed-Precision** | Phase 2 | Mixing FP32, SQ8, and SQ4 across nodes within the same graph created severe numerical regressions and memory alignment penalties. | Replaced by verified uniform SQ8 quantization. |
| **Autonomous Graph Rewiring** | Out of Scope | Online edge mutation during query execution threatens thread safety and causes non-deterministic query latencies. | Explicitly prohibited by development boundaries. |
