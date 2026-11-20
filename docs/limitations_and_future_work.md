# AdaptiveVec — Operational Limitations, Failure Modes, and Future Work

**Document Status:** Phase 6.2 Engineering Closure & Presentation Ready  
**Branch:** `phase-6.2-comprehensive-engineering-closure`  
**Parent Checkpoint:** `7f2a862` (Phase 6.1 Defense Freeze)  
**Target Milestone:** Comprehensive Correctness, Performance and Engineering Closure  

---

## 1. Executive Summary

Scientific maturity requires an unambiguous declaration of where an architecture succeeds, where it reaches its physical boundaries, under what conditions it degrades, and what architectural capabilities remain out of scope.

AdaptiveVec is an embedded, single-node C++ vector search library engineered for memory-efficient, dynamic retrieval on heterogeneous high-dimensional manifolds. It is not an enterprise distributed database, nor is it a universal silver bullet for all vector distributions.

---

## 2. Operational Boundaries and Failure Modes

### 2.1 Data Distribution Boundaries
1. **Uniform Hyperspherical Vector Spaces:**
   - *Condition:* Synthetic datasets uniformly sampled from a unit hypersphere $\mathbb{S}^{D-1}$ where local density is constant ($\rho(x_i) \approx \text{const}$) and local intrinsic dimensionality is homogeneous ($\widehat{\text{LID}}(x_i) \approx D$).
   - *Behavior:* The variance of geometric signals approaches zero ($\sigma_{\text{LID}} \to 0$). The normalized exponential budget allocation collapses to uniform allocation ($M_i = M_{\text{base}}$).
   - *Consequence:* AdaptiveVec incurs a negligible preprocessing cost ($20\text{ ms}$ per 100K vectors) with zero throughput advantage over canonical uniform HNSW.
2. **Extreme Intrinsic Dimensionality ($D_{\text{int}} > 512$):**
   - *Condition:* Data spaces where the curse of dimensionality is absolute and distances concentrate completely ($d_{\text{max}} / d_{\text{min}} \to 1$).
   - *Behavior:* Relative Neighborhood Graph pruning loses discriminative power because angle conditions degenerate.

### 2.2 Algorithmic Connectivity Boundaries
1. **Extreme Degree Skew ($M_{\text{max}} / M_{\text{min}} > 4$):**
   - *Condition:* Setting $M_{\text{min}} = 4$ and $M_{\text{max}} = 64$.
   - *Behavior:* Hub nodes become over-connected "gravity wells" where greedy search loops extensively evaluating dozens of local neighbors before escaping. Peripheral nodes become disconnected islands.
   - *Safe Envelope:* Degree modulation must be bounded within $[8, 24]$ for $M_{\text{base}} = 16$ (a maximum ratio of $3.0$).
2. **Upper-Layer Geometric Modulation:**
   - *Condition:* Applying geometric degree scaling to levels $l \ge 1$ ($\beta_{\text{upper}} > 0$).
   - *Behavior:* Distorts long-range highway routing and degrades overall navigability. Upper layers must remain strictly uniform ($\beta_{\text{upper}} = 0.0$).

### 2.3 Dynamic Churn Boundaries
1. **Catastrophic Batch Deletions ($>80\%$ Churn):**
   - *Condition:* More than 80% of index elements are deleted simultaneously.
   - *Behavior:* Iterative local healing evaluates millions of severed edges, taking more time than a full index rebuild from remaining active vectors.
   - *Operational Recommendation:* If deletion volume exceeds 80% of index cardinality in a single operation, trigger a full background index rebuild instead of iterative healing.
2. **Post-Compaction Skip-Layer Thinning:**
   - *Condition:* Under 50% churn, post-compaction recall drops slightly (0.9944 healed $\to$ 0.9884 post-compaction).
   - *Root Cause:* Compaction repacks surviving edges but does not construct new skip links on upper levels. Rebuilding from scratch achieves 0.9978 recall by generating fresh global skip highways.

---

## 3. Performance Tradeoffs

### 3.1 Raw Vector Re-ranking Latency Regression
- **The Issue:** Native SQ8 with 2x FP32 re-ranking achieves 99.54% recall, but drops throughput to 5,036 QPS (compared to 47,576 QPS for Pure SQ8 without re-ranking).
- **The Mechanism:** Re-ranking requires non-sequential memory lookups to fetch full 128D FP32 vectors for $2k$ candidate IDs, causing severe L3 CPU cache misses that dominate total search latency.
- **Guideline:** Deploy Pure SQ8 for memory-constrained, high-throughput search. Reserve re-ranking only for applications that strictly require exact float precision at lower query concurrency.

### 3.2 Compaction Write-Lock Pause Duration
- **The Issue:** Physical compaction is not wait-free or "zero downtime."
- **The Mechanism:** Acquiring `std::unique_lock<std::shared_mutex>` pauses concurrent search readers while active buffers are remapped and repointed.
- **Measured Duration:** On SIFT-100K, the exclusive write-lock pause is **6.5 to 7.2 milliseconds**. While brief enough to prevent client timeouts, it represents a discrete latency spike during compaction execution.

---

## 4. Unsupported Capabilities (Explicit Non-Goals)

To prevent scope creep and maintain research focus, the following capabilities are explicitly classified as unsupported:

1. **Distributed Clustering & Sharding:** AdaptiveVec is strictly an embedded, single-node library. It does not provide network protocols (gRPC/HTTP), Raft consensus, or distributed partition management.
2. **GPU / Hardware Acceleration:** All implementations operate natively on x86-64 CPUs with AVX2/FMA vector intrinsics. CUDA, ROCm, or TPU offloading are not supported.
3. **Product Quantization (PQ):** AdaptiveVec supports 8-bit (SQ8) and 4-bit (SQ4) scalar affine quantization. Product Quantization, Inverted File (IVF) indexing, and learned codebooks are unsupported.
4. **Lockless Shared-Memory Cross-Process Concurrency:** Multi-threaded concurrency is fully supported within a single process via C++ standard synchronization primitives, but multi-process shared-memory memory-mapped IPC is out of scope.

---

## 5. Future Research Directions

1. **Progressive Post-Compaction Highway Regeneration:** Developing a lightweight, asynchronous background routine that re-links upper skip layers after compaction without halting search readers.
2. **Dynamic Sub-Vector Dimensionality Adaptation:** Extending geometric allocation from node out-degrees to quantized bit-depths (allocating 8-bit precision to high-LID hubs and 4-bit precision to low-LID peripherals).
3. **Hardware-Accelerated ARM NEON & AVX-512 Kernels:** Implementing dedicated 512-bit SIMD kernels for emerging enterprise server architectures.
