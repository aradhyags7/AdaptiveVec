# AdaptiveVec — Operational Limitations, Failure Modes, and Future Work

**Document Status:** Phase 6.4 Final SLA Integrity & Presentation Ready  
**Branch:** `phase-6.4-final-correctness-closure`  
**Parent Checkpoint:** `cece8fc` (Phase 6.3 Integrated Acceptance)  
**Target Milestone:** Residual Engineering Defects and Final Integrated Acceptance  

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
1. **Compaction Quality and the 35% Rebuild Threshold:**
   - *Condition:* High volume of cumulative streaming deletions.
   - *Behavior:* At 10% churn on SIFT-100K, multi-layer degree replenishment preserves recall within **0.18 percentage points** (0.0018 loss vs healed index), strictly satisfying the $\le 0.20$ pp research target with a **14.46x maintenance speedup** over full rebuild. At 25% churn, fixed-$ef$ recall loss is **0.82 pp** (search recalibration via `calibrate_search()` restores recall to 95.98% at a 6.89x speedup).
   - *Operational Recommendation:* When cumulative churn exceeds **35%**, structural dilution of upper-layer skip lists causes local replenishment to experience diminishing returns. Furthermore, surviving dataset cardinality is smaller, making full reconstruction fast (3.31 s at 50% churn). For churn $>35\%$, full background index reconstruction is formally recommended.
2. **Post-Compaction Maintenance vs Rebuild:**
   - *Condition:* For moderate churn ($\le 35\%$), local healing, compaction, and search recalibration restore the index in 0.43 to 0.75 seconds, delivering a 7x to 14x maintenance speedup over rebuilding from scratch.

---

## 3. Performance Tradeoffs

### 3.1 Raw Vector Re-ranking Latency Regression
- **The Issue:** Native SQ8 with 2x FP32 re-ranking achieves high precision, but drops throughput to 5,036 QPS on SIFT-10K and drives P99 latency to 26.75 ms on Transformer-384D. In Phase 6.4, purging raw FP32 vectors via `drop_raw_vectors()` eliminated cache thrashing, restoring sub-10 ms P99 latency (7.38 ms).
- **The Mechanism:** Re-ranking requires non-sequential memory lookups to fetch full 128D FP32 vectors for $2k$ candidate IDs, causing severe L3 CPU cache misses that dominate total search latency.
- **Guideline:** Deploy Pure SQ8 for memory-constrained, high-throughput search. Reserve re-ranking only for applications that strictly require exact float precision at lower query concurrency.

### 3.2 Compaction Write-Lock Pause Duration
- **The Issue:** Physical compaction is not wait-free or "zero downtime."
- **The Mechanism:** Acquiring `std::unique_lock<std::shared_mutex>` pauses concurrent search readers while active buffers are remapped and repointed.
- **Measured Duration:** On SIFT-100K ($N=100,000$), the exclusive write-lock pause is **129.2 to 147.0 milliseconds** (and **127.5 milliseconds** on Transformer-57K). The previously reported 6.5–7.2 ms reflected a 10K vector slice. While brief enough to prevent client timeouts, it represents a discrete latency spike during compaction execution.

### 3.3 4-Bit Scalar Quantization (SQ4) Accuracy Trade-off
- **The Issue:** Pure SQ4 compresses vector payloads to 4 bits per dimension (0.50x of SQ8, 6.10 MB vs 12.21 MB for SIFT-100K, reducing steady-state process working set from 56.70 MB to 49.30 MB).
- **The Trade-off:** Asymmetric 4-bit scalar quantization incurs significant information loss, reducing Recall@10 from 97.70% (SQ8) to 88.10% (SQ4).
- **Guideline:** Pure SQ8 is confirmed as the primary production default. Pure SQ4 is retained as a specialized ultra-low-memory option for edge devices or severely memory-constrained nodes where sub-90% recall is acceptable.

---

## 4. Unsupported Capabilities (Explicit Non-Goals)

To prevent scope creep and maintain scientific integrity, the following capabilities are explicitly classified as unsupported:

1. **Disk Persistence and File-Based Serialization:** Formally classified as an `UNSUPPORTED_CAPABILITY`. AdaptiveVec is an in-memory proximity graph engine. No mock memory-mapped disk persistence, write-ahead logging (WAL), or background disk checkpointing is claimed or deployed.
2. **Distributed Clustering & Sharding:** AdaptiveVec is strictly an embedded, single-node library. It does not provide network protocols (gRPC/HTTP), Raft consensus, or distributed partition management.
3. **GPU / Hardware Acceleration:** All implementations operate natively on x86-64 CPUs with AVX2/FMA vector intrinsics. CUDA, ROCm, or TPU offloading are not supported.
4. **Product Quantization (PQ):** AdaptiveVec supports 8-bit (SQ8) and 4-bit (SQ4) scalar affine quantization. Product Quantization, Inverted File (IVF) indexing, and learned codebooks are unsupported.
5. **Lockless Shared-Memory Cross-Process Concurrency:** Multi-threaded concurrency is fully supported within a single process via C++ standard synchronization primitives, but multi-process shared-memory memory-mapped IPC is out of scope.
6. **Query-Time Runtime Controllers:** Online difficulty prediction and adaptive beam controllers were permanently retired in Phase 5.0 to protect the query hot-path.

---

## 5. Future Research Directions

1. **Progressive Post-Compaction Highway Regeneration:** Developing a lightweight, asynchronous background routine that re-links upper skip layers after compaction without halting search readers.
2. **Dynamic Sub-Vector Dimensionality Adaptation:** Extending geometric allocation from node out-degrees to quantized bit-depths (allocating 8-bit precision to high-LID hubs and 4-bit precision to low-LID peripherals).
3. **Hardware-Accelerated ARM NEON & AVX-512 Kernels:** Implementing dedicated 512-bit SIMD kernels for emerging enterprise server architectures.
