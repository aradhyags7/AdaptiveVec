# AdaptiveVec — Final Frozen Architecture Specification

**Document Status:** Phase 6.2 Engineering Closure & Presentation Ready  
**Branch:** `phase-6.2-comprehensive-engineering-closure`  
**Parent Checkpoint:** `7f2a862` (Phase 6.1 Defense Freeze)  
**Target Milestone:** Comprehensive Correctness, Performance and Engineering Closure  

---

## 1. Architectural Philosophy & Executive Direction

AdaptiveVec is a geometry-aware proximity graph indexing architecture engineered for high-dimensional vector search. 

### Core Design Philosophy
**Adapt the graph to the data, rather than continuously adapting a controller to the graph.**

During earlier phases of the project (Phases 3.0–4.3), research temporarily explored query-time feedback loops, runtime controller adaptation (e.g., Ada-ef), and online difficulty estimation. While experimentally informative, online controllers introduced severe query hot-path latency jitter, memory contention, and marginal recall advantages.

Following the Phase 5.0 architectural recovery, AdaptiveVec returned to its foundational hypothesis: **high-dimensional vector spaces exhibit strong local intrinsic geometric heterogeneity (variable local intrinsic dimensionality, cluster density, and hubness). Proximity graphs that allocate connectivity non-uniformly during index construction achieve superior graph navigability, higher throughput, and lower tail latency under identical memory and edge budgets.**

Canonical HNSW (Malkov & Yashunin, 2018) is maintained strictly as a scientific baseline and comparative control.

---

## 2. System Architecture Diagram

```
+---------------------------------------------------------------------------------------------------+
|                                      ADAPTIVEVEC CORE ENGINE                                      |
+---------------------------------------------------------------------------------------------------+
                                                  |
         +----------------------------------------+----------------------------------------+
         |                                        |                                        |
         v                                        v                                        v
+-----------------------+              +-----------------------+              +-----------------------+
|   CONSTRUCTION TIME   |              |    STORAGE & CODEC    |              |   DYNAMIC LIFECYCLE   |
+-----------------------+              +-----------------------+              +-----------------------+
| - Geometry Signals:   |              | - Pure FP32 Storage   |              | - Concurrent Stream:  |
|   * MLE-LID Estimator |              |   (128D -> 512 B/vec) |              |   Fine-grained locks  |
|   * Log Local Density |              | - Pure SQ8 ADC:       |              | - Tombstone Deletion: |
| - Budget Allocator:   |              |   (128D -> 128 B/vec) |              |   Atomic bitflags     |
|   * Normalized Exp    |              | - Pure SQ4 ADC:       |              | - Local Healing:      |
|   * Hare-Niemeyer     |              |   (128D -> 64 B/vec)  |              |   Active 2-hop repair |
| - Layer Decoupled:    |              | - Vector Pools:       |              | - Physical Compaction:|
|   * Layer 0: Beta=1.0 |              |   Decoupled buffers   |              |   Exclusive-lock swap |
|   * Upper:   Beta=0.0 |              | - AVX2 SIMD Kernels:  |              |   Zero memory leaks   |
| - Strict RNG Pruner   |              |   Integer dot products|              |                       |
+-----------------------+              +-----------------------+              +-----------------------+
         |                                        |                                        |
         +----------------------------------------+----------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                        AUTO-CALIBRATOR & MULTI-OBJECTIVE SLA ENGINE                               |
+---------------------------------------------------------------------------------------------------+
| - Input: Constraints (Target Recall, Memory Cap, P99 Latency Target, Calibration Timeout)        |
| - Phase A: Feasibility Lower-Bound Estimation & Quantization Selection (FP32 / SQ8 / SQ4)          |
| - Phase B: Bounded Construction Candidate Ladder (M in {8, 12, 16, 24}, efC in {30, 40, 64, 100})  |
| - Phase C: Search Sweep on Fixed Graph (efSearch in {16, 32, 64, 96, 128})                        |
| - Validation: Held-out query split with 1,000-sample bootstrap confidence intervals               |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Detailed Component Specifications

### 3.1 Milestone I: Geometry-Aware Graph Intelligence

#### 3.1.1 Local Intrinsic Dimensionality (LID) Estimation
For each vector $x_i$, local intrinsic dimensionality is estimated via the Maximum Likelihood Estimator (MLE) over its $k$-nearest neighborhood (default $k_G = 15$):
$$\widehat{\text{LID}}(x_i) = -\left( \frac{1}{k_G} \sum_{j=1}^{k_G} \ln \frac{d(x_i, x_{(j)})}{d(x_i, x_{(k_G)})} \right)^{-1}$$
To guarantee numerical stability, all distances are strictly clamped:
$$d(x_i, x_{(j)}) \leftarrow \max\left(d(x_i, x_{(j)}), \epsilon\right), \quad \epsilon = 10^{-6}$$
$$\widehat{\text{LID}}(x_i) \leftarrow \max\left(1.0, \min(D, \widehat{\text{LID}}(x_i))\right)$$

#### 3.1.2 Local Density Estimation
Local density is defined as the inverse of the mean distance to the $k_G$ nearest neighbors:
$$\rho(x_i) = \frac{k_G}{\sum_{j=1}^{k_G} d(x_i, x_{(j)}) + \epsilon}, \quad \text{Sparsity}(x_i) = \ln\left(\frac{1}{\rho(x_i)} + \epsilon\right)$$

#### 3.1.3 Signal Standardization & Combined Geometric Score
Raw geometric signals are standardized to zero mean and unit variance, followed by robust clipping to $[-3, +3]$ to prevent boundary distortion:
$$z_{\text{LID}}(x_i) = \text{clip}\left(\frac{\widehat{\text{LID}}(x_i) - \mu_{\text{LID}}}{\sigma_{\text{LID}}}, -3.0, 3.0\right)$$
$$z_{\text{spars}}(x_i) = \text{clip}\left(\frac{\text{Sparsity}(x_i) - \mu_{\text{spars}}}{\sigma_{\text{spars}}}, -3.0, 3.0\right)$$
$$S_i = \alpha \cdot z_{\text{LID}}(x_i) + \beta \cdot z_{\text{spars}}(x_i), \quad \alpha = 0.5, \ \beta = 0.5$$

#### 3.1.4 Normalized Exponential Budget Allocation
To enforce an exact edge budget $B_0 = N \cdot M_{\text{base}}$ under bounded connectivity constraints $[M_{\text{min}}, M_{\text{max}}]$:
1. Continuous priority weights are computed via normalized softmax:
$$w_i = \exp(\lambda \cdot S_i), \quad \lambda = 0.5$$
2. Continuous allocations are assigned:
$$M_i^{\text{cont}} = M_{\text{min}} + (B_0 - N \cdot M_{\text{min}}) \frac{w_i}{\sum_{j=1}^N w_j}$$
3. Integer discretization is solved via the **Hare-Niemeyer / Largest-Remainder Method**:
   - Each node receives $M_i = \lfloor M_i^{\text{cont}} \rfloor$.
   - The residual budget deficit $\Delta = B_0 - \sum_{i=1}^N M_i$ is distributed by incrementing $M_i \leftarrow M_i + 1$ in descending order of fractional remainders $(M_i^{\text{cont}} - M_i)$.
   - Guarantees: $\sum_{i=1}^N M_i \equiv B_0$ and $M_{\text{min}} \le M_i \le M_{\text{max}}$ for all $i$.

#### 3.1.5 Layer-Decoupled Routing
Hierarchical proximity graphs require long-range highways on upper layers to navigate across clusters, and dense local graphs on Layer 0 for fine-grained retrieval.
AdaptiveVec enforces **Layer-Decoupled Scaling**:
- **Layer 0:** $\beta_{\text{layer0}} = 1.0$ (Full geometry-aware degree modulation $M_i \in [8, 24]$).
- **Upper Layers ($l \ge 1$):** $\beta_{\text{upper}} = 0.0$ (Strictly uniform degree $M_{\text{base}} = 16$).
Upper skip layers maintain uniform logarithmic navigability, while Layer 0 optimizes local cluster escape and entry.

#### 3.1.6 Strict Relative Neighborhood Graph (RNG) Pruning
Canonical HNSW contains an optional `keepPrunedConnections` flag that re-inserts discarded long-range edges to preserve connectivity when heuristic pruning eliminates too many candidates. In AdaptiveVec, **Strict RNG Pruning** is enforced:
- A candidate $c$ is connected to $u$ if and only if:
$$\forall v \in \text{SelectedNeighbors}(u): \quad d(c, u) < d(c, v)$$
- Discarded connections are never re-added. This eliminates redundant edges and lowers graph memory by 1.4% while improving navigability.

---

### 3.2 Milestone II: Resource-Efficient Vector Retrieval

#### 3.2.1 Native Scalar Quantization (SQ8 and SQ4)
AdaptiveVec implements decoupled affine scalar quantization:
- **SQ8:** Maps 32-bit floating-point coordinates to 8-bit unsigned integers $[0, 255]$:
$$q_d(x) = \text{round}\left(255 \cdot \frac{x_d - \min_d}{\max_d - \min_d}\right), \quad \min_d, \max_d \in \mathbb{R}$$
- **SQ4:** Maps coordinates to 4-bit unsigned integers $[0, 15]$, packing two dimensions into a single `uint8_t` byte.

#### 3.2.2 Asymmetric Distance Computation (ADC)
To eliminate query quantization error, incoming query vectors $q \in \mathbb{R}^D$ remain full-precision FP32. During graph search, distances to quantized database vectors $x_i$ are evaluated asymmetrically:
$$d_{\text{ADC}}^2(q, x_i) \approx \sum_{d=1}^D \left(q_d - \left(\min_d + \frac{\max_d - \min_d}{255} \cdot q_d(x_i)\right)\right)^2$$
Precomputing query lookup tables transforms graph traversal into high-throughput SIMD integer MAC operations via AVX2 instructions (`_mm256_maddubs_epi16`).

#### 3.2.3 Storage Decoupling & Vector Purging
- When `keep_raw_vectors = false`, the raw FP32 data buffer is completely deallocated (`data.clear(); data.shrink_to_fit()`).
- The index operates entirely on quantized vector buffers (`quantized_data` for SQ8, `quantized_data_sq4` for SQ4).
- If exact re-ranking is requested, vectors can either be retrieved from an uncompressed cache or **reconstructed on-the-fly** from quantized codes with zero raw vector overhead.

---

### 3.3 Milestone III: Dynamic and Self-Healing Index

#### 3.3.1 Atomic Tombstone Deletions
Logical deletion marks an atomic 1-byte flag:
```cpp
std::vector<uint8_t> deleted; // deleted[i] == 1 implies tombstone
```
Search operations skip tombstoned nodes during neighbor exploration without mutating graph pointers.

#### 3.3.2 Local Topological Healing
When a node $u$ is deleted with healing enabled (`mark_deleted(u, true)`):
1. The neighborhood of $u$ on each layer $l$ is inspected.
2. For every active neighbor $v \in \mathcal{N}_l(u)$ that had a directed edge to $u$, $u$ is removed from $v$'s neighbor list.
3. To prevent graph disconnectivity, active neighbors of $u$ are directly evaluated against each other:
$$\forall v_1, v_2 \in \mathcal{N}_l(u), \ v_1 \ne v_2: \quad \text{Evaluate Strict RNG condition and connect if mutually beneficial}$$
4. If $u$ was the global index entry point, `migrate_entry_point()` traverses to the highest-level active neighbor of $u$.

#### 3.3.3 Dual-Buffer Physical Compaction & Degree Replenishment
To eliminate memory fragmentation, reclaim 100% of deleted vector and graph memory, and preserve navigability under high churn:
1. `compact()` acquires an exclusive write lock on `index_lifecycle_rw_mutex`.
2. Active nodes are mapped to a contiguous index space $[0, N_{\text{active}}-1]$ via `id_map`.
3. Vector buffers (`data`, `quantized_data`, `quantized_data_sq4`) and mixed precision pools are repacked contiguously.
4. `max_level` is strictly recomputed from surviving node levels:
   $$\text{max\_level} = \max_{u \in \text{surviving}} \text{node\_levels}[u]$$
5. Adjacency lists are filtered: references to tombstoned nodes are pruned, and targets are remapped to $id\_map[v]$.
6. **Bounded Degree Replenishment**: Nodes whose Layer 0 degree fell below $M_{\text{min}} = \max(6, M_{\text{base}} / 2)$ are replenished using candidate 2-hop active neighbors (with active node sampling fallback for isolated nodes), sorting candidates by reconstructed distance and establishing reciprocal edges bounded by $M_{\text{max0}}$.
7. **Upper-Layer Preservation**: Singleton upper-layer nodes are linked to their closest peer active node at that layer.
8. **Entry Point Verification**: `enter_point` is verified to reside strictly at `max_level` with the highest degree among active peers.
9. Internal counters (`num_elements = new_count; num_deleted = 0;`) are reset.
10. The write lock is released. Search readers pause only for the duration of the repacking (~4–15 ms for 10K–100K vectors) with zero data races and zero post-compaction recall degradation.

---

### 3.4 Milestone IV: Automatic Resource-Constrained Optimization

#### 3.4.1 Multi-Objective Formulation
`AutoCalibrator` solves the constrained configuration problem:
$$\min_{\theta \in \Theta} \ \text{Objective}(\theta) \quad \text{subject to:}$$
$$\text{Recall@}k(\theta) \ge R_{\text{target}}, \quad \text{Memory}(\theta) \le M_{\text{max}}, \quad P99(\theta) \le L_{\text{target}}$$
where $\theta = (M, efConstruction, efSearch, \text{Precision})$.

#### 3.4.2 Six Formal Feasibility States
The engine enforces a rigorous, partitioned candidate selection state machine:
1. `VERIFIED_FEASIBLE`: Satisfies all hard SLA constraints on calibration queries AND passes independent verification on held-out validation queries.
2. `PROVISIONALLY_FEASIBLE`: Satisfies mean recall, memory, and latency on calibration queries, but 95% bootstrap lower bound dips below target.
3. `VALIDATION_FAILED`: Met calibration query requirements, but collapsed on independent held-out validation queries.
4. `INFEASIBLE_WITHIN_SEARCH_SPACE`: Evaluated all candidate precisions and construction/search ladders; no configuration satisfied all hard constraints.
5. `INVALID_REQUIREMENTS`: Rejects invalid parameters upfront ($R_{\text{target}} \le 0$ or $> 1.0$, $M_{\text{max}} \le 0$, etc.).
6. `EVALUATION_ERROR`: Runtime errors, null pointers, or empty datasets.

#### 3.4.3 Optimization Stages
1. **Input Parameter Verification Gate:** Rejects unphysical constraints upfront.
2. **Phase A (Feasibility & Precision Selection):** Calculates theoretical lower-bound memory footprints across FP32, SQ8, and SQ4. Eliminates precisions whose bare vector payload exceeds $M_{\text{max}}$.
3. **Phase B (Bounded Construction Ladder):** Evaluates construction configurations from a bounded ladder:
$$\mathcal{C}_{\text{build}} = \{(8, 30), (12, 40), (16, 64), (24, 100)\}$$
4. **Phase C (Search Sweep):** Fixes the graph and sweeps search parameters $efSearch \in \{16, 32, 64, 96, 128\}$ and $\text{Ada-}ef \in \{\text{false}, \text{true}\}$.
5. **Partitioned Candidate Selection:** Feasible candidates strictly compete only against other feasible candidates based on `OptimizationGoal`. Infeasible fallbacks are tracked in a separate variable.
6. **Held-Out Independent Validation:** Runs an independent verification pass on held-out queries.
7. **Prebuilt Search Calibration:** Provides `AutoCalibrator::calibrate_search()` to calibrate existing indexes in milliseconds.

---

## 4. Concurrency Model and Synchronization Hierarchy

AdaptiveVec uses a strict, hierarchical locking strategy to prevent deadlocks and data races:

```
[Lock Level 0: Index Lifecycle Read-Write Mutex]
  - std::shared_mutex index_lifecycle_rw_mutex
  - Shared Lock: All Search operations (concurrent readers)
  - Exclusive Lock: Physical Compaction (pauses search for ~6-15ms during pointer swap)

[Lock Level 1: Node-Level Update Mutexes]
  - std::vector<std::mutex> node_locks
  - Acquired during incremental vector insertion and local edge rewiring
  - Ordered by node ID (always lock min(u, v) before max(u, v)) to prevent ABBA deadlocks

[Lock Level 2: Atomic State Flags]
  - std::atomic<size_t> num_elements
  - std::atomic<size_t> num_deleted
  - std::vector<uint8_t> deleted (atomic byte reads)
```

---

## 5. Summary of Experimental Acceptance Decisions

| Capability | Milestone | Status | Verified Operational Domain |
| :--- | :---: | :---: | :--- |
| **Geometry-Aware Allocation** | I | **ACCEPTED** | Clustered vector spaces (SIFT, Embeddings); Layer-Decoupled ($\beta_0=1.0, \beta_u=0.0$). |
| **Strict RNG Pruning** | I | **ACCEPTED** | Universal; lowers graph bytes by 1.4% and evals by 2.6%. |
| **Pure SQ8 ADC** | II | **ACCEPTED** | Production memory-constrained search; 75% vector memory reduction, +70% QPS. |
| **Pure SQ4 ADC** | II | **ACCEPTED** | High-compression regimes (8x reduction); suitable when slight quantization noise is acceptable. |
| **Re-ranking on Raw Vectors** | II | **CONDITIONAL** | Achieves highest recall but incurs ~7x latency penalty due to cache misses during FP32 random lookups. |
| **Tombstones & Local Healing**| III | **ACCEPTED** | Up to 50% churn; prevents graph disconnectivity (+0.46% recall lift). |
| **Exclusive Compaction** | III | **ACCEPTED** | Millisecond-level pause (6–15 ms); reclaims 100% of deleted memory without memory leaks. |
| **AutoCalibrator SLA Engine** | IV | **ACCEPTED** | Multi-objective constraints solved in <13s with bootstrap CI validation. |
