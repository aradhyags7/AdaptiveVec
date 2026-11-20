# Architecture Comparison: AdaptiveVec Historical, Optimized, and Current Variants

This document establishes a rigorous side-by-side comparative analysis of the three internal AdaptiveVec architectural lineages and external industry baselines, specifying their data structures, algorithmic mechanics, computational complexity, and runtime trade-offs.

---

## 1. Architectural Taxonomy

| Dimension | Architecture A: Original AdaptiveVec | Architecture B: Optimized Original | Architecture C: Current AdaptiveVec (Phase 4.3 Control) | External Baselines (hnswlib / FAISS) |
|:---|:---|:---|:---|:---|
| **Primary Philosophy** | Manifold-adaptive proximity graph with online LID/density probing | Geometry-aware structural adaptation + zero-overhead native systems core | Canonical HNSW + offline parameter calibration; runtime controllers disabled | Static uniform proximity graph ($M$, $efC$ globally uniform) |
| **Commit Origin** | `9c22f4b` (`v1.0.0-submission`) | Phase 5.0 Recovery Synthesis | `51b7f46` (Phase 4.3 Final) | Upstream releases |
| **Node Degree $M_i$** | Dynamic: $M(x) \propto z_{\text{LID}} + z_D \in [8, 24]$ | Dynamic: $M(x) \propto z_{\text{LID}} + z_D \in [8, 24]$ | Uniform: $M=16$ (or calibrated static) | Uniform: $M=16$ |
| **Construction $efC_i$** | Dynamic: $efC(x) \in [40, 220]$ | Dynamic: $efC(x) \in [40, 220]$ | Uniform: $efC=120$ (or calibrated static) | Uniform: $efC=200$ |
| **Layer Scaling** | Decoupled decay: $M^{(l)} = M \cdot (0.75)^l$ | Decoupled decay: $M^{(l)} = M \cdot (0.75)^l$ | Disabled (uniform across levels) | Canonical $M_{\text{max}} = M$, $M_{\text{max}0} = 2M$ |
| **Hubness Regulation** | In-degree penalty: $d_{\text{eff}} = d \cdot [1 + \mu \frac{\text{deg}_{\text{in}}}{\bar{\text{deg}}}]$ | In-degree penalty: $d_{\text{eff}} = d \cdot [1 + \mu \frac{\text{deg}_{\text{in}}}{\bar{\text{deg}}}]$ | Disabled (canonical RNG heuristic) | None (canonical RNG heuristic) |
| **Visited Set Structure** | `std::unordered_set<tableint>` (Heap-allocated per search) | `VisitedTagPool` (Generation-tagged flat buffer, zero alloc) | `VisitedTagPool` (Generation-tagged flat buffer, zero alloc) | Flat byte array / bitset / generation tag |
| **Distance Kernels** | AVX2 unsquared $L_2$ (`sqrt` inside inner loop) | AVX2/FMA squared $L_2$ (monotonic, no `sqrt`) | AVX2/FMA squared $L_2$ | AVX2/AVX-512 squared $L_2$ |
| **Search Termination** | Stagnation early-exit ($p=6, \epsilon=10^{-4}$) | Calibrated stagnation exit ($p=8$) or static beam | Fixed stagnation patience ($p=8$) | Priority queue exhaustion ($W$ exhausted) |
| **Query-Time Adaptation** | Dynamic $efSearch(q)$ via entry point probing | Pure static beam search (zero online estimator overhead) | Disabled (controllers decoupled) | Pure static beam search |
| **Quantization** | Uniform SQ8 & 4-bit SQ4 with FP32 reranking | Uniform SQ8 (calibrated per-dim affine) | Uniform SQ8 (calibrated per-dim affine) | FP32 / FAISS SQ8 |
| **Search Overhead** | High (heap allocations + `sqrt` + set lookup) | Minimal (flat memory + zero allocations) | Minimal (flat memory + zero allocations) | Minimal (hand-tuned assembly / C++) |
| **Concurrency Core** | 4,096 striped mutexes (`std::mutex`) | Striped spinlocks / mutexes + thread-local stats | Striped mutexes + stack-allocated `QuerySearchStats` | OpenMP coarse grain / fine-grained locks |

---

## 2. In-Depth Architectural Profiles

### Architecture A: Original AdaptiveVec (`v1.0.0-submission`)

#### Core Characteristics:
- **Strengths**: 
  - Genuinely adapts graph topology to local vector geometry.
  - Achieves $7.4\%$ to $10.7\%$ total edge reduction on SIFT-100K and SIFT-500K.
  - Introduces effective in-degree hubness regulation and two-tier streaming deletion with localized graph healing.
- **Weaknesses & Implementation Regressions**:
  - `search_layer` instantiates a `std::unordered_set<tableint>` on every call, triggering millions of small heap allocations and hash lookups during search.
  - Evaluates `std::sqrt()` in the innermost distance loop, wasting CPU vector execution cycles on monotonic comparisons.
  - Multi-threaded search suffered from atomic counter cache-line bouncing.
  - Memory accounting in `get_total_memory_bytes()` reported raw payload sizes rather than realized heap allocation capacities.

---

### Architecture B: Optimized Original AdaptiveVec (Phase 5.0 Future Foundation)

#### Core Characteristics:
- **Philosophy**: Restore the structural intelligence of Architecture A while eliminating all runtime inefficiencies identified during Phases 0–4.3.
- **Key Upgrades**:
  1. **Zero-Allocation Visited Tracking**: Replaces `std::unordered_set` with a pre-allocated, thread-local `VisitedTagPool` using generation tagging. Visited checks require a single memory read and write without clearing the buffer between queries.
  2. **SIMD Squared Distance Acceleration**: Retains exact metric distances for MLE LID and Density estimation during probing, but executes graph traversal and candidate ranking purely under squared Euclidean distance ($L_2^2$), eliminating `sqrt` instructions entirely.
  3. **Thread-Local Instrumentation**: Passes lightweight `QuerySearchStats` by reference on the thread stack, avoiding all atomic contention during distance calculations.
  4. **Contiguous Vector Alignment**: Vector coordinates are stored in 64-byte aligned contiguous flat memory blocks, maximizing hardware prefetcher efficiency.
  5. **Pure Construction-Time Adaptation**: Modulates $M(x)$ and $efC(x)$ during ingestion; executes search without runtime difficulty predictors or query-time reference monitors.

---

### Architecture C: Current AdaptiveVec (Phase 4.3 Production Control)

#### Core Characteristics:
- **Philosophy**: A hardened, statically configured HNSW index with uniform node degrees ($M=16, efC=120$), uniform SQ8 quantization, and optional fixed stagnation patience ($p=8$).
- **Role in Phase 5.0**: Serves as the internal control representing the culmination of the Phase 0–4.3 engineering efforts. It provides a stable, highly optimized baseline to determine whether geometry-aware connectivity (Architecture B) confers any genuine algorithmic benefit over uniform connectivity under equalized resource budgets.

---

### External Baselines: hnswlib & FAISS HNSW

- **hnswlib**: The reference C++ implementation by Yury Malkov. Utilizes contiguous binary graph storage, flattened vector data, and generation-tagged visited bitsets. Sets the performance benchmark for CPU cache efficiency and single-query latency.
- **FAISS HNSW**: The Meta AI vector search library. Highly optimized for batch vector search, AVX2/AVX-512 distance kernels, and optional scalar quantization.
