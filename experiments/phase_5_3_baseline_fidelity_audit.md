# AdaptiveVec — Phase 5.3: Baseline Fidelity Audit and Scientific Corrections

**Author:** Google DeepMind / AdaptiveVec Research Team  
**Phase:** 5.3 — Budget-Constrained Geometric Connectivity Allocation  
**Date:** 2026-11-20  
**Repository Branch:** `phase-5.3-budget-constrained-geometry`  
**Status:** Mandatory Stage 0 Audit Closure  

---

## 1. Executive Direction & Purpose

This audit resolves critical discrepancies identified in the Phase 5.2 report regarding:
1. Architectural fidelity between native HNSW controls and official `hnswlib`.
2. Mathematical distinction between geometric terminology and candidate occlusion.
3. Reconciliation of empirical edge reductions, fallback counters, search costs, and memory accounting.
4. Formal correction of the Phase 5.2 "matched-budget" claim.

---

## 2. Official HNSW Neighbor Selection Audit

### 2.1. Code Inspection: Official `hnswlib` vs Native AdaptiveVec

We performed a line-by-line inspection of official `hnswlib` (`v0.8.0` / `master`, commit `7480ee3`, source: [`hnswalg.h`](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h#L475-L515)).

#### Official `hnswlib` Implementation (`getNeighborsByHeuristic2`):
```cpp
// third_party/hnswlib/hnswlib/hnswalg.h (lines 475-515)
void getNeighborsByHeuristic2(
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    const size_t M) {
    if (top_candidates.size() < M) return;

    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0) {
        queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
        top_candidates.pop();
    }

    while (queue_closest.size()) {
        if (return_list.size() >= M) break;
        std::pair<dist_t, tableint> curent_pair = queue_closest.top();
        dist_t dist_to_query = -curent_pair.first;
        queue_closest.pop();
        bool good = true;

        for (std::pair<dist_t, tableint> second_pair : return_list) {
            dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                          getDataByInternalId(curent_pair.second),
                                          dist_func_param_);
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }
        if (good) return_list.push_back(curent_pair);
    }

    for (std::pair<dist_t, tableint> curent_pair : return_list) {
        top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
}
```

#### Crucial Audit Finding:
1. **Absence of Fallback Retention in Official `hnswlib`:**  
   In official `hnswlib`, when `return_list` has fewer than $M$ elements after processing `queue_closest`, **no rejected candidates are re-inserted**. The algorithm simply returns whatever candidates passed the triangle inequality test (`good == true`).
2. **Origin of Fallback Retention in AdaptiveVec:**  
   In Malkov & Yashunin's original 2016/2018 paper (*"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"*), Algorithm 4 (*SELECT-NEIGHBORS-HEURISTIC*) defines a theoretical parameter: `keepPrunedConnections: flag indicating whether to keep pruned connections`. The paper's pseudocode specified:
   $$\text{if } \text{keepPrunedConnections and } |R| < M: \; R \leftarrow R \cup \text{extractNearest}(W_d)$$
   When AdaptiveVec implemented its native `select_neighbors_heuristic` in Phase 0, it faithfully coded Malkov & Yashunin's Algorithm 4 pseudocode with `keep_pruned_connections = true` by default.
3. **Implication:**  
   AdaptiveVec's "Canonical HNSW" control was **not** identical to official `hnswlib`; it was executing Algorithm 4 with `keep_pruned_connections = true`. Conversely, AdaptiveVec's "Strict RNG" (`keep_pruned_connections = false`) is **functionally equivalent to official `hnswlib`'s `getNeighborsByHeuristic2`**!
4. **Correction:**  
   Disabling fallback retention is **not** a novel AdaptiveVec pruning algorithm; it is the exact behavior of official `hnswlib`. All claims that "Strict RNG is a novel pruning discovery" are retracted.

### 2.2. Architectural Taxonomy

| Index Configuration | Implementation Reference | Heuristic Diversity Test | Fallback Refill (`keep_pruned`) | Degree Policy |
| :--- | :--- | :---: | :---: | :---: |
| **Official `hnswlib`** | `third_party/hnswlib/hnswlib/hnswalg.h:475` | Standard RNG ($d_{uv} < d_{uq}$) | **No** (Strict RNG by default) | Uniform ($M=16, M_0=32$) |
| **Native HNSW (with fallback)** | `cpp/adaptive_hnsw.hpp:1703` | Standard RNG ($d_{uv} < d_{uq}$) | **Yes** (Malkov Algorithm 4) | Uniform ($M=16, M_0=32$) |
| **Native HNSW (without fallback)** | `cpp/adaptive_hnsw.hpp:1695` | Standard RNG ($d_{uv} < d_{uq}$) | **No** (Direct `hnswlib` parity) | Uniform ($M=16, M_0=32$) |
| **Architecture B** | `cpp/adaptive_hnsw.hpp` | Standard RNG ($d_{uv} < d_{uq}$) | **Yes** (Historical default) | Adaptive $M(x) \in [8, 24]$, coupled |
| **Layer-Decoupled AdaptiveVec** | `cpp/adaptive_hnsw.hpp` | Standard RNG ($d_{uv} < d_{uq}$) | **No** (Matches official `hnswlib`) | Adaptive $M(x)$ on L0, fixed Upper |

---

## 3. Mathematical Terminology and Complexity Corrections

### 3.1. Rigorous Geometric Definitions

Previous reports used "collinearity", "acute angles", and "candidate occlusion" interchangeably. We establish precise mathematical definitions:

1. **Acute Angle:**  
   Two candidate vectors $u, v \in \mathbb{R}^D$ subtend an acute angle with respect to base vector $q$ if:
   $$\cos \theta = \frac{\langle u - q, v - q \rangle}{\|u - q\| \|v - q\|} > 0$$
   An acute angle is a necessary condition for triangle inequality occlusion in Euclidean space, but does not imply collinearity.
2. **Collinearity:**  
   Vectors $u, v, q$ are collinear if and only if $\text{rank}(\{u - q, v - q\}) = 1$, i.e., $v - q = c(u - q)$ for scalar $c$. In high-dimensional spaces ($D \ge 128$), exact collinearity has probability measure zero. The term "collinear edges" was colloquial and mathematically inaccurate; the correct term is **directionally redundant or occluded neighbors**.
3. **Relative Neighborhood Candidate Occlusion (RNG Diversity Condition):**  
   A candidate $v$ is occluded by an already selected neighbor $u$ with respect to base point $q$ if:
   $$d(u, v) < d(v, q)$$
   This geometrically implies that $u$ is strictly closer to $v$ than $q$ is to $v$. Therefore, any path from $q$ to $v$ can be routed through $u$, rendering a direct edge $(q, v)$ redundant in the Relative Neighborhood Graph.
4. **Directional Redundancy:**  
   Multiple edges that point into the same angular Voronoi cone without providing additional metric reachability.

### 3.2. Parameter Tuning and Search Complexity Corrections

1. **Tuning Requirements:**  
   While Strict RNG eliminates the need for an artificial pruning threshold parameter ($\alpha$ in GCP), it **does not** eliminate graph hyperparameter tuning. The practitioner must still tune nominal degree $M$, construction exploration depth $efConstruction$, search depth $efSearch$, and layer multiplier $m_L$.
2. **Complexity Claims:**  
   Proximity graphs do not provide mathematically guaranteed $\mathcal{O}(\log N)$ search complexity in arbitrary metric spaces. Logarithmic search holds only under specific structural conditions: bounded doubling dimension, expansion constants, and absence of severe hubness/clustering bottlenecks.

---

## 4. Reconciliation of Phase 5.2 Measurements

### 4.1. Edge Reduction Reconciliation (28% vs 12%)
- In the pure uniform experiment (`Canonical HNSW with fallback` vs `Canonical HNSW without fallback`), setting `keep_pruned_connections = false` reduced edges from $2,666,780$ to $1,910,348$ (**$-28.37\%$**).
- In the combined Layer-Decoupled architecture (`Scheme 4` vs `Scheme 1`), total edges were $2,361,573$ vs $2,694,602$ (**$-12.36\%$**).
- **Explanation:** In Scheme 4, Layer 0 applies geometric degree scaling $M_i \in [8, 24]$ with $M_{\max}^{(0)} = 48$. Nodes in high-LID and boundary regions are allocated up to $50\%$ higher nominal degree capacity ($M=24$). Even with Strict RNG pruning active, these nodes retain more edges than in uniform $M=16$. The geometric degree expansion offsets approximately $16\%$ of the edge reduction.

### 4.2. Fallback Counter Interpretation
- `fallback_edges_added` and `fallback_retention_invocations` in `adaptive_hnsw.hpp` are incremented during every invocation of `select_neighbors_heuristic`.
- Because heuristic neighbor selection occurs repeatedly during forward node insertion and during bidirectional reverse neighbor updates (`mutuallyConnectNewElement`), and edges can subsequently be replaced or re-pruned, these counters represent **cumulative insertion attempts**, not surviving edges in the final graph topology.

### 4.3. Upper-Layer Search Cost Invariance
- The reported upper-layer distance evaluation figure ($68.59$ evals on SIFT-100K) is the **empirical mean** across the 10,000 query evaluation workload, not a static hardware constant. Greedy routing hops vary slightly per query depending on entry point alignment.

### 4.4. Memory Accounting Clarification
- In Phase 5.2, reported "Layer-0 Memory" of $11.44\text{ MB}$ accounted **strictly for graph adjacency lists and headers** ($\approx 2.4\text{M edges} \times 4\text{ bytes} + \text{metadata}$).
- It excluded raw FP32 vector data storage ($100,000 \times 128 \times 4\text{ bytes} = 51.2\text{ MB}$).
- Phase 5.3 establishes a unified, two-tier memory accounting schema: **Owned Graph Memory** vs **Total Index Memory**.

### 4.5. Feature Sign and Density Convention
- In `adaptive_hnsw.hpp`, `Density(x)` is computed as the inverse average distance to $k$ nearest neighbors: $\rho(x) = k / \sum_{j=1}^k d(x, n_j)$.
- High $\rho(x)$ denotes dense clusters; low $\rho(x)$ denotes sparse regions.
- To prioritize sparse boundary and bridge regions, the geometric complexity score must positively weight sparsity, i.e., $S_i = \alpha \cdot \text{LID}(x_i) + \beta \cdot \text{Sparsity}(x_i)$, where $\text{Sparsity}(x_i) = -\log(\rho(x_i))$.

### 4.6. Correction of the Phase 5.2 "Matched-Budget" Claim
- In Phase 5.2, Scheme 6 was described as a "Matched Edge Budget Control".
- However, Scheme 6 generated **2,801,586 edges**, whereas Canonical HNSW generated **2,694,602 edges** ($+106,984$ edges, or **$+3.97\%$**).
- **Correction:** Scheme 6 did **not** match the edge budget of the baseline. Its recall advantage ($99.56\%$ vs $99.45\%$) was partially funded by an inflated edge budget. In Phase 5.3, budget normalization is enforced strictly so that $\sum M_i \le B_0$ with zero budget drift.

---
*End of Stage 0 Baseline Fidelity Audit.*
