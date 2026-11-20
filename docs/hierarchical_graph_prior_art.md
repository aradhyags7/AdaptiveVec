# Hierarchical Proximity Graphs and Navigable Routing: Prior Art and Theoretical Foundations

**Author:** Google DeepMind / AdaptiveVec Team  
**Phase:** 5.2 — Layer-Decoupled Geometric Routing and Hierarchical Optimization  
**Date:** 2026-11-20  
**Branch:** `phase-5.2-layer-decoupled-routing`  
**Status:** Theoretical Survey & Literature Analysis  

---

## 1. Introduction and Architectural Taxonomy

Proximity graphs have become the dominant algorithmic paradigm for billion-scale Approximate Nearest Neighbor (ANN) search in high-dimensional vector spaces. They consistently outperform tree-based space-partitioning structures (e.g., KD-trees, VP-trees) and locality-sensitive hashing (LSH) by orders of magnitude on high-recall Pareto frontiers.

At their core, proximity graphs approximate a Delaunay triangulation or Relative Neighborhood Graph (RNG) over a metric dataset $X \subset \mathbb{R}^D$. Navigational structures fall into two primary architectural families:

1. **Flat Monolithic Graphs:** Examples include KGraph (Dong et al., 2011), NSG (Fu et al., VLDB 2019), and Vamana / DiskANN (Subramanya et al., NeurIPS 2019). These indices construct a single, highly refined Layer-0 graph, relying on a pre-designated medoid or random entry point, and compensate for the absence of upper highway layers through long-range directional edge pruning ($\alpha$-pruning).
2. **Hierarchical Proximity Graphs:** Pioneered by Malkov & Yashunin (TPAMI 2018) in HNSW (Hierarchical Navigable Small World), this family layers graphs $\{G_0, G_1, \dots, G_{L_{\max}}\}$ into a multi-scale hierarchy inspired by 1D probabilistic skip-lists (Pugh, CACM 1990). Search traverses top-down: greedily homing in on the query neighborhood in sparse upper layers before descending into dense lower layers.

AdaptiveVec Phase 5.2 synthesizes the structural insights of flat graphs with the computational efficiency of hierarchical skip-lists through **Layer-Decoupled Geometric Routing**.

---

## 2. Foundational HNSW Architecture (Malkov & Yashunin, 2018)

### 2.1. Probabilistic Hierarchy Assignment
In canonical HNSW, each vector $x_i$ is assigned an integer maximum layer $l_i \in [0, \infty)$ upon insertion using an exponential probability distribution parameterized by scale factor $m_L$:

$$l_i = \left\lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \right\rfloor, \quad m_L = \frac{1}{\ln(M)}$$

This ensures that the expected vertex count decays exponentially with layer index:

$$\mathbb{E}[N_l] = N \cdot e^{-l / m_L} = N \cdot M^{-l}$$

For $M = 16$, layer $l=1$ contains $6.25\%$ of vertices, layer $l=2$ contains $0.39\%$, and layer $l=3$ contains $0.024\%$.

### 2.2. Neighbor Selection Heuristic and the Diversity Criterion
When inserting node $q$ into layer $l$ with candidate pool $W$, HNSW selects at most $M$ neighbors using an RNG-inspired diversity pruning heuristic. Candidate $c \in W$ is added to selected set $R$ if and only if:

$$\forall r \in R, \quad \text{dist}(c, q) < \text{dist}(c, r)$$

If $c$ is closer to an already-selected neighbor $r$ than to base node $q$, it is deemed **directionally redundant** (forming an acute-angled triangle dominated by edge $(q, r)$) and is discarded to prevent edge clustering.

### 2.3. The Origin and Cost of `keep_pruned_connections`
In the reference C++ implementation of HNSW (`hnswlib`), a secondary fallback loop is executed after diversity pruning:

```cpp
if (keep_pruned_connections && result.size() < M) {
    for (auto& discarded_node : discarded) {
        if (result.size() >= M) break;
        result.push_back(discarded_node);
    }
}
```

This fallback policy was originally introduced as a defensive safeguard to guarantee that node degree matches nominal capacity $M$ even when diverse candidates are scarce. However, our Phase 5.1 and Phase 5.2 investigations reveal that:
- In high-dimensional spaces ($D \ge 128$), fallback retention forces the inclusion of collinear, shadowed edges.
- On SIFT-100K, disabling fallback retention (**Strict RNG**) purges **769,338 redundant edges** ($-28.55\%$ edge reduction), improves search throughput by $+6.4\%$ (from 23,618 QPS to 25,138 QPS), accelerates build time by $31.5\%$ (12.51s vs 18.26s), and retains identical 99.30% Recall@10.

---

## 3. Flat Proximity Graph Innovations: NSG and Vamana/DiskANN

### 3.1. Navigating Spreadout Graph (NSG, Fu et al., 2019)
Fu et al. identified that HNSW's multi-layer hierarchy introduces memory overhead and inter-layer pointer indirection. NSG constructs a flat monotonic graph starting from an approximate nearest-neighbor graph (KNNG) and prunes edges using the **Monotonic Relative Neighbor Graph (MRNG)** rule:

$$\text{Edge } (p, v) \text{ retained iff } \forall u \in N(p), \; \angle(pv, pu) \ge 60^\circ$$

NSG proved that with a well-chosen virtual medoid entry point, single-layer graphs can achieve competitive search speed without maintaining upper layers.

### 3.2. Vamana and DiskANN (Subramanya et al., 2019)
DiskANN generalized NSG's pruning rule by introducing the parameter $\alpha \ge 1.0$:

$$\alpha \cdot \text{dist}(p, v) < \text{dist}(u, v)$$

When $\alpha = 1.0$, the rule reduces to standard RNG. As $\alpha$ increases ($\alpha \in [1.2, 1.5]$), the pruning condition is relaxed, intentionally retaining longer "shortcut" edges across clusters. DiskANN demonstrated that a single flat graph with $\alpha$-pruning could be routed from a central medoid across disk blocks without upper skip-list layers.

---

## 4. Theoretical Limits of Hierarchical Routing in High Dimensions

### 4.1. Polylogarithmic Complexity on Bounded Doubling Dimension
Kleinberg (STOC 2000) and Malkov & Yashunin (2018) demonstrated that on metric spaces with bounded doubling dimension $d_{\text{double}}$, navigable small-world graphs achieve polylogarithmic search time $\mathcal{O}(\log^2 N)$ or $\mathcal{O}(\log N)$. 

The proof relies on two structural invariants:
1. **Long-Range Links:** Upper layers contain edges spanning large spatial distances, halving the metric distance to the target at each step (geometric skip-list property).
2. **Short-Range Manifold Links:** Lower layers provide dense, locally navigable meshes guaranteeing that greedy search does not encounter local minima.

### 4.2. High-Dimensional Breakdown: The "Highway Fallacy"
In practice, when intrinsic dimensionality is high ($D_{\text{intrinsic}} \ge 30$), the doubling dimension assumption collapses. Radovanović et al. (JMLR 2010) proved that high dimensions induce **hubness**: a small subset of nodes appear in the $k$-nearest-neighbor lists of an exponentially large fraction of the dataset.

In hierarchical graphs, hubness creates the **Highway Fallacy**:
- If upper layers are over-connected ($M_{\text{upper}}$ large) or if discarded edges are forcibly retained (`keep_pruned = true`), high-degree hub nodes in upper layers accumulate incoming edges from all directions.
- Greedy routing through these hubs causes premature convergence or oscillatory routing detours, where the search beam zigzags between distant hubs rather than descending straight toward the query's spatial basin.
- Because upper layers use strictly greedy 1-NN ($ef=1$), redundant edges in upper layers do not improve recall; they merely multiply distance evaluations during vector insertion and corrupt entry-point handover.

---

## 5. Positioning of AdaptiveVec Phase 5.2 in the Literature

AdaptiveVec Phase 5.2 establishes a principled synthesis between HNSW's skip-list routing and geometric manifold adaptation:

```
[HNSW (2018)] -------------------+
(Multi-layer skip list,           |
 uniform M, keep_pruned=true)     |
                                  v
[Vamana/DiskANN (2019)] ---> [AdaptiveVec Phase 5.2]
(Flat graph, alpha-pruning,   (Layer-Decoupled Geometric Routing:
 no upper skip-list)           - Upper Layers: Sparse Strict RNG Highways (beta_u = 0)
                               - Layer 0: Geometry-Aware Manifold Adaptation (beta_0 = 1.0)
                               - Pure Construction-Time Adaptivity (Zero Hot-Path Cost))
```

1. **Decoupled Specialization:** Rather than treating all layers identically (as in HNSW) or abandoning hierarchy entirely (as in DiskANN), Phase 5.2 decouples the operational objectives:
   - **Upper Layers ($l \ge 1$):** Operate as minimal, high-diversity highway skip-lists with $\beta_{\text{upper}} = 0$ and Strict RNG (`keep_pruned = false`).
   - **Layer 0 ($l = 0$):** Operates as a geometry-adaptive manifold mesh where $M_0(x_i) \propto S_i(x_i)$, allocating higher connectivity to sparse boundary nodes while maintaining hubness regulation.
2. **Deterministic Query Hot-Path:** Unlike runtime adaptive search controllers (tested and concluded in Phases 4.0–4.3), Phase 5.2 injects all adaptive intelligence strictly during graph construction, preserving deterministic, SIMD-accelerated linear graph traversal during query execution.
