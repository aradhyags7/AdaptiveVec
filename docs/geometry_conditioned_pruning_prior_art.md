# AdaptiveVec — Prior Art & Literature Review: Proximity Graph Edge Selection, Pruning Heuristics, and Intrinsic Dimensionality

**Document Identifier**: `docs/geometry_conditioned_pruning_prior_art.md`  
**Phase**: 5.1 (Stage 1 Research Foundation)  
**Author**: Antigravity Autonomous Systems Research  
**Status**: Authoritative Reference Monograph  

---

## 1. Executive Summary & Purpose

The central hypothesis of AdaptiveVec Phase 5.1 is:
> **Proximity graph edge selection and pruning can be conditioned on local geometric properties (Local Intrinsic Dimensionality, local density, directional dispersion, and anisotropy) to prune redundant co-linear edges in dense clusters while preserving long-range routing conduits across heterogeneous boundaries.**

To establish rigorous scientific boundaries, prevent duplicate invention, and isolate algorithmic novelty, this document examines the historical evolution of proximity graph construction, neighbor selection heuristics, diversity pruning rules, and the role of intrinsic dimensionality in high-dimensional Approximate Nearest Neighbor (ANN) search.

---

## 2. Theoretical Foundations of Proximity Graphs

### 2.1 Delaunay Triangulation and Relative Neighborhood Graphs (RNG)
The gold standard for geometric proximity graphs in $\mathbb{R}^d$ is the Delaunay Triangulation ($\text{DT}$), which guarantees monotonic search paths for any query under convex metric spaces. However, $\text{DT}$ suffers from an exponential complexity curse: in dimension $d$, the edge complexity scales as $\mathcal{O}(n^{\lceil d/2 \rceil})$, rendering exact Delaunay graphs intractable for $d > 3$.

To approximate Delaunay navigability with sparse bounded degree, computational geometry introduced the **Relative Neighborhood Graph (RNG)** (Toussaint, 1980). The RNG includes an edge between nodes $p$ and $q$ if and only if their lune (the intersection of hyper-balls centered at $p$ and $q$ of radius $d(p, q)$) contains no other vertex:
$$\text{Lune}(p, q) = B(p, d(p, q)) \cap B(q, d(p, q))$$
$$(p, q) \in E_{\text{RNG}} \iff \forall r \in V \setminus \{p, q\}, \max(d(p, r), d(q, r)) \ge d(p, q)$$

The RNG condition eliminates "occluded" or "shadowed" edges: if node $r$ is closer to both $p$ and $q$ than $d(p, q)$, the edge $(p, q)$ is redundant because traveling $p \to r \to q$ is geometrically feasible.

```
                  Standard RNG Lune Pruning Condition
                  
                            .  .  .  .
                         .              .
                       .     Lune(p, q)   .
                      .                    .
                     p --------[ r ]-------- q
                      .                    .
                       .                  .
                         .              .
                            .  .  .  .
                 
       Vertex r inside Lune(p, q) shadows edge (p, q):
       d(p, r) < d(p, q) AND d(q, r) < d(p, q) ==> Prune (p, q)
```

### 2.2 Navigable Small World (NSW) and HNSW Diversification
Malkov et al. (2014, 2018) operationalized Kleinberg's small-world navigation principle in high dimensions. In Hierarchical Navigable Small World (HNSW), candidate neighbor selection at each layer adapts the RNG lune condition via `SELECT-NEIGHBORS-HEURISTIC`:

Given query vector $q$, candidate priority queue $C$ (sorted by increasing distance to $q$), and budget $M$:
1. Pop closest candidate $c \in C$.
2. Check diversity against already selected neighbor set $R$:
   $$\forall r \in R, \quad d(c, r) \ge d(c, q)$$
3. If condition holds for all $r \in R$, append $c$ to $R$. Otherwise, add $c$ to $W_{\text{discarded}}$.
4. If $|R| < M$ and `keepPrunedConnections` is enabled, append from $W_{\text{discarded}}$ until $|R| = M$.

#### Critical Nuance in HNSW:
HNSW's condition $d(c, r) < d(c, q)$ is a **one-sided angular occluded-cone test**. In Euclidean space:
$$\cos \angle(q - c, r - c) = \frac{\|q - c\|^2 + \|r - c\|^2 - \|q - r\|^2}{2 \|q - c\| \|r - c\|}$$
When $d(c, r) < d(c, q)$, the angle $\theta = \angle(q - c, r - c)$ satisfies $\theta < 60^\circ$ whenever $\|r - c\| \approx \|q - c\|$. Thus, HNSW enforces an angular diversity cone of roughly $60^\circ$ between selected neighbors.

---

## 3. Modern Pruning Heuristics in State-of-the-Art Proximity Graphs

### 3.1 Navigating Spreading-out Graph (NSG)
Fu et al. (VLDB 2019) introduced the Navigating Spreading-out Graph (NSG). NSG identified two fundamental deficiencies in HNSW's layer-0 graph:
1. **Redundant Short Edges**: In dense regions, HNSW forms dense triangular cliques that slow down long-distance greedy traversal.
2. **Missing Long-Range Edges**: HNSW's strict RNG condition prunes long-range express links if any intermediate node exists in the lune.

NSG replaced HNSW's selection with the **Monotonic Relative Neighborhood Graph (MRNG)**:
$$(p, q) \in E_{\text{MRNG}} \iff \nexists r \in V \text{ s.t. } d(p, r) < d(p, q) \text{ and } d(q, r) < d(p, q)$$
Furthermore, NSG constructs the graph from a single global navigation center (medoid), ensuring all vertices are reachable along monotonically distance-decreasing paths from the navigation center.

### 3.2 Vamana and DiskANN: RobustPrune
Subramanya et al. (NeurIPS 2019) developed Vamana and DiskANN to optimize proximity graphs for out-of-core SSD traversal and low-latency in-memory search. DiskANN identified that strict RNG ($\alpha = 1.0$) prunes long-range shortcut edges that cross sparse voids between clusters.

DiskANN introduced **RobustPrune**, governed by a global relaxation parameter $\alpha \ge 1.0$:
$$\text{Prune candidate } c \iff \exists r \in R \text{ such that } \alpha \cdot d(r, c) \le d(p, c)$$

$$\iff d(r, c) \le \frac{1}{\alpha} d(p, c)$$

```
                       DiskANN RobustPrune Lune
                       
         Strict RNG (alpha = 1.0)           Relaxed RobustPrune (alpha = 1.2)
         
               .  .  .                            .  .  .
             .   Lune  .                        . Shrunk .
            p ----r---- c                      p ---r--- c
             .         .                        .  Lune .
               .  .  .                            .  .  .
           r shadows c                       r does NOT shadow c;
           (c is PRUNED)                     (c is RETAINED as shortcut)
```

#### Mathematical Properties of RobustPrune:
- **$\alpha = 1.0$**: Strict RNG. Lune radius is $d(p, c)$. Maximum edge pruning.
- **$\alpha > 1.0$ (e.g., $\alpha = 1.2$ or $1.4$)**: The shadow lune shrinks to radius $\frac{1}{\alpha} d(p, c)$. Candidate $c$ is retained even if intermediate node $r$ exists, unless $r$ is significantly closer to $c$ than $p$ is.
- **$\alpha \to \infty$**: Pure k-NN graph. No diversity pruning is performed.

**Crucial Limitation of DiskANN**: DiskANN treats $\alpha$ as a single, uniform global constant across the entire dataset (typically $\alpha \in [1.2, 1.5]$). It completely ignores that vector spaces exhibit heterogeneous density and varying intrinsic dimensionality.

---

## 4. Intrinsic Dimensionality and Hubness in High-Dimensional Spaces

### 4.1 The Curse of Dimensionality & Distance Concentration
As nominal dimension $D$ increases, Euclidean distances concentrate:
$$\lim_{D \to \infty} \frac{\text{Var}(\|x - y\|)}{\mathbb{E}[\|x - y\|]^2} = 0$$
In concentrated metric spaces, the ratio $\frac{d_{\max}}{d_{\min}} \to 1$, making greedy navigation fragile because local distance gradients become flat.

### 4.2 Local Intrinsic Dimensionality (LID)
Houle et al. (2012, 2017) formalized that real-world embeddings (e.g. text passages, image features) do not uniformly occupy ambient dimension $D$. Instead, they lie on low-dimensional manifolds whose local dimension varies across space.

Using Extreme Value Theory, the continuous Local Intrinsic Dimensionality (LID) at point $x$ over distance $r$ is defined as:
$$\text{LID}(x) = \lim_{r \to 0} \frac{\ln F(r)}{\ln r}$$
Where $F(r)$ is the cumulative distribution of distances from $x$.
In practice, given the $k$ nearest neighbors with distances $r_1 \le r_2 \le \dots \le r_k = r_{\max}$, the Maximum Likelihood Estimator (MLE) of Levina and Bickel (2004) is:
$$\widehat{\text{LID}}(x) = - \left[ \frac{1}{k - 1} \sum_{i=1}^{k - 1} \ln \left( \frac{r_i}{r_{\max}} \right) \right]^{-1}$$

#### Algorithmic Complexity:
Given pre-sorted neighbor distances $r_1 \dots r_k$, computing $\widehat{\text{LID}}$ requires $\mathcal{O}(k)$ operations. When performed during greedy insertion search, $k$ is fixed (e.g., $k=20$ or $25$), so the computation takes $\approx 150\ \text{ns}$ per vector, representing $<0.8\%$ of insertion time. It is **$\mathcal{O}(k)$ with respect to sample size, and $\mathcal{O}(1)$ with respect to total corpus size $N$**.

### 4.3 The Hubness Phenomenon
Radovanovic et al. (JMLR 2010) proved that high intrinsic dimensionality inevitably produces **hubs**—a small number of vertices that appear as nearest neighbors to an exponentially large number of other points.
In proximity graphs:
- Hubs accumulate excessive in-degree ($\text{deg}_{\text{in}} \gg M$).
- Hubs become traffic bottlenecks, increasing search path lengths and causing premature search termination.
- HNSW's reciprocal edge insertion (`graphs[lc][neighbor].push_back(q_idx)`) exacerbates hubness unless strictly regulated.

---

## 5. Comparative Taxonomy of Proximity Graph Pruning Heuristics

| System / Algorithm | Core Pruning Mechanism | Pruning Parameter | Spatial Adaptivity | Intrinsic Metric | Hubness Regulation |
|:---|:---|:---:|:---:|:---:|:---:|
| **HNSW** (Malkov & Yashunin 2018) | One-sided RNG lune diversity | None ($\alpha \equiv 1.0$) | Uniform across all nodes | None | None (degree capped post-hoc) |
| **NSG** (Fu et al. 2019) | Monotonic RNG (MRNG) from medoid | Angle threshold | Uniform | None | None |
| **DiskANN / Vamana** (Subramanya 2019) | RobustPrune ($\alpha \cdot d(r, c) \le d(p, c)$) | Global $\alpha \in [1.2, 1.5]$ | Uniform across all nodes | None | None |
| **FANNG** (Harwood & Drummond 2016) | Angular occlusion pruning | Global angle $\theta$ | Uniform | None | In-degree cap |
| **Original AdaptiveVec** (2026) | RNG with in-degree hubness penalty | Fixed $\alpha=1.0$, Dynamic $M(x)$ | Node-level $M(x), efC(x)$ | LID + Local Density | Linear in-degree penalty |
| **AdaptiveVec Phase 5.1 (Proposed)** | **Geometry-Conditioned Pruning (GCP)** | **Node-specific $\alpha_i = f(\text{LID}_i, \rho_i)$** | **Fully non-uniform** | **LID + Density + Anisotropy** | **Hubness + Geometry joint selection** |

---

## 6. Open Research Gaps Identified for Phase 5.1

1. **Gap 1: Uniform Pruning Fallacy**: DiskANN, HNSW, and NSG all apply a single, fixed pruning threshold globally. In dense low-LID subspaces, this retains redundant parallel edges that waste memory; in sparse high-LID manifold boundaries, it excessively prunes essential escape routes.
2. **Gap 2: Decoupled Connectivity vs Pruning**: Prior systems treat neighbor budget $M$ and pruning aggressiveness $\alpha$ as completely independent hyper-parameters. Phase 5.1 will explore whether nodes with high geometric complexity require **jointly** calibrated $(M_i, \alpha_i)$.
3. **Gap 3: Directional Coverage Blind Spots**: Standard RNG checks pairwise scalar distances without verifying whether the retained neighbor set covers the full spherical coordinates of surrounding space. Investigating directional vector dispersion ($u_{ij} \cdot u_{ik}$) during construction can close this gap.

---

## 7. Conclusions & Research Directives

1. **Algorithmic Honesty**: We do not claim invention of the RNG lune or RobustPrune conditions. DiskANN established the foundation of relaxed pruning ($\alpha \ge 1.0$).
2. **True Algorithmic Novelty**: The genuine scientific opportunity lies in **conditioning the pruning threshold $\alpha_i$ on local intrinsic dimensionality and density**, enabling the graph to automatically vary its structural selectivity across the vector manifold.
