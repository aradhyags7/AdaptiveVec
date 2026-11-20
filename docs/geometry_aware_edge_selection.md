# AdaptiveVec — Geometry-Conditioned Edge Selection and Constrained Resource Allocation

**Document Identifier**: `docs/geometry_aware_edge_selection.md`  
**Phase**: 5.1 (Stage 3 & Stage 4 Theoretical & Algorithmic Design)  
**Author**: Antigravity Autonomous Systems Research  
**Status**: Authoritative Architectural Design Monograph  

---

## 1. Executive Summary & Design Vision

In high-dimensional Approximate Nearest Neighbor (ANN) search, proximity graph construction has historically relied on **uniform heuristic rules**:
1. Uniform out-degree limits ($M, M_{\max0}$) across all vertices.
2. Uniform beam search widths ($efConstruction$) across all insertions.
3. Uniform diversity pruning thresholds ($\alpha \equiv 1.0$ in HNSW; $\alpha \in [1.2, 1.5]$ in DiskANN).

However, real-world vector spaces (such as 384-dimensional dense sentence embeddings or 128-dimensional image descriptors) exhibit severe geometric heterogeneity:
- **Dense Subspace Clusters**: Saturated local density, low Local Intrinsic Dimensionality (LID), and high directional collinearity. In these regions, standard RNG creates redundant parallel edges, inflating index memory and wasting distance calculations during search.
- **Sparse Boundary Clouds & Manifold Bridges**: Low density, high LID, and high susceptibility to hubness. In these regions, overly aggressive pruning drops critical long-range exit links, leading to disconnected components or navigation traps.

AdaptiveVec Phase 5.1 introduces **Geometry-Conditioned Edge Selection (GCES)** and formalizes graph construction as a **constrained geometric resource allocation problem**.

---

## 2. Mathematical Formulation of Constrained Edge Resource Allocation

Let $V = \{x_1, x_2, \dots, x_N\} \subset \mathbb{R}^D$ be the dataset of $N$ vectors.
Let $G = (V, E)$ be the proximity graph with directed edge set $E$. For each node $i \in V$, let $\mathcal{N}_i = \{j \in V : (i, j) \in E\}$ denote its outgoing adjacency list.

The total allocated graph memory $\mathcal{M}(G)$ is governed by:
$$\mathcal{M}(G) = \sum_{i=1}^N \left( \text{sizeof}(\text{NodeHeader}) + \text{Capacity}(\mathcal{N}_i) \cdot \text{sizeof}(\text{VertexID}) \right) + \mathcal{M}_{\text{vectors}} + \mathcal{M}_{\text{aux}}$$
Where $\text{Capacity}(\mathcal{N}_i)$ accounts for allocator headroom and reciprocal back-edge reservations ($M_{\max0}$).

### 2.1 The Optimization Problem
The global objective is to minimize expected query search work at matched recall:

$$\min_{G} \quad \mathbb{E}_{q \sim \mathcal{D}_Q} [ C_{\text{search}}(q; G) ]$$

Subject to:
1. **Target Recall Constraint**:
   $$\text{Recall}@k(G; \mathcal{D}_Q) \ge R_{\text{target}} \quad (\text{e.g., } \ge 98.0\%)$$
2. **Total Memory / Edge Budget Constraint**:
   $$\sum_{i=1}^N |\mathcal{N}_i| \le B_{\text{edges}}$$
3. **Hardware & Allocator Invariant**:
   $$\forall i \in V, \quad |\mathcal{N}_i| \le M_{\max}(i) \le M_{\text{cap}}$$
4. **Navigability Invariant**:
   $$\text{WeaklyConnectedComponents}(G) = 1, \quad \text{HierarchicalSuccess}(G) = 100\%$$
5. **Amortized Construction Cost**:
   $$T_{\text{build}}(G) \le (1 + \epsilon) \cdot T_{\text{baseline}} \quad (\epsilon < 0.15)$$

---

## 3. Directional Displacement and Neighborhood Geometry

To measure geometric redundancy, for any node $i$ and neighbor candidate $j$, define the normalized directional displacement vector:
$$u_{ij} = \frac{x_j - x_i}{\|x_j - x_i\|_2} \quad \in \mathbb{S}^{D-1}$$
For coincident or near-zero displacement ($\|x_j - x_i\| < 10^{-7}$), $u_{ij} = 0$.

For two neighbor candidates $j, l \in \mathcal{N}_i$, their directional correlation is given by the cosine of the angle between their displacements:
$$\rho_{jl}^{(i)} = \cos \theta_{jl} = u_{ij} \cdot u_{il}$$

```
                           Directional Diversity
                           
            High Collinear Redundancy          Diverse Angular Coverage
            (rho_jl ~ +1.0, theta < 30°)       (rho_jl <= 0.0, theta >= 90°)
                     
                       x_j                                  x_j
                      /                                    /
                     /                                    /
             x_i ---/                             x_i ---*
                     \                                    \
                      \                                    \
                       x_l                                  x_l
```

### 3.1 Directional Dispersion & Redundancy Score
For node $i$, the mean pairwise directional redundancy among candidate neighbors $C$ is:
$$\bar{\rho}_i = \frac{2}{|C|(|C| - 1)} \sum_{j < l \in C} \max\left(0, \cos \theta_{jl}\right)$$
- When $\bar{\rho}_i \to 1.0$, candidates cluster within a narrow cone. Retaining multiple edges in this cone is redundant.
- When $\bar{\rho}_i \to 0.0$, candidates disperse broadly across spherical dimensions, requiring greater connectivity to support multi-directional greedy routing.

---

## 4. Geometry-Conditioned Edge Selection (GCES) Formulation

In standard HNSW and DiskANN, candidate $c$ is checked against already accepted neighbor $r \in \mathcal{R}_i$:
$$\text{Prune } c \iff d(c, r)^2 < \alpha^2 \cdot d(c, x_i)^2$$
Where $\alpha$ is a global constant.

### 4.1 Node-Specific Pruning Parameter $\alpha_i$
AdaptiveVec generalizes $\alpha$ to a bounded, node-specific function:
$$\alpha_i = \text{clamp}\left( \alpha_{\text{base}} + \Delta \alpha(\text{LID}_i, \rho_i, \bar{\rho}_i), \quad \alpha_{\min}, \quad \alpha_{\max} \right)$$
Where:
- $\alpha_{\text{base}} = 1.0$ (standard RNG baseline).
- $\text{LID}_i$ is the standardized Local Intrinsic Dimensionality $z_{\text{LID}}(i) = \frac{\text{LID}_i - \mu_{\text{LID}}}{\sigma_{\text{LID}}}$.
- $\rho_i$ is the standardized local density $z_{\text{dens}}(i) = \frac{\text{density}_i - \mu_{\text{dens}}}{\sigma_{\text{dens}}}$.

### 4.2 Candidate Structural Directions

#### Direction A: Geometry-Aware Pruning Strength
$$\alpha_i = \alpha_{\text{base}} + \gamma_{\text{LID}} \cdot z_{\text{LID}}(i) - \gamma_{\text{dens}} \cdot z_{\text{dens}}(i)$$
- **High LID / Low Density (Boundary Manifolds)**: $\alpha_i$ increases (e.g. $\alpha_i = 1.25$). The shadow lune shrinks, making pruning harder and preserving essential escape shortcuts.
- **Low LID / High Density (Internal Clusters)**: $\alpha_i$ decreases or remains strict ($\alpha_i = 0.95 - 1.0$). Redundant co-linear edges are aggressively pruned.

#### Direction B: Directional Coverage Selection
In addition to distance diversity, candidate $c$ is rejected if its angular separation with an already selected neighbor $r \in \mathcal{R}_i$ is less than a minimum angle threshold $\theta_{\min}$:
$$\cos \angle(x_c - x_i, x_r - x_i) > \cos \theta_{\min}$$
Where $\theta_{\min}$ is adapted based on local intrinsic dimensionality:
$$\theta_{\min}(i) = \theta_{\text{base}} \cdot \left[ 1 + \omega \cdot \frac{1}{\text{LID}_i} \right]$$
In low LID spaces, candidates must be separated by wider angles (e.g., $75^\circ$), preventing edge crowding. In high LID spaces, angles are allowed to be tighter (e.g., $45^\circ$) to accommodate higher topological degrees.

#### Direction C: Joint Connectivity Budget and Pruning Modulation
Jointly modulates node capacity $M(x_i)$ and pruning parameter $\alpha(x_i)$:
$$M_i = \text{clamp}\left( M_{\text{base}} + \beta \cdot S(x_i), \quad M_{\min}, \quad M_{\max} \right)$$
$$\alpha_i = \text{clamp}\left( \alpha_{\text{base}} + \kappa \cdot S(x_i), \quad \alpha_{\min}, \quad \alpha_{\max} \right)$$
Where $S(x_i) = \alpha_{\text{LID}} z_{\text{LID}} + \beta_{\text{dens}} z_{\text{dens}}$.

#### Direction D: Boundary Cross-Neighborhood Conduit Protection
Identifies candidates that cross density gradients:
$$\Delta \rho(c, x_i) = |\text{density}(c) - \text{density}(x_i)|$$
If candidate $c$ acts as a bridge between a dense cluster and an adjacent sparse manifold ($\Delta \rho > \tau$), candidate $c$ is marked as a **Protected Conduit** and immune from secondary diversity pruning.

---

## 5. Construction Complexity & Algorithmic Guardrails

To prevent the construction slowdowns observed in Phase 1 (where bridge scores added $+45\%$ build time), GCES enforces strict computational bounds:
1. **Zero Matrix Inversion**: Avoid computing full covariance matrices or high-dimensional PCA.
2. **$\mathcal{O}(1)$ Directional Checks**: Directional angles are computed only between the top $M$ selected candidates, requiring at most $\frac{M(M-1)}{2}$ scalar dot products ($\approx 120$ ops), executed with AVX2 FMA.
3. **Zero Dynamic Allocation**: Candidate arrays, directional vectors, and visited sets use thread-local flat stack buffers.
4. **Deterministic Behavior**: Graph construction results are bitwise reproducible under pinned worker threads and fixed pseudo-random seeds.

---

## 6. Scientific Hypotheses & Acceptance Criteria for Phase 5.1

1. **Hypothesis 1 (Distance Evaluation Reduction)**:
   Under identical graph memory budgets, Geometry-Conditioned Edge Selection will reduce mean distance evaluations during search by $\ge 10\%$ compared to independently tuned global pruning baselines.
2. **Hypothesis 2 (High-Dimensional Navigability)**:
   On Transformer-384D, adapting $\alpha_i$ to local LID will resolve navigation bottlenecks, achieving higher Recall@10 at equal or higher throughput.
3. **Hypothesis 3 (Safe Edge Pruning)**:
   Conditioned pruning will eliminate $\ge 8\%$ of redundant internal cluster edges without degrading global reachability ($100\%$ WCC) or out-degree variance.
