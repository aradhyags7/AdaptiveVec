# AdaptiveVec — Budget-Constrained Geometric Connectivity Allocation
## Architectural Specification and Mathematical Formulation

**Author:** Google DeepMind / AdaptiveVec Research Team  
**Phase:** 5.3 — Budget-Constrained Geometric Connectivity Allocation  
**Date:** 2026-11-20  
**Status:** Approved Architectural Design  

---

## 1. Problem Formulation: Constrained Proximity Graph Optimization

Proximity graph construction has traditionally relied on unnormalized heuristic parameters (e.g., nominal degree $M=16$). In Phase 1–5.2, AdaptiveVec independently scaled node degree via:
$$M_i = \text{clip}\left( M_{\text{base}} \times [1 + \beta S_i], \; M_{\min}, \; M_{\max} \right)$$
While intuitive, this independent assignment has a critical flaw: **it does not normalize the total graph degree budget**. If a dataset has an intrinsically high complexity distribution, $\sum M_i$ expands uncontrollably, bloating memory and invalidating fair baseline comparisons.

Phase 5.3 reformulates connectivity allocation as a formal **constrained optimization problem**:

$$\min_{G} \mathbb{E}_{q \sim \mathcal{Q}} [C(q, G)]$$
$$\text{subject to:}$$
1. **Target Recall Constraint:** $\text{Recall}@k(q, G) \ge \tau$
2. **Layer-0 Total Budget Constraint:** $\sum_{i=1}^N M_i(0) \le B_0$
3. **Upper-Layer Highway Budget Constraint:** $\sum_{i=1}^N M_i(l) \le B_{\text{upper}}$
4. **Node Degree Bounds:** $M_{\min}(0) \le M_i(0) \le M_{\max}(0) \quad \forall i \in \{1, \dots, N\}$
5. **Physical Memory Bound:** $\text{AllocatedGraphBytes}(G) \le \mathcal{M}_{\max}$
6. **Graph Navigability Invariants:** Weakly Connected Components $\text{WCC}(G) = 1$

where:
- $N$ is the number of indexed dataset vectors.
- $S_i$ is the local geometric complexity score of vector $x_i$.
- $M_i(l)$ is the assigned forward connectivity budget of vector $x_i$ at layer $l$.
- $B_0 = N \times M_{\text{base}}$ is the strict total Layer-0 connectivity budget.
- $C(q, G)$ is the actual search cost (measured in distance evaluations and CPU cycles).

---

## 2. Distinction Between Graph Budget and Physical Topology

To prevent accounting fallacies, we establish rigorous distinctions between five structural quantities:

1. **Assigned Forward Budget ($M_i$):**  
   The target number of outgoing neighbors vector $x_i$ attempts to select during its forward insertion into the graph.
2. **Maximum Adjacency Capacity ($M_{\max}^{(l)}$):**  
   The physical buffer capacity allocated per node in memory to accommodate both forward edges and reverse connections. In HNSW, $M_{\max}^{(0)} = 2 \times M_i$ (or a fixed max limit) to prevent buffer overflows during bidirectional edge updates.
3. **Realized Outgoing Edge Count ($|E_{\text{out}}(i)|$):**  
   The actual number of surviving outgoing edges in node $i$'s neighbor list after Strict RNG diversity pruning. Because Strict RNG discards occluded neighbors, $|E_{\text{out}}(i)| \le M_i$.
4. **Reciprocal Edge Insertions ($E_{\text{rev}}$):**  
   Edges added to node $i$'s list when a later-inserted node $j$ selects $i$ as one of its nearest neighbors. If $|E_{\text{out}}(i)|$ exceeds $M_{\max}^{(l)}$, heuristic pruning is re-triggered on node $i$.
5. **Allocated Graph Memory Bytes:**  
   The physical resident memory allocated by the memory allocator, which depends on adjacency buffer sizing, vector array storage, alignment padding, and allocator fragmentation.

> [!IMPORTANT]
> The sum of forward budgets $\sum M_i$ is a construction constraint, **not** the realized edge count. Because Strict RNG drops directionally redundant edges, the realized edge count is strictly less than $\sum M_i$.

---

## 3. Budget-Normalized Allocation Strategies

Let $B_0 = N \times M_{\text{base}}$ be the total target Layer-0 forward budget, and $M_{\min}, M_{\max}$ be the hard node degree bounds.  
The guaranteed baseline budget is $N \times M_{\min}$, leaving an available residual budget:
$$R = B_0 - N \times M_{\min}$$

### Approach A — Uniform Allocation (Scientific Baseline)
$$M_i = M_{\text{base}} \quad \forall i \in \{1, \dots, N\}$$
Every node receives identical connectivity regardless of geometry.

### Approach B — Original AdaptiveVec (Unconstrained Clipped)
$$M_i = \text{clip}\left( \text{round}\left( M_{\text{base}} \times [1 + \beta S_i] \right), \; M_{\min}, \; M_{\max} \right)$$
Subject to arbitrary total budget drift ($\sum M_i \ne B_0$).

### Approach C — Normalized Linear Allocation
Define non-negative linear weights:
$$w_i = \max(\epsilon, \; 1.0 + \beta S_i), \quad \epsilon = 10^{-4}$$
Continuous ideal residual allocation:
$$r_i^* = R \times \frac{w_i}{\sum_{j=1}^N w_j}$$
Preliminary continuous allocation: $\tilde{M}_i = M_{\min} + r_i^*$.

### Approach D — Normalized Exponential Allocation
High-dimensional manifolds often exhibit exponential volume growth as intrinsic dimensionality rises. We evaluate exponential weighting:
$$w_i = \exp(\beta \cdot S_i)$$
Continuous ideal residual allocation:
$$r_i^* = R \times \frac{w_i}{\sum_{j=1}^N w_j}$$
Preliminary continuous allocation: $\tilde{M}_i = M_{\min} + r_i^*$.

### Bounded Integer Discretization & Largest-Remainder Algorithm:
To guarantee that $M_i \in [M_{\min}, M_{\max}]$ and $\sum_{i=1}^N M_i = B_0$ exactly, we implement the **Hare-Niemeyer (Largest-Remainder) Algorithm with Saturation Redistribution**:
1. Compute integer floors: $m_i = \min\left(M_{\max}, \max\left(M_{\min}, \lfloor \tilde{M}_i \rfloor\right)\right)$.
2. Compute remainders: $\delta_i = \tilde{M}_i - m_i$ for unsaturated nodes ($m_i < M_{\max}$).
3. Calculate residual deficit: $\Delta = B_0 - \sum_{i=1}^N m_i$.
4. Sort unsaturated nodes by descending remainder $\delta_i$ with stable tie-breaking on vector index $i$.
5. Add $+1$ to the top $\Delta$ nodes: $m_i \leftarrow m_i + 1$.
6. **Invariant:** $\sum_{i=1}^N M_i \equiv B_0$ with exact precision, and $M_i \in [M_{\min}, M_{\max}]$ for all $i$.

### Approach E — Geometry Quantile Allocation
Rather than continuous functions, partition the dataset into 5 complexity quintiles ($Q_1 \dots Q_5$) based on geometric score percentiles ($0-20\%, 20-40\%, 40-60\%, 60-80\%, 80-100\%$):
- Assign fixed discrete integer degrees:
  $$M(Q_1) = 10, \quad M(Q_2) = 13, \quad M(Q_3) = 16, \quad M(Q_4) = 19, \quad M(Q_5) = 22$$
- Because each quintile contains exactly $N/5$ vectors:
  $$\sum_{i=1}^N M_i = \frac{N}{5} (10 + 13 + 16 + 19 + 22) = \frac{N}{5} \times 80 = 16 N = B_0$$
- This guarantees exact budget normalization with maximum implementation simplicity.

---

## 4. Geometric Scoring Function

Following Stage 0 reconciliation, the geometric complexity score $S_i$ is formulated to reward both topological dimensionality and metric sparsity:

$$S_i = \alpha_{\text{LID}} \cdot \widehat{\text{LID}}(x_i) + \beta_{\text{spars}} \cdot \widehat{\text{Sparsity}}(x_i)$$

where:
1. **Local Intrinsic Dimensionality (Levina-Bickel MLE):**
   $$\text{LID}(x_i) = \left( \frac{1}{k-1} \sum_{j=1}^{k-1} \ln \frac{d(x_i, n_k)}{d(x_i, n_j)} \right)^{-1}$$
2. **Metric Sparsity (Logarithmic k-NN Distance):**
   $$\text{Sparsity}(x_i) = \ln \left( \frac{1}{k} \sum_{j=1}^k d(x_i, n_j) \right)$$
3. **Robust Standardization:**
   $$\widehat{\text{LID}}(x_i) = \frac{\text{LID}(x_i) - \mu_{\text{LID}}}{\sigma_{\text{LID}}}, \quad \widehat{\text{Sparsity}}(x_i) = \frac{\text{Sparsity}(x_i) - \mu_{\text{spars}}}{\sigma_{\text{spars}}}$$
   with clipping to $[-3.0, +3.0]$ to prevent outlier distortion.
4. Default parameters: $\alpha_{\text{LID}} = 0.5, \beta_{\text{spars}} = 0.5, k = 30$.

---

## 5. Summary of Architectural Invariants

- **Upper Layers ($l \ge 1$):** Decoupled from geometry ($\beta_{\text{upper}} = 0$), pinned to uniform highway skip-lists ($M_{\text{upper}} = 16$).
- **Layer 0 ($l = 0$):** Budget-constrained geometric degree allocation ($\sum M_i(0) = B_0$).
- **Pruning Heuristic:** Strict RNG (`keep_pruned_connections = false`), perfectly matching official `hnswlib`.
- **Query Execution:** Zero query-time adaptive overhead; purely branchless SIMD linear traversals.

---
*End of Budget-Constrained Geometric Connectivity Allocation Design.*
