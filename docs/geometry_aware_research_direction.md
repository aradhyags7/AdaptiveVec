# AdaptiveVec: Restored Geometry-Aware Research Direction
**Formal Problem Formulation, Mathematical Objectives, and Phase 5.x Research Strategy**

---

## 1. Executive Direction Reset

The historical trajectory of AdaptiveVec across Phases 0–4.3 attempted to compensate for graph inefficiencies by adding dynamic runtime complexity:
- Query-time workload feedback loops
- Telemetry drift detectors
- Resumable search state machines
- Online local intrinsic dimensionality predictors in the search path
- Dynamic uncertainty-aware early-exit heuristics

In Phase 4.3, this line of inquiry reached a definitive scientific stop rule: **computing online geometric difficulty signals on the query hot path costs significantly more CPU cycles than the distance calculations it avoids on modern SIMD kernels.**

AdaptiveVec therefore executes a fundamental reset to its original founding principle:

> **"Adapt the graph to the data, rather than continuously adapting a controller to the graph."**

All structural intelligence must be injected during **graph construction**, where its computational cost is amortized across millions of subsequent queries. Search execution must remain lean, deterministic, cache-friendly, and free from online monitoring overhead.

---

## 2. Proximity Graph Resource Allocation Formulation

We formalize the construction of an approximate nearest neighbor proximity graph as a **constrained graph resource allocation problem**.

### 2.1 The General Optimization Problem

Let:
- $\mathcal{X} = \{x_1, x_2, \dots, x_N\} \subset \mathbb{R}^D$ be the dataset of indexed vectors.
- $\mathcal{Q}$ be the distribution of incoming queries.
- $G = (V, E)$ be the directed proximity graph, where $V = \mathcal{X}$ and $E$ is the set of directed edges.
- $M_i = \text{deg}_{\text{out}}(x_i)$ denote the out-degree allocated to node $x_i$.
- $\text{LID}_i$ denote the local intrinsic dimensionality of the manifold at $x_i$.
- $\rho_i$ denote the local density or sparsity in the neighborhood of $x_i$.
- $B$ denote the total allowable edge quota or physical graph-memory budget.

The goal is to determine an allocation function $M_i = f(\text{LID}_i, \rho_i, \mathcal{N}_i)$ that solves:

$$\min_{G} \mathbb{E}_{q \sim \mathcal{Q}} \left[ C_{\text{search}}(q; G) \right]$$

$$\text{subject to:}$$
1. $\text{Recall}@k(q; G) \ge R_{\text{target}}$
2. $\sum_{i=1}^N M_i \le B$
3. $G$ maintains global reachability from entry point $v_0$ ($\text{Reachability}(G) \ge 1 - \delta$).

Where $C_{\text{search}}(q; G)$ is the actual retrieval cost (measured in distance evaluations and memory accesses).

---

## 3. Theoretical Hypotheses of Geometry-Aware Allocation

### Hypothesis 1: Subspace Redundancy Elimination in Low-LID Regions
In a local neighborhood where vectors lie on a low-dimensional manifold ($\text{LID} \ll D$), points are approximately co-planar. In canonical HNSW ($M=16$), maintaining 16 neighbors creates redundant transit paths connecting mutually reachable vertices. 
- *Conjecture*: Reducing $M_i$ from 16 to 8 in low-LID regions preserves graph reachability and greedy path existence while reducing edge storage by up to $50\%$ locally.

### Hypothesis 2: Hubness and Starvation Mitigation in High-LID Regions
In high-dimensional sparse regions ($\text{LID} \ge 24$), distances concentrate and the hubness phenomenon emerges. A small subset of boundary vectors become nearest neighbors to a disproportionate number of query points.
- *Conjecture*: Increasing $M_i$ from 16 to 24 (and expanding construction exploration $efC_i$) in high-LID and sparse regions provides topologically diverse exit links, preventing greedy routing from getting trapped in local minima.

### Hypothesis 3: Amortized Construction Complexity
Probing the local geometry requires $k_{\text{probe}} = 15$ distance computations during insertion. Because insertion already evaluates $efConstruction \times M \approx 120 \times 16 \approx 1,920$ distances, the probing overhead is:

$$\frac{15}{1920} \approx 0.78\%$$

By amortizing this negligible construction overhead, the resulting graph achieves structural compression that accelerates every subsequent search query with **zero runtime overhead**.

---

## 4. Phase 5.x Research Roadmap

```
+---------------------------------------------------------------------------------+
|                               AdaptiveVec Phase 5.x Roadmap                      |
+---------------------------------------------------------------------------------+
                                         |
                                         v
   +---------------------------------------------------------------------------+
   | Phase 5.0: Architectural Recovery and Geometry-Aware Research Restoration |
   | - Recover original algorithms from v1.0.0-submission (Commit 9c22f4b)     |
   | - Establish Architecture A (Original), B (Optimized), C (Phase 4.3 Control)|
   | - Conduct 8-way ablation (Uniform, Original, LID, Density, Random, etc.)  |
   | - Establish definitive empirical verdict on geometry-aware connectivity   |
   +---------------------------------------------------------------------------+
                                         |
                                         v
   +---------------------------------------------------------------------------+
   | Phase 5.1: Geometry-Aware Edge Selection and Non-Uniform Pruning           |
   | - Investigate whether geometry should guide NEIGHBOR SELECTION (RNG)      |
   |   rather than merely degree count M_i                                     |
   | - Formulate angular diversity heuristics informed by local curvature      |
   +---------------------------------------------------------------------------+
                                         |
                                         v
   +---------------------------------------------------------------------------+
   | Phase 5.2: Layer-Decoupled Expressway Optimization                        |
   | - Rigorously optimize higher-layer express link allocation (lambda decay) |
   | - Prevent upper-layer topological disconnects while maximizing compression|
   +---------------------------------------------------------------------------+
                                         |
                                         v
   +---------------------------------------------------------------------------+
   | Phase 5.3: Unified Memory-Budgeted Production Engine                      |
   | - Combine verified geometry-aware topology with uniform SQ8 quantization  |
   | - Final release candidate: high-recall, cache-efficient, production-grade |
   +---------------------------------------------------------------------------+
```
