# AdaptiveVec — Research Novelty, Algorithmic Contributions, and Prior Art

**Document Status:** Final Research Freeze (Release Defense)  
**Branch:** `release/december-12-research-defense`  
**Commit:** `4fcbd40`  
**Target Milestone:** Defense Package Presentation & Scientific Integrity Closure  

---

## 1. Executive Summary

A central requirement of rigorous academic defense is the precise differentiation between **established prior art**, **unsubstantiated claims**, **systems engineering improvements**, and **genuine algorithmic contributions**.

AdaptiveVec does not claim to have invented proximity graphs, hierarchical navigation, scalar quantization, or local intrinsic dimensionality. Rather, AdaptiveVec's scientific novelty lies in the **formal synthesis of intrinsic geometric signal estimation with proximity graph connectivity allocation**, establishing that adapting graph degree to local manifold curvature during index construction produces superior search efficiency compared to uniform baselines.

---

## 2. Prior Art Taxonomy

| Foundation Domain | Foundational Works | Prior Art Principles Used in AdaptiveVec | What AdaptiveVec Does Differently |
| :--- | :--- | :--- | :--- |
| **Hierarchical Proximity Graphs** | Malkov & Yashunin (TPAMI 2018: HNSW) | Logarithmic hierarchical skip layers; greedy routing on upper layers; beam search on Layer 0. | Canonical HNSW enforces uniform degree $M$ across all nodes. AdaptiveVec allocates degree non-uniformly on Layer 0 based on local geometry. |
| **Monotonic Neighborhood Graphs** | Fu et al. (VLDB 2019: NSG), Subramanya et al. (NeurIPS 2019: DiskANN / Vamana) | Relative Neighborhood Graph (RNG) angle condition for edge pruning. | AdaptiveVec enforces Strict RNG without heuristic connection over-retention (`keepPrunedConnections = false`), coupled with non-uniform node budgets. |
| **Intrinsic Dimensionality Theory** | Levina & Bickel (NeurIPS 2004), Houle et al. (ICDM 2012) | Maximum Likelihood Estimation (MLE) of Local Intrinsic Dimensionality (LID) over $k$-NN samples. | LID was historically used for dimensionality reduction or out-of-distribution detection; AdaptiveVec uses LID as a routing capacity signal for graph synthesis. |
| **Vector Compression & Quantization** | Jégou et al. (TPAMI 2011: PQ), Johnson et al. (IEEE Trans. Big Data 2019: FAISS) | Asymmetric Distance Computation (ADC); uniform scalar affine mapping (SQ8, SQ4). | AdaptiveVec decouples quantized vector storage entirely from graph topology, enabling zero-raw-vector storage, dynamic remapping, and physical compaction. |
| **Automated Index Tuning** | Li et al. (SIGMOD 2020), AutoIndex / PyNNDescent autotune | Heuristic grid sweeps over search parameters ($efSearch$). | `AutoCalibrator` formulates a constrained multi-objective optimization problem with theoretical feasibility bounding and 95% bootstrap confidence intervals. |

---

## 3. The Central Research Hypothesis

Canonical proximity graphs (such as HNSW) treat high-dimensional vector spaces as statistically isotropic and homogeneous: every point receives an identical out-degree budget ($M$).

**The AdaptiveVec Hypothesis:**
Real-world vector spaces (e.g., visual feature embeddings, dense text representations) lie on complex, lower-dimensional non-linear sub-manifolds characterized by:
1. **Heterogeneous Local Intrinsic Dimensionality:** Vectors in dense cluster centers exhibit high local intrinsic dimensionality and serve as "hubs" through which many shortest paths pass. Vectors in peripheral clusters exhibit low intrinsic dimensionality.
2. **Routing Bottlenecks:** Imposing a uniform degree $M$ across all nodes starves hub nodes of outward routing diversity, creating artificial bottlenecks where greedy search gets trapped. Conversely, assigning $M$ edges to peripheral nodes wastes memory on redundant short-range connections.
3. **Construction-Time Resolution:** Allocating graph connectivity non-uniformly during construction according to standardized geometric priority:
$$M_i \propto \exp\left(0.5 \cdot (\alpha \cdot z_{\text{LID}}(x_i) + \beta \cdot z_{\text{spars}}(x_i))\right)$$
optimizes graph navigability and slashes tail latency under an identical total memory budget.

---

## 4. Algorithmic Novelty vs Systems Engineering

To maintain scientific integrity, AdaptiveVec explicitly delineates between algorithmic contributions and systems engineering innovations:

### 4.1 Genuine Algorithmic Contributions
1. **Layer-Decoupled Geometric Connectivity Allocation:**
   - Adapting out-degree on Layer 0 according to normalized exponential geometric priority while strictly preserving uniform logarithmic connectivity on upper highway layers ($\beta_{\text{layer0}} = 1.0, \beta_{\text{upper}} = 0.0$).
   - Proof of effectiveness: Achieves **+24.01% higher QPS** and **-37.62% lower P95 tail latency** on SIFT-100K at matched memory and recall against Strict RNG Uniform control.
2. **Hare-Niemeyer Integer Discretization for Graph Synthesis:**
   - Translating continuous geometric priority weights into exact integer degrees satisfying both individual node bounds $[M_{\text{min}}, M_{\text{max}}]$ and exact global edge budgets $\sum M_i \equiv B_0$.
3. **Active 2-Hop Topological Healing:**
   - Preventing graph disconnectivity during dynamic churn by evaluating mutual Strict RNG conditions across surviving active neighbors of deleted vertices, yielding a **+0.46% recall lift** under 50% churn.

### 4.2 Systems Engineering Innovations
1. **Decoupled Scalar Quantization & AVX2 Kernel:**
   - Native SQ8/SQ4 memory layout decoupled from adjacency lists, allowing total raw FP32 deallocation while accelerating traversal throughput by +70.1% via integer SIMD dot products.
2. **Dual-Buffer Physical Compaction:**
   - Reclaiming 100% of deleted vector and graph memory via index defragmentation with an exclusive write-lock pause of only 6.5 to 7.2 ms, executing **79x to 117x faster** than re-indexing from scratch.
3. **Multi-Objective AutoCalibrator:**
   - An integrated 3-stage optimization engine that automatically identifies infeasible configurations via theoretical memory lower-bounding and returns validated Pareto configurations with bootstrap confidence intervals in <13 seconds.

---

## 5. Negative Findings & Discarded Hypotheses

A hallmark of genuine scientific research is the honest documentation of failed hypotheses:

1. **Query-Time Adaptation (Ada-ef / Feedback Controllers):**
   - *Hypothesis:* Dynamically adjusting $efSearch$ based on entry-point distance or runtime candidate stability would improve the Pareto frontier.
   - *Outcome:* Incurred severe reference-monitoring overhead, memory contention, and failed recall requirements across all datasets (Phases 4.1–4.3). Discarded in Phase 5.0.
2. **Geometry-Conditioned Pruning (GCP):**
   - *Hypothesis:* Relaxing or tightening RNG pruning thresholds based on local manifold curvature would prune more edges without sacrificing recall.
   - *Outcome:* Reduced throughput by 13.6% and lowered recall by 0.26% relative to Architecture B (Phase 5.1). Discarded in Phase 5.2.
3. **Upper-Layer Geometric Modulation:**
   - *Hypothesis:* Allocating non-uniform degrees to upper skip layers would improve global routing.
   - *Outcome:* Distorted long-range highways and degraded navigability. Decoupled scaling ($\beta_{\text{upper}} = 0.0$) proved strictly superior.
4. **Universal Recall Superiority:**
   - *Hypothesis:* Geometry-aware allocation universally achieves higher recall than all controls.
   - *Outcome:* Inverted/Reversed allocation achieved +0.22% higher recall by over-connecting peripheral boundaries, but at the expense of +5.6% more distance computations and lower throughput. Claim corrected.
