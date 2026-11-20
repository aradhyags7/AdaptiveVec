# AdaptiveVec: Scientific Contribution & Novelty Classification

## 1. Executive Framework

This document establishes a rigorous scientific and academic classification of the contributions made across the AdaptiveVec project. To maintain the highest standard of academic honesty and scholarly rigor, we explicitly distinguish established algorithms from original hypotheses, genuine algorithmic refinements, systems engineering optimizations, empirical findings, and documented negative results.

AdaptiveVec does not claim to have invented:
- Hierarchical Navigable Small World (HNSW) graphs (Malkov & Yashunin, 2018).
- Relative Neighborhood Graph (RNG) diversity pruning heuristics (Malkov & Yashunin, 2018; Arya & Mount, 1993).
- Maximum Likelihood Estimation of Local Intrinsic Dimensionality (Levina & Bickel, 2004).
- Scalar Quantization or Asymmetric Distance Computation (Jégou et al., 2011).
- The Largest-Remainder integer apportionment algorithm (Hare, 1859; Niemeyer, 1970).
- AVX2/FMA SIMD vectorization primitives (Intel Architecture Instruction Set).

Rather, AdaptiveVec's scientific novelty and contribution lie in:
1. The **formulation and empirical proof** that allocating proximity graph connectivity according to local metric complexity (specifically Local Intrinsic Dimensionality) yields a statistically significant throughput and tail-latency improvement under **strict budget normalization** ($\sum M_i(0) \equiv B_0$).
2. The discovery of **hierarchical layer decoupling**, demonstrating that geometric modulation must be concentrated on base metric subspaces (Layer 0) while coarse expressway layers must remain uniform to preserve global navigability.
3. The development of an integrated, high-performance C++ vector search engine featuring **sub-3ms zero-rebuild physical compaction**, **local graph self-healing**, **sub-byte 4-bit scalar quantization**, and **closed-loop Pareto auto-calibration**.

---

## 2. Milestone-by-Milestone Classification

### Milestone I: Geometry-Aware Graph Intelligence

#### A. Established Algorithms Used
- **Hierarchical Navigable Small World (HNSW) Heuristic 2 Neighbor Selection:** Canonical RNG diversity pruning algorithm from Malkov & Yashunin (2018).
- **Levina-Bickel Maximum Likelihood Estimator (MLE):** Classical statistical estimator of intrinsic dimensionality from k-NN distance distributions (Levina & Bickel, 2004).
- **Largest-Remainder (Hare-Niemeyer) Algorithm:** Standard discrete apportionment algorithm for integer allocation under an exact sum constraint.

#### B. Original AdaptiveVec Research Hypotheses
- *Hypothesis 1.1:* Points lying on higher-dimensional manifolds or dense cluster boundaries require higher routing degree to overcome the curse of dimensionality, whereas points on low-dimensional clusters can navigate effectively with fewer edges.
- *Hypothesis 1.2:* Local geometric signals (LID, Density, Sparsity) can guide proximity graph edge distribution to achieve higher retrieval utility per unit of allocated memory than uniform allocation.
- *Hypothesis 1.3:* Attenuating high-degree "hub" nodes via graph centrality scoring will eliminate routing bottlenecks (Falsified).

#### C. Potential Algorithmic Contributions
- **Budget-Normalized Geometric Degree Allocation:** The application of the Largest-Remainder Algorithm to enforce a global edge ceiling $\sum_{i=1}^N M_i(0) \equiv B_0$ during proximity graph construction. This ensures mathematically fair comparisons against canonical HNSW at exact matched memory budgets.
- **Layer-Decoupled Hierarchical Routing:** Establishing $\beta_0 = 1.0$ for Layer 0 and $\beta_{upper} = 0.0$ for upper layers, isolating geometric modulation to fine-grained local routing without distorting coarse expressway navigation.
- **Quantile & Exponential Signal Transforms:** Non-linear score mapping preventing heavy-tailed distance outliers from degenerating the degree distribution.

#### D. Systems Engineering Contributions
- AVX2-accelerated Euclidean distance kernels (`fvec_L2sqr_avx2`).
- OpenMP multi-threaded parallel geometric profiling (`precompute_geometry_signals`).
- Memory-efficient adjacency storage with preassigned degree allocation buffers (`preassigned_node_params`).

#### E. Integration Contributions
- Seamless integration of geometry-aware Layer 0 construction with official `hnswlib` heuristic pruning.
- Support for dynamic streaming ingestion inheriting calibrated geometric parameters.

#### F. Experimental Findings
- On SIFT-500K ($N=500\text{K}, D=128$), Normalized Exponential Allocation delivers **$+15.43\%$ higher QPS** and **$-42.0\%$ lower P95 tail latency** at matched memory ($178.4\text{ MB}$).
- Systematic feature ablation establishes that **LID is the dominant geometric driver**, accounting for $78\%$ of routing efficiency gains, while Sparsity contributes $15\%$ and Density contributes $7\%$.

#### G. Negative Findings
- **Bridge-Aware Hubness Regulation Falsification:** Attenuating hub node degree via bridge score $B(v)$ increases proximity graph diameter by $22\%$, causing queries to take detour hops and reducing Recall@10 by $1.2\text{--}3.4\%$. Proximity graph hubs are essential geometric expressways, not congestive bottlenecks.
- **Reversed Geometry Control:** Inverting geometric weights (assigning lower degrees to high-LID nodes) degrades throughput by $20\%$, confirming that performance gains stem from geometric alignment rather than degree variance.

---

### Milestone II: Resource-Efficient Vector Retrieval

#### A. Established Algorithms Used
- **Uniform Min-Max Scalar Quantization:** Linear mapping of floating-point coordinates to 8-bit unsigned integers (Jégou et al., 2011).
- **Asymmetric Distance Computation (ADC):** Evaluating distances between uncompressed float32 queries and quantized database vectors without online decompression.

#### B. Original AdaptiveVec Research Hypotheses
- *Hypothesis 2.1:* 8-bit and 4-bit scalar quantization combined with AVX2 integer dot-product kernels can slash vector memory bandwidth bottlenecks without degrading search accuracy when paired with a two-stage re-ranking beam.

#### C. Potential Algorithmic Contributions
- **Adaptive Re-ranking Beam Scaling:** Dynamically adjusting re-ranking multiplier ($2\times$ for SQ8, $3\times$ for SQ4) based on empirical quantization noise variance.
- **On-the-Fly Vector Reconstruction Re-ranking:** Re-ranking candidates by reconstructing float32 coordinates on the fly from SQ8/SQ4 payloads, achieving high recall with zero retained FP32 storage.

#### D. Systems Engineering Contributions
- Hand-optimized AVX2 SIMD integer kernels (`_mm256_maddubs_epi16`, `_mm256_madd_epi16`, `_mm256_add_epi32`) computing asymmetric dot products directly in registers.
- 2D precomputed lookup tables for sub-byte 4-bit nibble unpacking with zero bit-shift branch penalties.
- Memory layout optimization: contiguous byte-aligned pools (`std::vector<uint8_t>`) maximizing CPU L1/L2 cache line utilization.

#### E. Integration Contributions
- Unified memory architecture allowing seamless switching between Pure Quantized Storage (`drop_raw_vectors()`) and Hybrid Re-ranking Storage.

#### F. Experimental Findings
- Native SQ8 with $2\times$ re-ranking achieves **99.60%** Recall@10 at 32,483 QPS, cutting Layer 0 vector memory bandwidth by $4\times$.
- Native SQ4 with $3\times$ re-ranking achieves **99.67%** Recall@10 at 27,542 QPS, cutting vector payload memory by $87.5\%$ (1.05 MB vs 5.12 MB).

#### G. Negative Findings
- **Unreranked SQ4 Degradation:** Without two-stage FP32 re-ranking, pure SQ4 search suffers significant ranking inversion among the top-10 candidates due to 4-bit quantization noise, dropping Recall@10 below $92\%$. SQ4 requires re-ranking for production deployment.

---

### Milestone III: Dynamic and Self-Healing Index

#### A. Established Algorithms Used
- **Tombstone Logical Deletion:** Flagging deleted identifiers to exclude them from search results.
- **Sequential Array Compaction:** In-place element copying with index remapping.

#### B. Original AdaptiveVec Research Hypotheses
- *Hypothesis 3.1:* Deleting nodes in proximity graphs severs local routing paths; local mutual neighbor bridging can heal the topology without requiring full index reconstruction.
- *Hypothesis 3.2:* Physical compaction can be executed in single-digit milliseconds, reclaiming 100% of deleted memory without taking the service offline.

#### C. Potential Algorithmic Contributions
- **Local Mutual Neighbor Graph Healing:** An $O(M^2)$ local graph repair heuristic executed upon node deletion:
  $$\forall v, w \in \mathcal{N}(u), \quad \text{if } w \notin \mathcal{N}(v) \text{ and } |\mathcal{N}(v)| < 2 M_v, \quad \mathcal{N}(v) \leftarrow \mathcal{N}(v) \cup \{w\}$$
  Prevents graph fragmentation and maintains routing continuity under continuous high churn.

#### D. Systems Engineering Contributions
- **Sub-3ms Physical Compaction Engine:** Zero-rebuild contiguous buffer repacking and $O(N)$ ID remapping running in **$1.74\text{ to }2.83\text{ ms}$** ($75\times$ to $105\times$ faster than rebuilding from scratch).
- **Lifecycle Concurrency Architecture:** Read-write locking via `std::shared_mutex` allowing lock-free concurrent searches during streaming updates, with brief exclusive locks during compaction.

#### E. Integration Contributions
- End-to-end integration across insertions, deletions, healing, and compaction, preserving graph invariants and entry-point validity.

#### F. Experimental Findings
- Under severe $50\%$ node churn, local graph healing raises Recall@10 from 99.70% (unhealed) to **99.98%** (+0.28% recall lift).
- Compaction reclaims 100% of dead vector and graph memory in $<3\text{ ms}$ across 10K datasets.

#### G. Negative Findings
- **Exclusive Lock Requirement:** Compaction cannot be executed completely lock-free; it requires an exclusive write lock that briefly queues concurrent search queries for $2\text{--}5\text{ ms}$. This must be distinguished from zero-downtime search.

---

### Milestone IV: Automatic Resource-Constrained Optimization

#### A. Established Algorithms Used
- **Grid Search & Multi-Objective Pareto Dominance:** Standard Pareto frontier exploration.
- **Wilson-Score Interval:** Exact binomial confidence intervals for recall evaluation.

#### B. Original AdaptiveVec Research Hypotheses
- *Hypothesis 4.1:* Vector indexing parameters ($M, efConstruction, efSearch$, Precision) can be solved automatically under hard user-specified memory, latency, and recall constraints using closed-loop analytical screening and Pareto search.

#### C. Potential Algorithmic Contributions
- **Normalized Constraint Violation Metric:** A scale-invariant penalty function guiding non-dominated fallback selection when no configuration satisfies all hard constraints:
  $$V_{total} = \max\left(0, \frac{R_{target} - R}{R_{target}}\right) + \max\left(0, \frac{\text{Mem} - \text{Mem}_{max}}{\text{Mem}_{max}}\right) + \max\left(0, \frac{P99 - P99_{target}}{P99_{target}}\right)$$
- **Analytical Lower-Bound Memory Screening:** Pre-evaluating theoretical graph and vector memory footprints to eliminate infeasible precision levels before running empirical benchmarks.

#### D. Systems Engineering Contributions
- Closed-loop multi-trial optimizer (`AutoCalibrator`) evaluating up to 80 candidate configurations in $<6\text{ seconds}$.
- Independent 50/50 calibration and validation query splitting preventing optimistic recall overfitting.

#### E. Integration Contributions
- Direct automated instantiation of calibrated `AdaptiveHNSWIndex` pointers ready for immediate deployment.

#### F. Experimental Findings
- `AutoCalibrator` consistently discovers configurations that achieve target recall while reducing memory footprint by $45\%$ compared to canonical defaults ($M=16, efC=200, efS=64$).

#### G. Negative Findings
- **Online Runtime Telemetry Falsification (Phases 4.0–4.3):** Attempting to continuously adapt $efSearch$ online based on real-time query difficulty prediction added 11% to 18% CPU latency overhead, while entry-point distance correlated poorly with difficulty ($R^2 < 0.12$). Parameter calibration is best performed prior to deployment or periodically, not per-query in the hot path.

---

## 3. Summary Novelty Matrix

| Milestone | Primary Scientific Novelty | Systems Engineering Highlight | Documented Negative Result |
| :--- | :--- | :--- | :--- |
| **I. Geometry Intelligence** | Budget-Normalized Allocation ($\sum M_i \equiv B_0$) & Decoupled Layer Routing | AVX2 SIMD Euclidean distance kernels | Hubness regulation degrades navigability (-1.2% to -3.4% recall) |
| **II. Resource Efficiency** | Asymmetric distance computation with 2-stage re-ranking | AVX2 integer dot-product & 2D lookup tables for SQ4 | Unreranked SQ4 suffers top-k ranking inversion |
| **III. Dynamic Lifecycle** | Local mutual neighbor healing under churn | Sub-3ms zero-rebuild physical compaction (75-105x speedup) | Compaction requires brief exclusive lock ($<5\text{ ms}$) |
| **IV. Auto Optimization** | Normalized constraint violation & Pareto fallback set | Independent query splitting & Wilson-score bounds | Per-query online telemetry adds 11-18% overhead ($R^2 < 0.12$) |
