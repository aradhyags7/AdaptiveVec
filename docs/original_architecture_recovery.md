# AdaptiveVec: Original Architecture Recovery Monograph
**Comprehensive Reconstruction of the Original Research Philosophy, Mathematical Formulations, and System Mechanisms**

---

## 1. Executive Direction & Historical Intent

The fundamental research objective of **AdaptiveVec** as originally conceived in the conference research monograph (*v1.0.0-submission*, commit `9c22f4b`) was:

> **"Adapt the graph to the data, rather than continuously adapting a controller to the graph."**

Canonical Hierarchical Navigable Small World (HNSW) proximity graphs enforce a rigid, globally uniform parameterization ($M=16, efConstruction=200$) across the entire vector space. This uniform allocation exhibits severe structural pathologies across real-world, non-homogeneous manifolds:

1. **Redundant Edge Bloat in Dense Subspaces**: In dense clusters with low Local Intrinsic Dimensionality (LID), vectors populate low-dimensional sub-manifolds. Enforcing uniform degree quotas ($M=16$ or $M=32$) allocates redundant, co-linear proximity links that squander DRAM and memory bus bandwidth with zero marginal routing utility.
2. **Topological Starvation & Hubness in Sparse Boundary Subspaces**: In sparse high-dimensional regions, uniform edge budgets cause capacity starvation. Under the **Hubness Phenomenon**, central vertices accumulate disproportionate in-degrees and act as routing bottlenecks, degrading tail latency ($p95/p99$) and trapping greedy search trajectories.
3. **The Memory-Bandwidth Bottleneck**: High-dimensional vector search on commodity hardware (laptops, edge instances with 8–16 GB RAM and constrained memory buses) is strictly memory-bandwidth bound. Traversal requires fetching uncompressed vectors ($512\ \text{bytes}$ per 128D FP32 vector), saturating memory channels.
4. **Query-Time Static Budget Penalty**: Enforcing a uniform $efSearch$ forces easy queries in dense clusters to compute excessive redundant hops, while difficult boundary queries fail to discover true nearest neighbors.

AdaptiveVec resolved these challenges through **structural intelligence embedded directly into the proximity graph during construction**, augmented by dual-stage scalar quantization and lightweight stagnation early-exit.

---

## 2. Mathematical Formulation & Online Geometric Signals

Let $\mathcal{X} = \{x_1, x_2, \dots, x_N\} \subset \mathbb{R}^D$ denote a corpus of $N$ vectors embedded in ambient dimension $D$, with metric distance function $d(x, y)$ ($L_2$ Euclidean or Cosine distance).

### 2.1 Local Intrinsic Dimensionality (LID) Estimation

Given a local neighborhood of candidate distances sorted in ascending order $d(x, v_1) \le d(x, v_2) \le \dots \le d(x, v_k)$, AdaptiveVec computes the Maximum Likelihood Estimator (MLE) of LID:

$$\widehat{\text{LID}}(x) = -\left[ \frac{1}{k - 1} \sum_{i=1}^{k-1} \ln \left( \frac{d(x, v_i)}{d(x, v_k)} \right) \right]^{-1}$$

Where:
- $d(x, v_k) = r_k$ represents the maximum radius of the local exploratory sphere.
- $k$ is the neighborhood sampling size (default $k=15$).
- In low-dimensional sub-manifolds, $\widehat{\text{LID}}(x) \in [2.0, 8.0]$. In high-dimensional diffuse clouds or boundary zones, $\widehat{\text{LID}}(x) \ge 24.0$.

#### Numerical Stability & Invariants:
- **Zero-Distance / Duplicate Vector Handling**: If $r_k \le 10^{-7}$, the neighborhood consists of identical or co-located points; the estimator immediately returns $\widehat{\text{LID}} = 1.0$.
- **Ratio Clamping**: Ratios $\frac{d(x, v_i)}{r_k}$ are clamped to $[10^{-7}, 1.0 - 10^{-7}]$ to prevent $\ln(0)$ singularity or $\ln(1) = 0$ division-by-zero artifacts.
- **Log Sum Clamping**: If $|\sum \ln(\text{ratio})| < 10^{-7}$, the estimator returns $1.0$.
- **Output Bounds**: The estimated LID is bounded to $[1.0, 1000.0]$.
- **Distance Metric Semantics**: The original implementation evaluated Euclidean distance ($d = \sqrt{\sum (x_i - y_i)^2}$) rather than squared Euclidean distance ($d^2$). Note that in MLE LID, since $\ln(d^2 / r_k^2) = 2 \ln(d / r_k)$, using squared distances without dividing by 2 halves the estimated LID. The original code strictly used unsquared metric distances.

### 2.2 Local Density Estimation

To capture localized point spacing independent of dimensional expansion rate, Local Density $D(x)$ is formulated as the mean metric distance to the $k$ nearest probed neighbors:

$$D(x) = \frac{1}{k} \sum_{i=1}^k d(x, v_i)$$

#### Semantic Direction:
- A **larger** $D(x)$ corresponds to greater average distance to neighbors, representing a **sparser** neighborhood.
- A **smaller** $D(x)$ represents a tight, **dense** cluster.
- In AdaptiveVec's difficulty score, higher values represent harder/sparser regions requiring more graph connectivity.

### 2.3 Streaming Online Welford Tracking

To eliminate expensive offline pre-clustering passes, AdaptiveVec maintains running distribution moments online in $\mathcal{O}(1)$ time using Welford's algorithm:

$$\delta_n = z_n - \bar{M}_{n-1}$$
$$\bar{M}_n = \bar{M}_{n-1} + \frac{\delta_n}{n}$$
$$S_n = S_{n-1} + \delta_n \cdot (z_n - \bar{M}_n)$$
$$\sigma_n = \sqrt{\frac{S_n}{n - 1}}$$

AdaptiveVec maintains two independent Welford trackers:
1. `lid_tracker`: Tracks running mean $\mu_{\text{LID}}$ and standard deviation $\sigma_{\text{LID}}$.
2. `density_tracker`: Tracks running mean $\mu_D$ and standard deviation $\sigma_D$.

Optionally, an Exponential Moving Average (EMA) mode with decay $\alpha_{\text{ema}} = 0.05$ adapts to non-stationary vector ingestion streams.

### 2.4 Geometric Complexity Score

The normalized geometric complexity score $S(x)$ combines standardized z-scores of intrinsic dimensionality and sparsity:

$$z_{\text{LID}}(x) = \frac{\widehat{\text{LID}}(x) - \mu_{\text{LID}}}{\sigma_{\text{LID}} + 10^{-6}}$$
$$z_D(x) = \frac{D(x) - \mu_D}{\sigma_D + 10^{-6}}$$
$$S(x) = \text{clip}\left( \alpha \cdot z_{\text{LID}}(x) + \beta \cdot z_D(x), -2.5, +2.5 \right)$$

Default configuration: $\alpha = 0.5$, $\beta = 0.5$.
- When a node is in a dense, low-dimensional cluster: $z_{\text{LID}} < 0$ and $z_D < 0 \implies S(x) < 0$.
- When a node is in a sparse, high-dimensional boundary region: $z_{\text{LID}} > 0$ and $z_D > 0 \implies S(x) > 0$.

---

## 3. Adaptive Graph Construction & Topology Modulation

### 3.1 Online Zero-Clustering Probing ($<0.8\%$ Overhead)

During vector insertion of $x$:
1. The vector descends greedily from top level $L_{\text{max}}$ down to insertion level $l_{\text{target}} = \min(L_{\text{max}}, l_{\text{new}})$ using standard 1-NN routing.
2. At level $l_{\text{target}}$, AdaptiveVec executes a single lightweight probe beam search with budget:
   $$k_{\text{probe}} = \min\left(25, \max\left(10, \left\lfloor \frac{efC_{\text{base}}}{4} \right\rfloor \right)\right)$$
3. The discovered candidate distances directly feed the LID and Density estimators. Because $k_{\text{probe}} \le 25 \ll efC_{\text{base}} = 120$, probing adds less than $0.8\%$ computational overhead to total index construction time.

### 3.2 Dynamic Degree and Construction Allocation

AdaptiveVec dynamically maps the complexity score $S(x)$ to a node-specific base degree quota $M(x)$ and construction beam $efC(x)$:

$$M(x) = \text{clip}\left( \left\lfloor M_{\text{base}} \cdot (1 + \gamma \cdot S(x)) \right\rceil, M_{\text{min}}, M_{\text{max}} \right)$$
$$efC(x) = \text{clip}\left( \left\lfloor efC_{\text{base}} \cdot (1 + \gamma \cdot S(x)) \right\rceil, efC_{\text{min}}, efC_{\text{max}} \right)$$

Default parameters:
- $M_{\text{base}} = 16$, $M_{\text{min}} = 8$, $M_{\text{max}} = 24$ (or $32$).
- $efC_{\text{base}} = 120$, $efC_{\text{min}} = 40$, $efC_{\text{max}} = 220$.
- Policy sensitivity $\gamma = 0.40$.

In low-complexity clusters ($S(x) \approx -2.5$), $M(x) \to 8$ and $efC(x) \to 40$. In high-complexity boundary regions ($S(x) \approx +2.5$), $M(x) \to 24$ and $efC(x) \to 220$.

### 3.3 Layer-Decoupled Dynamic Link Scaling

In hierarchical proximity graphs, higher layers ($l \ge 1$) serve purely as coarse expressway routing layers. Allocating large edge budgets at higher levels is redundant. AdaptiveVec geometrically compresses higher-layer edge capacity:

$$M^{(l)}(x) = \max\left( M_{\text{min}}^{(l)}, \left\lfloor M(x) \cdot \lambda^l \right\rfloor \right)$$
$$M_{\text{max}}^{(l)}(x) = \max\left( M_{\text{min}}^{(l)}, \left\lfloor M_{\text{max}}(x) \cdot \lambda^l \right\rfloor \right)$$

Where decay factor $\lambda = 0.75$ and minimum layer degree $M_{\text{min}}^{(l)} = 4$. Layer 0 retains full unscaled degree $M^{(0)}(x) = M(x)$ and $M_{\text{max}}^{(0)}(x) = 2 \cdot M(x)$.

### 3.4 Hubness-Aware In-Degree Regulation

Under canonical Relative Neighborhood Graph (RNG) neighbor pruning (Algorithm 4 of Malkov & Yashunin), vertices with high centrality accumulate excessive in-degree links, creating topological hubs that trap routing paths.

AdaptiveVec penalizes over-connected hubs by evaluating candidate edges under an effective distance metric:

$$d_{\text{eff}}(u, v) = d(u, v) \cdot \left[ 1 + \mu \cdot \left( \frac{\text{deg}_{\text{in}}(v)}{\bar{\text{deg}}_{\text{in}}} \right) \right]$$

Where:
- $\text{deg}_{\text{in}}(v)$ is the tracked in-degree of candidate neighbor $v$ at Layer 0.
- $\bar{\text{deg}}_{\text{in}}$ is the current mean in-degree across all indexed nodes.
- Penalty coefficient $\mu = 0.15$.

This penalty disfavors over-connected hubs during candidate selection without distorting true metric distances, shrinking in-degree variance by $>55\%$ and eliminating bottleneck vertices.

---

## 4. Search Optimizations and Quantization Regimes

### 4.1 Distance Stagnation Early Exit

During query-time beam search at Layer 0, candidate expansion often exhibits severe diminishing returns after the top-$k$ nearest neighbors have been trapped. AdaptiveVec monitors the continuous improvement of the best discovered distance $d_{\text{best}}$:

$$\Delta = d_{\text{best}}^{(\text{old})} - d_{\text{best}}^{(\text{new})}$$

Search terminates if $\Delta < \epsilon$ ($\epsilon = 10^{-4}$) for $p = 6$ consecutive expansion steps after the candidate beam $W$ has accumulated at least $efSearch$ candidates.

Crucially, $d_{\text{best}}$ is updated against all newly discovered neighbors ($d_{\text{best}} \leftarrow \min(d_{\text{best}}, d_{\text{new}})$) rather than evaluating consecutive popped elements from the candidate queue (which are monotonically non-decreasing).

### 4.2 Query-Time Dynamic efSearch (Frontier 1)

Rather than enforcing a static candidate beam width $efSearch$ across heterogeneous queries, AdaptiveVec modulates the search budget online according to estimated query difficulty:

$$S(q) = 0.7 \cdot S(\text{entry\_node}) + 0.3 \cdot \text{clip}\left( \frac{d(q, \text{entry}) - \mu_D}{\sigma_D}, -2.0, 2.0 \right)$$
$$efSearch(q) = \text{clip}\left( \left\lfloor ef_{\text{base}} \cdot (1 + \kappa \cdot S(q)) \right\rceil, ef_{\text{min}}, ef_{\text{max}} \right)$$

Default: $\kappa = 0.60$, $ef_{\text{base}} = 64$, $ef_{\text{min}} = 24$, $ef_{\text{max}} = 128$.

### 4.3 Dual-Stage Vector Quantization: SQ8 & Ultra-Edge SQ4

1. **8-Bit Scalar Quantization (SQ8)**: Maps each dimension to $[0, 255]$ via per-dimension min/max affine scaling. Reduces vector storage by $4\times$ ($128\ \text{bytes}$ per 128D vector).
2. **4-Bit Scalar Quantization (SQ4)**: Maps dimensions to 16 levels ($[0, 15]$). Pairs of 4-bit nibbles are packed into contiguous uint8 bytes:
   $$\text{PackedByte}_k = q_{2k}^{(4)} \mid (q_{2k+1}^{(4)} \ll 4)$$
   Reduces vector memory by $8\times$ ($64\ \text{bytes}$ per 128D vector). AVX2 SIMD unpacks nibbles into 32-bit registers for hardware dot-product evaluation.
3. **Exact Float32 Re-Ranking**: Graph beam search retrieves an expanded candidate set $R = \alpha \cdot k$ ($\alpha = 2.0$, $R = 20$) using quantized distances, followed by exact FP32 Euclidean re-ranking.

---

## 5. Lifecycle Architecture: Deletions, Healing, and Compaction

### 5.1 Two-Tier Node Lifecycle
1. **Tier 1: Localized Graph Healing**:
   - Marking node $v$ deleted sets a bit in `deleted[v] = true` and decrements active count.
   - Surviving outgoing neighbors of $v$ are retained as temporary **soft transit bridges** during online search.
   - For every incoming predecessor $u \in \text{preds}(v)$, edge $(u \to v)$ is severed, and replacement candidates are drawn from $v$'s surviving neighbors via RNG heuristic pruning up to $M_{\max}(u)$.
   - Mutual neighbor bridging connects pairs of surviving neighbors $(w_1, w_2)$, preserving local biconnectivity.
   - 4,096 striped mutexes (`node_locks[i % 4096]`) guarantee deadlock-free concurrency.
2. **Automatic Entry-Point Migration**:
   - If entry point $ep$ is marked deleted, a top-down search identifies the surviving node with the highest degree on the highest active level.
3. **Tier 2: Online Zero-Downtime Compaction (`compact()`)**:
   - When cumulative tombstone churn reaches $20\%$--$50\%$, `compact()` packs surviving vectors into contiguous physical memory buffers $[0, N_{\text{active}}-1]$.
   - Adjacency tables are remapped, dead edges are purged, and vector buffers are re-allocated. In benchmark evaluations, compaction executed in $62.1\ \text{ms}$ on SIFT-50K ($46\times$ faster than rebuilding from scratch).

---

## 6. Historical Performance Claims & Verification Status

| Benchmark Domain | Metric Claimed in v1.0.0 Monograph | Observed Empirical Reality (Phases 0–4.3) | Root Cause / Discrepancy Note |
|:---|:---|:---|:---|
| **SIFT-100K Edge Pruning** | $-7.4\%$ to $-10.6\%$ total graph edges | Verified ($-7.4\%$ to $-10.7\%$) | Pruning confirmed; graph edge count decreases as intended. |
| **SIFT-100K Search QPS** | $+50.3\%$ QPS ($7,075$ vs $4,708$ QPS) | Unverified under fair baseline | Comparison was made against an un-optimized $L_2$ baseline with `std::unordered_set` visited sets. Under AVX2 squared $L_2$ with flat arrays, canonical HNSW achieves $>20,000$ QPS. |
| **SQ4 Memory Reduction** | $-72.7\%$ total index memory ($16.1$ vs $58.8$ MB) | Verified for raw vector storage | Vector storage shrinks from 48.8 MB to 6.1 MB; graph edge memory remains standard uint32 adjacency. |
| **SQ4 Throughput Acceleration** | $+36.7\%$ QPS over FP32 ($10,809$ vs $7,906$ QPS) | Partially verified | Nibble unpacking SIMD kernel adds instruction overhead that partially offsets memory bandwidth savings on CPU cache-resident workloads. |
| **Dynamic efSearch** | $+35.1\%$ QPS at zero recall loss ($\kappa=0.6$) | Confirmed on synthetic queries, degrades under shift | Online difficulty probing via entry node is noisy and sensitive to query manifold shift. |
| **MiniLM-50K-MultiDomain (384D)** | $54,624$ QPS at $0.838$ Recall@10 | Dataset Discrepancy Identified | Evaluated on 57,893 real passage embeddings (SciFact, NFCorpus, ArguAna, Webmasters, Android), NOT the 100K synthetic dataset used in later phases. |
| **Streaming Deletions** | $+1.13\%$ recall retention over unhealed tombstones | Verified | Localized healing prevents graph fragmentation under high churn. |

---

## 7. Recovery Blueprint for Architecture B

To restore the original research philosophy while eliminating historical implementation defects:
1. **Retain the Geometry-Aware Construction Principle**: Allocate $M(x)$ and $efC(x)$ based on local geometric complexity ($S(x) \propto z_{\text{LID}} + z_D$).
2. **Import Validated Low-Level Systems Engineering**:
   - Zero-allocation generation-tagged `VisitedTagPool` (eliminating `std::unordered_set`).
   - AVX2/FMA squared Euclidean distance kernels.
   - Contiguous flat vector storage.
   - Exact thread-local search instrumentation (`QuerySearchStats`).
   - Safe striped concurrency.
3. **Purge Expensive Online Search Controllers**:
   - Eliminate runtime workload controllers, query-path reference monitors, and neural/online uncertainty predictors.
   - The graph must be optimized **at construction time** so that query execution remains zero-overhead, deterministic, and fast.
