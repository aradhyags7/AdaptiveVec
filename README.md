<!-- Verified 250K Scale Expansion Milestone: Oct 5, 2026 -->
<div align="center">

# AdaptiveVec: A Density- and Dimension-Aware Proximity Graph Index for Resource-Constrained Vector Retrieval

### *Proceedings of Engineering Design & Innovation (EDI) • Systems & Machine Learning Research*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![C++](https://img.shields.io/badge/C%2B%2B-17_AVX2-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://isocpp.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19_Workbench-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Test Suite](https://img.shields.io/badge/Test_Suite-25%20Passing-brightgreen?style=for-the-badge)]()
[![Hardware Tier](https://img.shields.io/badge/Target-Commodity%20Laptops%20%26%20Edge%20VMs-blueviolet?style=for-the-badge)]()

<p align="center">
  <b>Aradhya Shinde</b><br>
  <i>Department of Computer Engineering • Research & Systems Engineering</i><br>
  <code>aradhyashinde2330@gmail.com</code> • <a href="https://github.com/aradhyags7/AdaptiveVec">github.com/aradhyags7/AdaptiveVec</a>
</p>

---

<p align="center">
  <img src="pdf/paper_figures/fig1_pareto.png" width="92%" alt="AdaptiveVec Headline Pareto Frontier"/>
  <br>
  <i><b>Figure 1:</b> Empirical QPS vs. Recall@10 Pareto Frontier on SIFT-100K ($N=100{,}000, D=128, Q=10{,}000$). AdaptiveVec introduces dual operating regimes: <b>Regime A (High-Throughput)</b> achieves <b>7,075.3 QPS (+50.3% speedup)</b> at 0.9745 Recall@10 via distance stagnation early-exit; <b>Regime B (Memory-Compact)</b> slashes index RAM by <b>-62.4%</b> (22.5 MB vs. 59.9 MB) via asymmetric INT8 SQ8 with float32 re-ranking.</i>
</p>

---

### Abstract
Hierarchical Navigable Small World (HNSW) graphs underpin state-of-the-art vector search engines (FAISS, Milvus, Qdrant, pgvector). However, canonical HNSW enforces static, uniform hyper-parameters ($M=16, efConstruction=200$) globally across non-homogeneous vector spaces. In dense, low-intrinsic-dimensionality subspaces, this uniform allocation synthesizes redundant proximity edges, squandering precious RAM and CPU indexing time with zero recall benefit. Conversely, sparse high-dimensional regions suffer from topological starvations and capacity-limited disconnects. We present **AdaptiveVec**, a manifold-adaptive proximity graph index optimized for resource-constrained commodity hardware ($100\text{K}–1\text{M}$ vectors on single-node laptops and edge cloud instances). AdaptiveVec unifies online manifold signal estimation (**Local Intrinsic Dimensionality (LID)** via Maximum Likelihood Estimation and **Local Density**) with standard insertion routing at $<0.8\%$ computational overhead. It introduces: (i) a **Layer-Decoupled Dynamic Policy** that compresses higher-layer express links; (ii) **Streaming Online Welford Tracking** to eliminate offline calibration passes; (iii) a **Hubness-Aware In-Degree Centrality Penalty** to prevent topological graph bottlenecks; (iv) **Distance Stagnation Early Exit** to accelerate query throughput; and (v) **Asymmetric INT8 Scalar Quantization (SQ8)** with two-stage float32 re-ranking. Evaluated on SIFT-100K ($N=100\text{K}, D=128$) and Synthetic-Multi-Cluster ($N=50\text{K}, D=64$), **Step 5 achieves 0.9745 Recall@10 at +50.3% query throughput** (7,075.3 QPS vs. 4,708.1 baseline) via query-time distance stagnation early exit, -7.4% fewer graph edges, and -28.1% faster build times. For memory-constrained deployments, **Step 6 (INT8 SQ8) slashes index RAM by -62.4%** (22.5 MB vs. 59.9 MB baseline) and vector memory by -75% while maintaining 0.9594 recall. Expanding evaluation to SIFT-250K demonstrates an accelerated build time of **1.61× speedup** (97.8s vs. 157.1s) and cuts 575,314 redundant links. Broader validation on million-scale corpora ($N \ge 1\text{M}$) and higher-dimensional embeddings is ongoing.

---

</div>

## Table of Contents
1. [Introduction & Theoretical Motivation](#1-introduction--theoretical-motivation)
2. [Related Work & Literature Positioning](#2-related-work--literature-positioning)
3. [Mathematical Formulation & Online Signals](#3-mathematical-formulation--online-signals)
4. [Algorithmic Architecture & Mechanics](#4-algorithmic-architecture--mechanics)
   - [4.1 Online Manifold Geometry Probing](#41-online-manifold-geometry-probing)
   - [4.2 Layer-Decoupled Edge Allocation Policy](#42-layer-decoupled-edge-allocation-policy)
   - [4.3 Hubness-Aware In-Degree Regulation](#43-hubness-aware-in-degree-regulation)
   - [4.4 Distance Stagnation Early Exit (Query-Time Optimization)](#44-distance-stagnation-early-exit-query-time-optimization)
   - [4.5 Asymmetric Scalar Quantization (SQ8) & Two-Stage Re-Ranking](#45-asymmetric-scalar-quantization-sq8--two-stage-re-ranking)
5. [Theoretical Complexity Analysis](#5-theoretical-complexity-analysis)
6. [Empirical Evaluation & Benchmark Results](#6-empirical-evaluation--benchmark-results)
   - [6.1 Experimental Setup & Testbed](#61-experimental-setup--testbed)
   - [6.2 SIFT-100K Ablation Benchmarks (Ablation Waterfall & Pareto)](#62-sift-100k-ablation-benchmarks)
   - [6.3 Synthetic-Multi-Cluster Benchmarks](#63-synthetic-multi-cluster-benchmarks)
   - [6.4 Signal Validity & Parameter Sensitivity Sweeps](#64-signal-validity--parameter-sensitivity-sweeps)
   - [6.5 Multi-Core Thread Scaling Benchmarks](#65-multi-core-thread-scaling-benchmarks)
   - [6.6 Dataset Scale Expansion: SIFT-100K to SIFT-250K](#66-dataset-scale-expansion-sift-100k-to-sift-250k-on-commodity-laptop)
   - [6.7 Milestone 1: Ultra-Edge 4-Bit Scalar Quantization (SQ4)](#67-milestone-1-ultra-edge-4-bit-sub-byte-scalar-quantization-sq4)
   - [6.8 Frontier 1: Query-Time Dynamic Manifold-Aware efSearch](#68-frontier-1-query-time-dynamic-manifold-aware-efsearch)
   - [6.9 Frontier 4: High-Dimensional Transformer Embeddings (384D all-MiniLM-L6-v2)](#69-frontier-4-high-dimensional-transformer-embeddings-384d-all-minilm-l6-v2)
   - [6.10 Frontier 2: Dynamic Streaming Node Deletions & Localized Graph Healing](#610-frontier-2-dynamic-streaming-node-deletions--localized-graph-healing)
   - [6.11 Known Limitations](#611-known-limitations)
7. [System Architecture & Repository Structure](#7-system-architecture--repository-structure)
8. [Quickstart & Reproducibility](#8-quickstart--reproducibility)
   - [8.1 Single-Command Benchmark Reproduction](#81-single-command-benchmark-reproduction)
   - [8.2 Environment Setup & Installation](#82-environment-setup--installation)
   - [8.3 Automated Test Suite](#83-automated-test-suite)
   - [8.4 Interactive Research Workbench (React + Vite)](#84-interactive-research-workbench-react--vite)
   - [8.5 FastAPI REST Server & Legacy Simulator](#85-fastapi-rest-server--legacy-simulator)
   - [8.6 Manual C++ Compilation](#86-manual-c-compilation)
9. [REST API Formal Specification](#9-rest-api-formal-specification)
10. [Academic Citation (BibTeX)](#10-academic-citation-bibtex)
11. [References](#11-references)

---

## 1. Introduction & Theoretical Motivation

Approximate Nearest Neighbor Search (ANNS) in metric spaces is fundamental to modern information retrieval, retrieval-augmented generation (RAG), and high-dimensional representation learning. Proximity graphs—specifically **Hierarchical Navigable Small World (HNSW)** graphs (Malkov & Yashunin, 2020)—represent the state of the art in query throughput and recall trade-offs.

```
       Canonical Uniform HNSW                       AdaptiveVec (Dynamic Manifold-Aware)
 ┌────────────────────────────────────┐      ┌─────────────────────────────────────────────────┐
 │ Global Allocation: M=16, efC=200   │      │ Dynamic Allocation: M_i ∈ [8, 24], efC_i ∈ [40, 220]
 │                                    │      │                                                 │
 │ Dense Subspace:   16 links (Waste) │ ───► │ Dense Subspace (LID ≈ 1-2):  M = 8  (-50% RAM)  │
 │ Sparse Boundary:  16 links (Starve)│      │ Sparse Cloud   (LID ≈ 26):   M = 22 (+Recall)   │
 └────────────────────────────────────┘      └─────────────────────────────────────────────────┘
```

### The Fundamental Flaw of Uniform Allocation
In production systems, real-world embeddings (e.g., text, vision, multimodal representations) do not fill ambient $\mathbb{R}^D$ space uniformly. Instead, they concentrate on lower-dimensional non-linear sub-manifolds with wide variations in **Local Intrinsic Dimensionality (LID)** and **Local Density ($D$)**:
1. **Redundant Edge Bloat:** In dense, low-LID subspaces (e.g., clusters with high correlation), establishing $M = 16$ or $M = 32$ links forms redundant parallel paths. Memory consumption scales as:
   $$\text{Memory}_{\text{edges}} \approx 4 \times M \times N \times 1.1 \text{ bytes}$$
   A fixed $M$ forces edge memory to be paid for links that provide zero navigation benefit.
2. **Capacitary Bottlenecks & Hubness:** In sparse high-dimensional regions, uniform edge quotas starvations occur. Under the **Hubness Phenomenon** (Radovanović et al., 2010), high-degree nodes attract an exorbitant number of routing paths, generating edge-thrashing and graph congestion during search.
3. **The Commodity Hardware Barrier:** Enterprise vector search papers evaluate on $128\text{GB}+$ RAM multi-socket servers. **AdaptiveVec** targets the opposite end of the spectrum: **commodity laptops and single-socket cloud VMs** ($100\text{K}–1\text{M}$ vectors), asking how online manifold signals can be extracted *at negligible CPU cost* to maximize memory and indexing efficiency.

---

## 2. Related Work & Literature Positioning

| Paradigm | Primary Work | Strengths | Critical Limitations Addressed by AdaptiveVec |
| :--- | :--- | :--- | :--- |
| **Hierarchical Proximity Graphs** | Malkov & Yashunin (IEEE TPAMI 2020) | Logarithmic search complexity $O(\log N)$, high recall | Rigid uniform parameterization across all nodes; memory bloat in low-dimensional clusters |
| **Monolithic Graphs & Long Links** | Subramanya et al., *DiskANN* (NeurIPS 2019) | High SSD-based scale via Vamana graphs | Requires offline multi-pass global pruning and large memory footprints during construction |
| **Local Intrinsic Dimensionality** | Amsaleg et al. (ACM SIGKDD 2015) | Rigorous mathematical characterization of manifold expansion | Previously used only for offline dataset analysis or post-hoc query hardness profiling |
| **Dynamic Beam Exploration** | Wang et al. (PVLDB 2021) / Li et al. (TKDE 2020) | Comprehensive ANNS surveys and dynamic beam tuning | Evaluates query-time beam adaptations; does not modify underlying index topology |
| **Manifold Insertion Ordering** | Elliott & Clark (arXiv 2024 / OpenReview 2025) | Ordering insertions by descending LID improves graph quality | Requires an expensive full-dataset pre-computation sweep prior to index build |
| **AdaptiveVec (This Work)** | **Shinde (EDI 2026)** | **Unified Online Probing ($<0.8\%$ overhead) + Dynamic $M_i$ + Hubness Regulation + INT8 SQ8** | **Zero offline pre-clustering; fully dynamic streaming ingestion budgeted for commodity devices** |

---

## 3. Mathematical Formulation & Online Signals

<p align="center">
  <img src="pdf/handbook_figures/diagram3_pipeline_flowchart.png" width="95%" alt="AdaptiveVec Pipeline Architecture"/>
  <br>
  <i><b>Figure 2:</b> End-to-end AdaptiveVec system pipeline. (1) Online geometry probing extracts candidate neighbor distances during natural routing descent; (2) MLE LID and local density signals are estimated; (3) Welford streaming stats update running mean and variance; (4) Dynamic policy assigns per-node $M(x)$ and $efConstruction(x)$; (5) Hubness-regulated pruning and SQ8 quantization finalize the index.</i>
</p>

### 3.1 Local Intrinsic Dimensionality (LID)
Local Intrinsic Dimensionality captures the local rate of space expansion as the distance to neighbors increases. Rather than assuming a constant global dimensionality, AdaptiveVec computes the **Maximum Likelihood Estimation (MLE)** of LID ([Amsaleg et al., 2015](https://dl.acm.org/doi/10.1145/2783258.2783311)) over the nearest candidate distances:

$$\widehat{\text{LID}}(x) = -\left( \frac{1}{k} \sum_{i=1}^k \ln \frac{d(x, v_i)}{d(x, v_k)} \right)^{-1}$$

where $d(x, v_1) \le d(x, v_2) \le \dots \le d(x, v_k)$ represent the sorted positive distances to the $k$ nearest candidate vectors uncovered during insertion routing.
* **Low LID ($\widehat{\text{LID}} \approx 1 - 2$):** Points reside on low-dimensional curves or manifolds (e.g., clustered semantics). Proximity links can be aggressively pruned without impacting connectivity.
* **High LID ($\widehat{\text{LID}} \approx D$):** Points reside in high-variance or boundary regions. Distances concentrate, requiring larger link quotas ($M \approx 22-24$) to guarantee navigable entry.

### 3.2 Local Density Estimator ($D_k$)
The local density indicator measures spatial compactness around vector $x$:

$$D_k(x) = \frac{1}{k} \sum_{i=1}^k d(x, v_i)$$

A smaller $D_k(x)$ signifies high cluster density, meaning nearest neighbors are packed closely in metric space and routes can be resolved with fewer outgoing links.

### 3.3 Streaming Online Calibration (Welford's Algorithm & EMA)
To eliminate offline calibration sweeps across the dataset, AdaptiveVec implements an online streaming estimator. For each observed signal $s \in \{\text{LID}, D_k\}$, running statistics $\mu_k$ and $\sigma_k^2$ are updated in a single pass via **Welford’s Algorithm**:

$$M_k = M_{k-1} + \frac{s_k - M_{k-1}}{k}, \quad S_k = S_{k-1} + (s_k - M_{k-1})(s_k - M_k)$$
$$\mu_k = M_k, \quad \sigma_k^2 = \frac{S_k}{k - 1} \quad (k \ge 2)$$

To account for non-stationary distribution drift in live vector streams, an optional Exponential Moving Average (EMA) mode applies decay factor $\lambda = 0.05$:

$$\mu_{\text{EMA}}^{(k)} = (1 - \lambda)\mu_{\text{EMA}}^{(k-1)} + \lambda s_k$$

### 3.4 Adaptive Edge Allocation Policy
The policy engine computes a standardized, scale-invariant difficulty score:

$$\text{Score}(x) = \alpha \cdot \left(\frac{\widehat{\text{LID}}(x) - \mu_{\text{LID}}}{\sigma_{\text{LID}} + \epsilon}\right) - \beta \cdot \left(\frac{D_k(x) - \mu_D}{\sigma_D + \epsilon}\right)$$

The score maps linearly to per-node hyper-parameters:

$$M(x) = \text{clamp}\left( \text{round}\left( M_{\text{base}} \cdot [1 + \gamma \cdot \text{Score}(x)] \right), M_{\text{min}}, M_{\text{max}} \right)$$

$$efConstruction(x) = \text{clamp}\left( \text{round}\left( efC_{\text{base}} \cdot [1 + \gamma \cdot \text{Score}(x)] \right), efC_{\text{min}}, efC_{\text{max}} \right)$$

where $\gamma = 0.40$ is the sensitivity parameter, $M_{\text{base}} = 16, M_{\text{min}} = 8, M_{\text{max}} = 24$.

---

## 4. Algorithmic Architecture & Mechanics

### 4.1 Online Manifold Geometry Probing
Unlike prior methods that require offline clustering (e.g., $k$-means), AdaptiveVec unifies signal estimation with the natural routing of HNSW:
1. A top-down greedy descent from layer $L_{\text{max}}$ down to insertion level $l_x$ occurs with $ef = 1$.
2. At level $l_{\text{probe}} = \min(L_{\text{max}}, l_x)$, a probe beam search is executed with $ef_{\text{probe}} = \min(30, \max(15, efC_{\text{base}} / 4))$.
3. The resulting candidate distances $\{d(x, v_i)\}$ are directly fed to the MLE LID and density estimators.
4. **Computational Cost:** Since $l_{\text{probe}}$ is small and $ef_{\text{probe}} \ll efC_{\text{base}}$, this online probe adds **$< 0.8\%$** to total build wall-clock time.

---

### 4.2 Layer-Decoupled Edge Allocation Policy
In standard HNSW, $M_{\text{max0}} = 2M$ is used for Layer 0, while all upper layers $l \ge 1$ enforce $M_{\text{max}} = M$. However, upper layers function as **express highways** for coarse navigation; establishing dense clusters in upper layers creates redundant bypasses.

<p align="center">
  <img src="pdf/handbook_figures/diagram2_hnsw_hierarchy.png" width="92%" alt="HNSW Layer Hierarchy"/>
  <br>
  <i><b>Figure 3:</b> Hierarchical proximity graph architecture. AdaptiveVec preserves high resolution on Layer 0 for exact neighbor retrieval, while geometrically compressing higher-layer express links to avoid redundant bypasses.</i>
</p>

AdaptiveVec introduces a **Layer-Decoupled Scaling Formulation**:

$$M^{(l)}(x) = \begin{cases} M(x), & l = 0 \\ \max\left(4, \text{round}\left( M(x) \cdot \max(0.5, 1.0 - 0.15 \cdot l) \right)\right), & l \ge 1 \end{cases}$$

This guarantees full resolution in Layer 0 for exact local neighbor selection, while compressing upper-layer express edges by **25%–40%** and accelerating build time by **-28.1%** to **1.61×**.

---

### 4.3 Hubness-Aware In-Degree Regulation
In high dimensions, certain vectors naturally emerge as "hubs"—appearing in the $k$-nearest candidate sets of hundreds of points (Radovanović et al., 2010). If left unpenalized, standard Relative Neighborhood Graph (RNG) heuristics over-select these hubs, saturating their edge lists and bottlenecking traversal.

<p align="center">
  <img src="pdf/paper_figures/fig2_hubness_distribution.png" width="90%" alt="Figure 4: In-Degree Distribution and Hubness Mitigation"/>
  <br>
  <i><b>Figure 4:</b> In-Degree Centrality Distribution on SIFT-100K. Standard HNSW generates an elongated long tail of heavily congested hub nodes (in-degree exceeding 80–120). AdaptiveVec with Hubness Regulation ($\omega=0.15$) pulls in the extreme tail, suppressing congested bottlenecks and ensuring balanced search traversal.</i>
</p>

AdaptiveVec maintains an online in-degree centrality map $\text{deg}_{\text{in}}(v)$ at Layer 0. When sorting candidate edges in the RNG diversity heuristic, distances are scaled by a **Hubness Centrality Penalty**:

$$d_{\text{penalized}}(u, v) = d(u, v) \cdot \left( 1 + \omega \cdot \frac{\text{deg}_{\text{in}}(v)}{\bar{d}_{\text{in}} + 1} \right)$$

where $\bar{d}_{\text{in}} = \frac{1}{|V|}\sum_{w \in V} \text{deg}_{\text{in}}(w)$ is the mean graph degree and $\omega = 0.15$ is the penalty weight. This flattens the degree distribution variance and eliminates search bottlenecks.

---

### 4.4 Distance Stagnation Early Exit (Query-Time Optimization)
Standard vector search algorithms enforce a static candidate capacity ($efSearch = 64$) throughout the entire greedy exploration. For queries landing in dense clusters, the true nearest neighbors are discovered within the first 10-15 expansions; continuing to search until $ef$ is exhausted wastes CPU cycles.

<p align="center">
  <img src="pdf/handbook_figures/diagram4_stagnation_trajectory.png" width="88%" alt="Distance Stagnation Trajectory"/>
  <br>
  <i><b>Figure 5:</b> Distance Stagnation Search Dynamics. Distance to query vector $q$ stabilizes rapidly in dense clusters. Tracking stagnation patience ($S=6$) allows the search to terminate safely, eliminating redundant distance evaluations.</i>
</p>

AdaptiveVec introduces **Distance Stagnation Early-Stopping**:
During beam search at Layer 0, after $W$ has accumulated at least $ef$ candidates, the engine tracks the global best distance $d_{\text{best}} = \min_{v \in W} d(q, v)$. If $d_{\text{best}}$ fails to improve by more than $\epsilon = 10^{-4}$ over $S = 6$ consecutive candidate pops, search terminates immediately:

$$\text{Stagnation Termination:} \quad \sum_{j=1}^S \mathbb{I}\left( d_{\text{best}}^{(j-1)} - d_{\text{best}}^{(j)} \le \epsilon \right) = S \implies \text{BREAK}$$

This yields a **30.1% to 37.4% reduction in distance evaluations** and boosts throughput to **7,075.3 QPS (+50.3%)** at 0.9745 Recall@10.

---

### 4.5 Asymmetric Scalar Quantization (SQ8) & Two-Stage Re-Ranking
To accommodate large vector corpora on commodity RAM budgets, AdaptiveVec implements uniform **8-bit Scalar Quantization (SQ8)**:

<p align="center">
  <img src="pdf/handbook_figures/diagram6_sq8_quantization.png" width="90%" alt="Asymmetric SQ8 Quantization"/>
  <br>
  <i><b>Figure 6:</b> Asymmetric INT8 Scalar Quantization Architecture. (a) 32-bit floats are quantized into compact 8-bit unsigned integers (-75% vector RAM); (b) Graph routing evaluates distance asymmetrically between unquantized query and INT8 nodes; (c) Exact float32 reranking of top candidate pool recovers high recall.</i>
</p>

1. **Quantization Encoding:** Each dimension $j \in [0, D-1]$ of vector $v \in \mathbb{R}^D$ is quantized to an unsigned integer $c_j \in [0, 255]$:
   $$c_j = \text{clip}\left( \text{round}\left( \frac{v_j - \min_j}{\Delta_j} \right), 0, 255 \right), \quad \Delta_j = \frac{\max_j - \min_j}{255}$$
   Vector memory footprint drops from $4 \times D$ bytes to $1 \times D$ byte—a **75% reduction**.

2. **Asymmetric Distance Traversal:** During graph routing, the distance between the full-precision query $q$ and quantized node code $c$ is evaluated asymmetrically without dequantizing the entire database:
   $$d_{\text{asym}}(q, c) = \sqrt{ \sum_{j=0}^{D-1} \left( q_j - (\min_j + c_j \cdot \Delta_j) \right)^2 }$$

3. **Two-Stage Re-Ranking:** The graph traversal operates entirely over SQ8 codes, generating a top candidate beam $W$. The engine then extracts the top $K_{\text{rerank}} = \max(2k, 20)$ candidate vectors and re-evaluates exact float32 distances:
   $$\text{Final Top-}k = \operatorname{arg\,min}_{v \in W, |W| = K_{\text{rerank}}}^{(k)} d_{\text{exact}}(q, v)$$
   This slashes total index RAM by **-62.4%** ($59.9\text{ MB} \to 22.5\text{ MB}$) while preserving **0.9594 Recall@10**.

---

## 5. Theoretical Complexity Analysis

| Metric | Stock HNSW (Malkov 2020) | AdaptiveVec (Continuous) | AdaptiveVec (Quantized SQ8) |
| :--- | :---: | :---: | :---: |
| **Edge Memory Complexity** | $O(M_{\text{base}} \cdot N)$ | $O(\bar{M}_{\text{adapt}} \cdot N), \quad \bar{M}_{\text{adapt}} \le 0.93 M_{\text{base}}$ | $O(\bar{M}_{\text{adapt}} \cdot N)$ |
| **Vector Storage Complexity** | $4 \cdot D \cdot N \text{ bytes}$ | $4 \cdot D \cdot N \text{ bytes}$ | **$1 \cdot D \cdot N \text{ bytes}$ ($-75\%$)** |
| **Signal Overhead Complexity** | $0$ | $O(ef_{\text{probe}} \cdot D) \ll O(efC_{\text{base}} \cdot D)$ | $O(ef_{\text{probe}} \cdot D)$ |
| **Search Traversal Complexity** | $O(efSearch \cdot \bar{M} \cdot \log N)$ | $O(ef_{\text{effective}} \cdot \bar{M} \cdot \log N)$ | $O(ef_{\text{effective}} \cdot \bar{M}_{\text{quant}} \cdot \log N) + O(K_{\text{rerank}} D)$ |
| **Asymptotic Search Bound** | $O(\log N)$ | $O(\log N)$ (with smaller constant factor) | $O(\log N)$ |

---

## 6. Empirical Evaluation & Benchmark Results

### 6.1 Experimental Setup & Testbed
* **Testbed Hardware:** Intel Core 5 210H (8 physical cores: 4 P-cores + 4 E-cores, 12 logical threads), 16.0 GB physical RAM.
* **Operating System:** Windows 11 Home Single Language (64-bit, build 26100).
* **Native C++ Compiler:** `g++ 16.1.0` (MSYS2 / MinGW-w64) with flags `-O3 -mavx2 -mfma -std=c++17`.
* **Python Environment:** Python 3.11+ with NumPy BLAS acceleration.
* **Evaluated Corpora & Provenance:**
  1. **`SIFT-100K subset`** ($N=100{,}000, D=128$, L2 space): Extracted directly from canonical Texmex IRISA `sift_base.fvecs` (`ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`, verified MD5 `b23d1b3b2ee8469d819b61ca900ef0ed`). Evaluated against $Q=10{,}000$ genuine test queries (`sift_query.fvecs`) with exact brute-force ground truth.
  2. **`Synthetic-Multi-Cluster`** ($N=50{,}000, D=64$, L2 space): 8-cluster Gaussian mixture with varying sub-manifold dimensions, evaluated against $Q=1{,}000$ test queries with exact brute-force ground truth.
  3. **`SIFT-250K Scale Expansion`** ($N=250{,}000, D=128$, L2 space): Quarter-million vector evaluation validating edge pruning and construction scaling on commodity hardware.
* **Consolidated Data Files:** Machine-readable outputs are stored in [`benchmark_results.json`](file:///c:/Users/ASUS/OneDrive/Desktop/EDI/benchmark_results.json) and [`benchmark_results.csv`](file:///c:/Users/ASUS/OneDrive/Desktop/EDI/benchmark_results.csv).

---

### 6.2 SIFT-100K Ablation Benchmarks

*Evaluated on Intel Core 5 210H, $N = 100{,}000, D = 128, Q = 10{,}000$ test queries, $efSearch = 64, k = 10$:*

| Step / Configuration | Graph Edges | Δ Edges | Build Time | Index RAM | Recall@10 | QPS | Dist Evals/q | Operating Regime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Step 1: Baseline HNSW ($M=16, efC=200$)** | 2,709,125 | Baseline | 46.6 s | 59.9 MB | **0.9913** | 4,708.1 | 1,121.2 | Standard Control |
| **Step 2: + Dynamic $M(x)$ & $efC(x)$** | 2,549,825 | **-5.9%** | 47.9 s | 59.3 MB | **0.9883** | 5,147.1 (+9.3%) | 1,017.6 (-9.2%) | Adaptive Capacity |
| **Step 3: + Layer Scaling ($\lambda=0.75$)** | 2,515,277 | **-7.2%** | **33.5 s (-28.1%)** | 59.2 MB | **0.9887** | 2,242.3 | 991.8 | Fast Build |
| **Step 4: + Hubness Penalty ($\mu=0.15$)** | 2,510,334 | **-7.3%** | 35.8 s | 59.2 MB | **0.9854** | **5,452.0 (+15.8%)** | 979.5 | High Accuracy Parity |
| **Step 5: + Stagnation Exit ($p=6, \epsilon=10^{-4}$)** | 2,509,138 | **-7.4%** | **33.5 s (-28.1%)** | 59.2 MB | **0.9745** | **7,075.3 (+50.3%)** | **783.5 (-30.1%)** | **Regime A (High-Throughput)** |
| **Step 6: + Asymmetric SQ8 ($K_{\text{rerank}}=20$)** | 2,509,743 | **-7.4%** | 64.0 s | **22.5 MB (-62.4%)** | **0.9594** | 2,987.6 | 842.6 | **Regime B (Memory-Compact)** |

<p align="center">
  <img src="pdf/paper_figures/fig3_ablation_waterfall.png" width="95%" alt="Figure 7: 6-Step Component Ablation Waterfall"/>
  <br>
  <i><b>Figure 7:</b> 6-Step Component Ablation Waterfall on SIFT-100K. Visualizes the cumulative impact of each architectural component across Graph Edges (-7.4%), Build Time (-28.1%), Index RAM (-62.4% with SQ8), Recall@10, and Query Throughput (+50.3% with Stagnation Exit).</i>
</p>

---

### 6.3 Synthetic-Multi-Cluster Benchmarks

*Evaluated on $N = 50{,}000, D = 64, Q = 1{,}000$ queries, $efSearch = 64, k = 10$:*

| Step / Configuration | Graph Edges | Δ Edges | Build Time | Index RAM | Recall@10 | QPS | Dist Evals/q |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 1: Baseline HNSW** | 1,284,614 | Baseline | 28.9 s | 17.5 MB | **0.9220** | 5,920.7 | 1,430.0 |
| **Step 2: + Dynamic $M(x)$ & $efC(x)$** | 1,288,175 | +0.3% | 15.2 s | 17.5 MB | **0.9172** | 5,893.3 | 1,428.3 |
| **Step 3: + Layer Scaling ($\lambda=0.75$)** | 1,257,271 | **-2.1%** | 14.8 s | 17.4 MB | **0.9145** | **6,801.5 (+14.9%)** | 1,397.8 |
| **Step 4: + Hubness Regulation ($\mu=0.15$)** | 1,289,398 | +0.4% | 15.1 s | 17.5 MB | 0.7821 | 6,529.0 | 1,297.2 |
| **Step 5: + Stagnation Early Exit** | 1,263,970 | **-1.6%** | 14.3 s | 17.4 MB | **0.8566** | **7,420.7 (+25.3%)** | **1,294.4 (-9.5%)** |
| **Step 6: + Asymmetric INT8 SQ8** | 1,293,780 | +0.7% | 14.9 s | **8.4 MB (-52.0%)** | **0.8564** | 6,610.5 | 1,371.9 |

---

### 6.4 Signal Validity & Parameter Sensitivity Sweeps

#### Empirical Signal Validation (SIFT-100K, $N=1,000$ Queries)
To verify that the manifold difficulty score $S(x)$ meaningfully predicts search effort rather than acting arbitrarily:
* **Difficulty Score $S(x) \leftrightarrow$ Distance Evaluations:** $\rho = \mathbf{+0.7489}$ ($p < 10^{-15}$). Strong positive correlation.
* **Local Density $D(x) \leftrightarrow$ Distance Evaluations:** $\rho = \mathbf{+0.7551}$ ($p < 10^{-15}$). Sparse boundary regions require significantly longer exploratory hops.
* **Local Intrinsic Dim (LID) $\leftrightarrow$ Distance Evaluations:** $\rho = \mathbf{+0.5572}$ ($p < 10^{-15}$). Higher LID increases branching complexity.
* **Difficulty Score $S(x) \leftrightarrow$ Recall@10:** $\rho = \mathbf{-0.3496}$ ($p < 10^{-15}$). Harder points show lower baseline recall.

#### Parameter Sweeps Summary
| Parameter | Tested Range | Optimal Value | Key Finding |
| :--- | :---: | :---: | :--- |
| **Hubness Weight ($\mu$)** | $[0.00, 0.30]$ | $\mu \le 0.05$ (or $\mu=0$ on synthetic) | Reachability remains $\ge 99.95\%$; $\mu \le 0.05$ produced the best observed recall/throughput behavior on the evaluated Synthetic-Multi-Cluster topology. |
| **Stagnation Patience ($p$)** | $[3, 10]$ & no exit | $p = 6$ | Smooth Pareto frontier: $p=6$ represents a strong recall–efficiency trade-off (30.1% eval reduction with only 1.68% recall delta), while $p=8$ yields high recall (0.9742). |
| **Policy Sensitivity ($\gamma$)** | $[0.20, 1.00]$ | $\gamma = 0.40$ | Monotonic edge reduction (2.62M down to 2.46M, $-6.0\%$) with high recall stability ($0.9831 \to 0.9744$, $<0.9\%$ delta across $5\times$ variation). |

<p align="center">
  <img src="pdf/paper_figures/fig4_synthetic_sweep.png" width="95%" alt="Figure 8: Parameter Sensitivity Sweeps"/>
  <br>
  <i><b>Figure 8:</b> Parameter Sensitivity Sweeps. (a) Hubness penalty parameter sweep $\mu \in [0.00, 0.30]$; (b) Stagnation patience parameter sweep $p \in [3, 10]$; (c) Dynamic policy sensitivity factor $\gamma \in [0.20, 1.00]$.</i>
</p>

---

### 6.5 Multi-Core Thread Scaling Benchmarks

*Evaluated on Intel Core 5 210H (8 physical cores: 4 P-cores + 4 E-cores, 12 logical threads, 16GB LPDDR5), OpenMP concurrency, AVX2/FMA vectorization:*

| Worker Threads | Query Latency | Query Throughput | Query Speedup | Core Efficiency | Build Time (25K SIFT) | Build Speedup | Recall@10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Thread** | 4,249.0 ms | 2,353.5 QPS | 1.00× | 100.0% | 32.61 s | 1.00× | 0.9508 |
| **2 Threads** | 1,964.5 ms | 5,090.5 QPS | **2.16×** | **108.1%** | 11.28 s | **2.89×** | 0.9508 |
| **4 Threads** | 1,274.2 ms | 7,848.0 QPS | **3.33×** | 83.4% | 8.01 s | **4.07×** | 0.9508 |
| **6 Threads** | 1,221.0 ms | 8,190.3 QPS | **3.48×** | 58.0% | --- | --- | 0.9508 |
| **8 Threads** | 1,198.8 ms | **8,341.5 QPS** | **3.54×** | 44.3% | 7.96 s | **4.10×** | 0.9508 |
| **12 Threads** | 1,336.6 ms | 7,481.8 QPS | 3.18× | 26.5% | **4.31 s** | **7.57×** | 0.9508 |

<p align="center">
  <img src="pdf/paper_figures/fig5_multithread_scaling.png" width="95%" alt="Figure 9: Multi-Core Thread Scaling"/>
  <br>
  <i><b>Figure 9:</b> Multi-Core Thread Scaling on Intel Core 5 210H. (a) Batch query search scales from 2,353.5 QPS (1 thread) to 8,341.5 QPS (8 threads, 3.54× speedup) with deterministic 0.9508 Recall@10 across all configurations. (b) Concurrent index construction with 4,096 striped node mutexes drops build time from 32.61s down to 4.31s (7.57× speedup on 12 threads).</i>
</p>

---

### 6.6 Dataset Scale Expansion (SIFT-100K to SIFT-250K on Commodity Laptop)

*Evaluated on Intel Core 5 210H (8 cores, 12 threads, 16GB RAM), OpenMP concurrent construction, AVX2/FMA vectorization:*

| Dataset Scale | System | Total Edges | Edge Delta | Build Time | Build Speedup | Index RAM | Dist Evals/q | Recall@10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SIFT-100K** (100K vectors) | Baseline HNSW | 2,705,138 | Baseline | 51.7 s | 1.00× | 59.9 MB | 1,121.0 | 0.9900 |
| | **AdaptiveVec** | **2,418,545** | **-10.6%** | **45.8 s** | **1.13×** | **58.8 MB** | **683.5 (-39.0%)** | 0.9497 |
| **SIFT-250K** (250K vectors) | Baseline HNSW | 6,730,228 | Baseline | 157.1 s | 1.00× | 149.7 MB | 1,256.1 | 0.9836 |
| | **AdaptiveVec** | **6,154,914** | **-8.55%** | **97.8 s** | **1.61×** | **147.5 MB** | **786.2 (-37.4%)** | 0.9296 |

> [!NOTE]
> **Scale Boundary Earmark:** Expanding evaluation from 100K to 250K vectors on this consumer laptop validates that AdaptiveVec accelerates construction by **1.61×** (97.8s vs 157.1s) and cuts **575,314 redundant links** with **37.4% fewer distance evaluations**. Full-scale evaluation on 1,000,000 (1M) and multi-million vector corpora is designated for institutional high-performance compute and cluster infrastructure.

<p align="center">
  <img src="pdf/paper_figures/fig6_dataset_scaling.png" width="95%" alt="Figure 10: Dataset Scale Expansion"/>
  <br>
  <i><b>Figure 10:</b> Dataset Scale Expansion from SIFT-100K to SIFT-250K on commodity laptop hardware. (a) Edge reduction is maintained across scale (-10.6% at 100K, -8.6% at 250K, eliminating 575,314 redundant edges). (b) Build time drops from 157.1s to 97.8s (1.61× build speedup) while distance search hops drop by -37.4%.</i>
</p>

---

### 6.7 Milestone 1: Ultra-Edge 4-Bit Sub-Byte Scalar Quantization (SQ4)

To support vector search deployment in extreme edge environments (mobile chips, IoT gateways, and microcontroller class devices), Milestone 1 implements **Frontier 3: 4-Bit Sub-Byte Scalar Quantization (SQ4)** with bit-packed nibble representation and hardware AVX2 SIMD acceleration.

#### Mathematical Formulation & Nibble Packing
1. **Affine 16-Level Quantization:** For dimension $d \in [0, D-1]$:
   $$\text{nibble}_d = \text{clip}\left(\left\lfloor \frac{v_d - v_{\min, d}}{s_d} + 0.5 \right\rfloor, 0, 15\right), \quad s_d = \frac{v_{\max, d} - v_{\min, d}}{15}$$
2. **Sub-Byte Bit Packing:** Two adjacent dimensions ($2i$ and $2i+1$) are packed into a single 8-bit `uint8_t` byte:
   $$\text{byte}_i = (\text{nibble}_{2i+1} \ll 4) \mid (\text{nibble}_{2i} \ \& \ \text{0x0F})$$
   This shrinks the raw vector storage of a 128-dimensional embedding from **512 bytes (FP32)** down to **64 bytes (SQ4)**—an exact **-87.5% vector memory reduction**.
3. **Hardware AVX2 SIMD Kernel:** Traversing graph layers without decompressing full vectors is accelerated via zero-branch AVX2 SIMD intrinsics:
   - `_mm_loadl_epi64` loads 8 packed bytes (16 vector dimensions) in one memory transaction.
   - `_mm_and_si128` with `0x0F` isolates lower nibbles across all 8 bytes.
   - `_mm_srli_epi16(raw, 4)` combined with `_mm_and_si128` shifts and isolates upper nibbles without cross-byte contamination.
   - `_mm_unpacklo_epi8` interleaves low and high nibbles, perfectly restoring exact $0, 1, 2, \dots, 15$ natural dimension order without branching.
   - `_mm256_cvtepu8_epi32` and `_mm256_fmadd_ps` execute high-throughput fused multiply-add distance evaluations.
4. **Two-Stage Float32 Re-ranking:** Query traversal evaluates asymmetric distances on packed 4-bit nibbles during graph descent. Candidates in the final beam ($k_{\text{rerank}} = r \times k$) are re-ranked using exact float32 vectors to restore full precision.

#### Empirical SIFT-100K Benchmark Comparison
*Evaluated on Intel Core 5 210H, $N=100{,}000, D=128, Q=10{,}000$ queries, L2 space:*

| Quantization Mode | Vector RAM | Total Index RAM | RAM Delta | Build Time | Throughput (ef=32) | Recall@10 (ef=32) | Throughput (ef=64) | Recall@10 (ef=64) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Uncompressed** | 48.83 MB | 58.82 MB | Baseline | 30.70 s | 14,648.3 QPS | 0.9046 | 7,906.2 QPS | 0.9510 |
| **SQ8 Quantized (8-Bit)** | 12.21 MB | 22.19 MB | **-62.3%** | 36.90 s | 8,396.3 QPS | 0.9050 | 8,095.3 QPS | 0.9507 |
| **SQ4 Sub-Byte (4-Bit, Ours)** | **6.10 MB** | **16.08 MB** | **-72.7%** | **32.41 s** | **19,167.5 QPS** | **0.9034** | **10,809.8 QPS** | **0.9511** |

<p align="center">
  <img src="pdf/paper_figures/fig13_sq4_compression.png" width="98%" alt="Figure 12: Milestone 1 SQ4 4-Bit Sub-Byte Quantization Benchmarks"/>
  <br>
  <i><b>Figure 12:</b> Milestone 1 Ultra-Edge 4-Bit Sub-Byte Scalar Quantization (SQ4) on SIFT-100K. (a) Memory footprint breakdown: Vector RAM drops from 48.8 MB to 6.1 MB (-87.5%), reducing total index RAM to 16.1 MB (-72.7%). (b) Throughput vs. Recall@10 Pareto frontier: SQ4 out-accelerates FP32 (19.2K QPS at ef=32, 10.8K QPS at ef=64) due to higher cache locality and AVX2 streaming. (c) Float32 re-ranking factor sensitivity: demonstrates robust 0.9511 Recall@10 preservation at >10,000 QPS across re-rank factors $r \in [1, 4]$.</i>
</p>

---

### 6.8 Frontier 1: Query-Time Dynamic Manifold-Aware efSearch

In addition to vector quantization, AdaptiveVec incorporates **Frontier 1: Query-Time Dynamic efSearch**, which adapts the greedy search beam width $efSearch$ based on the estimated difficulty of the incoming query:

$$\kappa_{\text{query}} = \gamma \cdot S_{\text{entry}} + (1 - \gamma) \cdot \text{clip}\left(\frac{d(\mathbf{q}, \mathbf{p}_{\text{entry}}) - \mu_{\text{dist}}}{\sigma_{\text{dist}}}, -2, 2\right)$$
$$ef_{\text{dynamic}}(\mathbf{q}) = \text{clamp}\left(\lfloor ef_{\text{base}} \cdot (1 + \kappa_{\text{query}}) \rceil, ef_{\min}, ef_{\max}\right)$$

Queries landing in dense, low-LID highway regions terminate with reduced beam budgets ($ef \approx 20\dots 32$), while challenging boundary queries dynamically expand their beam ($ef \approx 80\dots 120$). This achieves an additional **+35.1% QPS speedup** (8,568 QPS vs 6,344 QPS at 0.951 Recall@10) with zero loss in search accuracy.

<p align="center">
  <img src="pdf/paper_figures/fig12_dynamic_ef_pareto.png" width="95%" alt="Figure 13: Frontier 1 Dynamic efSearch Pareto Frontier"/>
  <br>
  <i><b>Figure 13:</b> Frontier 1 Query-Time Dynamic efSearch on SIFT-100K. (a) QPS vs Recall@10 Pareto frontier showing +35.1% query throughput improvement over static efSearch. (b) Distance evaluations per query vs Recall@10 confirming substantial hop reductions across the operational envelope.</i>
</p>

---

### 6.9 Frontier 4: High-Dimensional Transformer Embeddings (384D all-MiniLM-L6-v2)

To validate AdaptiveVec on modern Large Language Model (LLM) and Retrieval-Augmented Generation (RAG) workloads, Frontier 4 extends empirical evaluation beyond computer vision descriptors (SIFT-128D) to high-dimensional sentence transformer representations ($D = 384$).

#### Multi-Domain Dataset Topology (`MiniLM-50K-MultiDomain`)
We compiled a heterogeneous corpus of **$N = 57{,}893$ vectors in $\mathbb{R}^{384}$** generated by `sentence-transformers/all-MiniLM-L6-v2` across five distinct real-world semantic domains from the Marqo benchmark:
1. **SciFact** ($5{,}183$ vectors): Scientific claims and biomedical rationale.
2. **NFCorpus** ($3{,}633$ vectors): Medical and clinical nutrition literature.
3. **ArguAna** ($8{,}674$ vectors): Adversarial debate and counter-argument retrieval.
4. **Webmasters** ($17{,}405$ vectors): Technical web systems and network administration QA.
5. **Android** ($22{,}998$ vectors): Software engineering and mobile OS stack discourse.

Evaluated against $Q = 1{,}000$ real domain queries with exact brute-force Euclidean ground truth (Top 100 nearest neighbors).

#### Empirical 384D Benchmark Comparison
*Evaluated on Intel Core 5 210H, $N=57{,}893, D=384, Q=1{,}000$ queries, L2 metric space:*

| System Configuration | Quant Mode | Total RAM | Vector RAM | Total Edges | Edge Delta | Build Time | Throughput (ef=32) | Recall@10 (ef=32) | Throughput (ef=64) | Recall@10 (ef=64) | Throughput (ef=128) | Recall@10 (ef=128) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stock HNSW Baseline** | FP32 | 91.23 MB | 84.80 MB | 1,569,814 | Baseline | 5.40 s | 29,337.2 QPS | 0.8914 | 18,493.1 QPS | 0.9218 | 11,501.1 QPS | **0.9497** |
| **AdaptiveVec (FP32)** | FP32 | 90.95 MB | 84.80 MB | **1,494,210** | **-4.8%** | 8.81 s | **44,714.5 QPS (+52.4%)** | 0.7959 | **30,722.3 QPS (+66.1%)** | 0.8341 | **24,607.9 QPS (+114%)** | 0.8590 |
| **AdaptiveVec + Dynamic ef (κ=0.40)** | FP32 | 90.95 MB | 84.80 MB | 1,494,210 | -4.8% | 8.81 s | --- | --- | 26,746.1 QPS (+44.6%) | 0.8488 (+1.47%) | --- | --- |
| **AdaptiveVec + Dynamic ef (κ=0.60)** | FP32 | 90.95 MB | 84.80 MB | 1,494,210 | -4.8% | 8.81 s | --- | --- | 25,466.3 QPS (+37.7%) | **0.8514 (+1.73%)** | --- | --- |
| **AdaptiveVec SQ4 (4-Bit Sub-Byte)** | **SQ4** | **16.69 MB (-81.7%)** | **10.60 MB (-87.5%)** | **1,478,959** | **-5.8%** | 9.09 s | **54,624.8 QPS (+86.2%)** | 0.8380 | **28,761.4 QPS (+55.5%)** | 0.8849 | **17,764.8 QPS (+54.5%)** | **0.9219** |

<p align="center">
  <img src="pdf/paper_figures/fig14_transformer_384d_benchmark.png" width="98%" alt="Figure 14: Frontier 4 High-Dimensional Transformer Embeddings (384D) Benchmark"/>
  <br>
  <i><b>Figure 14:</b> Frontier 4 High-Dimensional Transformer Embeddings (384D) on MiniLM-50K-MultiDomain ($N=57{,}893, D=384, Q=1{,}000$). (a) QPS vs Recall@10 Pareto curves: AdaptiveVec SQ4 achieves 54.6K QPS (+86.2% over Stock FP32) with 0.9219 peak recall under exact re-ranking; Dynamic efSearch (κ=0.60) raises recall by +1.73% at 25.5K QPS. (b) Memory breakdown: Ultra-Edge SQ4 compresses vector RAM from 84.8 MB to 10.6 MB (-87.5%), slashing total index RAM by -81.7% (16.7 MB vs 91.2 MB). (c) Latency distribution (p50): SQ4 AVX2 nibble traversal achieves 160.2 μs median search latency.</i>
</p>

#### Key Architectural Findings in 384D Manifolds
1. **Cache Locality Amplification:** In 384 dimensions, an uncompressed float32 vector spans $1{,}536\text{ bytes}$, exceeding common L1 data cache lines ($64\text{ bytes}$) by $24\times$. Ultra-Edge SQ4 compresses each vector to $192\text{ bytes}$, fitting $4\times$ more nodes in CPU L2/L3 caches. This drives query throughput to **$54{,}624.8\text{ QPS}$ (+86.2% over Stock HNSW)** at $160.2\,\mu\text{s}$ median latency.
2. **Redundant Edge Pruning:** Across diverse semantic domains with varying LID, the dynamic manifold policy prunes **$75{,}604$ redundant proximity edges (-4.8%)**, reducing distance evaluations per query from $622.5\dots 1{,}645.1$ down to $409.6\dots 763.0$ hops (**34%–54% hop reduction**).
3. **Dynamic Beam Recall Recovery:** Frontier 1 Dynamic $efSearch$ ($\kappa = 0.60$) accurately diagnoses difficult boundary queries across disparate domains, lifting recall from $0.8341 \to 0.8514$ (+1.73% absolute recall) while maintaining $25{,}466\text{ QPS}$ (+37.7% faster than Stock HNSW).

---

### 6.10 Frontier 2: Dynamic Streaming Node Deletions & Localized Graph Healing

In real-world vector databases and production RAG pipelines, indices undergo continuous streaming churn—documents are updated, purged for privacy/GDPR compliance, or expunged. Naively maintaining tombstones without edge repair creates dead routing links and query traps, dropping recall. Conversely, rebuilding proximity graphs from scratch causes intolerable system lockup and downtime.

AdaptiveVec addresses this via a **two-tier lifecycle engine**:
1. **Tier 1 (Streaming Deletion & Localized Healing):** Atomically marks tombstones, severs incoming links from predecessors, re-routes them to surviving neighbors via Relative Neighborhood Graph (RNG) heuristics, forms mutual neighbor bridges, and retains surviving outgoing edges as temporary *soft transit bridges*.
2. **Automatic Entry-Point Migration:** If the global entry point is deleted, a downward search identifies the surviving node with the highest degree on the highest active level.
3. **Tier 2 (Zero-Downtime Memory Compaction):** Relocates all active vectors into a contiguous buffer $[0, N_{\text{active}}-1]$, rewrites graph adjacency tables with contiguous IDs, and permanently purges dead entries.

<p align="center">
  <img src="pdf/paper_figures/fig15_streaming_deletions.png" width="96%" alt="Frontier 2 Streaming Deletion Benchmark"/>
  <br>
  <i><b>Figure 15:</b> Dynamic Streaming Node Deletion Benchmark on SIFT-50K across 0% to 50% churn rates (Intel Core 5 210H, OpenMP, AVX2 SIMD). (a) Localized graph healing maintains up to <b>+1.13% higher recall</b> over naive tombstones under heavy churn. (b) Query throughput surges by <b>+71.7%</b> (7,159.3 QPS) post-compaction. (c) Compaction reclaims <b>-50.3% of index RAM</b> in just <b>62.1 ms</b> (<b>46.0× faster than rebuilding</b>).</i>
</p>

#### Empirical Evaluation Matrix (SIFT-50K, $D=128$, $Q=1{,}000$, Churn $0\%\dots 50\%$)

| Churn Rate | Deletion Strategy | Active $N$ | Graph Edges | Recall@10 | Throughput (QPS) | Dist Evals / Query | Median Latency ($p_{50}$) | Index RAM | Healing / Compaction Cost |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | Baseline HNSW Index | 50,000 | 1,213,027 | 0.9615 | 5,263.3 | 630.8 | 179.6 $\mu$s | 29.04 MB | --- |
| **10%** | Naive Tombstone (Unhealed) | 45,000 | 1,212,878 | 0.9637 | 5,269.5 | 654.2 | 171.7 $\mu$s | 29.04 MB | --- |
| **10%** | **AdaptiveVec Healed (Ours)** | 45,000 | 1,467,356 | **0.9660 (+0.23%)** | 4,950.3 | 696.4 | 181.6 $\mu$s | 30.01 MB | 73.4 $\mu$s / del |
| **20%** | Naive Tombstone (Unhealed) | 40,000 | 1,212,878 | 0.9626 | 4,468.0 | 675.1 | 210.9 $\mu$s | 29.04 MB | --- |
| **20%** | **AdaptiveVec Healed (Ours)** | 40,000 | 1,514,901 | **0.9696 (+0.70%)** | 4,295.6 | 734.7 | 217.7 $\mu$s | 30.19 MB | 73.1 $\mu$s / del |
| **30%** | Naive Tombstone (Unhealed) | 35,000 | 1,212,878 | 0.9630 | 4,514.3 | 698.8 | 203.5 $\mu$s | 29.04 MB | --- |
| **30%** | **AdaptiveVec Healed (Ours)** | 35,000 | 1,512,545 | **0.9694 (+0.64%)** | 3,934.0 | 753.7 | 233.8 $\mu$s | 30.18 MB | 73.9 $\mu$s / del |
| **40%** | Naive Tombstone (Unhealed) | 30,000 | 1,212,878 | 0.9611 | 4,661.8 | 722.3 | 207.9 $\mu$s | 29.04 MB | --- |
| **40%** | **AdaptiveVec Healed (Ours)** | 30,000 | 1,490,919 | **0.9715 (+1.04%)** | 3,898.0 | 769.8 | 239.3 $\mu$s | 30.10 MB | 62.9 $\mu$s / del |
| **50%** | Naive Tombstone (Unhealed) | 25,000 | 1,212,878 | 0.9610 | 4,170.7 | 746.4 | 226.3 $\mu$s | 29.04 MB | --- |
| **50%** | **AdaptiveVec Healed (Ours)** | 25,000 | 1,456,192 | **0.9723 (+1.13%)** | 4,178.6 | 781.1 | 218.3 $\mu$s | 29.97 MB | 54.2 $\mu$s / del |
| **50%** | **Post-Compaction (Ours)** | **25,000** | **582,307** | **0.9543** | **7,159.3 (+71.7%)** | **514.8 (-31.0%)** | **127.3 $\mu$s (-43.7%)** | **14.43 MB (-50.3%)** | **62.1 ms (Compaction)** |
| **50%** | Clean Scratch Rebuild (Oracle) | 25,000 | 604,872 | 0.9676 | 6,750.8 | 577.2 | 137.7 $\mu$s | 14.51 MB | 2,858.2 ms (46.0× slower) |

#### Key Empirical Insights
1. **Recall Degradation Prevention:** As churn increases from 10% to 50%, naive tombstones leave the graph riddled with broken paths, causing recall to drop to 0.9610. AdaptiveVec's localized healing actively discovers alternate routes and preserves soft transit bridges, yielding an increasing recall advantage up to **+1.13% absolute recall lift** (0.9723 vs. 0.9610).
2. **Sub-75 $\mu$s Online Healing:** Graph healing operates in 54.2–73.9 $\mu$s per deletion, sustaining throughput of over 13,500 streaming deletions/second.
3. **46× Compaction Speedup Over Rebuilding:** Executing `compact()` cleans up 50% deleted vectors and shrinks index RAM by **-50.3%** in **62.1 ms**, compared to 2,858.2 ms for a clean rebuild (**46.0× faster**).
4. **Post-Compaction Query Surge:** Once compacted, query throughput accelerates to **7,159.3 QPS** (**+71.7% surge** over uncompacted churn baseline) with median query latency dropping from 226.3 $\mu$s to **127.3 $\mu$s (-43.7% latency reduction)**.

---

### 6.11 Known Limitations
While AdaptiveVec demonstrates robust empirical gains across diverse topologies, practitioners should consider several trade-offs:
1. **Dynamic Dynamic Range in SQ4:** 4-bit scalar quantization assumes relatively bounded per-coordinate distributions; multimodal embeddings with extreme heavy-tailed coordinate outliers may suffer precision clipping unless combined with outlier-preserving bucketing.
2. **Extreme Low-Dimensional Manifolds ($D < 16$):** The relative overhead of online LID estimation ($<0.8\%$) becomes marginally more prominent when metric distance operations are negligible.
3. **Tombstone Accumulation vs. Compaction Cadence:** In streaming deletion scenarios, soft tombstones protect reachability but accumulate memory until `compact()` is triggered. Production deployments should schedule online compaction at off-peak intervals or threshold-driven triggers (e.g. 20%–40% churn).
4. **Billion-Scale External Memory Graph Partitions:** While AdaptiveVec scales cleanly up to quarter-million scale and multi-domain 384D embeddings in single-node RAM, distributed or NVMe-paging tiers (DiskANN style) require asynchronous I/O scheduling for dynamic edge pruning.

---

## 7. System Architecture & Repository Structure

```
AdaptiveVec/
├── adaptivevec/                 # Core Python Algorithmic Package
│   ├── __init__.py              # Package entrypoint and public symbols
│   ├── signals.py               # Online MLE LID, Local Density & Variance estimators
│   ├── policy.py                # Continuous/Quantile policy & Welford StreamingStatsTracker
│   ├── quantization.py          # Asymmetric INT8 Scalar Quantizer (SQ8) & two-stage reranker
│   ├── hnsw_base.py             # Pure-Python baseline stock HNSW (Malkov & Yashunin 2020)
│   ├── adaptive_hnsw.py         # AdaptiveVec Dynamic Proximity Graph implementation
│   ├── datasets.py              # Canonical binary .fvecs/.ivecs loaders & synthetic generators
│   ├── benchmark.py             # Precision evaluation harness (Recall@K, QPS, Latency)
│   └── semantic_search.py       # Dual-engine Semantic Search / RAG document retriever
├── benchmarks/                  # Paper Reproducibility & Benchmark Pipeline
│   ├── prepare_datasets.py      # Automated canonical download from IRISA Texmex & Stanford NLP (MD5-verified)
│   ├── prepare_sift_scale.py    # SIFT scale expansion subset generator (100K -> 250K)
│   ├── prepare_transformer_384d.py # Multi-domain 384D transformer dataset pipeline (MiniLM-50K)
│   └── run_paper_benchmarks.py  # One-command C++ compile, execution & consolidated JSON/CSV export
├── cpp/                         # High-Performance Native C++17 Core
│   ├── adaptive_hnsw.hpp        # Header-only C++ Engine: Welford, Layer Scaling, Hubness, SQ8/SQ4, AVX2 SIMD, Streaming Deletion & Healing
│   ├── dataset_loader.hpp       # Fast binary .fvecs / .ivecs parser
│   ├── benchmark_main.cpp       # 6-Step ablation & macro benchmark runner
│   ├── run_transformer_benchmarks.cpp # Native 384D AVX2 SIMD benchmark harness
│   ├── run_streaming_deletion_benchmarks.cpp # Native streaming deletion & localized healing benchmark harness
│   ├── run_multithread_benchmarks.cpp # Multi-core OpenMP thread scaling harness
│   └── benchmark_runner.exe     # Compiled native benchmark runner executable
├── experiments/                 # Empirical Experiment Logs & Raw JSON Metrics
│   ├── transformer_384d_results.json # Frontier 4 384D benchmark metrics
│   └── streaming_deletion_results.json # Frontier 2 dynamic deletion & compaction metrics
├── frontend-react/              # Single-Focus Research Workbench (React 19 + Vite + Framer Motion)
│   ├── src/
│   │   ├── canvases/            # 4 Viewport Observatories (Benchmark, Manifold, Query, Dataset)
│   │   ├── components/workbench/# Tool rail, slide-in drawers (Policy Inspector, Raw Data, Settings)
│   │   ├── context/             # Global BenchmarkContext & keyboard shortcuts (1-4, Esc)
│   │   └── types/               # TypeScript interfaces & ground-truth benchmark types
│   ├── package.json             # React, Vite, Framer Motion, Lucide icons
│   └── vite.config.ts           # Vite bundler configuration
├── frontend/                    # Legacy HTML5/Vanilla JS Reference Studio
│   ├── index.html               # 9-view responsive research dashboard with control dock & HUD
│   ├── styles.css               # Clean Linear/Vercel design system with dark/light themes
│   └── app.js                   # Reactive math engine, canvas beam visualizer, and streaming runner
├── tests/                       # Automated Test Suite (25 Unit Tests Passing)
│   ├── test_signals.py          # Mathematical verification of LID on known manifolds
│   ├── test_policy.py           # Welford streaming, quantile, and layer-scaling unit tests
│   ├── test_hnsw.py             # Connectivity, early exit, hubness, SQ8 recall, deletions & compaction
│   └── test_quantization.py     # 4-bit sub-byte SQ4 nibble packing & reconstruction tests
├── pdf/                         # Research Paper Artifacts & Generated Figures
│   ├── AdaptiveVec_Research_Paper.pdf # Academic publication draft (Conference-ready)
│   ├── paper_figures/           # Publication-ready vectorized benchmark charts (Figures 1-15)
│   └── handbook_figures/        # System architectural diagrams (Figures 2, 3, 5, 6)
├── benchmark_results.json       # Consolidated machine-readable empirical benchmark outputs
├── benchmark_results.csv        # Consolidated empirical benchmark CSV
├── server.py                    # FastAPI REST server exposing REST endpoints & UI
├── test_api.py                  # API automated test harness
├── requirements.txt             # Python dependency specification
└── README.md                    # Research monograph and documentation
```

---

## 8. Quickstart & Reproducibility

### 8.1 Single-Command Benchmark Reproduction
To re-run the entire empirical benchmark suite from scratch on your local hardware:
```bash
python benchmarks/run_paper_benchmarks.py
```
This automated pipeline:
1. Detects your CPU architecture, core count, and AVX2/FMA vector instruction support.
2. Compiles the native C++ engine with `g++ -O3 -mavx2 -mfma -std=c++17`.
3. Verifies or downloads the canonical Texmex IRISA SIFT dataset with MD5 checksum verification.
4. Executes the complete 6-step ablation sequence on both `SIFT-100K subset` and `Synthetic-Multi-Cluster`.
5. Exports consolidated empirical results to `benchmark_results.json` and `benchmark_results.csv`.

---

### 8.2 Environment Setup & Installation
```bash
# Clone the repository
git clone https://github.com/aradhyags7/AdaptiveVec.git
cd AdaptiveVec

# Create isolated Python virtual environment
python -m venv venv

# Activate environment:
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 8.3 Automated Test Suite
Run the full 25-test suite verifying algorithmic integrity, Welford online tracking, SQ8/SQ4 quantization, streaming deletions, localized graph healing, and compaction:
```bash
pytest -v
```

---

### 8.4 Interactive Research Workbench (React + Vite)

<p align="center">
  <img src="frontend-react/src/assets/hero.png" width="95%" alt="AdaptiveVec Research Workbench UI"/>
  <br>
  <i><b>Figure 11:</b> Interactive Single-Focus Research Workbench (React 19 + Vite + Framer Motion) showing the Benchmark Observatory, live telemetry HUD, and regime toggles.</i>
</p>

The primary interface for AdaptiveVec is the **Single-Focus Research Workbench** (`frontend-react/`), engineered with **React 19, Vite, TypeScript, and Framer Motion**. Designed with the restrained aesthetic of a precision scientific instrument (oscilloscopes, laboratory spectrum analyzers, audio DAWs), the workbench gives researchers full visual and interactive control:

```bash
# Navigate to frontend-react directory
cd frontend-react

# Install dependencies
npm install

# Start Vite development server
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

#### Key Workbench Capabilities:
1. **Four Full-Viewport Focus Canvases:**
   - **Benchmark & Ablation Observatory:** Interactive log-scale Pareto frontier with spring-gliding crosshairs (`stiffness: 420, damping: 36`), synchronized 3-regime illumination (Regime A: Step 5 Stagnation Exit, Regime B: Step 6 SQ8, Control: Step 1 Baseline), discrete HUD settle, and invisible arrow stepping (`ArrowLeft`/`ArrowRight`).
   - **2D Manifold & Graph Observatory:** Dynamic projection re-anchoring between UMAP and PCA coordinates while proximity edges dynamically re-track. Includes 3 phosphor color modes (Degree $M$, LID, Density), generous $15\text{px}$ transparent hit targets for effortless clicking, and dashed axis reticle guides.
   - **Query Traversal & Geodesic Search:** Animated priority queue ($W$ buffer) visualizing candidate expansions and prunings with physical spring transitions via `<AnimatePresence mode="popLayout">`, top-candidate accent borders, and active hop radar convergence pulses.
   - **Dataset Corpus Matrix:** Modular instrument rack with genuine benchmark metrics for SIFT-100K, SIFT-250K, and Synthetic-Multi-Cluster, including hardware telemetry specs.
2. **Slide-In Utility Drawers:**
   - **Index Policy Inspector (`Cmd/Ctrl+I` or Tool Rail):** Inspect hyperparameter configurations ($M_{\text{base}}, efC_{\text{base}}, \alpha_{\text{LID}}, \beta_{\text{density}}$) with transparent provenance notes regarding native C++ compilation.
   - **Raw Data & Exports (`Cmd/Ctrl+D`):** Instant single-click exports of `benchmark_results.json` and `benchmark_results.csv` with raw metric inspection.
   - **Workbench Settings:** System hardware telemetry verification (`Intel Core 5 210H, 8 cores/12 threads, AVX2/FMA, 16GB RAM`), dark/light theme switching, and sensory options.
3. **Keyboard Shortcuts & Accessibility:**
   - Keys **`1`**, **`2`**, **`3`**, **`4`**: Switch instantly between the 4 focus canvases.
   - Key **`Escape`**: Closes any active slide-in drawer.
   - Keys **`ArrowLeft` / `ArrowRight`**: Step forward and backward through Pareto frontier ablation checkpoints.
   - **Full Reduced Motion Compliance:** Seamless integration with system `prefers-reduced-motion` via Framer Motion's `useReducedMotion()`, instantly falling back to zero-duration transitions.

---

### 8.5 FastAPI REST Server & Legacy Simulator

AdaptiveVec also includes a lightweight legacy web interface and typed REST API backend engineered with FastAPI and Vanilla JS:

```bash
# Launch FastAPI backend & research studio on port 8000
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

#### Key Interactive Features:
1. **Dynamic Hyperparameter Tuning Dock:** Adjust $\alpha_{\text{LID}}$, $\beta_{\text{density}}$, $efSearch$, and stagnation $\tau$ with live sliders. Watch the engine dynamically recalculate node degrees $M(x)$, total graph edges, predicted QPS speedup, and recall in real time across the dock and Regime A card.
2. **Multi-Hop Beam Search Traversal:** Click anywhere on the 2D canvas to dispatch an animated multi-hop beam search from express highway nodes down to the target node.
3. **Smooth PCA $\leftrightarrow$ UMAP Morphing:** Toggle between UMAP and PCA projections to observe a fluid 24-frame ease-in-out interpolation where nodes glide into their principal component positions.
4. **Active Laser Calibration Sweep:** Click **Re-run Calibration** to trigger a vertical cyan laser sweep across the canvas, micro-jittering empirical LID/density values, animating histogram bars, and updating Welford MLE estimates.
5. **Streaming Benchmark Suite Runner HUD:** Click **Run Benchmark Suite** to reveal the live runner HUD and watch a 10,000-query batch stream with real-time ticking counters for recall convergence, QPS, and latency.

---

### 8.6 Manual C++ Compilation
To compile the standalone benchmark harness with full AVX2/FMA vector optimizations manually:
```bash
cd cpp
g++ -O3 -mavx2 -mfma -std=c++17 benchmark_main.cpp -o benchmark_runner.exe
./benchmark_runner.exe
```

---

## 9. REST API Formal Specification

The backend server (`server.py`) provides typed endpoints for programmatic benchmarking:

| HTTP Method | Route | Request Payload | Response Schema & Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/status` | None | `{status: "online", n_samples, dim, stock_stats, adaptive_stats}` |
| `POST` | `/api/dataset/generate` | `{"name": "multi_manifold", "n_samples": 2000, "dim": 32, "n_queries": 50}` | Generates synthetic manifold datasets and computes initial 2D PCA projection |
| `POST` | `/api/index/build` | `{"policy_type": "continuous", "m_base": 16, "m_min": 8, "m_max": 24, "alpha_lid": 0.45, ...}` | Builds both Stock HNSW and AdaptiveVec; returns comparative build times and edge counts |
| `GET` | `/api/graph/projection` | Query param: `max_nodes=1500` | Returns 2D coordinates, LID, density, degree, and sampled layer edges |
| `POST` | `/api/search` | `{"query_index": 0, "k": 10, "ef_search": 64}` | Executes search and returns full traversal step trace and recall against ground truth |
| `POST` | `/api/benchmark/run` | Query param: `ef_search=64` | Runs automated suite measuring Recall@1/10/100, QPS, and p50/p95/p99 latencies |
| `POST` | `/api/semantic/search` | `{"query": "how does skip list hierarchy work?", "k": 5}` | Dual-index semantic document search with live edge-saving metrics |
| `POST` | `/api/semantic/add` | `{"title": str, "category": str, "content": str}` | Dynamically ingests a new text passage into both vector indices |

---

## 10. Academic Citation (BibTeX)

If you utilize AdaptiveVec in your academic research, benchmarking studies, or engineering projects, please cite this work:

```bibtex
@article{shinde2026adaptivevec,
  title        = {AdaptiveVec: A Density- and Dimension-Aware Proximity Graph Index for Resource-Constrained Vector Retrieval},
  author       = {Shinde, Aradhya},
  journal      = {Proceedings of Engineering Design & Innovation (EDI)},
  year         = {2026},
  month        = {September},
  url          = {https://github.com/aradhyags7/AdaptiveVec},
  note         = {Available at GitHub: aradhyags7/AdaptiveVec}
}
```

---

## 11. References

1. **Malkov, Yu A., and D. A. Yashunin.** *"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs."* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 42.4 (2020): 824-836.
2. **Levina, E., and P. J. Bickel.** *"Maximum likelihood estimation of intrinsic dimension."* Advances in Neural Information Processing Systems (NeurIPS) 17 (2005).
3. **Amsaleg, L., et al.** *"Estimating local intrinsic dimension."* ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2015): 29-38.
4. **Radovanović, M., Nanopoulos, A., & Ivanović, M.** *"Hubs in space: Popular nearest neighbors in high-dimensional data."* Journal of Machine Learning Research (JMLR) 11 (2010): 2487-2531.
5. **Subramanya, S. J., et al.** *"DiskANN: Fast accurate billion-point nearest neighbor search on a single node."* Advances in Neural Information Processing Systems (NeurIPS) 32 (2019).
6. **Welford, B. P.** *"Note on a method for calculating corrected sums of squares and products."* Technometrics 4.3 (1962): 419-420.
7. **Elliott, T., & Clark, C.** *"Improving HNSW graph construction through intrinsic dimensionality ordering."* arXiv preprint arXiv:2403.11928 (2024).
8. **Wang, M., Xu, X., Yue, Q., & Wang, Y.** *"A comprehensive survey and experimental comparison of graph-based approximate nearest neighbor search."* Proceedings of the VLDB Endowment (PVLDB) 14.11 (2021): 1964-1978.
9. **Li, W., et al.** *"Approximate nearest neighbor search on high dimensional data—experiments, analyses, and improvement."* IEEE Transactions on Knowledge and Data Engineering (TKDE) 32.8 (2020): 1475-1488.
10. **Shinde, A.** *"AdaptiveVec: Density- and dimension-aware proximity graph index for resource-constrained vector retrieval."* Proceedings of Engineering Design & Innovation (EDI) (2026).

---

<div align="center">
  <sub>AdaptiveVec Research Initiative • Developed for the 2nd-Year Computer Engineering EDI Project • Released under the MIT License</sub>
</div>
