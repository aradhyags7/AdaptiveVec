# AdaptiveVec — Phase 5.2: Stage 0 Acceptance Decisions & Experimental Correction Audit

**Document Identifier**: `experiments/phase_5_2_phase_5_1_acceptance_audit.md`  
**Phase**: 5.2 (Stage 0 Governance Reset & Empirical Reclassification)  
**Author**: Antigravity Autonomous Systems Research  
**Status**: Authoritative Experimental Correction Monograph  

---

## 1. Executive Direction & Reclassification of Geometry-Conditioned Pruning (GCP)

In Phase 5.1, Geometry-Conditioned Pruning (GCP) was evaluated as a candidate mechanism for edge selection. While GCP satisfied its isolated metric gate of reducing distance evaluations relative to Canonical Uniform HNSW (1,130.1 vs 1,264.7 evals, $-10.64\%$), a rigorous head-to-head comparison against **Architecture B (Optimized Original AdaptiveVec)** reveals critical trade-offs that preclude adopting GCP as a default production mechanism.

### 1.1 Direct Head-to-Head Comparison: GCP vs Architecture B on SIFT-100K ($efSearch=64$)

| Performance Metric | Architecture B (Control 2) | Phase 5.1 GCP (Exp-8 Full) | Delta ($\Delta$) | Directional Impact |
|:---|:---:|:---:|:---:|:---:|
| **Recall@10** | **99.25%** | 98.99% | **-0.26%** | Regressed |
| **Recall@1** | **99.59%** | 99.34% | **-0.25%** | Regressed |
| **Search Throughput (QPS)** | **26,032.2 QPS** | 22,483.4 QPS | **-13.63%** | Regressed |
| **Mean Distance Evals/query** | 1,168.8 | **1,130.1** | **-3.31%** | Marginal Reduction |
| **Total Graph Edges** | 2,418,571 | **2,379,061** | **-1.63%** | Marginal Reduction |
| **Allocated Memory** | 100.88 MB | 100.63 MB | **-0.25%** | Negligible ($\approx 250\ \text{KB}$) |
| **In-Degree Variance** | 66.3 | **55.0** | **-17.0%** | Improved |
| **p95 Latency** | **0.714 ms** | 0.859 ms | **+20.3%** | Slower Tail |

### 1.2 Formal Governance Disposition:
1. **Reclassification**: Geometry-Conditioned Pruning (GCP) is formally **reclassified as an experimental mechanism**. It is disabled by default in production and scientific baselines.
2. **Scientific Finding**: Modulating pruning aggressiveness $\alpha_i$ per node yields only a marginal $-1.63\%$ edge reduction and $-3.31\%$ distance evaluation reduction relative to Architecture B, while inducing a substantial **$-13.63\%$ throughput penalty** due to branch divergence and slightly degraded routing conduits.
3. **Primary Research Foundation**: **Architecture B remains the sole primary research foundation** for AdaptiveVec.

---

## 2. Discovery and Significance of the Strict RNG Control

In Phase 5.1, Control 5 evaluated **Strict RNG** (`keep_pruned_connections = false`). In canonical HNSW, `select_neighbors_heuristic` checks candidate diversity: if candidate $c$ is shadowed by an already accepted neighbor $r$ ($d(c, r) < d(c, q)$), $c$ is added to a `discarded` queue. If the diverse result set has fewer than $M$ elements, HNSW fills the remaining slots from `discarded` (fallback neighbor retention).

When this fallback retention is disabled (`keep_pruned_connections = false`), the graph retains **only strictly diverse RNG edges**.

### 2.1 Remarkable Empirical Performance of Strict RNG (Control 5)

| Metric | Canonical HNSW (Ctrl-1) | Architecture B (Ctrl-2) | Strict RNG Control (Ctrl-5) | Strict RNG vs Canonical | Strict RNG vs Arch B |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Total Graph Edges** | 2,694,602 | 2,418,571 | **1,925,264** | **-28.55% (-769,338 edges)** | **-20.40% (-493,307 edges)** |
| **Layer-0 Mean Degree** | 25.87 | 23.22 | **18.44** | **-28.7%** | **-20.6%** |
| **Recall@10** | 99.45% | 99.25% | **99.30%** | **-0.15% (Within Noise)** | **+0.05%** |
| **Search QPS** | 23,631.8 | **26,032.2** | **25,137.7** | **+6.37% Faster** | **-3.4%** |
| **Mean Distance Evals** | 1,264.7 | **1,168.8** | **1,184.4** | **-6.35% Fewer** | **+1.3%** |
| **Allocated Memory** | 102.57 MB | 100.88 MB | **97.57 MB** | **-5.00 MB Saved** | **-3.31 MB Saved** |
| **In-Degree Variance** | 38.4 | 66.3 | **35.2** | **-8.3% Lower** | **-46.9% Lower** |
| **Build Time** | 18.26 s | 19.03 s | **12.51 s** | **-31.5% Faster Build** | **-34.2% Faster Build** |

### 2.2 Critical Research Implications for Phase 5.2
1. **Fallback Retention is Redundant Clutter**: Retaining non-diverse discarded candidates to artificially saturate node degree to $M$ adds **$\approx 770,000$ redundant edges** in SIFT-100K that contribute virtually zero retrieval benefit (Recall@10 is 99.30% without them vs 99.45% with them).
2. **Speed & Efficiency**: Strict RNG prunes edges naturally, accelerating build time by **31.5%** and reducing graph memory without requiring complex online geometry estimators.
3. **Mandatory Investigation in Phase 5.2**: Before developing complex layer-specific geometric adaptations, Stage 1 mandates a systematic ablation of Strict RNG:
   - Does Strict RNG compromise upper-layer routing?
   - Does applying Strict RNG to Layer 0 while keeping standard retention on upper layers provide the optimal balance between long-range navigability and local efficiency?
   - How does Strict RNG interact with Architecture B's geometric $M(x)$ allocation?

---

## 3. Disentangling Logical Edge Count from Physical Allocated Memory

A recurring reporting issue in earlier phases was conflating the **number of logical edges** ($\sum |\mathcal{N}_i|$) with **physical allocated memory** ($\mathcal{M}_{\text{alloc}}$).

### 3.1 Allocator Architecture & Capacity Headroom
In modern standard library allocators (and flat contiguous chunk arenas):
- `std::vector<tableint>` capacity grows geometrically (typically doubling: 1, 2, 4, 8, 16, 32, 64).
- If a node has a degree limit of $M_{\max0} = 32$, its adjacency vector reserves capacity for 32 elements ($32 \times 4 = 128$ bytes).
- If Strict RNG or GCP reduces the realized degree from 26 down to 22, the logical edge count drops from 26 to 22 ($-15.4\%$). However, unless the allocation buffer is explicitly compacted or sized to fit, `vector.capacity()` remains 32, and the physical allocated memory remains identical!

### 3.2 Rigorous Accounting Contract for Phase 5.2
1. **Report Both Metrics Explicitly**: Every table must report:
   - Logical Edge Count ($\sum |E_l|$)
   - Actual Adjacency Allocated Bytes ($\sum \text{Capacity}(\mathcal{N}_i) \times \text{sizeof}(\text{tableint})$)
   - Total Process Allocated RAM (including vectors, generation arrays, metadata, and runtime heaps).
2. **Compact Adjacency Storage**: Phase 5.2 investigates whether flat, packed adjacency representations (e.g., CSR or flat arrays sized to realized degree) can convert logical edge reductions directly into equivalent DRAM byte savings.

---

## 4. Summary of Phase 5.2 Directives

1. **Architecture B** is preserved as the active research foundation.
2. **GCP** is classified as experimental and non-default.
3. **Strict RNG** is elevated to a primary comparator and investigated layer-by-layer.
4. **Layer-Decoupled Resource Allocation** will be tested against fixed, uniform, and Strict RNG baselines to isolate whether upper layers benefit from geometric adaptation.
