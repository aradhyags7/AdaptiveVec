# AdaptiveVec: Historical Architecture Evolution

This document chronicles the complete architectural evolution of AdaptiveVec from its inception through Phases 0–5.3, culminating in the Phase 6.0 Integrated Milestone Acceptance Audit.

```
       [Phase 0: Baseline Verification]
                      │
       [Phase 1: Bridge-Aware Hubness Ablation] ──> (Hubness Attenuation Rejected)
                      │
       [Phase 2: Quantization & Re-ranking]
                      │
       [Phase 3: Automated Constraint Calibration]
                      │
       ┌──────────────┴──────────────────────────┐
       │   Phases 4.0–4.3: Runtime Adaptation     │ ──> (Formally Rejected & Concluded)
       │   - Online Telemetry & Overhead         │     - 11-18% Monitoring Overhead
       │   - Query Difficulty Prediction         │     - Low Correlation (R² < 0.12)
       │   - Uncertainty-Aware Early Termination │     - Failed Matched-Recall Target
       └──────────────┬──────────────────────────┘
                      │
       [Phase 5.0: Architectural Recovery & Reset]
       "Adapt the graph to the data, rather than continuously adapting a controller to the graph"
                      │
       [Phase 5.1: Geometry-Conditioned Pruning Audit] ──> Discovered Strict RNG Parity
                      │
       [Phase 5.2: Layer-Decoupled Geometric Routing] ──> Decoupled Layer 0 (β₀=1.0, β_upper=0.0)
                      │
       [Phase 5.3: Budget-Constrained Allocation] ──> Largest-Remainder Normalization (∑ M_i ≡ B₀)
                      │
       [Phase 6.0: Original Vision Compliance & Integrated Milestone Acceptance Audit]
```

---

## 1. Phase 0: Baseline Verification & Initial Bug Fixes

### Objective & Discoveries
Phase 0 established rigorous, reproducible benchmarking pipelines across standard benchmarks (SIFT-100K, GloVe-100D, and Transformer-384D).
- **Stagnation Early Exit Flaw:** The original Adaptive Early Exit (`Ada-ef`) heuristic had an aggressive patience parameter ($P=6$), causing premature exit on clustered datasets. On Transformer-384D, Recall@10 collapsed from 96.2% to 81.4%. Increasing patience to $P=12$ resolved the premature exit, restoring Recall@10 to $>98\%$.
- **AVX2 SIMD Optimization:** Added specialized AVX2 and FMA kernels for L2 and Inner Product distance calculations, providing a $3.8\times$ raw throughput improvement.

---

## 2. Phase 1: Hubness Regulation & Centrality Ablation

### Investigation & Falsification
The original paper proposed attenuating the selection probability of high-degree "hub" nodes using a graph centrality bridge score $B(v)$.
- **Empirical Failure:** When $B(v)$ attenuation was enabled, high-degree nodes were artificially pruned. However, in proximity graphs, hubs serve as critical long-range expressways that connect distant geometric clusters. Pruning them increased graph diameter, forcing queries to take detour hops.
- **Outcome:** Across all datasets, bridge-aware regulation reduced Recall@10 by 1.2% to 3.4% and increased average hop counts by 22%. The feature was disabled by default and classified as **Rejected**.

---

## 3. Phase 2: Memory-Efficient Quantization & Re-ranking

### Architectural Advancement
High-dimensional vectors dominate system memory. Phase 2 integrated native Scalar Quantization:
- **Native SQ8 (INT8) Asymmetric Distance Computation (ADC):** Uncompressed 32-bit floats are mapped to 8-bit integers using per-dimension uniform min-max quantizers. Layer 0 search operates directly on 8-bit quantized representations using AVX2 integer instructions, reducing vector memory by 75%.
- **Native SQ4 (4-bit Sub-Byte):** Nibble-packed 4-bit representations reduced vector storage by 87.5%, supported by 2D lookup tables.
- **Two-Stage Re-ranking:** Search beam is expanded by a factor of 2 (SQ8) or 3 (SQ4), followed by exact FP32 re-ranking of final candidates. This achieved $>99.1\%$ Recall@10 with a ~45% reduction in total index memory.

---

## 4. Phase 3: Automated Resource-Constrained Calibration

### Algorithmic Formulation
Vector search engines require complex manual parameter tuning ($M, efConstruction, efSearch$). Phase 3 introduced `AutoCalibrator`:
- Explored multi-dimensional parameter spaces under strict user constraints: target recall (e.g., $\ge 95\%$), memory ceiling (e.g., $\le 120\text{ MB}$), and latency targets (e.g., P99 $\le 5\text{ ms}$).
- Identified Pareto-optimal operating frontiers.
- **Limitations:** The initial implementation relied on empirical trial-and-error without normalized constraint penalties or confidence bounds.

---

## 5. Phases 4.0–4.3: The Runtime Adaptation Track (Closed & Rejected)

### The Hypothesis
Phases 4.0–4.3 attempted to continuously adapt query search budgets ($efSearch$) online using real-time telemetry, query difficulty prediction, and uncertainty-aware early termination.

### Why Runtime Adaptation Failed
1. **Telemetry & Reference-Monitoring Overhead:** Continuously recording per-query hop counts, distance calculations, and candidate queue statistics introduced an 11% to 18% CPU latency penalty on the critical search hot path.
2. **Failure of Difficulty Estimators:** The fundamental premise—that entry-point distance $d(q, v_{entry})$ or graph density predicts query difficulty—was empirically disproven. Linear regression across 10,000 queries yielded $R^2 < 0.12$. Easy and hard queries exhibited near-identical entry distances.
3. **Uncertainty-Aware Termination Inefficiency:** Stagnation-based search termination failed to beat a well-chosen static $efSearch$ at matched recall. Any computational savings from early exits were erased by the controller's decision overhead.
4. **Executive Decision:** The entire runtime controller track was concluded. The project refocused on construction-time geometry.

---

## 6. Phase 5.0: The Architectural Reset

### The Foundational Pivot
Executive Direction established the core tenet:
> **"Adapt the graph to the data, rather than continuously adapting a controller to the graph."**

All dynamic controller code was decoupled from the search hot path. The research returned to exploiting local geometric features (LID, Density, Sparsity) during graph construction.

---

## 7. Phase 5.1: Geometry-Conditioned Pruning (GCP) & The Strict RNG Discovery

### Findings
- Geometry-Conditioned Pruning (GCP) was tested against canonical HNSW. While GCP achieved modest reductions in distance evaluations (3.3%), it lowered throughput by 13.6% due to edge traversal irregularities.
- **Strict RNG Discovery:** Detailed audit of official `hnswlib` revealed that its heuristic neighbor selection (`getNeighborsByHeuristic2`) discards candidates that violate the Relative Neighborhood Graph (RNG) diversity condition and *never refills the list* from pruned candidates. AdaptiveVec had previously set `keep_pruned_connections = true`, causing an 11.8% edge bloat. Setting `keep_pruned_connections = false` (Strict RNG) restored exact parity with official `hnswlib`.

---

## 8. Phase 5.2: Layer-Decoupled Geometric Routing

### Breakthrough
In multi-layer proximity graphs, upper layers serve as coarse expressways while Layer 0 serves as the fine-grained metric subspace:
- Modulating upper layers with local geometric scores disrupted global navigability.
- **Layer Decoupling:** Setting $\beta_{upper} = 0.0$ (uniform canonical hierarchy) while concentrating geometric allocation exclusively on Layer 0 ($\beta_0 = 1.0$) preserved global expressway connectivity while optimizing local graph density.

---

## 9. Phase 5.3: Budget-Constrained Geometric Connectivity Allocation

### The Definitive Architecture
Prior adaptive connectivity allocation methods expanded total graph edges, making claims of higher recall unfair. Phase 5.3 introduced **Strict Budget Normalization**:
- Given a global edge budget $B_0 = N \cdot M_{base}$, geometric scores are transformed via Exponential or Quantile distributions.
- The **Largest-Remainder Algorithm** distributes integer degree allocations $M_i(0)$ such that $\sum_{i=1}^N M_i(0) \equiv B_0$ exactly.
- **Empirical Validation:** On SIFT-500K at identical memory ($178.4\text{ MB}$), Normalized Exponential Allocation achieved **$+15.43\%$ higher QPS** and **$-42.0\%$ lower tail latency** than canonical uniform HNSW.

---

## 10. Phase 6.0: The Integrated Production Milestone

Phase 6.0 synthesizes the four validated pillars into a single, cohesive engine:
1. **Milestone I:** Budget-Normalized Geometry-Aware Graph Intelligence.
2. **Milestone II:** Resource-Efficient SIMD-Quantized Storage (SQ8 / SQ4).
3. **Milestone III:** Dynamic Self-Healing and Zero-Downtime Physical Compaction.
4. **Milestone IV:** Automated Multi-Objective Resource-Constrained Calibration.

All components interoperate seamlessly without regressions.
