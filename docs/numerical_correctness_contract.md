# Numerical Correctness and Architecture Contract
**AdaptiveVec Engine Specification — Phase 2.1 Release**

This document establishes the binding architectural and numerical correctness contracts for AdaptiveVec. All future phases (including automatic calibration, workload-aware graph evolution, and hardware acceleration) must strictly adhere to these specifications.

---

## 1. Storage Pool and Memory Layout Contract

### 1.1 Vector Storage Descriptors
Nodes in the index are mapped to representations via a 64-bit descriptor:
```cpp
struct VectorStorageDescriptor {
    uint32_t offset = 0;                     // Pool element/byte offset
    VectorPrecision prec = VectorPrecision::FP32; // Tag: FP32, SQ8, SQ4
};
```
- **Tag Integrity:** `prec` must be explicitly initialized upon node assignment.
- **Offset Semantics:**
  - For `VectorPrecision::FP32`: `offset` represents the **float element index** into `pool_fp32` (e.g., `&pool_fp32[offset]`).
  - For `VectorPrecision::SQ8`: `offset` represents the **byte index** into `pool_sq8` (e.g., `&pool_sq8[offset]`), incrementing by `dim`.
  - For `VectorPrecision::SQ4`: `offset` represents the **byte index** into `pool_sq4` (e.g., `&pool_sq4[offset]`), incrementing by `packed_dim = (dim + 1) / 2`.
- **Vector ID Stability:** The index of `vector_descriptors[i]` corresponds strictly to `tableint i`. Under node compaction or tombstone removal, descriptors must be remapped in synchronized lock-step with graph adjacency lists.

### 1.2 Memory Accounting Equations
For an index containing $N$ vectors of dimensionality $D$:
- **FP32 Logical Payload:** $4 \times N \times D$ bytes.
- **SQ8 Logical Payload:** $1 \times N \times D$ bytes.
- **SQ4 Logical Payload:** $1 \times N \times \lceil D / 2 \rceil$ bytes.
- **Mixed Precision Realized Payload:**
  $$M_{\text{payload}} = 4 \times N_{\text{fp32}} \times D + 1 \times N_{\text{sq8}} \times D + 1 \times N_{\text{sq4}} \times \left\lceil \frac{D}{2} \right\rceil \text{ bytes}$$
- **Verification Rule:** All memory reports must compute payload from realized descriptor counts ($N_{\text{fp32}}, N_{\text{sq8}}, N_{\text{sq4}}$), never from requested target budget ratios.

---

## 2. Quantization Numerical Correctness Contract

### 2.1 Mathematical Reconstruction
Scalar quantization maps floating-point coordinates $x_j \in [\text{min}_j, \text{max}_j]$ to discrete integer codes:
- **SQ8 Code:**
  $$c_j = \text{round}\left( \frac{x_j - \text{min}_j}{\text{scale}_j} \right), \quad c_j \in [0, 255], \quad \text{scale}_j = \frac{\text{max}_j - \text{min}_j}{255}$$
- **SQ4 Code:**
  $$c_j = \text{round}\left( \frac{x_j - \text{min}_j}{\text{scale}_j} \right), \quad c_j \in [0, 15], \quad \text{scale}_j = \frac{\text{max}_j - \text{min}_j}{15}$$
- **Reconstruction:**
  $$\hat{x}_j = \text{min}_j + c_j \times \text{scale}_j$$
- **Degenerate Subspaces:** If $\text{max}_j - \text{min}_j \le 10^{-7}$, $\text{scale}_j \equiv 1.0$, preventing zero-division and NaN propagation.

### 2.2 Prohibited Online Training Pattern
> **CRITICAL INVARIANT:** Quantizers must **NEVER** be trained on single vectors ($N=1$) inside node insertion loops (`insert()`).
- Training on $N=1$ collapses $[\text{min}_j, \text{max}_j]$ to a single point, freezing $\text{scale}_j = 1.0$ and $\text{min}_j = x_{0, j}$.
- All quantizers must be trained on a representative calibration sample ($N \ge 1,000$, default $50,000$) before quantized encoding begins.
- Graph construction must proceed on pure FP32 raw vectors unless explicitly operating in a pre-quantized offline mode.

---

## 3. SIMD Distance Kernel & Search Beam Contract

### 3.1 Kernel Precision Gates
- **Scalar Reference Parity:** For any query vector $q$ and quantized code $c$, the asymmetric SIMD kernel result $D_{\text{simd}}(q, c)$ must match the scalar reconstruction distance $D_{\text{scalar}}(q, \hat{x})$ within single-precision floating point tolerance:
  $$\frac{|D_{\text{simd}} - D_{\text{scalar}}|}{\max(1.0, |D_{\text{simd}}|)} \le 10^{-4}$$
- **Nibble Unpacking (SQ4):** AVX2 nibble expansion must strictly unpack byte $i$ as:
  - Low nibble (bits 0–3): dimension $2i$.
  - High nibble (bits 4–7): dimension $2i + 1$.
  - Odd-dimension tails must be handled cleanly without out-of-bounds byte reads.

### 3.2 Search Beam Width Contract
- **User ef Primacy:** The search beam width $W$ must be governed by user search parameters:
  $$W = \max(k, \text{effective\_ef})$$
- **Reranking Protection:** In pure quantized search (`enable_rerank = false`) or mixed precision without raw FP32 vectors, $W$ must **never** be multiplied by `rerank_factor * k`. Multiplicative beam expansion is reserved solely for two-stage candidate re-ranking against resident FP32 buffers.
- **Dynamic Ada-ef Compatibility:** When `enable_ada_ef = true`, stagnation early exit must only trigger at Layer 0 after $|W| \ge \text{effective\_ef}$ candidates have been evaluated.

---

## 4. Verification Acceptance Gates

Before any new PR or research phase is accepted into the engine:
1. **Canonical Parity:** Canonical FP32 retrieval must reproduce verified baselines ($>92.5\%$ Recall@10 at ef=64 on SIFT-100K and Transformer-384D).
2. **Ground-Truth Parity:** System A brute-force search must achieve $100\%$ match against independently computed exact ground truth.
3. **Kernel Verification:** `test_numerical_correctness` must achieve 100% pass rate across all benchmark dimensions ($D \in \{1, 7, 16, 63, 64, 65, 127, 128, 384, 768, 1536\}$).
4. **4-Way Decomposition:** System C (Asymmetric SIMD) vs System B (Reconstructed Scalar) must show $100\%$ ranking agreement.
