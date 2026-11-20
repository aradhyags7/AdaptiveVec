# AdaptiveVec — Uncertainty-Aware Search and Dynamic Patience Architecture

**Author**: Senior Systems Research Engineer  
**Date**: November 20, 2026  
**Document Status**: Production Architecture Specification (Phase 4.3)  
**System Target**: Resource-Constrained Graph-Based Vector Retrieval (ANNS)

---

## 1. Architectural Motivation and Hypothesis

### 1.1 The Stagnation-Termination Trade-Off in Proximity Graphs
In approximate nearest neighbor search over Hierarchical Navigable Small World (HNSW) graphs, fixed search budgets ($efSearch$) waste computation on easy queries that converge rapidly into distinct clusters. Conversely, heuristic stagnation termination (Ada-ef) terminates traversal after a fixed number of consecutive non-improving steps ($patience$).

However, fixed stagnation patience exhibits two fundamental failure modes:
1. **False Early Termination on Manifold Boundaries**: When navigating high-dimensional or ambiguous boundary regions, search often encounters local topological plateaus where 4 to 8 consecutive candidate expansions fail to improve the furthest distance in the beam, yet further exploration would escape the plateau and discover the true nearest neighbors. A small fixed patience prematurely terminates the search, causing recall loss.
2. **Futile Distance Evaluations in Converged Clusters**: In dense, well-separated clusters, the search discovers the true top-$k$ neighbors within the first few hops. Continuing to expand candidates up to a conservative patience budget (e.g. $p=16$) computes dozens of futile distance evaluations against suboptimal neighbors.

### 1.2 The Uncertainty-Aware Search Hypothesis
Phase 4.3 investigates the hypothesis that **search patience should be dynamically modulated based on real-time topological uncertainty and candidate convergence**:
$$p(q, t) = \text{clip}\left(p_{base} + \alpha \cdot \text{Uncertainty}(q, t) - \beta \cdot \text{Convergence}(q, t), p_{min}, p_{max}\right)$$
- When local topological uncertainty is high (tight neighbor margins, high intrinsic dimensionality), patience expands ($p \to p_{max}$), permitting the search to escape local plateaus.
- When candidate convergence is decisive (wide margins, low intrinsic dimensionality, rapidly decaying improvement velocity), patience contracts ($p \to p_{min}$), cutting off redundant distance evaluations.

---

## 2. Zero-Contention Traversal Instrumentation

### 2.1 The Critical Flaw of Prior Instrumentation
Phase 4.3 Stage 1 revealed that previous fixed-budget benchmarks reported synthetic placeholders ($6 \times efSearch$) because `search()` lacked per-query instrumentation and index-level atomic counters suffered multi-threaded cache-line bouncing.

### 2.2 Thread-Local Search Instrumentation
AdaptiveVec Phase 4.3 implements zero-allocation, thread-local instrumentation passed directly via stack pointers:
```cpp
struct QuerySearchStats {
    uint32_t upper_layer_dist_evals = 0; // Greedy descent (L_max down to 1)
    uint32_t layer0_dist_evals = 0;      // Layer-0 beam exploration
    uint32_t rerank_dist_evals = 0;      // Two-stage FP32 re-ranking
    uint32_t visited_nodes = 0;          // Nodes marked in VisitedTagPool
    uint32_t hops = 0;                   // Candidate expansions popped from queue
    uint32_t early_exit_stagnation_count = 0; // Stagnation early exits
    
    inline uint32_t total_dist_evals() const {
        return upper_layer_dist_evals + layer0_dist_evals + rerank_dist_evals;
    }
};
```
Every distance calculation is recorded without atomic synchronization, ensuring zero multi-threading overhead while capturing the exact hardware distance evaluation breakdown.

---

## 3. Mathematical Formulation of Search Utility Signals

All utility features are computed strictly from already-evaluated distances in CPU registers without memory allocations or ground-truth access:

```
[ Active Candidate Beam W ]
            |
            +---> Top-k Distance Margin: Delta_margin = (d_k - d_1) / (d_k + eps)
            |
            +---> Local Intrinsic Dim:   LID = [ 1/(k-1) * sum ln(d_k / d_i) ]^(-1)
            |
            +---> Improvement Velocity:  v_imp = Delta d_best / (d_best * steps)
            |
            v
+--------------------------------------------------------------------------+
|  Uncertainty = clip( LID / 10.0 + (1.0 - Delta_margin), 0.0, 3.0 )       |
|  Convergence = clip( stagnation / p_base + (1.0 - 10 * v_imp), 0.0, 3.0 )|
+--------------------------------------------------------------------------+
                                    |
                                    v
       p(q, t) = clip( p_base + alpha * Uncertainty - beta * Convergence, p_min, p_max )
```

### 3.1 Local Intrinsic Dimensionality (LID)
Computed via the Levina-Bickel Maximum Likelihood Estimator on the active top-$k$ candidates:
$$\widehat{\text{LID}}(q, t) = \left(\frac{1}{k-1} \sum_{i=1}^{k-1} \ln \left(\frac{d_k(q)}{d_i(q) + \epsilon}\right)\right)^{-1}$$
- High LID ($\ge 10.0$) indicates dense, high-dimensional manifolds requiring deeper beam expansion.
- Low LID ($\le 5.0$) indicates low-dimensional linear manifolds where greedy routing is highly efficient.

### 3.2 Top-k Relative Distance Margin
$$\Delta_{margin}(q, t) = \frac{d_k(q) - d_1(q)}{d_k(q) + \epsilon}$$
- A large margin ($\Delta_{margin} > 0.35$) indicates that the nearest neighbor is clearly separated from the background cluster.
- A small margin ($\Delta_{margin} < 0.10$) indicates severe neighborhood ambiguity near cluster boundaries.

### 3.3 Candidate Improvement Velocity
$$\nu_{imp}(q, t) = \frac{\sum \Delta d_{improvement}}{d_{best} \cdot N_{improvement}}$$
Measures the percentage rate at which the search frontier is continuing to discover closer candidates.

---

## 4. Search Traversal and Explicit Termination Taxonomy

### 4.1 Traversal Loop Logic
During layer-0 beam search:
1. When beam capacity reaches $efSearch$, stagnation monitoring activates.
2. Dynamic patience $p(q, t)$ is evaluated in $O(1)$ CPU operations.
3. If stagnation counter exceeds $p(q, t)$, search terminates immediately.
4. **Decisive Convergence Fast-Exit**: If $\Delta_{margin} > 0.40$ and $\widehat{\text{LID}} < 6.0$ after minimum initial exploration ($p \ge p_{min}$), search terminates immediately under `HighMarginConvergence`.

### 4.2 Termination Taxonomy
Every query termination records an explicit categorical reason:
- `FrontierExhausted`: Candidate queue emptied or frontier distance exceeded furthest beam distance.
- `FixedPatienceEarlyExit`: Stagnation counter exceeded static patience threshold.
- `UncertaintyAwareEarlyExit`: Dynamic patience $p(q, t)$ was exhausted.
- `HighMarginConvergence`: Decisive high-margin, low-LID convergence triggered.
- `BudgetExhausted`: Traversal reached hard beam boundary.

---

## 5. Invariant Preservation and Safety Guarantees

1. **State Invariance**: Candidate priority queues and furthest heaps maintain strict partial ordering at all times.
2. **Zero Dynamic Memory Allocation**: Traversal operates exclusively on pre-allocated thread-local tag pools and register-backed heaps ($\Delta \text{RAM} = 0$).
3. **Bounded Patience Envelope**: Enforced clamping ensures:
   $$2 \le p(q, t) \le 16$$
   Eliminates infinite loops or premature single-hop dropouts.
4. **No Test Ground-Truth Leakage**: All signals are derived strictly from query distances computed during the live traversal.
