# AdaptiveVec — Lightweight Adaptive Search and Controller Architecture

**Author**: Senior Systems Research Engineer  
**Date**: November 20, 2026  
**Document Status**: Production Architecture Specification (Phase 4.2)  
**System Target**: Resource-Constrained Approximate Nearest Neighbor Search (ANNS)

---

## 1. Architectural Motivation and System Overview

### 1.1 Limitations of Prior Runtime Adaptation (Phase 4.1 Audit)
In Phase 4.1, empirical audits revealed two severe operational bottlenecks in the original continuous adaptation architecture:
1. **Excessive Reference Monitoring Overhead**: The Phase 4 controller evaluated a batch of $M_{ref} = 100$ offline reference queries every 50 production queries. On 384-dimensional vectors, evaluating 100 full graph traversals every 50 production requests incurred a **$190\%$ to $200\%$ computational penalty**, collapsing query throughput by up to $81\%$ (from 3,100 QPS down to 579 QPS on Transformer-384D).
2. **Ineffective Difficulty Prediction**: Global entry-point distance $d(q, ep)$ exhibited near-zero rank correlation with actual query hardness ($\rho = -0.07$). HNSW's upper-layer greedy descent prunes macroscopic distance exponentially in $O(\log N)$ hops, rendering initial global distance uninformative by the time traversal reaches layer 0.

### 1.2 Two-Level Decoupled Architecture
Phase 4.2 resolves these deficiencies by decoupling fast, per-query execution from slow, background workload-level calibration:

```
                            [ Incoming Query q ]
                                     |
                                     v
                 +---------------------------------------+
                 |       Upper Layers Greedy Descent      |
                 +---------------------------------------+
                                     |
                         (Layer-0 Entry Node ep_L0)
                                     v
               +-------------------------------------------+
  FAST PATH    |     Phase 1: Conservative Beam (ef_init)  |
  (Per-Query)  +-------------------------------------------+
                                     |
                         [ Intermediate Signals ]
                    Margin, Local LID, Distance Rate
                                     |
                                     v
                        /-------------------------\
                       <     Should Expand Beam?   >
                        \-------------------------/
                               /            \
                       [No: Easy]          [Yes: Hard]
                             /                \
                 +-------------------+  +------------------------------------+
                 | Terminate at ef_init|  | Phase 2: Resume Beam to ef_expanded|
                 | (15-20% Work Saved)|  | (Reuse Heaps, Tags & Distances)   |
                 +-------------------+  +------------------------------------+
                             \                /
                              v              v
                         [ Final Top-K Result ]
                                     |
=====================================|========================================
SLOW PATH                            v
(Workload Level) +---------------------------------------+
                 | Incremental Reference Monitor (< 5%)  | (1 ref query / 25 req)
                 +---------------------------------------+
                                     |
                            [ Rolling Recall ]
                                     |
                                     v
                 +---------------------------------------+
                 | Workload Controller (Hysteresis, Deadband)
                 | Adapts ef_init, ef_expanded, or Rollback
                 +---------------------------------------+
```

---

## 2. Resumable Search Architecture

### 2.1 The Need for Resumable Graph Traversal
Traditional dynamic beam search implementations evaluate queries with a fixed budget $efSearch$. If intermediate inspection determines that a query requires deeper exploration, conventional implementations restart traversal from scratch, discarding all previously calculated vector distances and visited states.

AdaptiveVec Phase 4.2 implements **Resumable Graph Traversal**:
- Layer-0 search begins with a conservative initial search beam $ef_{init}$ (e.g. 32).
- Candidate priority queues (`candidate_queue`), furthest-candidate heaps (`w_furthest`), and visited tags (`VisitedTagPool`) are maintained in thread-local storage.
- If multi-signal analysis classifies the query as hard, the target capacity of `w_furthest` is expanded from $ef_{init}$ to $ef_{expanded}$ (e.g. 64), and traversal resumes directly from the active candidate frontier.

### 2.2 Algorithmic Formulation

```text
Algorithm: RESUMABLE-LAYER0-SEARCH(Index, q, k, ef_init, ef_expanded, Estimator)
Input: Index graph G, query vector q, target k, initial beam ef_init, expanded beam ef_expanded
Output: Top-k nearest neighbors

1.  curr_ep <- GREEDY-DESCENT-UPPER-LAYERS(G, q, enter_point)
2.  visited <- GET-THREAD-VISITED-POOL()
3.  visited.reset(N)
4.  visited.mark(curr_ep)
5.  candidates <- MIN-HEAP({(dist(q, curr_ep), curr_ep)})
6.  w_furthest <- MAX-HEAP({(dist(q, curr_ep), curr_ep)})
7.  best_dist <- dist(q, curr_ep)
8.  
9.  // Phase 1: Explore with conservative budget ef_init
10. EXPLORE-FRONTIER(candidates, w_furthest, ef_init, visited, best_dist)
11. 
12. // Phase 2: Multi-Signal Difficulty Evaluation
13. signals <- EXTRACT-SIGNALS(candidates, w_furthest, q, curr_ep)
14. if Estimator.should_expand(signals) then:
15.     // Seamlessly expand beam without resetting visited tags
16.     EXPLORE-FRONTIER(candidates, w_furthest, ef_expanded, visited, best_dist)
17. 
18. return EXTRACT-TOP-K(w_furthest, k)
```

### 2.3 Proof of Invariant Preservation
Let $C_t \subseteq V$ denote the set of frontier candidates in `candidates` at time step $t$, and let $W_t \subseteq V$ denote the $ef$-nearest nodes discovered so far in `w_furthest`.
1. **Frontier Monotonicity**: At all times, every node $u \in C_t$ satisfies $d(q, u) \le \max_{v \in W_t} d(q, v)$.
2. **Tag Invariance**: Every evaluated vertex is marked in `visited`. Increasing the capacity ceiling of $W$ from $ef_{init}$ to $ef_{expanded}$ relaxes the pruning criterion $\max_{v \in W} d(q, v)$, allowing valid unexpanded candidates in $C$ to be explored.
3. **No Redundant Computations**: Because `visited` tags are preserved, no node is evaluated twice:
   $$\text{Total Distances} = \text{Distances}(ef_{init}) + \text{Distances}(ef_{expanded} \setminus ef_{init})$$
   Redundant work is strictly zero.

---

## 3. Multi-Signal Query Difficulty Estimation

### 3.1 Feature Vector Specification
Rather than relying on uninformative global entry distance, the estimator extracts orthogonal geometric features during traversal:

| Signal | Feature Name | Mathematical Definition | Spearman Rank Correlation ($\rho$) | Physical Meaning |
|---|---|---|---:|---|
| $s_1$ | **Candidate Improvement Velocity** | $\frac{\Delta d_{furthest}}{d_{best} \cdot \Delta t}$ | **$-0.4709$** | Rate at which candidate beam is advancing at $ef_{init}$ boundary. |
| $s_2$ | **Estimated Local Intrinsic Dim (LID)** | $\left(\frac{1}{k-1} \sum_{i=1}^{k-1} \ln \frac{d_k}{d_i}\right)^{-1}$ | **$+0.3277$** | Local topological dimensionality of candidate neighborhood. |
| $s_3$ | **Layer-0 Entry Distance** | $d(q, ep_{L0})$ | **$+0.3135$** | Proximity to target cluster after upper-layer descent. |
| $s_4$ | **Top-k Distance Margin** | $\frac{d_k - d_1}{d_k + \epsilon}$ | **$-0.2923$** | Relative clustering tightness (small margin = boundary ambiguity). |
| $s_5$ | **Layer-0 Entry Node Out-Degree** | $\text{deg}(ep_{L0})$ | $-0.0669$ | Local hubness of entry candidate. |

### 3.2 Decision Rules
The decision to expand from $ef_{init}$ to $ef_{expanded}$ is computed in $O(1)$ arithmetic operations ($< 0.1 \mu s$):
$$\text{Expand} \iff (s_4 < \theta_{margin} \land s_2 > \theta_{lid}) \lor (s_1 > \theta_{imp}) \lor (s_3 > \theta_{dist} \land s_4 < 1.5 \theta_{margin})$$
- Easy queries in distinct, low-dimensional clusters terminate early at $ef_{init}$, saving $15\% - 20\%$ of distance computations.
- Hard queries near manifold boundaries automatically trigger resumption to $ef_{expanded}$.

---

## 4. Bounded Incremental Reference Monitoring

### 4.1 Bounded Cost Principle
To eliminate throughput collapse, monitoring work is strictly capped to $< 5.0\%$ of CPU budget:
- **Sampling Strategy**: Exactly 1 offline reference query is evaluated every $K = 25$ production queries.
- **Overhead Guarantee**:
  $$\text{Overhead}_{CPU} = \frac{1 \text{ ref query}}{25 \text{ prod queries}} = 4.0\% \text{ additional search work}$$
- **Rolling Accuracy Window**: The monitor maintains a rolling FIFO ring buffer of the last 100 evaluated reference queries, computing moving average recall:
  $$\bar{R}_{ref}(t) = \frac{1}{100} \sum_{i=0}^{99} \text{Recall}(q_{t-i}^{ref})$$
- **Smoothness**: Eliminates the latency spikes and CPU stalls associated with batch evaluation.

---

## 5. Workload-Level Adaptation and Safety Guarantees

### 5.1 Hysteresis and Deadband Control
The slower workload controller updates every $N_{eval} = 50$ production queries using the rolling accuracy estimate $\bar{R}_{ref}$:
- **Target Accuracy**: $R_{target}$ (e.g. $95.0\%$).
- **Deadband**: $[R_{target} - 0.01, R_{target} + 0.02]$. Zero adjustments occur within this window, preventing parameter flutter.
- **Cooldown**: $N_{cooldown} = 100$ queries must elapse between adaptations.

### 5.2 Deterministic Safety Rollback
If $\bar{R}_{ref} < R_{target} - 0.02$:
1. Immediately restore stable configuration: $ef_{init} \leftarrow ef_{init}^{stable}, ef_{expanded} \leftarrow ef_{expanded}^{stable}$.
2. Enforce double cooldown: $N_{cooldown} \leftarrow 200$ queries.
3. Record safety event in telemetry.

### 5.3 Memory and Concurrency Guarantees
- **Zero Runtime Allocations**: $\Delta \text{RAM}_{runtime} = 0 \text{ bytes}$.
- **Thread Safety**: Resumable search operates exclusively on thread-local heap queues and visited tag pools.
