# Historical Architectural Regression Audit across Phases 0–4.3

This audit examines every major architectural modification introduced into AdaptiveVec from the original research submission (`v1.0.0-submission` / commit `9c22f4b`) through Phase 4.3 (`51b7f46`), evaluating its theoretical intent, empirical impact, correctness consequences, and final governance disposition for Phase 5.0.

---

## 1. Executive Summary of Architectural Evolution

The development trajectory of AdaptiveVec diverged into two distinct phenomena:
1. **Low-Level Systems Optimization (Highly Beneficial)**:
   - Replacing `std::unordered_set` with generation-tagged flat arrays (`VisitedTagPool`).
   - Introducing AVX2/FMA vectorization with squared Euclidean metric traversal.
   - Eliminating atomic distance counters from the inner distance loop to prevent cache-line bouncing.
   - Reorganizing memory buffers for 64-byte alignment and contiguous access.
   These modifications accelerated single-thread and multi-thread search throughput by $3.5\times$ to $4.2\times$ without altering graph topology.
2. **Algorithmic Divergence and Online Control Drift (Detrimental)**:
   - In Phases 1–2, the clean in-degree hubness and uniform quantization were replaced with complex bridge-importance scores and mixed-precision pools, which caused severe numerical regressions (Transformer recall collapsed to 18%).
   - In Phases 4.0–4.3, rather than focusing on graph topology, development shifted entirely to online runtime controllers, dynamic difficulty estimation, reference monitors, and uncertainty predictors in the search path.
   - In Phase 4.3, rigorous instrumentation revealed that calculating online difficulty signals in the query hot path ($321\ \mu\text{s}$ per query on SIFT-100K) cost $20\times$ more execution time than the distance evaluations it avoided.

---

## 2. Comprehensive Architectural Change Matrix

| Change ID | Phase & Commit | Original Mechanism | Modified Mechanism | Engineering Motivation | Expected Impact | Observed Empirical Reality | Correctness & Invariants | Disposition in Phase 5.0 |
|:---:|:---:|:---|:---|:---|:---|:---|:---|:---:|
| **CHG-01** | Phase 0<br>`50ac107` | Unseeded RNG in `generate_random_level()` | Explicitly seeded thread-local PRNG (`std::mt19937 gen(1337)`) | Guarantee deterministic graph construction across repeated runs | Reproducible layer distribution and edge topology | Verified bitwise reproducibility; zero regression | Fixed non-deterministic height distribution | **RETAINED** |
| **CHG-02** | Phase 0.5<br>`414f105` | Heap-allocated `std::unordered_set<tableint>` in `search_layer` | Generation-tagged `VisitedTagPool` (flat array with incrementing epoch) | Eliminate millions of dynamic heap allocations and hash overhead | $+200\%$ to $+300\%$ search throughput | Search QPS increased from $\sim 5,000$ to $>20,000$ on SIFT-100K | Preserves exact visited invariant with zero leaks | **RETAINED** |
| **CHG-03** | Phase 0.5<br>`414f105` | Monotonic Euclidean metric evaluation with `sqrt()` | Squared Euclidean distance ($L_2^2$) in graph beam search | Eliminate expensive square root instructions on monotonic priority queues | $+15\%$ to $+25\%$ AVX2 SIMD throughput | Distance kernel latency reduced from $0.42\ \mu\text{s}$ to $0.25\ \mu\text{s}$ | Mathematically monotonic for $L_2$; preserves candidate order | **RETAINED** |
| **CHG-04** | Phase 1<br>`2fec2cb` | Simple in-degree centrality penalty: $1 + \mu \frac{\text{deg}_{\text{in}}}{\bar{\text{deg}}}$ | Bridge-importance score tracking ($b(u) \in [0, 1]$) with hubness modulation | Identify structural topological bridges and prevent severance | Improved reachability and recall in sparse graphs | Construction time increased by $45\%$; bridge scores saturated at $\sim 0.90$; no throughput benefit over simple in-degree | Saturated scores caused uniform penalties; complex graph state synchronization | **REVERTED**<br>(Restore original in-degree penalty) |
| **CHG-05** | Phase 2<br>`e9f4a0c` | Uniform SQ8 or SQ4 quantization with FP32 reranking | Geometry-aware mixed-precision pool (FP32, SQ8, SQ4 mixed across vertices) | Allocate high precision to high-LID nodes and low precision to low-LID nodes | $+40\%$ memory reduction at zero recall loss | Severe numerical regression: Transformer recall collapsed from $93\%$ to $18\%$; SIFT SQ8 collapsed to $60.4\%$ | Scale mismatch, variable byte stride, and memory misalignment | **REVERTED**<br>(Restore clean uniform SQ8) |
| **CHG-06** | Phase 2.1<br>`c87de4b` | Flawed mixed-precision kernels from Phase 2 | Clean uniform SQ8 with per-dimension affine calibration and FP32 reranking | Restore numerical correctness and baseline retrieval fidelity | Recover Recall@10 to $>97\%$ on SIFT, $>89\%$ on Transformer | Full recall recovery: SIFT SQ8 achieved $97.9\%$ recall; Transformer recovered to $89.4\%$ | Validated 4-way decomposition: exact parity with float ground truth | **RETAINED** |
| **CHG-07** | Phase 3<br>`acc0fb4` | Fixed static hyperparameters ($M=16, efC=120$) | Automatic resource-constrained calibration (grid search + bootstrap recall) | Automatically discover optimal static parameters for target hardware/SLA | Predictable latency-memory Pareto frontier | Provided strong static baseline; demonstrated that calibrated static HNSW is hard to beat | Safe offline search; zero runtime overhead | **RETAINED**<br>(As optional tool) |
| **CHG-08** | Phase 4.0<br>`5b8a766` | Static graph search with fixed $efSearch$ | Continuous workload-aware runtime controller (PID-like feedback loop) | Dynamically adapt candidate beam width to incoming query rate and recall drift | Adapt to fluctuating production workloads | Introduced thread synchronization overhead, telemetry latency spikes, and unpredictable p99 | Feedback loop oscillations under bursty traffic | **ARCHIVED**<br>(Disabled by default) |
| **CHG-09** | Phase 4.1<br>`66a0190` | Background workload telemetry | In-path reference query monitoring | Measure live query difficulty and recall in real-time | Real-time calibration to workload shifts | Reference queries consumed up to $18\%$ of worker thread cycles, severely depressing QPS | Degraded query throughput; noisy difficulty estimation | **ARCHIVED**<br>(Permanently removed) |
| **CHG-10** | Phase 4.2<br>`2962679` | Single-pass beam search | Resumable search state machine ($ef=32 \to 64$) | Expand candidate beam only for difficult queries | $+15\%$ throughput at matched recall | Resuming and priority queue re-insertion added high CPU cache-shuffling overhead | Failed recall requirements on non-synthetic distributions | **ARCHIVED**<br>(Permanently removed) |
| **CHG-11** | Phase 4.3<br>`51b7f46` | Fixed stagnation early exit ($p=6, \epsilon=10^{-4}$) | Online uncertainty-aware termination (LID + margin + velocity) | Achieve $\ge 10\%$ fewer distance evaluations than fixed stagnation | Lower distance evaluation count | Evaluator overhead ($321\ \mu\text{s}$) overwhelmed distance savings ($15\ \mu\text{s}$), degrading throughput by $-31.75\%$ on SIFT | False early exits on boundary queries; violation of acceptance criteria | **ARCHIVED**<br>(Enforced Stop Rule) |
| **CHG-12** | Phase 4.3<br>`51b7f46` | Global `std::atomic<uint64_t>` distance counter in distance functions | Thread-local stack-allocated `QuerySearchStats` structure | Eliminate cache-line bouncing and false sharing across OpenMP threads | $+15\%$ to $+25\%$ multi-threaded throughput | Verified: multi-threaded throughput increased by $+21.4\%$; exact per-stage distance tracking | Completely thread-safe with zero atomic locking | **RETAINED** |

---

## 3. Detailed Algorithmic & Topological Diagnostic

### 3.1 Graph Degree Distribution & Topology Regressions
- **Original Behavior (v1.0.0)**: On SIFT-100K, dynamic $M(x)$ allocated degrees ranging from 8 to 24 with a mean degree of $\approx 15.17$. Base layer edges totaled $2,509,138$ ($-7.4\%$ vs canonical HNSW's $2,709,125$).
- **Regressions in Phases 1–4**: In Phase 1, bridge-aware penalties distorted edge selection, causing nodes to reject high-quality nearest neighbors in favor of artificial bridges. In Phases 4.0–4.3, graph construction was left completely static while all focus moved to search controllers.
- **Phase 5.0 Resolution**: Restore the original $M(x) \propto z_{\text{LID}} + z_D$ allocation formula and verify its topological connectivity, clustering coefficients, and reachability.

### 3.2 Distance Evaluation Counter Discrepancy
- **Audit Findings**:
  - In Phase 4.1 (`phase4_1_workload_comparisons.cpp:305`), distance evaluations were hardcoded synthetic constants (`fixed_evals = 280.0`).
  - In Phase 4.2 (`phase4_2_matched_recall_benchmarks.cpp:214, 224`), distance evaluations were estimated as $6 \times efSearch$.
  - True measured evaluations are $2.33\times$ to $4.36\times$ higher ($1,073.3$ vs $384.0$ at $ef=64$).
- **Phase 5.0 Invariant**: All Phase 5.0 benchmarks record live, instrumented distance computations via `QuerySearchStats`. Synthetic formulas are strictly banned.

### 3.3 Transformer Dataset Discrepancy
- **Audit Findings**:
  - The original research monograph (Table~\ref{tab:transformer_384d}, Section 4.5) evaluated **MiniLM-50K-MultiDomain**, containing **57,893 real passage vectors** across 5 domains (SciFact, NFCorpus, ArguAna, Webmasters, Android).
  - In later phases, experiments inadvertently substituted a 100,000-vector synthetic embedding set, creating inexplicable recall and QPS divergence.
- **Phase 5.0 Resolution**: The 57,893-vector corpus (`data/transformer_384d/transformer_384d_base.fvecs`) has been verified and fingerprinted. All Phase 5.0 evaluations use the authentic original corpus with exact brute-force ground truth.
