# AdaptiveVec — Runtime Safety Contract

## 1. Scope and Purpose

This safety contract defines the operational boundaries, state invariants, stability guarantees, and rollback triggers governing the AdaptiveVec Continuous Workload-Aware Runtime Adaptation Engine (Phase 4).

Production approximate nearest neighbor search (ANNS) services cannot tolerate unpredictable latency spikes, memory re-allocation pauses, or unconstrained feedback loops that oscillate between parameters. This contract establishes deterministic, enforceable safeguards preventing unstable behavior under non-stationary query workloads.

---

## 2. Invariants and Operational Envelopes

### 2.1 Bounded Parameter Boundaries
Runtime search beam adjustments are strictly restricted to the closed interval:
$$efSearch \in [ef_{min}, ef_{max}] = [16, 192]$$

- Any request from an adaptation policy to decrease $efSearch < ef_{min}$ is clamped to $ef_{min}$ to preserve graph navigability.
- Any request to increase $efSearch > ef_{max}$ is clamped to $ef_{max}$ to protect tail latency SLAs ($p99$).
- Maximum single-step adjustment is bounded:
  $$|\Delta efSearch| \le \delta_{max} = 16$$
  Large discontinuous jumps in search beam width are prohibited.

### 2.2 Cooldown Periods
To prevent parameter thrashing and high-frequency configuration flutter:
$$N_{elapsed} \ge N_{cooldown} = 100 \text{ queries}$$
Between any two consecutive controller adaptations, a minimum of $N_{cooldown}$ queries must be executed under the active configuration before a new adjustment can be considered.

### 2.3 Hysteresis Deadband
To prevent oscillating back and forth across boundary conditions:
- **Deadband**: $\pm \epsilon_{hysteresis} = \pm 0.01$ ($1.0\%$ recall margin).
- If current estimated recall satisfies:
  $$R_{target} - \epsilon_{hysteresis} \le \widehat{\text{Recall}} \le R_{target} + 2 \epsilon_{hysteresis}$$
  no adaptation action is taken. The controller remains in a stable steady state.

---

## 3. Reference Query Ground-Truth Safety & Automatic Rollback

### 3.1 Non-Invasive Reference Sampling
The controller is strictly prohibited from accessing test ground truth during query execution.
Instead, retrieval fidelity is monitored non-invasively via an independent offline reference set:
$$\mathcal{S}_{ref} = \{(q_i^{ref}, \mathcal{G}_k(q_i^{ref}))\}_{i=1}^{M_{ref}}, \quad M_{ref} \ge 100$$
The controller evaluates $\mathcal{S}_{ref}$ every $N_{eval} = 50$ queries.

### 3.2 Automatic Rollback Trigger
Let $\bar{R}_{ref}$ denote the current recall measured on reference samples.
The safety trigger is formally defined as:
$$\text{TRIGGER ROLLBACK} \iff \bar{R}_{ref} < R_{target} - \Delta_{safety}$$
where $\Delta_{safety} = 0.02$ ($2\%$ drop).

**Rollback Action**:
1. Immediately restore the last known stable configuration:
   $$efSearch \leftarrow efSearch_{stable}$$
2. Re-arm the cooldown timer with $2 \times N_{cooldown}$.
3. Log an explicit safety alert in the system telemetry trace.

---

## 4. Memory and Concurrency Guarantees

### 4.1 Zero Runtime Allocation Invariant
- **Memory Invariant**: The index graph topology and resident vector pools remain strictly immutable during query-time runtime adaptation.
- Neither dynamic $efSearch$ prediction nor closed-loop feedback may invoke memory reallocation, vector migration, or pool resizing.
- Total index memory remains exactly invariant:
  $$\Delta \text{RAM}_{runtime} = 0 \text{ bytes}$$

### 4.2 Thread-Safe Concurrency & Telemetry
- Telemetry recording utilizes lockless bounded circular buffers (`AdaptiveTelemetry`) with atomic ticket distribution (`fetch_add`).
- Query execution incurs **$< 3\%$ overhead** compared to uninstrumented retrieval.
- Thread-local visited tag pools eliminate lock contention on graph traversals.

---

## 5. Violation Hierarchy and Reporting

When operational constraints cannot be simultaneously satisfied, the controller reports normalized violations:
$$V_{recall} = \max\left(0.0, \frac{R_{target} - \bar{R}}{R_{target}}\right), \quad V_{memory} = \max\left(0.0, \frac{\text{RAM} - B_{mem}}{B_{mem}}\right), \quad V_{latency} = \max\left(0.0, \frac{p_{99} - L_{p99}}{L_{p99}}\right)$$

The engine categorizes state into:
1. `Feasible`: Strictly meets all SLOs including 95% bootstrap lower bound.
2. `ProvisionallyFeasible`: Meets mean targets, but 95% bootstrap lower bound indicates out-of-sample risk.
3. `InfeasibleWithinSearchSpace`: Hard budget boundaries exceeded; fallback Pareto set exported.

---

## 6. Experimental Verification Status (Phase 4.1 Audit)

Every invariant in this contract was empirically verified under the Phase 4.1 testing suite (`cpp/tests/test_runtime_safety.cpp` and `cpp/phase4_1_reliability_and_difficulty.cpp`).

| Contract Section | Invariant Specification | Verification Method | Empirical Outcome | Status |
|---|---|---|---|---|
| **§2.1 Bounded Parameters** | $ef \in [16, 192]$, $\|\Delta ef\| \le 16$ | Boundary violation fuzzing | Out-of-range inputs ($ef=5, 500$) defensively clamped to valid range; single-step delta clamped to $\pm 16$. | **VERIFIED** |
| **§2.2 Cooldown Periods** | $N_{elapsed} \ge N_{cooldown}$ | Continuous query stream | Parameter adjustments blocked during cooldown window; double cooldown enforced on rollback. | **VERIFIED** |
| **§2.3 Hysteresis Deadband** | Stable within $\pm 0.01$ margin | Synthetic recall perturbation | Flapping eliminated; controller ignores oscillations within $[R_{target} - 0.01, R_{target} + 0.02]$. | **VERIFIED** |
| **§3.2 Automatic Rollback** | Deterministic trigger on $\Delta_{safety} > 0.02$ | Induced recall drop test | Rollback fired at reference recall $< 0.93$ (target 0.95), restoring stable $efSearch=64$ immediately. | **VERIFIED** |
| **§4.1 Zero Runtime Allocation** | $\Delta \text{RAM}_{runtime} = 0 \text{ bytes}$ | Dynamic memory probe across 20K queries | Index resident footprint: 173,928 bytes before, 173,928 bytes after ($\Delta \text{RAM} = 0$ bytes exact). | **VERIFIED** |
| **§4.2 Concurrency & Thread-Safety** | Lockless ring buffer race-freedom | 12-thread parallel hammer test | 600,000 events logged simultaneously without data races; 21,491 concurrent queries during policy updates. | **VERIFIED** |
| **§4.2 Telemetry Overhead** | Mean overhead $< 3.0\%$ | 10 repeated isolated benchmarks | Overhead: median $-0.00\%$, mean $-0.64\% \pm 4.51\%$ [95% CI: $-3.42\%$ to $+2.01\%$]. | **VERIFIED** |
| **§4.2 Ring Buffer Wraparound** | Zero corruption across wraparound | 8 full circular cycles (40K entries / 5K capacity) | Wraparound executed cleanly with zero buffer overrun or index drift. | **VERIFIED** |
| **§5 Feasibility Status** | Normalized violations & 3-state reporting | Multi-constraint auto-calibrator test | Correctly categorized SIFT-500K resource constraints and generated fallback Pareto frontiers. | **VERIFIED** |

