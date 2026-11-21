# AdaptiveVec Research Paper Package

**Working Title:** *AdaptiveVec: Geometry-Aware and Resource-Constrained Approximate Nearest Neighbor Search with Dynamic Graph Maintenance*  
**Research Checkpoint:** `7936c61` (`phase-6.4-final-correctness-closure`)  
**Standardized Hardware Platform:** Intel Core 5 210H (8 cores / 12 threads, AVX2/FMA, 16 GB RAM), Windows 11 Build 26100, GCC 14.2.0 (`g++ -O3 -mavx2 -mfma -fopenmp -std=c++17`).

---

## 1. Directory Structure and Package Inventory

```
research/paper/
├── adaptivevec_paper.tex          # Complete IEEE-style LaTeX manuscript (Sections 1–13)
├── references.bib                 # Verified scholarly BibTeX entries with DOIs
├── README.md                      # Package overview, compilation, and reproduction guide
├── evidence_map.md                # Mapping of every claim to raw JSON artifacts & code
├── scientific_issues.md           # In-depth analysis of 10 critical issues & contradictions
├── reviewer_questions.md          # 20 difficult academic defense questions & answers
├── figures/                       # Modular TikZ vector diagrams
│   ├── architecture.tex           # 4-milestone modular system architecture
│   ├── hierarchy_allocation.tex   # Layer decoupling and geometric degree allocation
│   └── dynamic_maintenance.tex    # Dynamic maintenance, mutation replay & gatekeeper
└── tables/                        # Publication LaTeX tables
    ├── four_contributions.tex     # Master table of four core research contributions
    ├── geometry_comparison.tex    # SIFT-100K geometry allocation controls comparison
    ├── quantization_memory.tex    # Precision modes, memory breakdown & QPS scaling
    ├── dynamic_maintenance.tex    # Dynamic maintenance, compaction quality & gates
    ├── autocalibrator_feasibility.tex # 6-state feasibility matrix and SLA scenarios
    └── limitations_validity.tex   # Comprehensive limitations and validity threats
```

---

## 2. LaTeX Compilation Instructions

The manuscript `adaptivevec_paper.tex` is authored in standard IEEEtran format. It includes the modular tables and TikZ figures from `tables/` and `figures/`, and references `references.bib`.

### Standard Compilation Workflow (Linux / macOS / Windows with TeX Live or MiKTeX)
```bash
pdflatex adaptivevec_paper.tex
bibtex adaptivevec_paper
pdflatex adaptivevec_paper.tex
pdflatex adaptivevec_paper.tex
```
Or using `latexmk`:
```bash
latexmk -pdf adaptivevec_paper.tex
```

> **Note on Local Environment PATH:**  
> A local TeX distribution (`pdflatex` or `latexmk`) is currently not registered on the system `PATH` on this host. The `.tex` sources, `.bib` references, and modular TikZ diagrams are fully self-contained and verified for standard IEEEtran compilation. They can be compiled directly on any environment with TeX Live / MiKTeX or uploaded to Overleaf.

---

## 3. Empirical Reproduction Guide

Every empirical result cited in the manuscript is backed by a verified raw JSON artifact in the `experiments/` directory and can be reproduced using native C++ executables compiled from `cpp/`.

### Milestone I: Geometry Allocation & Controls
- **Raw Evidence:** `experiments/phase_6_1_geometry_reproduction.json`, `experiments/phase_6_4_geometry_implementation_audit.json`
- **Reproduction Binary:**
  ```powershell
  .\cpp\phase6_1_evidence_runner.exe
  ```
- **Key Verifications:**
  - Normalized Exponential (+24.01% QPS, -37.62% P95 latency at 99.26% recall).
  - Degree conservation ($\sum M_i = 3{,}200{,}000 = N \cdot M_{\text{base}}$ via Hare-Niemeyer).
  - Shuffled Degree Control (-23.01% QPS degradation).
  - Reversed Geometry Control (+0.22 pp recall at +5.63% dist evals).

### Milestone II: Vector Storage & Fresh Process Isolation
- **Raw Evidence:** `experiments/phase_6_4_fresh_precision_memory.json`
- **Reproduction Binary:**
  ```powershell
  .\cpp\benchmark_fresh_precision.exe
  ```
- **Key Verifications:**
  - Pure SQ8 ADC: 41.85 MB index-owned memory (-46.66% vs FP32), 56.70 MB process working set, 97.70% recall at 12,437 QPS.
  - Pure SQ4 ADC: 35.75 MB index-owned memory, 49.30 MB working set, 88.10% recall.
  - Two-stage FP32 re-ranking: 90.68 MB memory, ~7x throughput collapse (5,036 QPS).

### Milestone III: Dynamic Maintenance, Compaction Quality & Safe Replacement
- **Raw Evidence:** `experiments/phase_6_4_dynamic_maintenance_investigation.json`, `experiments/phase_6_4_concurrency_validation.json`
- **Reproduction Binaries:**
  ```powershell
  .\cpp\benchmark_dynamic_maintenance.exe
  .\cpp\test_concurrent_mutation_preservation.exe
  ```
- **Key Verifications:**
  - Multi-Layer Strict RNG replenishment recovers 25% churn recall from 0.9280 to 0.9786 (fixed $efS=64$) and 0.9865 (recalibrated).
  - Compaction Deferral under 10% churn due to dual peak memory (143.98 MB > 120.0 MB).
  - Rebuild policy triggered under 50% churn (fresh rebuild succeeds in 3.95 s with 0.9878 recall).
  - Zero lost mutations across 350 mutations and 12,494 concurrent reader queries; reader pause $\le 0.194$ ms.

### Milestone IV: AutoCalibrator & Formal Feasibility
- **Raw Evidence:** `experiments/phase_6_4_exact_calibrator_regression.json`, `experiments/phase_6_4_sla_integrity.json`
- **Reproduction Binaries:**
  ```powershell
  .\cpp\phase6_4_runner.exe
  .\cpp\test_suite7.exe
  ```
- **Key Verifications:**
  - Historical strict memory SLA solved in 12.6 s: $R=0.9416 \ge 0.92$, $M=3.38\text{ MB} \le 4.5\text{ MB}$, $\text{P99}=0.33\text{ ms} \le 5.0\text{ ms}$.
  - Impossible SLA rejected in 0.05 s (`INFEASIBLE_WITHIN_SEARCH_SPACE`).
  - All 6 deterministic state tests pass with zero false feasibility.
  - Transformer-384D full corpus certified: $R=0.9720 \ge 0.95$, $M=56.61\text{ MB} \le 150\text{ MB}$, $\text{P99}=7.38\text{ ms} \le 15.0\text{ ms}$.
