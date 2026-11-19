"""
AdaptiveVec Production Engine Interface.
Supports Automatic Resource-Constrained Calibration (Phase 3).
"""

import numpy as np
from typing import Optional, Union, List, Tuple

try:
    import _adaptivevec_cpp as _cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

class AdaptiveVec:
    """
    AdaptiveVec Vector Retrieval Index.
    
    Provides high-performance approximate nearest neighbor retrieval with optional
    automatic resource-constrained calibration across memory, recall, and tail latency budgets.
    """
    def __init__(
        self,
        dimension: int,
        target_recall: float = 0.95,
        memory_budget_mb: float = 1024.0,
        target_p99_latency_ms: float = 5.0,
        space: str = "l2",
        auto_optimize: bool = True,
        max_calibration_duration_s: float = 60.0,
        optimization_goal: str = "MaxThroughput",
        threads: int = 0
    ):
        self.dimension = dimension
        self.target_recall = target_recall
        self.memory_budget_mb = memory_budget_mb
        self.target_p99_latency_ms = target_p99_latency_ms
        self.space_str = space.lower()
        self.auto_optimize = auto_optimize
        self.max_calibration_duration_s = max_calibration_duration_s
        self.optimization_goal = optimization_goal
        self.threads = threads
        
        self.native_index = None
        self.calibration_report = None
        self.calibrated_ef_search = 64
        self.is_built = False
        self._built_n_vectors = 0

    def _get_cpp_space(self):
        if not HAS_CPP:
            raise RuntimeError("AdaptiveVec native C++ extension (_adaptivevec_cpp) is not available.")
        if self.space_str in ("cosine", "angular"):
            return _cpp.SpaceType.Cosine
        return _cpp.SpaceType.L2

    def calibrate(
        self,
        vectors: np.ndarray,
        calibration_queries: np.ndarray,
        ground_truth: np.ndarray,
        dataset_name: str = "UserDataset"
    ):
        """
        Calibrate index hyperparameters against resource constraints.
        
        Evaluates precision feasibility, bounded graph construction, and query parameter sweeps
        with non-parametric 95% bootstrap confidence intervals.
        Pre-constructs the winning index instance to eliminate double-build overhead.
        """
        if not HAS_CPP:
            raise RuntimeError("Native C++ extension required for calibration.")
            
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        calibration_queries = np.ascontiguousarray(calibration_queries, dtype=np.float32)
        ground_truth = np.ascontiguousarray(ground_truth, dtype=np.int32)
        
        constraints = _cpp.CalibrationConstraints()
        constraints.target_recall = float(self.target_recall)
        constraints.max_memory_mb = float(self.memory_budget_mb)
        constraints.target_p99_latency_ms = float(self.target_p99_latency_ms)
        constraints.max_calibration_duration_s = float(self.max_calibration_duration_s)
        constraints.threads = int(self.threads)
        
        if self.optimization_goal in ("MinMemory", "min_memory"):
            constraints.optimization_goal = _cpp.OptimizationGoal.MinMemory
        elif self.optimization_goal in ("MinLatency", "min_latency"):
            constraints.optimization_goal = _cpp.OptimizationGoal.MinLatency
        elif self.optimization_goal in ("MaxRecall", "max_recall"):
            constraints.optimization_goal = _cpp.OptimizationGoal.MaxRecall
        else:
            constraints.optimization_goal = _cpp.OptimizationGoal.MaxThroughput
            
        report, calibrated_idx = _cpp.calibrate_index(
            vectors,
            self._get_cpp_space(),
            calibration_queries,
            ground_truth,
            constraints,
            dataset_name
        )
        
        self.calibration_report = report
        self.native_index = calibrated_idx
        if report and report.selected_config:
            self.calibrated_ef_search = report.selected_config.ef_search
            
        self.is_built = True
        self._built_n_vectors = vectors.shape[0]
        return report

    def build(self, vectors: np.ndarray):
        """
        Build the index. If calibrate() has already been called on the same vector corpus,
        this operation re-uses the pre-constructed index directly without redundant building.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n_vectors, dim = vectors.shape
        if dim != self.dimension:
            raise ValueError(f"Vector dimension {dim} does not match index dimension {self.dimension}")
            
        if self.is_built and self.native_index is not None and self._built_n_vectors == n_vectors:
            # Already built during calibration; avoid redundant index construction
            return
            
        if not HAS_CPP:
            raise RuntimeError("Native C++ extension required.")
            
        cfg = _cpp.PolicyConfig()
        if self.calibration_report and self.calibration_report.selected_config:
            sel = self.calibration_report.selected_config
            cfg.m_base = sel.m
            cfg.ef_c_base = sel.ef_construction
            cfg.enable_ada_ef = sel.enable_ada_ef
            precision = sel.precision
        else:
            cfg.m_base = 16
            cfg.ef_c_base = 64
            precision = "FP32"
            
        self.native_index = _cpp.AdaptiveHNSWIndex(dim, self._get_cpp_space(), False, cfg)
        self.native_index.batch_insert(vectors, self.threads)
        
        if precision == "SQ8":
            self.native_index.policy.keep_raw_vectors = False
            self.native_index.allocate_mixed_precision(_cpp.AllocationStrategy.UniformSQ8, 0.25)
        elif precision == "SQ4":
            self.native_index.policy.keep_raw_vectors = False
            self.native_index.allocate_mixed_precision(_cpp.AllocationStrategy.UniformSQ4, 0.125)
            
        self.is_built = True
        self._built_n_vectors = n_vectors

    def search(self, query: np.ndarray, k: int = 10, ef: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Search for top-k nearest neighbors.
        Uses calibrated efSearch if ef is not explicitly provided.
        """
        if not self.is_built or self.native_index is None:
            raise RuntimeError("Index has not been built. Call calibrate() or build() first.")
            
        query = np.ascontiguousarray(query, dtype=np.float32)
        effective_ef = ef if ef is not None else self.calibrated_ef_search
        
        if query.ndim == 1:
            candidates = self.native_index.search(query, k, effective_ef)
            return [(c.id, c.dist) for c in candidates]
        elif query.ndim == 2:
            batch_res = self.native_index.batch_search(query, k, effective_ef, self.threads)
            return [[(c.id, c.dist) for c in row] for row in batch_res]
        else:
            raise ValueError("Query must be 1D or 2D array.")
            
    def get_memory_breakdown(self):
        """Return memory breakdown of the active native index."""
        if not self.is_built or self.native_index is None:
            return None
        return self.native_index.get_memory_breakdown()
