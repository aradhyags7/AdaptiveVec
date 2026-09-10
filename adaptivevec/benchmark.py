"""
Comprehensive Benchmarking Harness for AdaptiveVec vs. Stock HNSW.
Measures Recall@K, Build Time, QPS, Latency (p50/p95/p99), Index Memory, and Edge Reduction.
"""

import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from .hnsw_base import StockHNSW
from .adaptive_hnsw import AdaptiveHNSW
from .policy import AdaptivePolicyConfig
from .datasets import compute_ground_truth

@dataclass
class BenchmarkResult:
    index_name: str
    dataset_name: str
    n_samples: int
    dim: int
    build_time_s: float
    total_edges: int
    avg_edges_per_node: float
    index_memory_mb: float
    memory_savings_pct: float
    recall_1: float
    recall_10: float
    recall_100: float
    qps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    avg_dist_comps_per_query: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BenchmarkHarness:
    """Automated benchmark runner."""
    
    def __init__(self, data: np.ndarray, queries: np.ndarray, space: str = "l2"):
        self.data = data
        self.queries = queries
        self.space = space
        self.dim = data.shape[1]
        self.n_samples = len(data)
        self.n_queries = len(queries)
        
        # Pre-compute exact ground truth
        self.gt_100_idx, self.gt_100_dist = compute_ground_truth(self.data, self.queries, k=100, space=self.space)

    def _calculate_recall(self, results: List[List[Tuple[float, int]]], k: int) -> float:
        """Calculates exact Recall@K against precomputed ground truth."""
        total_found = 0
        total_possible = self.n_queries * k
        
        for q_idx, res in enumerate(results):
            found_nodes = set(node for _, node in res[:k])
            gt_nodes = set(self.gt_100_idx[q_idx, :k])
            total_found += len(found_nodes.intersection(gt_nodes))
            
        return float(total_found / total_possible)

    def evaluate_stock_hnsw(
        self,
        m: int = 16,
        ef_construction: int = 150,
        ef_search: int = 50,
        k: int = 10
    ) -> BenchmarkResult:
        """Evaluates Stock HNSW baseline."""
        index = StockHNSW(dim=self.dim, m=m, ef_construction=ef_construction, space=self.space)
        
        # 1. Build Index
        start_build = time.perf_counter()
        for vec in self.data:
            index.insert(vec)
        build_time = time.perf_counter() - start_build
        
        stats = index.get_stats()
        
        # 2. Query Index
        latencies = []
        all_results = []
        init_dist_comps = index.total_dist_computations
        
        for q in self.queries:
            t0 = time.perf_counter()
            res = index.search(q, k=100, ef=ef_search)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
            all_results.append(res)
            
        total_query_time_s = sum(latencies) / 1000.0
        qps = float(self.n_queries / total_query_time_s) if total_query_time_s > 0 else 0.0
        dist_comps_per_q = (index.total_dist_computations - init_dist_comps) / self.n_queries
        
        recall_1 = self._calculate_recall(all_results, 1)
        recall_10 = self._calculate_recall(all_results, 10)
        recall_100 = self._calculate_recall(all_results, 100)
        
        return BenchmarkResult(
            index_name=f"Stock HNSW (M={m}, efC={ef_construction})",
            dataset_name="Vector Dataset",
            n_samples=self.n_samples,
            dim=self.dim,
            build_time_s=round(build_time, 4),
            total_edges=stats["total_edges"],
            avg_edges_per_node=round(stats["avg_edges_per_node"], 2),
            index_memory_mb=round(stats["total_memory_mb"], 3),
            memory_savings_pct=0.0,
            recall_1=round(recall_1, 4),
            recall_10=round(recall_10, 4),
            recall_100=round(recall_100, 4),
            qps=round(qps, 2),
            latency_p50_ms=round(float(np.percentile(latencies, 50)), 3),
            latency_p95_ms=round(float(np.percentile(latencies, 95)), 3),
            latency_p99_ms=round(float(np.percentile(latencies, 99)), 3),
            avg_dist_comps_per_query=round(dist_comps_per_q, 1),
            details=stats
        )

    def evaluate_adaptive_hnsw(
        self,
        policy_config: Optional[AdaptivePolicyConfig] = None,
        ef_search: int = 50,
        baseline_edges: Optional[int] = None,
        policy_name: str = "AdaptiveVec (Combined)"
    ) -> BenchmarkResult:
        """Evaluates AdaptiveVec with given policy configuration."""
        config = policy_config or AdaptivePolicyConfig()
        index = AdaptiveHNSW(dim=self.dim, policy_config=config, space=self.space)
        
        # Calibrate baseline statistics with random sample
        sample_subset = self.data[:min(500, len(self.data))]
        index.calibrate(sample_subset)
        
        # 1. Build Index
        start_build = time.perf_counter()
        for vec in self.data:
            index.insert(vec)
        build_time = time.perf_counter() - start_build
        
        stats = index.get_stats()
        
        # Calculate memory/edge savings relative to baseline
        savings_pct = 0.0
        if baseline_edges and baseline_edges > 0:
            savings_pct = ((baseline_edges - stats["total_edges"]) / baseline_edges) * 100.0
            
        # 2. Query Index
        latencies = []
        all_results = []
        init_dist_comps = index.total_dist_computations
        
        for q in self.queries:
            t0 = time.perf_counter()
            res = index.search(q, k=100, ef=ef_search)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            all_results.append(res)
            
        total_query_time_s = sum(latencies) / 1000.0
        qps = float(self.n_queries / total_query_time_s) if total_query_time_s > 0 else 0.0
        dist_comps_per_q = (index.total_dist_computations - init_dist_comps) / self.n_queries
        
        recall_1 = self._calculate_recall(all_results, 1)
        recall_10 = self._calculate_recall(all_results, 10)
        recall_100 = self._calculate_recall(all_results, 100)
        
        return BenchmarkResult(
            index_name=policy_name,
            dataset_name="Vector Dataset",
            n_samples=self.n_samples,
            dim=self.dim,
            build_time_s=round(build_time, 4),
            total_edges=stats["total_edges"],
            avg_edges_per_node=round(stats["avg_edges_per_node"], 2),
            index_memory_mb=round(stats["total_memory_mb"], 3),
            memory_savings_pct=round(savings_pct, 2),
            recall_1=round(recall_1, 4),
            recall_10=round(recall_10, 4),
            recall_100=round(recall_100, 4),
            qps=round(qps, 2),
            latency_p50_ms=round(float(np.percentile(latencies, 50)), 3),
            latency_p95_ms=round(float(np.percentile(latencies, 95)), 3),
            latency_p99_ms=round(float(np.percentile(latencies, 99)), 3),
            avg_dist_comps_per_query=round(dist_comps_per_q, 1),
            details=stats
        )

    def run_full_suite(self, ef_search: int = 50) -> List[BenchmarkResult]:
        """Runs comparative benchmark across all policy variations."""
        results = []
        
        # 1. Stock HNSW Baseline
        stock_res = self.evaluate_stock_hnsw(m=16, ef_construction=150, ef_search=ef_search)
        results.append(stock_res)
        baseline_edges = stock_res.total_edges
        
        # 2. Density-Only Adaptive
        cfg_density = AdaptivePolicyConfig(policy_type="density_only", m_base=16, m_min=8, m_max=28)
        results.append(self.evaluate_adaptive_hnsw(
            policy_config=cfg_density,
            ef_search=ef_search,
            baseline_edges=baseline_edges,
            policy_name="Density-Aware HNSW"
        ))
        
        # 3. LID-Only Adaptive
        cfg_lid = AdaptivePolicyConfig(policy_type="lid_only", m_base=16, m_min=8, m_max=28)
        results.append(self.evaluate_adaptive_hnsw(
            policy_config=cfg_lid,
            ef_search=ef_search,
            baseline_edges=baseline_edges,
            policy_name="LID-Aware HNSW"
        ))
        
        # 4. AdaptiveVec (Combined Continuous)
        cfg_combined = AdaptivePolicyConfig(policy_type="continuous", m_base=16, m_min=8, m_max=28, sensitivity=0.6)
        results.append(self.evaluate_adaptive_hnsw(
            policy_config=cfg_combined,
            ef_search=ef_search,
            baseline_edges=baseline_edges,
            policy_name="AdaptiveVec (Combined Continuous)"
        ))
        
        # 5. AdaptiveVec (Quantile Tiered)
        cfg_quantile = AdaptivePolicyConfig(policy_type="quantile", m_base=16, m_min=8, m_max=28)
        results.append(self.evaluate_adaptive_hnsw(
            policy_config=cfg_quantile,
            ef_search=ef_search,
            baseline_edges=baseline_edges,
            policy_name="AdaptiveVec (Quantile Tiered)"
        ))
        
        return results
