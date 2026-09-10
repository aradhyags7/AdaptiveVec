"""
AdaptiveVec: A Density- and Dimension-Aware Vector Index for Resource-Constrained Retrieval
"""

from .signals import compute_local_density, compute_mle_lid, compute_cluster_variance, estimate_node_signals
from .policy import AdaptivePolicy, AdaptivePolicyConfig, map_signals_to_params
from .hnsw_base import StockHNSW
from .adaptive_hnsw import AdaptiveHNSW
from .datasets import load_or_generate_dataset, generate_multi_manifold_dataset
from .benchmark import BenchmarkHarness, BenchmarkResult

__all__ = [
    "compute_local_density",
    "compute_mle_lid",
    "compute_cluster_variance",
    "estimate_node_signals",
    "AdaptivePolicy",
    "AdaptivePolicyConfig",
    "map_signals_to_params",
    "StockHNSW",
    "AdaptiveHNSW",
    "load_or_generate_dataset",
    "generate_multi_manifold_dataset",
    "BenchmarkHarness",
    "BenchmarkResult"
]
