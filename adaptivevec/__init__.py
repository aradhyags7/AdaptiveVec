"""
AdaptiveVec: A Density- and Dimension-Aware Vector Index for Resource-Constrained Retrieval
"""

import os
import sys

# On Windows, ensure MinGW / UCRT64 toolchain runtime DLLs are accessible
if sys.platform == "win32":
    mingw_bin = r"C:\msys64\ucrt64\bin"
    if os.path.exists(mingw_bin) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(mingw_bin)
        except OSError:
            pass

from .signals import compute_local_density, compute_mle_lid, compute_cluster_variance, estimate_node_signals
from .policy import AdaptivePolicy, AdaptivePolicyConfig, map_signals_to_params
from .hnsw_base import StockHNSW
from .adaptive_hnsw import AdaptiveHNSW
from .datasets import load_or_generate_dataset, generate_multi_manifold_dataset
from .benchmark import BenchmarkHarness, BenchmarkResult
from .quantization import ScalarQuantizer, ScalarQuantizer4Bit

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
    "ScalarQuantizer",
    "ScalarQuantizer4Bit",
    "load_or_generate_dataset",
    "generate_multi_manifold_dataset",
    "BenchmarkHarness",
    "BenchmarkResult"
]
