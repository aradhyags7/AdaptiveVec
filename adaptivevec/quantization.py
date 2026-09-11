"""
Scalar Quantization (SQ8) Module for AdaptiveVec.

Compresses 32-bit floating point embeddings into compact 8-bit integers (uint8),
reducing vector memory consumption by 75% while providing fast asymmetric distance
evaluations during graph traversal with exact float32 two-stage re-ranking.
"""

import numpy as np
from typing import Tuple, Optional, Union

class ScalarQuantizer:
    """
    Uniform 8-bit Scalar Quantizer (SQ8).
    Maps high-dimensional float32 vectors to uint8 codes with dimension-wise calibration.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.min_vals: np.ndarray = np.zeros(dim, dtype=np.float32)
        self.max_vals: np.ndarray = np.ones(dim, dtype=np.float32)
        self.scales: np.ndarray = np.ones(dim, dtype=np.float32)
        self.is_trained: bool = False

    def train(self, vectors: np.ndarray) -> None:
        """Calibrates min, max, and quantization step size per dimension."""
        if len(vectors) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        self.min_vals = np.min(vectors, axis=0)
        self.max_vals = np.max(vectors, axis=0)
        ranges = self.max_vals - self.min_vals
        # Avoid division by zero for degenerate dimensions
        self.scales = np.where(ranges > 1e-7, ranges / 255.0, 1.0).astype(np.float32)
        self.is_trained = True

    def encode(self, vector: np.ndarray) -> np.ndarray:
        """Quantizes a float32 vector into uint8 codes [0, 255]."""
        v = np.asarray(vector, dtype=np.float32)
        if not self.is_trained:
            # Auto-initialize with standard normal bounds [-3.0, 3.0] if untrained
            self.min_vals = np.full(self.dim, -3.0, dtype=np.float32)
            self.max_vals = np.full(self.dim, 3.0, dtype=np.float32)
            self.scales = np.full(self.dim, 6.0 / 255.0, dtype=np.float32)
            self.is_trained = True

        scaled = (v - self.min_vals) / self.scales
        return np.clip(np.round(scaled), 0, 255).astype(np.uint8)

    def decode(self, code: np.ndarray) -> np.ndarray:
        """Reconstructs approximate float32 vector from uint8 codes."""
        c = np.asarray(code, dtype=np.float32)
        return (self.min_vals + c * self.scales).astype(np.float32)

    def asymmetric_l2_distance(self, query_float: np.ndarray, code_uint8: np.ndarray) -> float:
        """
        Fast asymmetric Euclidean distance between exact float query and quantized vector code.
        d(q, c) = || q - (min + c * scale) ||_2
        """
        approx_vec = self.min_vals + code_uint8.astype(np.float32) * self.scales
        diff = query_float - approx_vec
        return float(np.dot(diff, diff) ** 0.5)

    def asymmetric_cosine_distance(self, query_float: np.ndarray, code_uint8: np.ndarray) -> float:
        """
        Asymmetric Cosine distance between exact float query and quantized vector code.
        """
        approx_vec = self.min_vals + code_uint8.astype(np.float32) * self.scales
        norm_q = np.linalg.norm(query_float)
        norm_c = np.linalg.norm(approx_vec)
        if norm_q < 1e-9 or norm_c < 1e-9:
            return 1.0
        cos_sim = np.dot(query_float, approx_vec) / (norm_q * norm_c)
        return float(max(0.0, 1.0 - cos_sim))
