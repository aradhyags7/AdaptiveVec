# Asymmetric Scalar Quantization (SQ8) Implementation
# Provides 8-bit integer quantization with linear scale and offset
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


class ScalarQuantizer4Bit:
    """
    Uniform 4-Bit Scalar Quantizer (SQ4) with Sub-Byte Nibble Packing.
    Compresses float32 vectors to 4-bit unsigned integers [0, 15], packing two dimensions
    per byte (uint8). Yields 87.5% vector memory reduction vs FP32 and 50% vs SQ8.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.packed_dim = (dim + 1) // 2
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
        self.scales = np.where(ranges > 1e-7, ranges / 15.0, 1.0).astype(np.float32)
        self.is_trained = True

    def encode(self, vector: np.ndarray) -> np.ndarray:
        """Quantizes a float32 vector into nibble-packed uint8 codes [0, 15]."""
        v = np.asarray(vector, dtype=np.float32)
        if not self.is_trained:
            self.min_vals = np.full(self.dim, -3.0, dtype=np.float32)
            self.max_vals = np.full(self.dim, 3.0, dtype=np.float32)
            self.scales = np.full(self.dim, 6.0 / 15.0, dtype=np.float32)
            self.is_trained = True

        scaled = (v - self.min_vals) / self.scales
        nibbles = np.clip(np.round(scaled), 0, 15).astype(np.uint8)
        
        packed = np.zeros(self.packed_dim, dtype=np.uint8)
        for i in range(self.packed_dim):
            d0 = i * 2
            d1 = d0 + 1
            n0 = nibbles[d0]
            n1 = nibbles[d1] if d1 < self.dim else 0
            packed[i] = (n1 << 4) | (n0 & 0x0F)
        return packed

    def decode(self, code: np.ndarray) -> np.ndarray:
        """Reconstructs approximate float32 vector from nibble-packed uint8 codes."""
        packed = np.asarray(code, dtype=np.uint8)
        vec = np.zeros(self.dim, dtype=np.float32)
        for i in range(self.packed_dim):
            d0 = i * 2
            d1 = d0 + 1
            byte = packed[i]
            n0 = byte & 0x0F
            n1 = (byte >> 4) & 0x0F
            vec[d0] = self.min_vals[d0] + float(n0) * self.scales[d0]
            if d1 < self.dim:
                vec[d1] = self.min_vals[d1] + float(n1) * self.scales[d1]
        return vec

    def asymmetric_l2_distance(self, query_float: np.ndarray, code_packed: np.ndarray) -> float:
        """
        Fast asymmetric Euclidean distance between exact float query and nibble-packed code.
        """
        approx_vec = self.decode(code_packed)
        diff = query_float - approx_vec
        return float(np.dot(diff, diff) ** 0.5)

    def asymmetric_cosine_distance(self, query_float: np.ndarray, code_packed: np.ndarray) -> float:
        """
        Asymmetric Cosine distance between exact float query and nibble-packed code.
        """
        approx_vec = self.decode(code_packed)
        norm_q = np.linalg.norm(query_float)
        norm_c = np.linalg.norm(approx_vec)
        if norm_q < 1e-9 or norm_c < 1e-9:
            return 1.0
        cos_sim = np.dot(query_float, approx_vec) / (norm_q * norm_c)
        return float(max(0.0, 1.0 - cos_sim))

