"""
Dataset Generation and Loading Utilities for AdaptiveVec.

Includes:
1. Multi-Manifold Synthetic Generator (heterogeneous clusters with varying true LID and Density)
2. SIFT-128d Benchmark Simulator / Loader
3. GloVe / Semantic Embedding Generator
4. Brute-Force Ground Truth Oracle
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

def generate_multi_manifold_dataset(
    n_samples: int = 10000,
    ambient_dim: int = 64,
    n_queries: int = 200,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates a synthetic dataset with distinct sub-manifolds exhibiting
    heterogeneous local densities and local intrinsic dimensionalities (LID):
    
    1. Region A (Dense 2D Swiss-Roll manifold in ambient_dim space) -> High Density, Low LID (~2)
    2. Region B (Dense 1D Spiral curve) -> Very High Density, Low LID (~1)
    3. Region C (Moderate 8D Hyper-Ellipsoid cluster) -> Medium Density, Medium LID (~8)
    4. Region D (Sparse 64D Uniform Cloud) -> Low Density, High LID (~ambient_dim)
    """
    np.random.seed(seed)
    
    n_per_region = n_samples // 4
    remainder = n_samples - (n_per_region * 4)
    
    vectors = []
    ground_truth_lids = []
    ground_truth_densities = []
    
    # 1. Region A: Swiss Roll (LID ~ 2, Dense)
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_per_region))
    height = 20 * np.random.rand(n_per_region)
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)
    swiss = np.zeros((n_per_region, ambient_dim))
    swiss[:, 0] = x
    swiss[:, 1] = y
    swiss[:, 2] = z
    swiss += np.random.normal(0, 0.05, size=swiss.shape)  # Small ambient noise
    vectors.append(swiss)
    ground_truth_lids.extend([2.0] * n_per_region)
    ground_truth_densities.extend(["high"] * n_per_region)
    
    # 2. Region B: 1D Archimedean Spiral (LID ~ 1, Dense)
    theta = np.linspace(0, 10 * np.pi, n_per_region) + np.random.normal(0, 0.02, n_per_region)
    r = 0.5 * theta
    spiral = np.zeros((n_per_region, ambient_dim))
    spiral[:, 3] = r * np.cos(theta) + 50.0
    spiral[:, 4] = r * np.sin(theta) + 50.0
    spiral += np.random.normal(0, 0.03, size=spiral.shape)
    vectors.append(spiral)
    ground_truth_lids.extend([1.0] * n_per_region)
    ground_truth_densities.extend(["very_high"] * n_per_region)
    
    # 3. Region C: 8D Hyper-Ellipsoid (LID ~ 8, Medium density)
    ellip_dims = 8
    ellip_raw = np.random.normal(0, 1, size=(n_per_region, ellip_dims))
    # Normalize to ball
    norms = np.linalg.norm(ellip_raw, axis=1, keepdims=True)
    radii = np.random.power(a=ellip_dims, size=(n_per_region, 1)) * 15.0
    ellip = np.zeros((n_per_region, ambient_dim))
    ellip[:, 5:5+ellip_dims] = (ellip_raw / (norms + 1e-9)) * radii - 40.0
    vectors.append(ellip)
    ground_truth_lids.extend([8.0] * n_per_region)
    ground_truth_densities.extend(["medium"] * n_per_region)
    
    # 4. Region D: Sparse 64D Gaussian Cloud (LID ~ ambient_dim, Sparse)
    n_d = n_per_region + remainder
    sparse_cloud = np.random.normal(0, 25.0, size=(n_d, ambient_dim)) + 80.0
    vectors.append(sparse_cloud)
    ground_truth_lids.extend([float(ambient_dim)] * n_d)
    ground_truth_densities.extend(["sparse"] * n_d)
    
    all_vectors = np.vstack(vectors).astype(np.float32)
    
    # Shuffle dataset
    indices = np.random.permutation(len(all_vectors))
    all_vectors = all_vectors[indices]
    
    # Sample queries
    query_indices = np.random.choice(len(all_vectors), size=n_queries, replace=False)
    queries = all_vectors[query_indices] + np.random.normal(0, 0.1, size=(n_queries, ambient_dim)).astype(np.float32)
    
    metadata = {
        "name": "Synthetic Multi-Manifold (Mixed LID & Density)",
        "dim": ambient_dim,
        "n_samples": len(all_vectors),
        "n_queries": n_queries,
        "regions": ["2D Swiss-Roll (Dense)", "1D Spiral (Dense)", "8D Hyper-Ellipsoid (Medium)", "64D Cloud (Sparse)"]
    }
    
    return all_vectors, queries, metadata

def generate_sift_like_dataset(
    n_samples: int = 10000,
    dim: int = 128,
    n_queries: int = 200,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates realistic 128-dimensional SIFT descriptor-like vectors with
    clustered gradient orientations and non-negative values.
    """
    np.random.seed(seed)
    n_clusters = max(5, n_samples // 500)
    centers = np.random.exponential(scale=20.0, size=(n_clusters, dim))
    cluster_scales = np.random.uniform(2.0, 15.0, size=n_clusters)
    
    cluster_assignments = np.random.choice(n_clusters, size=n_samples)
    vectors = []
    
    for i in range(n_samples):
        c_idx = cluster_assignments[i]
        center = centers[c_idx]
        scale = cluster_scales[c_idx]
        # SIFT features have positive values and exponential/gamma distributions
        v = np.abs(center + np.random.normal(0, scale, size=dim))
        # L2-normalize to emulate SIFT unit vectors
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v = v / norm
        vectors.append(v)
        
    all_vectors = np.array(vectors, dtype=np.float32)
    
    # Query vectors
    q_indices = np.random.choice(n_samples, size=n_queries, replace=False)
    queries = all_vectors[q_indices] + np.random.normal(0, 0.02, size=(n_queries, dim)).astype(np.float32)
    # Re-normalize queries
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / np.clip(q_norms, 1e-9, None)
    
    metadata = {
        "name": f"SIFT-{dim}D Clustered Benchmark",
        "dim": dim,
        "n_samples": n_samples,
        "n_queries": n_queries,
        "clusters": n_clusters
    }
    
    return all_vectors, queries, metadata

def compute_ground_truth(
    data: np.ndarray,
    queries: np.ndarray,
    k: int = 100,
    space: str = "l2"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Brute-force exact K-nearest neighbor calculation for recall evaluation.
    Returns (indices, distances) of shape (n_queries, k).
    """
    n_queries = len(queries)
    gt_indices = np.zeros((n_queries, k), dtype=np.int32)
    gt_distances = np.zeros((n_queries, k), dtype=np.float32)
    
    for q_idx in range(n_queries):
        query = queries[q_idx]
        if space == "cosine":
            # Cosine distance
            norm_q = np.linalg.norm(query)
            norms_data = np.linalg.norm(data, axis=1)
            sims = np.dot(data, query) / (np.clip(norms_data * norm_q, 1e-9, None))
            dists = 1.0 - sims
        else:
            # L2 Euclidean distance squared / root
            diff = data - query
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            
        # Top-K smallest distances
        top_k_idx = np.argsort(dists)[:k]
        gt_indices[q_idx] = top_k_idx
        gt_distances[q_idx] = dists[top_k_idx]
        
    return gt_indices, gt_distances

def load_or_generate_dataset(
    dataset_name: str = "multi_manifold",
    n_samples: int = 5000,
    dim: int = 64,
    n_queries: int = 100
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Universal dataset resolver."""
    name = dataset_name.lower()
    if "sift" in name:
        return generate_sift_like_dataset(n_samples=n_samples, dim=dim or 128, n_queries=n_queries)
    elif "glove" in name or "word" in name:
        # GloVe-style dense semantic vectors (normalized)
        data, queries, meta = generate_sift_like_dataset(n_samples=n_samples, dim=dim or 100, n_queries=n_queries)
        meta["name"] = f"GloVe-{dim}D Semantic Vector Simulator"
        return data, queries, meta
    else:
        return generate_multi_manifold_dataset(n_samples=n_samples, ambient_dim=dim or 64, n_queries=n_queries)
