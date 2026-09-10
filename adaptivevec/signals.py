"""
Signal Estimation Module for AdaptiveVec.

Computes:
1. Local Density (average distance to k-nearest neighbors in local sample)
2. Local Intrinsic Dimensionality (LID) using the Amsaleg et al. (2015) MLE estimator
3. Cluster Variance / Dispersion
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Union

EPSILON = 1e-9

def compute_local_density(distances: np.ndarray) -> float:
    """
    Computes local density indicator.
    Lower mean distance implies higher density.
    Returns the average distance to the k-nearest neighbors.
    """
    if len(distances) == 0:
        return 1.0
    return float(np.mean(distances))

def compute_mle_lid(distances: np.ndarray, k: Optional[int] = None) -> float:
    """
    Computes the Maximum Likelihood Estimation (MLE) of Local Intrinsic Dimensionality (LID).
    Reference:
        Amsaleg et al., "Estimating Local Intrinsic Dimension", SIGKDD 2015.
    
    Formula:
        LID(x) = - [ (1 / k) * sum_{i=1}^k ln(r_i / r_k) ]^(-1)
        where r_1 <= r_2 <= ... <= r_k are sorted distances to k nearest neighbors.
    
    Args:
        distances: 1D array of distances to neighboring points (positive values).
        k: Optional number of neighbors to consider. If None, uses len(distances).
    
    Returns:
        Estimated intrinsic dimensionality (float >= 1.0).
    """
    dists = np.sort(distances[distances > EPSILON])
    if k is not None and k < len(dists):
        dists = dists[:k]
    
    n = len(dists)
    if n < 3:
        # Not enough neighbors to reliably estimate LID; return neutral default
        return 2.0
    
    r_k = dists[-1]
    if r_k <= EPSILON:
        return 1.0
    
    # Avoid log(0) by clipping ratios to [EPSILON, 1.0 - EPSILON]
    ratios = np.clip(dists[:-1] / r_k, EPSILON, 1.0 - EPSILON)
    log_sum = np.sum(np.log(ratios))
    
    if abs(log_sum) < EPSILON:
        return 1.0
    
    # Negative inverse of average log ratio
    lid = - ( (n - 1) / log_sum )
    
    # Clamp to reasonable positive range [1.0, 1000.0]
    return float(np.clip(lid, 1.0, 1000.0))

def compute_cluster_variance(vectors: np.ndarray) -> float:
    """
    Computes the spatial dispersion / trace of covariance of the local sample vectors.
    """
    if len(vectors) < 2:
        return 0.0
    centroid = np.mean(vectors, axis=0)
    diffs = vectors - centroid
    return float(np.mean(np.sum(diffs ** 2, axis=-1)))

def estimate_node_signals(
    distances: np.ndarray,
    vectors: Optional[np.ndarray] = None,
    k: int = 15
) -> Dict[str, float]:
    """
    Computes all signals for a candidate neighborhood around a point.
    
    Args:
        distances: Array of Euclidean or Cosine distances to local neighbors.
        vectors: Optional array of neighbor vectors to compute spatial variance.
        k: Neighborhood size to use for estimation.
    
    Returns:
        Dictionary with 'density', 'lid', and 'variance'
    """
    sorted_dists = np.sort(distances)
    k_eff = min(k, len(sorted_dists))
    k_dists = sorted_dists[:k_eff]
    
    density_val = compute_local_density(k_dists)
    lid_val = compute_mle_lid(k_dists)
    variance_val = compute_cluster_variance(vectors[:k_eff]) if vectors is not None and len(vectors) > 0 else density_val ** 2
    
    return {
        "density": density_val,
        "lid": lid_val,
        "variance": variance_val
    }

def profile_dataset_signals(
    data: np.ndarray,
    sample_size: int = 500,
    k: int = 15
) -> Dict[str, Any]:
    """
    Profiles a dataset to obtain global reference statistics (mean, std, min, max, quantiles)
    for density and LID to calibrate adaptive policies.
    """
    n = len(data)
    if n == 0:
        return {}
    
    sample_indices = np.random.choice(n, size=min(sample_size, n), replace=False)
    sample = data[sample_indices]
    
    densities = []
    lids = []
    
    # Compute exact pairwise distances for the sample
    for i in range(len(sample)):
        diff = sample - sample[i]
        dists = np.linalg.norm(diff, axis=1)
        dists = dists[dists > EPSILON]
        if len(dists) >= k:
            d_k = np.sort(dists)[:k]
            densities.append(np.mean(d_k))
            lids.append(compute_mle_lid(d_k))
    
    densities = np.array(densities) if densities else np.array([1.0])
    lids = np.array(lids) if lids else np.array([2.0])
    
    return {
        "density_mean": float(np.mean(densities)),
        "density_std": float(np.std(densities)) if np.std(densities) > 0 else 1.0,
        "density_q25": float(np.percentile(densities, 25)),
        "density_q50": float(np.percentile(densities, 50)),
        "density_q75": float(np.percentile(densities, 75)),
        "lid_mean": float(np.mean(lids)),
        "lid_std": float(np.std(lids)) if np.std(lids) > 0 else 1.0,
        "lid_q25": float(np.percentile(lids, 25)),
        "lid_q50": float(np.percentile(lids, 50)),
        "lid_q75": float(np.percentile(lids, 75)),
        "sample_count": len(sample)
    }
