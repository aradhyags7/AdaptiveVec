"""
Unit Tests for Signal Estimation (Density, MLE LID, Cluster Variance).
"""

import unittest
import numpy as np
from adaptivevec.signals import compute_mle_lid, compute_local_density, compute_cluster_variance, estimate_node_signals

class TestSignals(unittest.TestCase):
    
    def test_mle_lid_known_dimensions(self):
        """
        Verify that the MLE LID estimator accurately estimates intrinsic dimensionality
        for known synthetic manifolds (e.g. 1D line, 2D plane, 5D hypercube).
        """
        np.random.seed(42)
        
        # Test 1D uniform line
        n = 1000
        line_pts = np.sort(np.random.uniform(0, 10, n))
        center_idx = n // 2
        dists_1d = np.abs(line_pts - line_pts[center_idx])
        dists_1d = np.sort(dists_1d[dists_1d > 1e-6])[:50]
        lid_1d = compute_mle_lid(dists_1d)
        self.assertAlmostEqual(lid_1d, 1.0, delta=0.4, msg=f"Expected ~1.0 for 1D line, got {lid_1d}")
        
        # Test 2D uniform disk
        r = np.sqrt(np.random.uniform(0, 1, n)) * 10
        theta = np.random.uniform(0, 2 * np.pi, n)
        disk_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        dists_2d = np.linalg.norm(disk_pts - disk_pts[0], axis=1)
        dists_2d = np.sort(dists_2d[dists_2d > 1e-6])[:50]
        lid_2d = compute_mle_lid(dists_2d)
        self.assertAlmostEqual(lid_2d, 2.0, delta=0.5, msg=f"Expected ~2.0 for 2D disk, got {lid_2d}")

    def test_local_density(self):
        """Verify that tight clusters have lower distance (higher density) than sparse clouds."""
        dense_dists = np.array([0.1, 0.12, 0.15, 0.18, 0.2])
        sparse_dists = np.array([5.0, 6.2, 7.1, 8.5, 9.0])
        
        d_dense = compute_local_density(dense_dists)
        d_sparse = compute_local_density(sparse_dists)
        
        self.assertLess(d_dense, d_sparse)
        self.assertAlmostEqual(d_dense, 0.15)

    def test_cluster_variance(self):
        """Verify spatial dispersion of points."""
        tight_pts = np.random.normal(0, 0.1, size=(50, 10))
        spread_pts = np.random.normal(0, 10.0, size=(50, 10))
        
        var_tight = compute_cluster_variance(tight_pts)
        var_spread = compute_cluster_variance(spread_pts)
        
        self.assertLess(var_tight, var_spread)

    def test_estimate_node_signals(self):
        dists = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        signals = estimate_node_signals(dists, k=5)
        self.assertIn("density", signals)
        self.assertIn("lid", signals)
        self.assertIn("variance", signals)
        self.assertGreater(signals["lid"], 0.5)

if __name__ == "__main__":
    unittest.main()
