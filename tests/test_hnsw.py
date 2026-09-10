"""
Unit Tests for Stock HNSW and AdaptiveHNSW Graph Construction and Search.
"""

import unittest
import numpy as np
from adaptivevec.hnsw_base import StockHNSW
from adaptivevec.adaptive_hnsw import AdaptiveHNSW
from adaptivevec.policy import AdaptivePolicyConfig
from adaptivevec.datasets import generate_multi_manifold_dataset, compute_ground_truth

class TestHNSW(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.dim = 16
        self.n_samples = 300
        self.n_queries = 20
        self.data, self.queries, _ = generate_multi_manifold_dataset(
            n_samples=self.n_samples,
            ambient_dim=self.dim,
            n_queries=self.n_queries
        )
        self.gt_idx, self.gt_dist = compute_ground_truth(self.data, self.queries, k=10)

    def test_stock_hnsw_construction_and_search(self):
        index = StockHNSW(dim=self.dim, m=8, ef_construction=60)
        for v in self.data:
            index.insert(v)
            
        stats = index.get_stats()
        self.assertEqual(stats["total_nodes"], self.n_samples)
        self.assertGreater(stats["total_edges"], 0)
        self.assertGreaterEqual(stats["max_level"], 0)
        
        # Search queries
        hits = 0
        total = self.n_queries * 10
        for i, q in enumerate(self.queries):
            results = index.search(q, k=10, ef=40)
            res_nodes = set(n for _, n in results)
            gt_nodes = set(self.gt_idx[i])
            hits += len(res_nodes.intersection(gt_nodes))
            
        recall = hits / total
        self.assertGreater(recall, 0.90, f"Stock HNSW Recall@10 was {recall}")

    def test_adaptive_hnsw_construction_and_search(self):
        config = AdaptivePolicyConfig(
            policy_type="continuous",
            m_base=8,
            m_min=4,
            m_max=14,
            ef_construction_base=60
        )
        index = AdaptiveHNSW(dim=self.dim, policy_config=config)
        index.calibrate(self.data[:100])
        
        for v in self.data:
            index.insert(v)
            
        stats = index.get_stats()
        self.assertEqual(stats["total_nodes"], self.n_samples)
        self.assertGreater(stats["total_edges"], 0)
        
        # Search queries
        hits = 0
        total = self.n_queries * 10
        for i, q in enumerate(self.queries):
            results = index.search(q, k=10, ef=40)
            res_nodes = set(n for _, n in results)
            gt_nodes = set(self.gt_idx[i])
            hits += len(res_nodes.intersection(gt_nodes))
            
        recall = hits / total
        self.assertGreater(recall, 0.90, f"Adaptive HNSW Recall@10 was {recall}")

    def test_search_trace_recording(self):
        index = AdaptiveHNSW(dim=self.dim)
        for v in self.data[:50]:
            index.insert(v)
            
        top_k, trace = index.search(self.queries[0], k=5, record_trace=True)
        self.assertIn("steps", trace)
        self.assertIn("total_dist_evals", trace)
        self.assertGreater(len(trace["steps"]), 0)

if __name__ == "__main__":
    unittest.main()
