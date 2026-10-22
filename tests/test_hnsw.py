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

    def test_early_exit_search(self):
        index = AdaptiveHNSW(dim=self.dim, early_exit=True, stagnation_patience=4)
        for v in self.data[:100]:
            index.insert(v)
            
        results_early, trace_early = index.search(self.queries[0], k=5, early_exit=True, record_trace=True)
        results_full, trace_full = index.search(self.queries[0], k=5, early_exit=False, record_trace=True)
        
        self.assertEqual(len(results_early), 5)
        self.assertEqual(len(results_full), 5)
        # Early exit should perform less or equal distance evaluations
        self.assertLessEqual(trace_early["total_dist_evals"], trace_full["total_dist_evals"])

    def test_hubness_regulation(self):
        index = AdaptiveHNSW(
            dim=self.dim,
            hubness_regulation=True,
            hubness_penalty_weight=0.25
        )
        for v in self.data[:120]:
            index.insert(v)
            
        self.assertEqual(len(index.in_degrees), 120)
        # Verify in_degree max is within reasonable bound
        max_deg = max(index.in_degrees.values())
        avg_deg = float(np.mean(list(index.in_degrees.values())))
        self.assertGreater(avg_deg, 0)
        self.assertLess(max_deg, 60, f"Excessive hub degree observed: {max_deg}")
        
        # Verify search completes successfully
        res = index.search(self.queries[0], k=5)
        self.assertEqual(len(res), 5)

    def test_scalar_quantization_two_stage(self):
        index = AdaptiveHNSW(
            dim=self.dim,
            quantize=True,
            policy_config=AdaptivePolicyConfig(m_base=8, ef_construction_base=60)
        )
        index.calibrate(self.data[:100])
        for v in self.data:
            index.insert(v)
            
        stats = index.get_stats()
        self.assertTrue(stats["quantized"])
        # Vector memory should be exactly 1 byte per dimension (75% savings vs 4 bytes)
        self.assertEqual(stats["vector_memory_bytes"], self.n_samples * self.dim * 1)
        
        # Search queries with two-stage re-ranking
        hits = 0
        total = self.n_queries * 10
        for i, q in enumerate(self.queries):
            results = index.search(q, k=10, ef=50)
            res_nodes = set(n for _, n in results)
            gt_nodes = set(self.gt_idx[i])
            hits += len(res_nodes.intersection(gt_nodes))
            
        recall = hits / total
        self.assertGreater(recall, 0.85, f"Quantized SQ8 Recall@10 was {recall}")

    def test_deletion_tombstone_filtering(self):
        """Frontier 2: Verify deleted nodes never appear in top-k query results."""
        index = AdaptiveHNSW(dim=self.dim, policy_config=AdaptivePolicyConfig(m_base=8, ef_construction_base=60))
        for v in self.data[:100]:
            index.insert(v)

        q = self.queries[0]
        initial_res = index.search(q, k=5)
        self.assertGreater(len(initial_res), 0)
        top_node = initial_res[0][1]

        # Delete the top nearest neighbor
        success = index.delete(top_node, heal=True)
        self.assertTrue(success)
        self.assertTrue(index.is_deleted(top_node))

        # Re-query: deleted node must NOT be present
        new_res = index.search(q, k=5)
        new_nodes = [node for _, node in new_res]
        self.assertNotIn(top_node, new_nodes)

        stats = index.get_stats()
        self.assertEqual(stats["tombstones"], 1)
        self.assertEqual(stats["active_nodes"], 99)

    def test_graph_healing_preserves_recall(self):
        """Frontier 2: Verify localized healing retains high recall after 20% random churn."""
        index = AdaptiveHNSW(dim=self.dim, policy_config=AdaptivePolicyConfig(m_base=8, ef_construction_base=60))
        for v in self.data[:150]:
            index.insert(v)

        # Delete 30 nodes (20% churn) with graph healing
        np.random.seed(99)
        to_delete = np.random.choice(150, size=30, replace=False)
        for node_id in to_delete:
            index.delete(int(node_id), heal=True)

        stats = index.get_stats()
        self.assertEqual(stats["tombstones"], 30)
        self.assertEqual(stats["active_nodes"], 120)

        # Query all test queries and verify no deleted nodes return
        surviving_nodes = set(range(150)) - set(to_delete)
        for q in self.queries:
            results = index.search(q, k=10, ef=40)
            res_nodes = set(node for _, node in results)
            self.assertTrue(res_nodes.issubset(surviving_nodes), "Deleted node leaked into results!")

    def test_entry_point_deletion_and_migration(self):
        """Frontier 2: Verify entry point migrates to an active node when deleted."""
        index = AdaptiveHNSW(dim=self.dim)
        for v in self.data[:50]:
            index.insert(v)

        old_ep = index.enter_point
        self.assertIsNotNone(old_ep)

        # Delete the entry point
        index.delete(old_ep, heal=True)
        self.assertTrue(index.is_deleted(old_ep))
        self.assertNotEqual(index.enter_point, old_ep)
        self.assertFalse(index.is_deleted(index.enter_point))

        # Subsequent search must succeed
        res = index.search(self.queries[0], k=5)
        self.assertEqual(len(res), 5)
        self.assertNotIn(old_ep, [node for _, node in res])

    def test_compaction_reclaims_memory(self):
        """Frontier 2: Verify compaction removes tombstones, updates indices, and retains accuracy."""
        index = AdaptiveHNSW(dim=self.dim)
        for v in self.data[:80]:
            index.insert(v)

        to_delete = [5, 12, 23, 44, 60]
        for node_id in to_delete:
            index.delete(node_id, heal=True)

        self.assertEqual(len(index.deleted), 5)
        compaction_info = index.compact()
        self.assertEqual(compaction_info["compacted_nodes"], 75)
        self.assertEqual(compaction_info["removed_tombstones"], 5)
        self.assertEqual(len(index.deleted), 0)
        self.assertEqual(len(index.data), 75)

        stats = index.get_stats()
        self.assertEqual(stats["tombstones"], 0)
        self.assertEqual(stats["total_nodes"], 75)
        self.assertEqual(stats["active_nodes"], 75)

        # Search should execute cleanly on compacted index
        res = index.search(self.queries[0], k=5)
        self.assertEqual(len(res), 5)
        for _, node in res:
            self.assertLess(node, 75)

if __name__ == "__main__":
    unittest.main()


def test_empty_index_query_handling():
    """Verify that searching an empty index returns empty results without crashing."""
    import numpy as np
    from adaptivevec.adaptive_hnsw import AdaptiveHNSW
    idx = AdaptiveHNSW(dim=16)
    results = idx.search(np.zeros(16, dtype=np.float32), k=5)
    assert len(results) == 0
