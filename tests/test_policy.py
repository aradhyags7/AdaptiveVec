"""
Unit Tests for Adaptive Policy Engine.
"""

import unittest
from adaptivevec.policy import AdaptivePolicy, AdaptivePolicyConfig, map_signals_to_params

class TestPolicy(unittest.TestCase):
    
    def test_continuous_policy_bounds(self):
        config = AdaptivePolicyConfig(
            policy_type="continuous",
            m_base=16,
            m_min=8,
            m_max=28,
            ef_construction_base=150,
            ef_construction_min=40,
            ef_construction_max=250,
            density_mean=1.0,
            density_std=0.5,
            lid_mean=4.0,
            lid_std=2.0
        )
        policy = AdaptivePolicy(config)
        
        # Test extreme low complexity (dense, low LID)
        p_low = policy.evaluate(density=0.1, lid=1.0)
        self.assertLessEqual(p_low.m, config.m_base)
        self.assertGreaterEqual(p_low.m, config.m_min)
        self.assertLessEqual(p_low.ef_construction, config.ef_construction_base)
        self.assertGreaterEqual(p_low.ef_construction, config.ef_construction_min)
        
        # Test extreme high complexity (sparse, high LID)
        p_high = policy.evaluate(density=3.0, lid=10.0)
        self.assertGreaterEqual(p_high.m, config.m_base)
        self.assertLessEqual(p_high.m, config.m_max)
        self.assertGreaterEqual(p_high.ef_construction, config.ef_construction_base)
        self.assertLessEqual(p_high.ef_construction, config.ef_construction_max)

    def test_quantile_policy(self):
        config = AdaptivePolicyConfig(
            policy_type="quantile",
            m_base=16,
            m_min=8,
            m_max=28,
            density_mean=1.0,
            density_std=1.0,
            lid_mean=4.0,
            lid_std=1.0
        )
        policy = AdaptivePolicy(config)
        
        p_low = policy.evaluate(density=0.1, lid=1.0)
        self.assertEqual(p_low.m, config.m_min)
        
        p_high = policy.evaluate(density=3.5, lid=8.0)
        self.assertEqual(p_high.m, config.m_max)

    def test_streaming_welford_tracker(self):
        import numpy as np
        from adaptivevec.policy import StreamingStatsTracker
        
        np.random.seed(42)
        samples = np.random.normal(loc=5.0, scale=2.0, size=200)
        
        tracker = StreamingStatsTracker()
        for x in samples:
            tracker.update(float(x))
            
        self.assertAlmostEqual(tracker.mean, float(np.mean(samples)), places=3)
        self.assertAlmostEqual(tracker.std, float(np.std(samples, ddof=1)), places=3)

    def test_streaming_online_calibration(self):
        config = AdaptivePolicyConfig(enable_streaming_calibration=True)
        policy = AdaptivePolicy(config)
        
        # Stream in 30 values of density and LID
        for i in range(1, 31):
            policy.observe(density=float(i * 0.1), lid=float(2.0 + i * 0.05))
            
        self.assertGreater(policy.density_tracker.count, 20)
        # Verify stats updated dynamically from default baseline
        self.assertNotEqual(config.density_mean, 1.0)
        self.assertGreater(config.density_std, 0.0)

    def test_layer_decoupled_scaling(self):
        config = AdaptivePolicyConfig(m_base=16)
        policy = AdaptivePolicy(config)
        params = policy.evaluate(density=1.0, lid=4.0)
        
        # Layer 0 retains full base allocation
        self.assertEqual(params.get_m_for_layer(0), params.m)
        self.assertEqual(params.get_m_max_for_layer(0), params.m_max0)
        
        # Upper layers dynamically compress express links
        self.assertLess(params.get_m_for_layer(2), params.m)
        self.assertLess(params.get_m_max_for_layer(2), params.m_max0)
        self.assertGreaterEqual(params.get_m_for_layer(5), 4)

if __name__ == "__main__":
    unittest.main()
