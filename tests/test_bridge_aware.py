"""
tests/test_bridge_aware.py
Tests for Stage 2: Bridge-Aware Hubness Regulation in AdaptiveVec.
Validates bridge score estimation, preservation ratio, and query routing across clusters.
"""

import pytest
import numpy as np
import adaptivevec._adaptivevec_cpp as ac


def test_bridge_aware_policy_config():
    cfg = ac.PolicyConfig()
    assert hasattr(cfg, "enable_bridge_aware")
    assert hasattr(cfg, "bridge_alpha")
    assert hasattr(cfg, "bridge_estimator")
    assert cfg.enable_bridge_aware is False
    assert cfg.bridge_alpha == 1.0
    assert cfg.bridge_estimator == 0

    cfg.enable_bridge_aware = True
    cfg.bridge_alpha = 0.8
    cfg.bridge_estimator = 1
    assert cfg.enable_bridge_aware is True
    assert cfg.bridge_alpha == pytest.approx(0.8)
    assert cfg.bridge_estimator == 1


def test_bridge_aware_index_construction_and_scores():
    np.random.seed(1337)
    dim = 16
    n_cluster = 100
    n_bridge = 20

    # Cluster 1 around -5.0, Cluster 2 around +5.0
    c1 = np.random.randn(n_cluster, dim).astype(np.float32) * 0.3 - 5.0
    c2 = np.random.randn(n_cluster, dim).astype(np.float32) * 0.3 + 5.0
    bridge = np.random.randn(n_bridge, dim).astype(np.float32) * 0.3  # near origin

    data = np.vstack([c1, bridge, c2])

    cfg = ac.PolicyConfig()
    cfg.enable_hubness_regulation = True
    cfg.enable_bridge_aware = True
    cfg.hubness_mu = 0.15
    cfg.bridge_alpha = 1.0
    cfg.bridge_estimator = 0

    index = ac.AdaptiveHNSWIndex(dim, ac.SpaceType.L2, True, cfg)
    index.batch_insert(data, num_threads=2)

    assert index.num_elements == len(data)
    scores = index.get_bridge_scores()
    assert len(scores) == len(data)

    for s in scores:
        assert 0.0 <= s <= 1.0

    avg_b = index.get_average_bridge_score()
    assert 0.0 <= avg_b <= 1.0

    ratio = index.get_bridge_preservation_ratio(threshold=0.3)
    assert ratio > 0.0

    # Query cluster 1
    q1 = np.zeros(dim, dtype=np.float32) - 5.0
    res1 = index.search(q1, k=5, ef=32)
    assert len(res1) == 5
    assert res1[0].id < n_cluster

    # Query cluster 2
    q2 = np.zeros(dim, dtype=np.float32) + 5.0
    res2 = index.search(q2, k=5, ef=32)
    assert len(res2) == 5
    assert res2[0].id >= (n_cluster + n_bridge)
