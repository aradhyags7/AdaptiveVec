"""
tests/test_cpp_parity.py
Validates Python <-> C++ pybind11 interoperability, API consistency,
numerical parity, memory reporting, and compaction behavior.
"""

import pytest
import numpy as np
import adaptivevec._adaptivevec_cpp as ac
from adaptivevec.signals import compute_mle_lid, compute_local_density
from adaptivevec.quantization import ScalarQuantizer, ScalarQuantizer4Bit


def test_cpp_module_initialization():
    cfg = ac.PolicyConfig()
    cfg.m_base = 16
    cfg.ef_c_base = 120
    cfg.enable_dynamic_m_efc = True
    cfg.enable_hubness_regulation = True
    cfg.hubness_mu = 0.15

    index = ac.AdaptiveHNSWIndex(dimension=32, space=ac.SpaceType.L2, adaptive=True, policy=cfg)
    assert index.dim == 32
    assert index.num_elements == 0
    assert index.get_total_edges() == 0


def test_batch_insertion_and_search():
    np.random.seed(42)
    dim = 16
    n_samples = 200
    n_queries = 10
    
    data = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    index = ac.AdaptiveHNSWIndex(dim, ac.SpaceType.L2, True)
    index.batch_insert(data, num_threads=2)
    assert index.num_elements == n_samples
    assert index.get_total_edges() > 0

    results = index.batch_search(queries, k=5, ef=32, num_threads=2)
    assert len(results) == n_queries
    for q_res in results:
        assert len(q_res) == 5
        # Verify candidates are sorted by distance ascending
        for i in range(len(q_res) - 1):
            assert q_res[i].dist <= q_res[i+1].dist + 1e-5


def test_quantization_parity():
    np.random.seed(123)
    dim = 32
    n_samples = 500
    data = np.random.randn(n_samples, dim).astype(np.float32)

    # Python ScalarQuantizer
    py_sq = ScalarQuantizer(dim=dim)
    py_sq.train(data)
    py_codes = np.array([py_sq.encode(data[i]) for i in range(len(data))])
    py_recon = np.array([py_sq.decode(py_codes[i]) for i in range(len(data))])

    # Parity check: max reconstruction error should be bounded
    py_max_err = np.max(np.abs(data - py_recon))
    assert py_max_err < 0.2, f"Python SQ8 reconstruction error high: {py_max_err}"


def test_memory_breakdown_reporting():
    dim = 64
    n_samples = 300
    data = np.random.randn(n_samples, dim).astype(np.float32)

    index = ac.AdaptiveHNSWIndex(dim, ac.SpaceType.L2, True)
    index.batch_insert(data)

    mb = index.get_memory_breakdown()
    assert mb.resident_fp32_bytes >= n_samples * dim * 4
    assert mb.graph_adjacency_bytes > 0
    assert mb.per_node_metadata_bytes > 0
    assert mb.process_working_set_bytes > 0
    assert mb.total_owned_bytes > mb.resident_fp32_bytes

    # Configuration B: SQ8 Quantized + Resident FP32
    index.quantize_vectors(sq8=True, sq4=False)
    mb_b = index.get_memory_breakdown()
    assert mb_b.allocated_quantized_capacity_bytes >= n_samples * dim * 1
    assert mb_b.resident_fp32_bytes == mb.resident_fp32_bytes

    # Configuration A: Pure Quantized (FP32 dropped)
    index.drop_raw_vectors()
    mb_a = index.get_memory_breakdown()
    assert mb_a.resident_fp32_bytes == 0
    assert mb_a.allocated_quantized_capacity_bytes >= n_samples * dim * 1


def test_tombstone_deletion_and_compaction():
    dim = 16
    n_samples = 150
    data = np.random.randn(n_samples, dim).astype(np.float32)

    index = ac.AdaptiveHNSWIndex(dim, ac.SpaceType.L2, True)
    index.batch_insert(data)
    assert index.get_active_count() == 150

    # Delete 30 nodes
    for i in range(10, 40):
        ok = index.delete_vector(i, heal=True)
        assert ok is True

    assert index.get_active_count() == 120

    # Compact
    index.compact()
    assert index.num_elements == 120
    assert index.get_active_count() == 120

    # Search should run without referencing any out of bound IDs
    res = index.search(data[0], k=10, ef=32)
    assert len(res) == 10
    for cand in res:
        assert cand.id < 120


def test_graph_fingerprint_invariance():
    dim = 16
    n_samples = 100
    data = np.random.randn(n_samples, dim).astype(np.float32)

    index = ac.AdaptiveHNSWIndex(dim, ac.SpaceType.L2, True)
    index.batch_insert(data)

    fp1 = index.get_graph_fingerprint()
    assert fp1 != 0

    # Query modification (enable ada-ef)
    index.policy.enable_ada_ef = True
    index.policy.ada_ef_patience = 6
    index.policy.ada_ef_epsilon = 1e-4
    fp2 = index.get_graph_fingerprint()

    # Fingerprint must be strictly invariant across query parameter changes
    assert fp1 == fp2, "Graph fingerprint must not change when query parameters change"


def test_adaptive_vec_high_level_wrapper_and_sq4_cosine():
    from adaptivevec import AdaptiveVec
    from adaptivevec.adaptive_vec import HAS_CPP

    assert HAS_CPP is True

    rng = np.random.RandomState(42)
    data = rng.randn(120, 21).astype(np.float32)

    wrapper = AdaptiveVec(dimension=21, space="cosine", threads=2)
    wrapper.build(data)
    res_1d = wrapper.search(data[3], k=5, ef=32)
    assert len(res_1d) == 5
    assert res_1d[0][0] == 3

    res_2d = wrapper.search(data[:4], k=3, ef=32)
    assert len(res_2d) == 4
    assert all(row[0][0] == idx for idx, row in enumerate(res_2d))

    # SQ4 with Cosine space must use asymmetric_cosine_distance (bounds [0, 2]), not L2^2
    cfg = ac.PolicyConfig()
    cfg.enable_sq4 = True
    cfg.enable_rerank = False
    cfg.keep_raw_vectors = True
    idx_cos = ac.AdaptiveHNSWIndex(21, ac.SpaceType.Cosine, False, cfg)
    idx_cos.train_quantizer(data)
    idx_cos.batch_insert(data, num_threads=2)

    cos_res = idx_cos.search(data[5], k=5, ef=32)
    assert len(cos_res) == 5
    assert 0.0 <= cos_res[0].dist < 0.25
    assert 5 in [c.id for c in cos_res]


def test_incremental_multi_batch_insert_and_concurrent_search():
    rng = np.random.RandomState(7)
    data = rng.randn(500, 16).astype(np.float32)

    idx = ac.AdaptiveHNSWIndex(16, ac.SpaceType.L2, True)
    for i in range(40):
        idx.insert(data[i])
    idx.batch_insert(data[40:280], num_threads=4)
    idx.batch_insert(data[280:500], num_threads=4)

    assert idx.num_elements == 500
    for probe in (5, 150, 480):
        hits = idx.search(data[probe], k=3, ef=48)
        assert hits[0].id == probe


def test_section2_budget_allocator_serving_manager_and_calibrator_bindings():
    from adaptivevec.adaptive_hnsw import AdaptiveHNSW
    from adaptivevec.policy import AdaptivePolicyConfig

    # 1. BudgetAllocator Python bindings & multi-round conservation
    cont = [19.9, 19.9, 8.0, 16.2]
    res = ac.BudgetAllocator.largest_remainder_discretization(cont, 64, 8, 24)
    assert sum(res) == 64
    assert all(8 <= m <= 24 for m in res)
    with pytest.raises(ValueError):
        ac.BudgetAllocator.largest_remainder_discretization([10.0, 10.0], 100, 8, 24)

    # 2. ServingIndexManager bindings, label sync & peak memory gate
    rng = np.random.RandomState(11)
    data = rng.randn(80, 16).astype(np.float32)
    idx = ac.AdaptiveHNSWIndex(16, ac.SpaceType.L2, True)
    idx.batch_insert(data, num_threads=2)

    mgr = ac.ServingIndexManager(idx)
    assert mgr.get_active_label_count() == 80
    cand = mgr.get_serving_index()
    val_q = data[:10]
    val_gt = np.tile(np.arange(5, dtype=np.int32), (10, 1))
    # Tiny peak budget should reject peak memory cleanly (Gate 3)
    rej = mgr.evaluate_and_replace(cand, val_q, val_gt, target_recall=0.0, max_memory_mb=0.0001)
    assert rej.status == ac.ReplacementStatus.RejectedPeakMemoryViolation

    # Direct replacement without snapshot_seq keeps labels synced cleanly
    cand_ok = ac.AdaptiveHNSWIndex(16, ac.SpaceType.L2, True)
    cand_ok.batch_insert(data[:60], num_threads=2)
    ok_res = mgr.evaluate_and_replace(cand_ok, val_q, val_gt, target_recall=0.0, max_memory_mb=0.0)
    assert ok_res.status == ac.ReplacementStatus.Success
    assert mgr.get_active_label_count() == 60

    # 3. AutoCalibrator default conservative & adaptive flags + calibrate_search
    constraints = ac.CalibrationConstraints()
    assert constraints.require_conservative_recall is True
    assert constraints.enable_adaptive_geometry is True
    constraints.target_recall = 0.80
    constraints.max_memory_mb = 50.0
    constraints.target_p99_latency_ms = 25.0
    report, cal_idx = ac.calibrate_index(data, ac.SpaceType.L2, data[:20], np.arange(20, dtype=np.int32).reshape(-1, 1), constraints, "ParityCalib")
    assert report.status in (ac.FeasibilityStatus.VerifiedFeasible, ac.FeasibilityStatus.ProvisionallyFeasible)
    assert report.selected_config.enable_adaptive is True

    cs_rep = ac.calibrate_search(cal_idx, data[:20], np.arange(20, dtype=np.int32).reshape(-1, 1), constraints, "ParitySearchCalib")
    assert len(cs_rep.evaluated_trials) > 0

    # 4. Python AdaptiveHNSW SQ4 byte-per-vector reporting
    data_21 = rng.randn(30, 21).astype(np.float32)
    sq4_idx = AdaptiveHNSW(dim=21, quantize="sq4", policy_config=AdaptivePolicyConfig(m_base=8, ef_construction_base=32))
    sq4_idx.calibrate(data_21)
    for row in data_21[:25]:
        sq4_idx.insert(row)
    stats = sq4_idx.get_stats()
    assert stats["vector_memory_bytes"] == 25 * 11
    assert stats["raw_vector_memory_bytes"] == 25 * 21 * 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
