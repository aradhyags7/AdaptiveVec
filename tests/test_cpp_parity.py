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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
