import pytest
import numpy as np
from adaptivevec import ScalarQuantizer, ScalarQuantizer4Bit, AdaptiveHNSW

def test_scalar_quantizer_8bit():
    np.random.seed(42)
    dim = 64
    vectors = np.random.randn(50, dim).astype(np.float32)
    
    sq8 = ScalarQuantizer(dim=dim)
    sq8.train(vectors)
    
    code = sq8.encode(vectors[0])
    assert code.shape == (dim,)
    assert code.dtype == np.uint8
    
    recon = sq8.decode(code)
    assert recon.shape == (dim,)
    
    # Check reconstruction error is bounded
    err = np.linalg.norm(vectors[0] - recon)
    assert err < 1.0
    
    dist = sq8.asymmetric_l2_distance(vectors[1], code)
    assert dist > 0.0

def test_scalar_quantizer_4bit():
    np.random.seed(42)
    dim = 64
    vectors = np.random.randn(50, dim).astype(np.float32)
    
    sq4 = ScalarQuantizer4Bit(dim=dim)
    sq4.train(vectors)
    assert sq4.packed_dim == 32
    
    code = sq4.encode(vectors[0])
    assert code.shape == (32,)
    assert code.dtype == np.uint8
    
    recon = sq4.decode(code)
    assert recon.shape == (dim,)
    
    # Check reconstruction error is reasonable for 4-bit (16 levels)
    err = np.linalg.norm(vectors[0] - recon)
    assert err < 2.5
    
    dist = sq4.asymmetric_l2_distance(vectors[1], code)
    assert dist > 0.0

def test_adaptive_hnsw_sq4_search():
    np.random.seed(42)
    dim = 16
    n = 30
    vectors = np.random.randn(n, dim).astype(np.float32)
    
    index = AdaptiveHNSW(dim=dim, quantize="sq4")
    index.calibrate(vectors[:10])
    for v in vectors:
        index.insert(v)
    
    assert len(index.quantized_data) == n
    assert index.quantized_data[0].shape == (8,)  # 16 dim packed into 8 bytes
    
    results = index.search(vectors[0], k=5, ef=16)
    assert len(results) > 0
    # Top result should be the query itself
    assert results[0][0] == 0

def test_mixed_precision_cpp_extension():
    import adaptivevec._adaptivevec_cpp as av
    np.random.seed(1337)
    dim = 32
    n = 200
    vectors = np.random.randn(n, dim).astype(np.float32)

    cfg = av.PolicyConfig()
    cfg.m_base = 16
    cfg.ef_c_base = 64
    cfg.enable_sq8 = True
    cfg.enable_sq4 = True

    index = av.AdaptiveHNSWIndex(dim, av.SpaceType.L2, True, cfg)
    index.batch_insert(vectors, 2)
    assert not index.is_mixed_precision_active()

    # Allocate mixed precision with combined sensitivity strategy
    pdist = index.allocate_mixed_precision(av.AllocationStrategy.CombinedSensitivity, 0.4)
    assert index.is_mixed_precision_active()
    assert pdist.n_fp32 + pdist.n_sq8 + pdist.n_sq4 == n
    assert pdist.savings_ratio > 0.25

    res = index.search(vectors[0], k=5, ef=32)
    assert len(res) > 0
    assert res[0].id == 0

    mb = index.get_memory_breakdown()
    assert mb.allocated_quantized_capacity_bytes > 0
    assert mb.total_owned_bytes > 0
