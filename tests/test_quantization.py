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
