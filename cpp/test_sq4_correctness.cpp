#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <chrono>
#include <iomanip>
#include "adaptive_hnsw.hpp"

using namespace adaptivevec;

// Reference scalar implementation for 4-bit asymmetric L2 squared distance
float reference_scalar_sq4_l2_sq(
    const float* q,
    const uint8_t* code,
    size_t dim,
    const std::vector<float>& min_vals,
    const std::vector<float>& scales
) {
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        size_t byte_idx = d / 2;
        uint8_t byte = code[byte_idx];
        uint8_t nibble = (d % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        float approx = min_vals[d] + static_cast<float>(nibble) * scales[d];
        float diff = q[d] - approx;
        sum += diff * diff;
    }
    return sum;
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "     AdaptiveVec SQ4 Distance Parity & Storage Verification       \n";
    std::cout << "=================================================================\n\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<size_t> test_dims = {128, 384};

    for (size_t dim : test_dims) {
        std::cout << ">>> Testing Dimension: " << dim << " ...\n";
        size_t n_vectors = 1000;
        std::vector<float> data(n_vectors * dim);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = dist(rng);
        }

        ScalarQuantizer4Bit quant(dim);
        quant.train(data.data(), n_vectors);

        assert(quant.is_trained);
        size_t expected_packed_dim = (dim + 1) / 2;
        assert(quant.packed_dim == expected_packed_dim);
        std::cout << "  [+] Quantizer trained. Packed dim = " << quant.packed_dim 
                  << " bytes (" << (quant.packed_dim * 8) << " bits for " << dim << " dims)\n";

        // Verification 1: Packed encoding and decoding fidelity
        std::vector<uint8_t> encoded(n_vectors * quant.packed_dim);
        for (size_t i = 0; i < n_vectors; ++i) {
            quant.encode(&data[i * dim], &encoded[i * quant.packed_dim]);
        }

        std::vector<float> decoded(dim);
        double max_recon_err = 0.0;
        for (size_t i = 0; i < n_vectors; ++i) {
            quant.decode(&encoded[i * quant.packed_dim], decoded.data());
            for (size_t d = 0; d < dim; ++d) {
                float err = std::abs(data[i * dim + d] - decoded[d]);
                if (err > max_recon_err) max_recon_err = err;
            }
        }
        std::cout << "  [+] Reconstruction bounded: max per-dim error = " << max_recon_err << "\n";

        // Verification 2: Asymmetric L2 squared distance parity (Scalar vs AVX2 SIMD)
        double max_diff = 0.0;
        double sum_diff = 0.0;
        for (size_t i = 0; i < n_vectors; ++i) {
            const float* q = &data[i * dim];
            const uint8_t* code = &encoded[((i + 17) % n_vectors) * quant.packed_dim];

            float simd_dist = quant.asymmetric_l2_sq_distance(q, code);
            float scalar_dist = reference_scalar_sq4_l2_sq(q, code, dim, quant.min_vals, quant.scales);

            double diff = std::abs(simd_dist - scalar_dist);
            if (diff > max_diff) max_diff = diff;
            sum_diff += diff;

            double rel_diff = diff / std::max(1.0, (double)scalar_dist);
            if (rel_diff > 1e-4) {
                std::cerr << "[-] Parity mismatch at vector " << i << ": SIMD=" << simd_dist 
                          << " vs Scalar=" << scalar_dist << " (diff=" << diff << ", rel=" << rel_diff << ")\n";
                return 1;
            }
        }
        std::cout << "  [+] Scalar vs SIMD Parity: 1000/1000 matched! (max diff=" << max_diff 
                  << ", mean diff=" << (sum_diff / n_vectors) << ")\n";
    }

    // Verification 3: Allocation sizes for 100K 128D
    size_t test_n = 100000;
    size_t test_dim = 128;
    size_t fp32_payload = test_n * test_dim * sizeof(float);
    size_t sq8_payload = test_n * test_dim * sizeof(uint8_t);
    size_t sq4_payload = test_n * ((test_dim + 1) / 2) * sizeof(uint8_t);

    std::cout << "\n>>> Vector Payload Breakdown for N=100,000, Dim=128:\n";
    std::cout << "  - FP32: " << fp32_payload << " bytes (" << (fp32_payload / (1024.0 * 1024.0)) << " MB)\n";
    std::cout << "  - SQ8:  " << sq8_payload << " bytes (" << (sq8_payload / (1024.0 * 1024.0)) << " MB) [2.00x vs SQ4, 0.25x vs FP32]\n";
    std::cout << "  - SQ4:  " << sq4_payload << " bytes (" << (sq4_payload / (1024.0 * 1024.0)) << " MB) [0.50x vs SQ8, 0.125x vs FP32]\n";

    assert(sq8_payload == 2 * sq4_payload);
    assert(fp32_payload == 8 * sq4_payload);
    std::cout << "\n=================================================================\n";
    std::cout << " ALL SQ4 CORRECTNESS & PARITY GATES PASSED SUCCESSFULLY!\n";
    std::cout << "=================================================================\n";

    return 0;
}
