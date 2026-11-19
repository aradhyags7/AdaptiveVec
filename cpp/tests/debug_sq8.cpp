#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace adaptivevec;

int main() {
    size_t dim = 16;
    std::mt19937 rng(42);
    size_t n_samples = 200;
    std::vector<float> data(n_samples * dim);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t d = 0; d < dim; ++d) data[i * dim + d] = norm_dist(rng);
    }

    ScalarQuantizer sq8(dim);
    sq8.train(data.data(), n_samples);

    std::vector<float> query(dim);
    for (size_t d = 0; d < dim; ++d) query[d] = norm_dist(rng);

    std::vector<uint8_t> code(dim);
    std::vector<float> recon(dim);

    for (size_t i = 0; i < 5; ++i) {
        const float* v = &data[i * dim];
        sq8.encode(v, code.data());
        sq8.decode(code.data(), recon.data());

        float scalar_d = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float r = sq8.min_vals[d] + (float)code[d] * sq8.scales[d];
            float diff = query[d] - r;
            scalar_d += diff * diff;
        }

        float true_recon_d = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float diff = query[d] - recon[d];
            true_recon_d += diff * diff;
        }

        float simd_d = sq8.asymmetric_l2_sq_distance(query.data(), code.data());

        std::cout << "Sample " << i << ":\n"
                  << "  scalar_d     = " << scalar_d << "\n"
                  << "  true_recon_d = " << true_recon_d << "\n"
                  << "  simd_d       = " << simd_d << "\n"
                  << "  simd - scalar= " << (simd_d - scalar_d) << "\n";
    }
    return 0;
}
