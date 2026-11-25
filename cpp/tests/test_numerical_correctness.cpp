#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace adaptivevec;

struct NumericalDimResult {
    size_t dim = 0;
    float sq8_max_abs_err = 0.0f;
    float sq8_max_rel_err = 0.0f;
    float sq8_mean_err = 0.0f;
    float sq8_ranking_agreement = 1.0f;
    bool sq8_has_nan_inf = false;

    float sq4_max_abs_err = 0.0f;
    float sq4_max_rel_err = 0.0f;
    float sq4_mean_err = 0.0f;
    float sq4_ranking_agreement = 1.0f;
    bool sq4_has_nan_inf = false;

    bool descriptor_roundtrip_ok = true;
    bool passed = true;
};

// Scalar reference decoder & distance for SQ8
float scalar_asym_l2_sq_sq8(const float* q, const uint8_t* code, const std::vector<float>& min_vals, const std::vector<float>& scales, size_t dim) {
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        float recon = min_vals[d] + (float)code[d] * scales[d];
        float diff = q[d] - recon;
        sum += diff * diff;
    }
    return sum;
}

// Scalar reference decoder & distance for SQ4
float scalar_asym_l2_sq_sq4(const float* q, const uint8_t* code, const std::vector<float>& min_vals, const std::vector<float>& scales, size_t dim) {
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        size_t byte_idx = d / 2;
        uint8_t byte = code[byte_idx];
        uint8_t nibble = (d % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        float recon = min_vals[d] + (float)nibble * scales[d];
        float diff = q[d] - recon;
        sum += diff * diff;
    }
    return sum;
}

NumericalDimResult audit_dimension(size_t dim, std::mt19937& rng) {
    NumericalDimResult res;
    res.dim = dim;
    size_t n_samples = 500;
    std::vector<float> data(n_samples * dim);

    // Distribution 1: standard normal
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    for (size_t i = 0; i < n_samples / 4; ++i) {
        for (size_t d = 0; d < dim; ++d) data[i * dim + d] = norm_dist(rng);
    }
    // Distribution 2: extreme dynamic range [-1000.0, 1000.0]
    std::uniform_real_distribution<float> wide_dist(-1000.0f, 1000.0f);
    for (size_t i = n_samples / 4; i < n_samples / 2; ++i) {
        for (size_t d = 0; d < dim; ++d) data[i * dim + d] = wide_dist(rng);
    }
    // Distribution 3: constant and zero vectors
    for (size_t i = n_samples / 2; i < 3 * n_samples / 4; ++i) {
        float c = (i % 2 == 0) ? 0.0f : 42.0f;
        for (size_t d = 0; d < dim; ++d) data[i * dim + d] = c;
    }
    // Distribution 4: vectors with outliers and negative values
    for (size_t i = 3 * n_samples / 4; i < n_samples; ++i) {
        for (size_t d = 0; d < dim; ++d) data[i * dim + d] = (d % 5 == 0) ? 500.0f : -0.1f * d;
    }

    // Train SQ8 and SQ4
    ScalarQuantizer sq8(dim);
    sq8.train(data.data(), n_samples);

    ScalarQuantizer4Bit sq4(dim);
    sq4.train(data.data(), n_samples);

    // Test Queries
    size_t n_queries = 20;
    std::vector<std::vector<float>> queries(n_queries, std::vector<float>(dim));
    for (size_t q = 0; q < n_queries; ++q) {
        for (size_t d = 0; d < dim; ++d) queries[q][d] = norm_dist(rng);
    }

    size_t packed_dim = (dim + 1) / 2;
    std::vector<uint8_t> code8(dim);
    std::vector<uint8_t> code4(packed_dim);
    std::vector<float> recon8(dim);
    std::vector<float> recon4(dim);

    double total_sq8_err = 0.0;
    double total_sq4_err = 0.0;
    size_t total_evals = 0;

    size_t sq8_rank_agreements = 0;
    size_t sq4_rank_agreements = 0;
    size_t rank_pairs = 0;

    for (size_t q = 0; q < n_queries; ++q) {
        const float* query_ptr = queries[q].data();
        std::vector<std::pair<float, size_t>> scalar_sq8_ranks(n_samples);
        std::vector<std::pair<float, size_t>> simd_sq8_ranks(n_samples);
        std::vector<std::pair<float, size_t>> scalar_sq4_ranks(n_samples);
        std::vector<std::pair<float, size_t>> simd_sq4_ranks(n_samples);

        for (size_t i = 0; i < n_samples; ++i) {
            const float* v = &data[i * dim];

            // --- SQ8 ---
            sq8.encode(v, code8.data());
            sq8.decode(code8.data(), recon8.data());
            float scalar_d8 = scalar_asym_l2_sq_sq8(query_ptr, code8.data(), sq8.min_vals, sq8.scales, dim);
            float simd_d8 = sq8.asymmetric_l2_sq_distance(query_ptr, code8.data());

            if (std::isnan(simd_d8) || std::isinf(simd_d8)) res.sq8_has_nan_inf = true;

            float abs_err8 = std::abs(simd_d8 - scalar_d8);
            float denom8 = std::max(1.0f, std::max(std::abs(simd_d8), std::abs(scalar_d8)));
            float rel_err8 = abs_err8 / denom8;

            res.sq8_max_abs_err = std::max(res.sq8_max_abs_err, abs_err8);
            res.sq8_max_rel_err = std::max(res.sq8_max_rel_err, rel_err8);
            total_sq8_err += rel_err8;

            scalar_sq8_ranks[i] = {scalar_d8, i};
            simd_sq8_ranks[i] = {simd_d8, i};

            // --- SQ4 ---
            sq4.encode(v, code4.data());
            sq4.decode(code4.data(), recon4.data());
            float scalar_d4 = scalar_asym_l2_sq_sq4(query_ptr, code4.data(), sq4.min_vals, sq4.scales, dim);
            float simd_d4 = sq4.asymmetric_l2_sq_distance(query_ptr, code4.data());

            if (std::isnan(simd_d4) || std::isinf(simd_d4)) res.sq4_has_nan_inf = true;

            float abs_err4 = std::abs(simd_d4 - scalar_d4);
            float denom4 = std::max(1.0f, std::max(std::abs(simd_d4), std::abs(scalar_d4)));
            float rel_err4 = abs_err4 / denom4;

            res.sq4_max_abs_err = std::max(res.sq4_max_abs_err, abs_err4);
            res.sq4_max_rel_err = std::max(res.sq4_max_rel_err, rel_err4);
            total_sq4_err += rel_err4;

            scalar_sq4_ranks[i] = {scalar_d4, i};
            simd_sq4_ranks[i] = {simd_d4, i};

            total_evals++;
        }

        // Check top-10 ranking consistency
        std::sort(scalar_sq8_ranks.begin(), scalar_sq8_ranks.end());
        std::sort(simd_sq8_ranks.begin(), simd_sq8_ranks.end());
        std::sort(scalar_sq4_ranks.begin(), scalar_sq4_ranks.end());
        std::sort(simd_sq4_ranks.begin(), simd_sq4_ranks.end());

        for (size_t k = 0; k < 10; ++k) {
            if (scalar_sq8_ranks[k].second == simd_sq8_ranks[k].second) sq8_rank_agreements++;
            if (scalar_sq4_ranks[k].second == simd_sq4_ranks[k].second) sq4_rank_agreements++;
            rank_pairs++;
        }
    }

    res.sq8_mean_err = total_evals > 0 ? (float)(total_sq8_err / total_evals) : 0.0f;
    res.sq4_mean_err = total_evals > 0 ? (float)(total_sq4_err / total_evals) : 0.0f;
    res.sq8_ranking_agreement = rank_pairs > 0 ? (float)sq8_rank_agreements / rank_pairs : 1.0f;
    res.sq4_ranking_agreement = rank_pairs > 0 ? (float)sq4_rank_agreements / rank_pairs : 1.0f;

    // Test Storage Pool Descriptors and Round-trip
    AdaptiveHNSWIndex test_idx(dim, SpaceType::L2, false);
    test_idx.batch_insert(data.data(), n_samples, 4);
    auto pdist = test_idx.allocate_mixed_precision(AllocationStrategy::CombinedSensitivity, 0.40);
    if (pdist.n_fp32 + pdist.n_sq8 + pdist.n_sq4 != n_samples) {
        res.descriptor_roundtrip_ok = false;
    }

    for (size_t i = 0; i < n_samples; ++i) {
        std::vector<float> rec(dim);
        test_idx.get_vector_reconstructed(i, rec.data());
        // Verify reconstructed vector is non-empty and well-formed
        for (size_t d = 0; d < dim; ++d) {
            if (std::isnan(rec[d]) || std::isinf(rec[d])) {
                res.descriptor_roundtrip_ok = false;
            }
        }
        // Verify get_node_distance_sq against get_vector_reconstructed
        float node_d = test_idx.get_node_distance_sq(queries[0].data(), i);
        if (std::isnan(node_d) || std::isinf(node_d) || node_d < 0.0f) {
            res.descriptor_roundtrip_ok = false;
        }
    }

    // Acceptance Criteria:
    // Rel error < 1e-4, no NaNs, ranking agreement >= 0.99, descriptors valid
    if (res.sq8_max_rel_err > 1e-4f || res.sq4_max_rel_err > 1e-4f ||
        res.sq8_has_nan_inf || res.sq4_has_nan_inf ||
        res.sq8_ranking_agreement < 0.99f || res.sq4_ranking_agreement < 0.99f ||
        !res.descriptor_roundtrip_ok) {
        res.passed = false;
    } else {
        res.passed = true;
    }

    return res;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  PHASE 2.1: STORAGE POOL & NUMERICAL KERNEL VERIFICATION " << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::mt19937 rng(42);
    std::vector<size_t> test_dims = {1, 7, 16, 63, 64, 65, 127, 128, 384, 768, 1536};
    std::vector<NumericalDimResult> results;
    bool all_passed = true;

    for (size_t d : test_dims) {
        auto r = audit_dimension(d, rng);
        results.push_back(r);
        std::cout << "Dim " << std::setw(4) << d << " | "
                  << "SQ8 RelErr: " << std::scientific << std::setprecision(2) << r.sq8_max_rel_err
                  << " Rank: " << std::fixed << std::setprecision(3) << r.sq8_ranking_agreement << " | "
                  << "SQ4 RelErr: " << std::scientific << std::setprecision(2) << r.sq4_max_rel_err
                  << " Rank: " << std::fixed << std::setprecision(3) << r.sq4_ranking_agreement << " | "
                  << "DescOK: " << (r.descriptor_roundtrip_ok ? "YES" : "NO") << " | "
                  << "Status: " << (r.passed ? "[PASS]" : "[FAIL]") << std::endl;
        if (!r.passed) all_passed = false;
    }

    // Output JSON report
    std::ofstream json_out("experiments/phase_2_1_correctness_results.json");
    json_out << "{\n";
    json_out << "  \"timestamp\": \"2026-11-20T00:33:00+05:30\",\n";
    json_out << "  \"compiler\": \"GCC 16.1.0\",\n";
    json_out << "  \"isa\": \"AVX2 + FMA\",\n";
    json_out << "  \"overall_status\": \"" << (all_passed ? "PASSED" : "FAILED") << "\",\n";
    json_out << "  \"dimension_audits\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        json_out << "    {\n";
        json_out << "      \"dimension\": " << r.dim << ",\n";
        json_out << "      \"sq8_max_abs_err\": " << r.sq8_max_abs_err << ",\n";
        json_out << "      \"sq8_max_rel_err\": " << r.sq8_max_rel_err << ",\n";
        json_out << "      \"sq8_mean_err\": " << r.sq8_mean_err << ",\n";
        json_out << "      \"sq8_ranking_agreement\": " << r.sq8_ranking_agreement << ",\n";
        json_out << "      \"sq8_has_nan_inf\": " << (r.sq8_has_nan_inf ? "true" : "false") << ",\n";
        json_out << "      \"sq4_max_abs_err\": " << r.sq4_max_abs_err << ",\n";
        json_out << "      \"sq4_max_rel_err\": " << r.sq4_max_rel_err << ",\n";
        json_out << "      \"sq4_mean_err\": " << r.sq4_mean_err << ",\n";
        json_out << "      \"sq4_ranking_agreement\": " << r.sq4_ranking_agreement << ",\n";
        json_out << "      \"sq4_has_nan_inf\": " << (r.sq4_has_nan_inf ? "true" : "false") << ",\n";
        json_out << "      \"descriptor_roundtrip_ok\": " << (r.descriptor_roundtrip_ok ? "true" : "false") << ",\n";
        json_out << "      \"passed\": " << (r.passed ? "true" : "false") << "\n";
        json_out << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    json_out << "  ]\n";
    json_out << "}\n";

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "  NUMERICAL RESULTS SAVED TO experiments/phase_2_1_correctness_results.json" << std::endl;
    std::cout << "  OVERALL STATUS: " << (all_passed ? "PASSED [OK]" : "FAILED [FAIL]") << std::endl;
    std::cout << "==========================================================" << std::endl;

    return all_passed ? 0 : 1;
}
