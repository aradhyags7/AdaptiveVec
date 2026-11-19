#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <set>
#include <algorithm>
#include <omp.h>

using namespace adaptivevec;

std::vector<float> load_fvecs(const std::string& path, size_t max_vectors, size_t& dim, size_t& count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    int d = 0; in.read((char*)&d, sizeof(int)); dim = d;
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t record_size = sizeof(int) + dim * sizeof(float);
    count = std::min(file_size / record_size, max_vectors);
    std::vector<float> data(count * dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < count; ++i) {
        int cur_d = 0; in.read((char*)&cur_d, sizeof(int));
        in.read((char*)&data[i * dim], dim * sizeof(float));
    }
    return data;
}

std::vector<std::vector<uint32_t>> load_ivecs(const std::string& path, size_t max_queries) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    std::vector<std::vector<uint32_t>> gt;
    size_t q = 0;
    while (in.peek() != EOF && q < max_queries) {
        int k = 0; in.read((char*)&k, sizeof(int));
        if (k <= 0 || k > 10000) break;
        std::vector<uint32_t> nns(k);
        in.read((char*)nns.data(), k * sizeof(uint32_t));
        gt.push_back(nns);
        q++;
    }
    return gt;
}

// System A: Exact FP32 Brute-Force Search
std::vector<std::vector<uint32_t>> brute_force_fp32(const float* base, size_t n_base, size_t dim,
                                                    const float* queries, size_t n_queries, int k) {
    std::vector<std::vector<uint32_t>> results(n_queries, std::vector<uint32_t>(k));
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t q = 0; q < n_queries; ++q) {
        const float* q_vec = queries + q * dim;
        std::vector<std::pair<float, uint32_t>> top_k(k, {1e30f, 0});
        for (size_t i = 0; i < n_base; ++i) {
            float dist = l2_sq_distance(q_vec, base + i * dim, dim);
            if (dist < top_k.back().first) {
                top_k.back() = {dist, (uint32_t)i};
                for (int j = k - 1; j > 0 && top_k[j].first < top_k[j - 1].first; --j) {
                    std::swap(top_k[j], top_k[j - 1]);
                }
            }
        }
        for (int j = 0; j < k; ++j) results[q][j] = top_k[j].second;
    }
    return results;
}

// System B: Exhaustive Quantized Search using Reconstructed Vectors
std::vector<std::vector<uint32_t>> brute_force_reconstructed(const AdaptiveHNSWIndex& index, size_t n_base, size_t dim,
                                                             const float* queries, size_t n_queries, int k) {
    std::vector<std::vector<uint32_t>> results(n_queries, std::vector<uint32_t>(k));
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t q = 0; q < n_queries; ++q) {
        const float* q_vec = queries + q * dim;
        std::vector<std::pair<float, uint32_t>> top_k(k, {1e30f, 0});
        std::vector<float> recon_vec(dim);
        for (size_t i = 0; i < n_base; ++i) {
            index.get_vector_reconstructed((tableint)i, recon_vec.data());
            float dist = l2_sq_distance(q_vec, recon_vec.data(), dim);
            if (dist < top_k.back().first) {
                top_k.back() = {dist, (uint32_t)i};
                for (int j = k - 1; j > 0 && top_k[j].first < top_k[j - 1].first; --j) {
                    std::swap(top_k[j], top_k[j - 1]);
                }
            }
        }
        for (int j = 0; j < k; ++j) results[q][j] = top_k[j].second;
    }
    return results;
}

// System C: Exhaustive Quantized Search using Production Asymmetric Kernels
std::vector<std::vector<uint32_t>> brute_force_asymmetric(const AdaptiveHNSWIndex& index, size_t n_base, size_t dim,
                                                          const float* queries, size_t n_queries, int k) {
    std::vector<std::vector<uint32_t>> results(n_queries, std::vector<uint32_t>(k));
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t q = 0; q < n_queries; ++q) {
        const float* q_vec = queries + q * dim;
        std::vector<std::pair<float, uint32_t>> top_k(k, {1e30f, 0});
        for (size_t i = 0; i < n_base; ++i) {
            float dist = index.get_node_distance_sq(q_vec, (tableint)i);
            if (dist < top_k.back().first) {
                top_k.back() = {dist, (uint32_t)i};
                for (int j = k - 1; j > 0 && top_k[j].first < top_k[j - 1].first; --j) {
                    std::swap(top_k[j], top_k[j - 1]);
                }
            }
        }
        for (int j = 0; j < k; ++j) results[q][j] = top_k[j].second;
    }
    return results;
}

// System D: HNSW Graph Traversal using Production Kernels
std::vector<std::vector<uint32_t>> graph_search_production(AdaptiveHNSWIndex& index, size_t dim,
                                                           const float* queries, size_t n_queries, int k, int ef) {
    std::vector<std::vector<uint32_t>> results(n_queries, std::vector<uint32_t>(k));
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t q = 0; q < n_queries; ++q) {
        auto res = index.search(queries + q * dim, k, ef);
        for (int j = 0; j < std::min((int)res.size(), k); ++j) {
            results[q][j] = res[j].id;
        }
    }
    return results;
}

double compute_overlap(const std::vector<std::vector<uint32_t>>& retrieved,
                       const std::vector<std::vector<uint32_t>>& reference,
                       int k) {
    size_t hits = 0;
    size_t total = retrieved.size() * k;
    for (size_t q = 0; q < retrieved.size(); ++q) {
        std::set<uint32_t> ref_set(reference[q].begin(), reference[q].begin() + std::min((size_t)k, reference[q].size()));
        for (int j = 0; j < std::min((int)retrieved[q].size(), k); ++j) {
            if (ref_set.count(retrieved[q][j])) hits++;
        }
    }
    return (double)hits / total;
}

void run_4way_decomposition(const std::string& name, const std::string& base_path, const std::string& query_path, const std::string& gt_path) {
    size_t dim = 0, n_base = 0, n_queries = 0;
    auto base_data = load_fvecs(base_path, 100000, dim, n_base);
    size_t q_dim = 0;
    auto query_data = load_fvecs(query_path, 100, q_dim, n_queries); // 100 test queries for exhaustive comparison
    auto external_gt = load_ivecs(gt_path, n_queries);

    std::cout << "\n========================================================" << std::endl;
    std::cout << "  4-WAY DECOMPOSITION AUDIT: " << name << " (N=" << n_base << ", Q=" << n_queries << ", D=" << dim << ")" << std::endl;
    std::cout << "========================================================" << std::endl;

    int k = 10;

    // --- SYSTEM A: Exact FP32 Brute Force ---
    auto t0 = std::chrono::high_resolution_clock::now();
    auto sys_A = brute_force_fp32(base_data.data(), n_base, dim, query_data.data(), n_queries, k);
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec_A = std::chrono::duration<double>(t1 - t0).count();
    double match_A_ext = compute_overlap(sys_A, external_gt, k);
    std::cout << "[System A] Exact FP32 Brute Force:" << std::endl;
    std::cout << "   Time: " << sec_A << " s | Match vs External Ground Truth: " << (match_A_ext * 100.0) << "%" << std::endl;

    // Build Master Graph with FP32
    PolicyConfig cfg;
    cfg.m_base = 16;
    cfg.ef_c_base = 64;
    cfg.enable_dynamic_m_efc = true;
    cfg.enable_layer_scaling = true;
    cfg.enable_hubness_regulation = true;
    cfg.hubness_mu = 0.003f;
    cfg.enable_bridge_aware = false;
    cfg.enable_ada_ef = false;
    cfg.enable_sq8 = false;
    cfg.enable_sq4 = false;
    cfg.keep_raw_vectors = true;

    AdaptiveHNSWIndex master(dim, SpaceType::L2, true, cfg);
    master.batch_insert(base_data.data(), n_base, 12);

    // Evaluate SQ8, SQ4, and Mixed Precision
    std::vector<std::pair<std::string, AllocationStrategy>> configurations = {
        {"Uniform SQ8", AllocationStrategy::UniformSQ8},
        {"Uniform SQ4", AllocationStrategy::UniformSQ4},
        {"Combined Sensitivity Mixed (40% Budget)", AllocationStrategy::CombinedSensitivity}
    };

    for (auto& [config_name, strat] : configurations) {
        std::cout << "\n--- Configuration: " << config_name << " ---" << std::endl;
        auto clone = master.clone();
        clone->policy.keep_raw_vectors = false; // Pure quantized storage
        double budget = (strat == AllocationStrategy::UniformSQ8) ? 0.25 :
                        ((strat == AllocationStrategy::UniformSQ4) ? 0.125 : 0.40);
        clone->allocate_mixed_precision(strat, budget);

        // System B: Exhaustive Quantized Search with Reconstructed Vectors
        auto sys_B = brute_force_reconstructed(*clone, n_base, dim, query_data.data(), n_queries, k);
        double match_B_vs_A = compute_overlap(sys_B, sys_A, k);

        // System C: Exhaustive Quantized Search with Asymmetric SIMD Kernels
        auto sys_C = brute_force_asymmetric(*clone, n_base, dim, query_data.data(), n_queries, k);
        double match_C_vs_B = compute_overlap(sys_C, sys_B, k);
        double match_C_vs_A = compute_overlap(sys_C, sys_A, k);

        // System D: Graph Traversal with Production Kernels (ef=64)
        auto sys_D_64 = graph_search_production(*clone, dim, query_data.data(), n_queries, k, 64);
        double match_D_vs_C = compute_overlap(sys_D_64, sys_C, k);
        double match_D_vs_A = compute_overlap(sys_D_64, sys_A, k);

        std::cout << "  [System B vs A] Quantization Distortion (Recon vs FP32): " << (match_B_vs_A * 100.0) << "%" << std::endl;
        std::cout << "  [System C vs B] SIMD Kernel Fidelity (Asym vs Recon):   " << (match_C_vs_B * 100.0) << "%" << std::endl;
        std::cout << "  [System C vs A] Quantized Exhaustive Recall vs FP32:     " << (match_C_vs_A * 100.0) << "%" << std::endl;
        std::cout << "  [System D vs C] Graph Routing Approximation (D vs C):    " << (match_D_vs_C * 100.0) << "%" << std::endl;
        std::cout << "  [System D vs A] Final End-to-End Recall (D vs FP32):     " << (match_D_vs_A * 100.0) << "%" << std::endl;
    }
}

int main() {
    run_4way_decomposition("Transformer-384D",
                           "data/transformer_384d/transformer_384d_base.fvecs",
                           "data/transformer_384d/transformer_384d_query.fvecs",
                           "data/transformer_384d/transformer_384d_groundtruth.ivecs");

    run_4way_decomposition("SIFT-100K",
                           "data/sift_subset_100k.fvecs",
                           "data/sift_query.fvecs",
                           "data/sift_subset_groundtruth.ivecs");
    return 0;
}
