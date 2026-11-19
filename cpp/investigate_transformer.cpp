#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <set>

using namespace adaptivevec;

std::vector<float> load_fvecs(const std::string& path, size_t max_vectors, size_t& dim, size_t& count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[-] Failed to open fvecs: " << path << std::endl;
        return {};
    }
    int d = 0;
    in.read((char*)&d, sizeof(int));
    dim = d;
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t record_size = sizeof(int) + dim * sizeof(float);
    size_t total_vectors = file_size / record_size;
    count = std::min(total_vectors, max_vectors);

    std::vector<float> data(count * dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < count; ++i) {
        int cur_d = 0;
        in.read((char*)&cur_d, sizeof(int));
        in.read((char*)&data[i * dim], dim * sizeof(float));
    }
    return data;
}

std::vector<std::vector<uint32_t>> load_ivecs(const std::string& path, size_t max_queries) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[-] Failed to open ivecs: " << path << std::endl;
        return {};
    }
    std::vector<std::vector<uint32_t>> gt;
    size_t q = 0;
    while (in.peek() != EOF && q < max_queries) {
        int k = 0;
        in.read((char*)&k, sizeof(int));
        if (k <= 0 || k > 10000) break;
        std::vector<uint32_t> nns(k);
        in.read((char*)nns.data(), k * sizeof(uint32_t));
        gt.push_back(nns);
        q++;
    }
    return gt;
}

double eval_recall(AdaptiveHNSWIndex& index, const std::vector<float>& queries, size_t n_queries, size_t dim,
                   const std::vector<std::vector<uint32_t>>& gt, int k, int ef) {
    size_t total_hits = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        auto res = index.search(&queries[q * dim], k, ef);
        std::set<uint32_t> gt_set(gt[q].begin(), gt[q].begin() + std::min((size_t)k, gt[q].size()));
        for (int i = 0; i < std::min((int)res.size(), k); ++i) {
            if (gt_set.count(res[i].id)) total_hits++;
        }
    }
    return (double)total_hits / (n_queries * k);
}

int main() {
    size_t dim = 0, n_base = 0, n_queries = 0;
    auto base_data = load_fvecs("data/transformer_384d/transformer_384d_base.fvecs", 60000, dim, n_base);
    size_t q_dim = 0;
    auto query_data = load_fvecs("data/transformer_384d/transformer_384d_query.fvecs", 1000, q_dim, n_queries);
    auto gt = load_ivecs("data/transformer_384d/transformer_384d_groundtruth.ivecs", n_queries);
    std::cout << "[*] Loaded Transformer-384D: " << n_base << " base, " << n_queries << " queries, dim: " << dim << std::endl;

    // Test 1: Phase 1 Canonical HNSW configuration (is_adaptive = false)
    std::cout << "\n--- Test 1: Canonical HNSW (is_adaptive = false) ---" << std::endl;
    {
        PolicyConfig p;
        p.enable_dynamic_m_efc = false;
        p.enable_layer_scaling = false;
        p.enable_hubness_regulation = false;
        p.enable_ada_ef = false;
        p.enable_bridge_aware = false;
        p.enable_sq8 = false;
        p.enable_sq4 = false;

        AdaptiveHNSWIndex canonical_index(dim, SpaceType::L2, false, p);
        canonical_index.batch_insert(base_data.data(), n_base, 12);
        std::cout << "  Edges: " << canonical_index.get_total_edges() << std::endl;

        for (int ef : {16, 32, 64, 128}) {
            double r10 = eval_recall(canonical_index, query_data, n_queries, dim, gt, 10, ef);
            std::cout << "  ef=" << ef << " Recall@10: " << (r10 * 100.0) << "%" << std::endl;
        }
    }

    // Test 2: Phase 2 master_index configuration (is_adaptive = true, cfg from phase2_mixed_benchmark.cpp)
    std::cout << "\n--- Test 2: Phase 2 master_index configuration ---" << std::endl;
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.ef_c_base = 64;
        cfg.enable_dynamic_m_efc = true;
        cfg.enable_layer_scaling = true;
        cfg.enable_hubness_regulation = true;
        cfg.hubness_mu = 0.003f;
        cfg.enable_bridge_aware = false;
        cfg.enable_ada_ef = true;
        cfg.enable_sq8 = true;
        cfg.enable_sq4 = true;
        cfg.keep_raw_vectors = true;

        AdaptiveHNSWIndex p2_index(dim, SpaceType::L2, true, cfg);
        p2_index.batch_insert(base_data.data(), n_base, 12);
        std::cout << "  Edges: " << p2_index.get_total_edges() << std::endl;

        std::cout << "  [Raw index before allocate_mixed_precision]" << std::endl;
        for (int ef : {16, 32, 64, 128}) {
            double r10 = eval_recall(p2_index, query_data, n_queries, dim, gt, 10, ef);
            std::cout << "  ef=" << ef << " Recall@10 (k=10): " << (r10 * 100.0) << "%" << std::endl;
        }

        std::cout << "  [With k=100 as in phase2_mixed_benchmark.cpp]" << std::endl;
        for (int ef : {16, 32, 64, 128}) {
            double r10 = eval_recall(p2_index, query_data, n_queries, dim, gt, 100, ef);
            std::cout << "  ef=" << ef << " Recall@100 (k=100): " << (r10 * 100.0) << "%" << std::endl;
        }

        std::cout << "  [Clone and allocate_mixed_precision: UniformFP32]" << std::endl;
        auto clone_fp32 = p2_index.clone();
        clone_fp32->policy.keep_raw_vectors = false;
        clone_fp32->allocate_mixed_precision(AllocationStrategy::UniformFP32, 1.0);
        for (int ef : {16, 32, 64, 128}) {
            double r10 = eval_recall(*clone_fp32, query_data, n_queries, dim, gt, 10, ef);
            std::cout << "  ef=" << ef << " Recall@10: " << (r10 * 100.0) << "%" << std::endl;
        }
        for (int ef : {16, 32, 64, 128}) {
            double r100 = eval_recall(*clone_fp32, query_data, n_queries, dim, gt, 100, ef);
            std::cout << "  ef=" << ef << " Recall@100: " << (r100 * 100.0) << "%" << std::endl;
        }
    }

    return 0;
}
