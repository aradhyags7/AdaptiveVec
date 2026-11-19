#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <set>

using namespace adaptivevec;

std::vector<float> load_fvecs(const std::string& path, size_t max_vectors, size_t& dim, size_t& count) {
    std::ifstream in(path, std::ios::binary);
    int d = 0; in.read((char*)&d, sizeof(int)); dim = d;
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t total_vectors = file_size / (sizeof(int) + dim * sizeof(float));
    count = std::min(total_vectors, max_vectors);
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

double eval_recall10(AdaptiveHNSWIndex& index, const std::vector<float>& queries, size_t n_queries, size_t dim,
                     const std::vector<std::vector<uint32_t>>& gt, int ef) {
    size_t hits = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        auto res = index.search(&queries[q * dim], 10, ef);
        std::set<uint32_t> gt_set(gt[q].begin(), gt[q].begin() + std::min((size_t)10, gt[q].size()));
        for (int i = 0; i < std::min((int)res.size(), 10); ++i) {
            if (gt_set.count(res[i].id)) hits++;
        }
    }
    return (double)hits / (n_queries * 10);
}

int main() {
    size_t dim = 0, n_base = 0, n_queries = 0;
    auto base_data = load_fvecs("data/transformer_384d/transformer_384d_base.fvecs", 60000, dim, n_base);
    size_t q_dim = 0;
    auto query_data = load_fvecs("data/transformer_384d/transformer_384d_query.fvecs", 1000, q_dim, n_queries);
    auto gt = load_ivecs("data/transformer_384d/transformer_384d_groundtruth.ivecs", n_queries);

    // Build master index with FP32 vectors (enable_sq8 = false, enable_sq4 = false)
    PolicyConfig cfg;
    cfg.m_base = 16;
    cfg.ef_c_base = 64;
    cfg.enable_dynamic_m_efc = true;
    cfg.enable_layer_scaling = true;
    cfg.enable_hubness_regulation = true;
    cfg.hubness_mu = 0.003f;
    cfg.enable_bridge_aware = false;
    cfg.enable_ada_ef = false; // test clean search first
    cfg.enable_sq8 = false;
    cfg.enable_sq4 = false;
    cfg.keep_raw_vectors = true;

    std::cout << "[*] Building FP32 Master Index..." << std::endl;
    AdaptiveHNSWIndex master(dim, SpaceType::L2, true, cfg);
    master.batch_insert(base_data.data(), n_base, 12);
    std::cout << "  Master Edges: " << master.get_total_edges() << std::endl;

    for (int ef : {16, 32, 64, 128}) {
        double r10 = eval_recall10(master, query_data, n_queries, dim, gt, ef);
        std::cout << "  FP32 Master ef=" << ef << " Recall@10: " << (r10 * 100.0) << "%" << std::endl;
    }

    // Now test Uniform FP32, Uniform SQ8, Uniform SQ4, and Mixed via allocate_mixed_precision
    std::vector<std::pair<std::string, AllocationStrategy>> tests = {
        {"1_Canonical_FP32", AllocationStrategy::UniformFP32},
        {"2_Uniform_SQ8", AllocationStrategy::UniformSQ8},
        {"3_Uniform_SQ4", AllocationStrategy::UniformSQ4},
        {"8_CombinedSensitivity_Mixed", AllocationStrategy::CombinedSensitivity}
    };

    for (auto& [name, strat] : tests) {
        auto clone = master.clone();
        clone->policy.keep_raw_vectors = false;
        double budget = (strat == AllocationStrategy::UniformFP32) ? 1.0 :
                        ((strat == AllocationStrategy::UniformSQ8) ? 0.25 :
                        ((strat == AllocationStrategy::UniformSQ4) ? 0.125 : 0.40));
        auto pdist = clone->allocate_mixed_precision(strat, budget);
        std::cout << "\n--- Testing " << name << " (FP32=" << pdist.n_fp32 << ", SQ8=" << pdist.n_sq8 << ", SQ4=" << pdist.n_sq4 << ") ---" << std::endl;
        for (int ef : {16, 32, 64, 128}) {
            double r10 = eval_recall10(*clone, query_data, n_queries, dim, gt, ef);
            std::cout << "  ef=" << ef << " Recall@10: " << (r10 * 100.0) << "%" << std::endl;
        }
    }

    return 0;
}
