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
    count = std::min(file_size / (sizeof(int) + dim * sizeof(float)), max_vectors);
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

int main() {
    size_t dim = 0, n_base = 0, q_dim = 0, n_queries = 0;
    auto base_data = load_fvecs("data/transformer_384d/transformer_384d_base.fvecs", 57893, dim, n_base);
    auto query_data = load_fvecs("data/transformer_384d/transformer_384d_query.fvecs", 1000, q_dim, n_queries);
    auto gt = load_ivecs("data/transformer_384d/transformer_384d_groundtruth.ivecs", n_queries);

    PolicyConfig cfg;
    cfg.m_base = 16;
    cfg.ef_c_base = 64;
    cfg.enable_dynamic_m_efc = true;
    cfg.enable_layer_scaling = true;
    cfg.enable_hubness_regulation = true;
    cfg.hubness_mu = 0.003f;
    cfg.keep_raw_vectors = true;

    AdaptiveHNSWIndex master(dim, SpaceType::L2, true, cfg);
    master.batch_insert(base_data.data(), n_base, 12);

    auto idx_rand = master.clone();
    idx_rand->policy.keep_raw_vectors = false;
    idx_rand->allocate_mixed_precision(AllocationStrategy::Random, 0.40);

    auto idx_rout = master.clone();
    idx_rout->policy.keep_raw_vectors = false;
    idx_rout->allocate_mixed_precision(AllocationStrategy::RoutingImportance, 0.40);

    // Compare retrieved results across all 1000 queries at ef=64
    size_t id_matches = 0;
    size_t total_ids = n_queries * 10;
    size_t hits_rand = 0, hits_rout = 0;

    for (size_t q = 0; q < n_queries; ++q) {
        auto res_rand = idx_rand->search(&query_data[q * dim], 10, 64);
        auto res_rout = idx_rout->search(&query_data[q * dim], 10, 64);

        std::set<uint32_t> gt_set(gt[q].begin(), gt[q].begin() + 10);
        for (int i = 0; i < 10; ++i) {
            if (gt_set.count(res_rand[i].id)) hits_rand++;
            if (gt_set.count(res_rout[i].id)) hits_rout++;
            if (res_rand[i].id == res_rout[i].id) id_matches++;
        }
    }

    std::cout << "Transformer-384D Random vs Routing Analysis at ef=64:\n"
              << "  Random Recall@10:  " << (double)hits_rand / total_ids * 100.0 << "%\n"
              << "  Routing Recall@10: " << (double)hits_rout / total_ids * 100.0 << "%\n"
              << "  Exact ID Agreement between Random and Routing: " << (double)id_matches / total_ids * 100.0 << "%\n";
    return 0;
}
