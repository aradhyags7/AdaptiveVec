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

    struct Case {
        std::string name;
        bool is_adaptive;
        bool dyn_m_efc;
        bool layer_scaling;
        bool hubness;
        bool ada_ef;
    };

    std::vector<Case> cases = {
        {"1. Canonical (all false)", false, false, false, false, false},
        {"2. is_adaptive=true, but all policy flags false", true, false, false, false, false},
        {"3. is_adaptive=true + dyn_m_efc", true, true, false, false, false},
        {"4. is_adaptive=true + layer_scaling", true, false, true, false, false},
        {"5. is_adaptive=true + hubness (mu=0.003)", true, false, false, true, false},
        {"6. Canonical graph, but search with ada_ef=true", false, false, false, false, true},
        {"7. Full adaptive except ada_ef (ada_ef=false)", true, true, true, true, false},
        {"8. Full adaptive with ada_ef (ada_ef=true)", true, true, true, true, true}
    };

    for (const auto& c : cases) {
        PolicyConfig p;
        p.enable_dynamic_m_efc = c.dyn_m_efc;
        p.enable_layer_scaling = c.layer_scaling;
        p.enable_hubness_regulation = c.hubness;
        p.hubness_mu = 0.003f;
        p.enable_ada_ef = c.ada_ef;
        p.enable_sq8 = false;
        p.enable_sq4 = false;

        AdaptiveHNSWIndex idx(dim, SpaceType::L2, c.is_adaptive, p);
        idx.batch_insert(base_data.data(), n_base, 12);
        double r10_64 = eval_recall10(idx, query_data, n_queries, dim, gt, 64);
        std::cout << c.name << " => Edges: " << idx.get_total_edges() 
                  << " | R@10 (ef=64): " << (r10_64 * 100.0) << "%" << std::endl;
    }
    return 0;
}
