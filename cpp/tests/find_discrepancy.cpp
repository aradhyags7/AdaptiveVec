#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <set>
#include <iomanip>

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
    auto query_data = load_fvecs("data/transformer_384d/transformer_384d_query.fvecs", 100, q_dim, n_queries);
    auto gt = load_ivecs("data/transformer_384d/transformer_384d_groundtruth.ivecs", n_queries);

    for (size_t q = 0; q < n_queries; ++q) {
        const float* q_vec = &query_data[q * dim];
        std::vector<std::pair<float, uint32_t>> all_dist(n_base);
        for (size_t i = 0; i < n_base; ++i) {
            all_dist[i] = {l2_sq_distance(q_vec, &base_data[i * dim], dim), (uint32_t)i};
        }
        std::sort(all_dist.begin(), all_dist.end());

        std::set<uint32_t> gt_set(gt[q].begin(), gt[q].begin() + 10);
        int hits = 0;
        for (int k = 0; k < 10; ++k) {
            if (gt_set.count(all_dist[k].second)) hits++;
        }
        if (hits < 10) {
            std::cout << "Query " << q << " Divergence: " << hits << "/10 matches\n";
            std::cout << "  Calculated 10th item (rank 9): ID=" << all_dist[9].second << " dist=" << std::setprecision(8) << all_dist[9].first << "\n";
            std::cout << "  Calculated 11th item (rank 10): ID=" << all_dist[10].second << " dist=" << std::setprecision(8) << all_dist[10].first << "\n";
            std::cout << "  GT 10th item: ID=" << gt[q][9] << " dist=" << l2_sq_distance(q_vec, &base_data[gt[q][9] * dim], dim) << "\n";
            std::cout << "  Distance difference: " << std::abs(all_dist[9].first - all_dist[10].first) << "\n";
        }
    }
    return 0;
}
