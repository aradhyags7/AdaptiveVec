#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <map>
#include <set>

using namespace adaptivevec;

std::vector<float> load_fvecs(const std::string& path, size_t max_vectors, size_t& dim, size_t& count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
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

std::vector<std::vector<uint32_t>> load_ivecs(const std::string& path, size_t max_records) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    std::vector<std::vector<uint32_t>> records;
    while (in && records.size() < max_records) {
        int k = 0;
        in.read((char*)&k, sizeof(int));
        if (!in) break;
        std::vector<uint32_t> rec(k);
        in.read((char*)rec.data(), k * sizeof(uint32_t));
        records.push_back(std::move(rec));
    }
    return records;
}

double compute_pearson_correlation(const std::vector<float>& x, const std::vector<float>& y) {
    size_t n = x.size();
    if (n < 2) return 0.0;
    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double num = 0.0, den_x = 0.0, den_y = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    double den = std::sqrt(den_x * den_y);
    return (den > 1e-9) ? (num / den) : 0.0;
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << " AdaptiveVec Phase 2: Stage 1 Verification & Bridge Audit\n";
    std::cout << "=================================================================\n\n";

    size_t dim = 0, n_base = 0, n_queries = 0;
    std::vector<float> base_data = load_fvecs("data/sift_subset_100k.fvecs", 100000, dim, n_base);
    size_t q_dim = 0;
    std::vector<float> query_data = load_fvecs("data/sift_query.fvecs", 2000, q_dim, n_queries);
    auto gt = load_ivecs("data/sift_subset_groundtruth.ivecs", n_queries);

    if (base_data.empty() || query_data.empty()) {
        std::cerr << "[-] Failed to load SIFT dataset files.\n";
        return 1;
    }
    std::cout << "[+] Loaded SIFT-100K: N=" << n_base << ", Q=" << n_queries << ", D=" << dim << "\n";

    // Build Index with Bridge-Aware Regulation
    PolicyConfig cfg;
    cfg.enable_dynamic_m_efc = true;
    cfg.enable_layer_scaling = true;
    cfg.enable_hubness_regulation = true;
    cfg.hubness_mu = 0.03f;
    cfg.enable_ada_ef = true;
    cfg.enable_bridge_aware = true;
    cfg.bridge_alpha = 1.0f;
    cfg.bridge_estimator = 0;

    AdaptiveHNSWIndex index(dim, SpaceType::L2, true, cfg);
    std::cout << "[*] Constructing Bridge-Aware Adaptive Index...\n";
    index.batch_insert(base_data.data(), n_base, 12);

    const auto& b_scores = index.get_bridge_scores();
    std::vector<float> sorted_b = b_scores;
    std::sort(sorted_b.begin(), sorted_b.end());

    double sum_b = std::accumulate(sorted_b.begin(), sorted_b.end(), 0.0);
    double mean_b = sum_b / sorted_b.size();
    double var_b = 0.0;
    for (float v : sorted_b) var_b += (v - mean_b) * (v - mean_b);
    var_b /= sorted_b.size();

    std::cout << "\n[1] Bridge-Score B(v) Distribution Metrics (N=" << sorted_b.size() << "):\n";
    std::cout << "    - Mean: " << std::fixed << std::setprecision(4) << mean_b << "\n";
    std::cout << "    - Variance: " << var_b << ", StdDev: " << std::sqrt(var_b) << "\n";
    std::cout << "    - Min: " << sorted_b.front() << ", Max: " << sorted_b.back() << "\n";
    std::cout << "    - p10: " << sorted_b[sorted_b.size() * 10 / 100] << "\n";
    std::cout << "    - p25: " << sorted_b[sorted_b.size() * 25 / 100] << "\n";
    std::cout << "    - p50 (Median): " << sorted_b[sorted_b.size() * 50 / 100] << "\n";
    std::cout << "    - p75: " << sorted_b[sorted_b.size() * 75 / 100] << "\n";
    std::cout << "    - p90: " << sorted_b[sorted_b.size() * 90 / 100] << "\n";
    std::cout << "    - p99: " << sorted_b[sorted_b.size() * 99 / 100] << "\n";

    // Track traversal hit frequency across 2,000 queries
    std::vector<uint32_t> visit_counts(n_base, 0);
    #pragma omp parallel for schedule(dynamic, 32)
    for (size_t q = 0; q < n_queries; ++q) {
        auto res = index.search(&query_data[q * dim], 10, 64);
        for (const auto& c : res) {
            #pragma omp atomic
            visit_counts[c.id]++;
        }
    }

    std::vector<float> visits_f(n_base);
    std::vector<float> in_degrees_f(n_base);
    for (size_t i = 0; i < n_base; ++i) {
        visits_f[i] = static_cast<float>(visit_counts[i]);
        visits_f[i] = static_cast<float>(visit_counts[i]);
    }

    double corr_bridge_visits = compute_pearson_correlation(b_scores, visits_f);
    std::cout << "\n[2] Correlation Analysis with Structural Search Routing Frequency:\n";
    std::cout << "    - Pearson Correlation between B(v) and Search Visitation Count: " << corr_bridge_visits << "\n";

    // Controlled Ablation: Bridge-Aware (mu=0.03, B) vs Uniform Equivalent Hubness (mu = 0.03 * (1 - mean_B) = 0.003)
    std::cout << "\n[3] Controlled Ablation: Bridge-Aware vs Equivalent Uniform Hubness:\n";
    
    // Config A: Bridge-Aware (mu=0.03, bridge_aware=true)
    auto res_ba = index.batch_search(query_data.data(), n_queries, 10, 64, 12);
    size_t hits_ba = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        std::set<uint32_t> gt_s(gt[q].begin(), gt[q].begin() + 10);
        for (int k = 0; k < 10 && k < (int)res_ba[q].size(); ++k) {
            if (gt_s.count(res_ba[q][k].id)) hits_ba++;
        }
    }
    double recall_ba = (double)hits_ba / (n_queries * 10);
    std::cout << "    - Bridge-Aware (mu=0.03, alpha=1.0) -> Recall@10 (ef=64): " << (recall_ba * 100.0) << "%\n";

    // Config B: Uniform Equivalent Hubness (mu = 0.003, bridge_aware=false)
    PolicyConfig cfg_equiv;
    cfg_equiv.enable_dynamic_m_efc = true;
    cfg_equiv.enable_layer_scaling = true;
    cfg_equiv.enable_hubness_regulation = true;
    cfg_equiv.hubness_mu = 0.003f; // Equivalent attenuated strength
    cfg_equiv.enable_ada_ef = true;
    cfg_equiv.enable_bridge_aware = false;

    AdaptiveHNSWIndex index_equiv(dim, SpaceType::L2, true, cfg_equiv);
    index_equiv.batch_insert(base_data.data(), n_base, 12);
    auto res_eq = index_equiv.batch_search(query_data.data(), n_queries, 10, 64, 12);
    size_t hits_eq = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        std::set<uint32_t> gt_s(gt[q].begin(), gt[q].begin() + 10);
        for (int k = 0; k < 10 && k < (int)res_eq[q].size(); ++k) {
            if (gt_s.count(res_eq[q][k].id)) hits_eq++;
        }
    }
    double recall_eq = (double)hits_eq / (n_queries * 10);
    std::cout << "    - Uniform Equivalent Hubness (mu=0.003, bridge=false) -> Recall@10 (ef=64): " << (recall_eq * 100.0) << "%\n";
    std::cout << "    - Delta (Bridge-Aware vs Uniform Equivalent): " << ((recall_ba - recall_eq) * 100.0) << " pp\n";

    // Adjacency vector capacity vs size analysis
    std::cout << "\n[4] Adjacency Vector Size vs Capacity Analysis:\n";
    size_t total_layer0_edges = index.get_layer0_edges();
    MemoryBreakdown mb = index.get_memory_breakdown();
    std::cout << "    - Total Layer-0 Edges (Logical Size): " << total_layer0_edges << "\n";
    std::cout << "    - Logical Edge Bytes: " << (total_layer0_edges * sizeof(tableint) / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "    - Allocated Graph Adjacency Bytes (Including Cap Overhead): " << (mb.graph_adjacency_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "    - Internal Vector Allocation Overhead: " << ((mb.graph_adjacency_bytes - total_layer0_edges * sizeof(tableint)) / (1024.0 * 1024.0)) << " MB\n";

    std::cout << "\n[+] Stage 1 verification audit complete.\n";
    return 0;
}
