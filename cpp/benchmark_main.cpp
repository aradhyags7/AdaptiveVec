/**
 * Standalone C++ Benchmark Runner for AdaptiveVec vs. Stock HNSW
 * Compiles with g++ -O3 -std=c++17 benchmark_main.cpp -o benchmark_runner
 */

#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace adaptivevec;

struct Dataset {
    std::vector<float> data;
    std::vector<float> queries;
    size_t n_samples;
    size_t n_queries;
    size_t dim;
    std::vector<std::vector<tableint>> gt_100;
};

Dataset generate_clustered_dataset(size_t n_samples = 10000, size_t dim = 64, size_t n_queries = 200) {
    Dataset ds;
    ds.n_samples = n_samples;
    ds.n_queries = n_queries;
    ds.dim = dim;
    ds.data.resize(n_samples * dim);
    ds.queries.resize(n_queries * dim);

    std::mt19937 rng(42);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> cluster_dist(0, 7);

    // Create 8 cluster centers
    std::vector<std::vector<float>> centers(8, std::vector<float>(dim));
    for (int c = 0; c < 8; ++c) {
        for (size_t d = 0; d < dim; ++d) {
            centers[c][d] = norm_dist(rng) * 15.0f;
        }
    }

    // Fill data with alternating dense (small variance) and sparse (large variance) clusters
    for (size_t i = 0; i < n_samples; ++i) {
        size_t c = cluster_dist(rng);
        float scale = (c % 2 == 0) ? 0.6f : 5.0f;
        for (size_t d = 0; d < dim; ++d) {
            ds.data[i * dim + d] = centers[c][d] + norm_dist(rng) * scale;
        }
    }

    // Fill queries
    for (size_t q = 0; q < n_queries; ++q) {
        size_t c = cluster_dist(rng);
        float scale = (c % 2 == 0) ? 0.6f : 5.0f;
        for (size_t d = 0; d < dim; ++d) {
            ds.queries[q * dim + d] = centers[c][d] + norm_dist(rng) * scale;
        }
    }

    // Compute ground truth for queries
    std::cout << "[*] Computing exact brute-force ground truth for " << n_queries << " queries..." << std::endl;
    ds.gt_100.resize(n_queries);
    for (size_t q = 0; q < n_queries; ++q) {
        const float* q_vec = &ds.queries[q * dim];
        std::vector<Candidate> all_dists(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            const float* d_vec = &ds.data[i * dim];
            all_dists[i] = {l2_distance(q_vec, d_vec, dim), (tableint)i};
        }
        std::partial_sort(all_dists.begin(), all_dists.begin() + 100, all_dists.end());
        ds.gt_100[q].resize(100);
        for (int k = 0; k < 100; ++k) {
            ds.gt_100[q][k] = all_dists[k].id;
        }
    }

    return ds;
}

float compute_recall(const std::vector<std::vector<Candidate>>& results, const std::vector<std::vector<tableint>>& ground_truth, int k) {
    size_t total_matches = 0;
    for (size_t q = 0; q < results.size(); ++q) {
        std::unordered_set<tableint> gt_set(ground_truth[q].begin(), ground_truth[q].begin() + k);
        for (int i = 0; i < std::min((int)results[q].size(), k); ++i) {
            if (gt_set.find(results[q][i].id) != gt_set.end()) {
                total_matches++;
            }
        }
    }
    return (float)total_matches / (float)(results.size() * k);
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  AdaptiveVec C++ Benchmark: Stock HNSW vs AdaptiveVec Engine   " << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    size_t n_samples = 10000;
    size_t dim = 64;
    size_t n_queries = 200;
    int ef_search = 50;

    Dataset ds = generate_clustered_dataset(n_samples, dim, n_queries);

    // 1. Benchmark Stock HNSW
    std::cout << "\n[1/2] Building Stock HNSW Index (M=16, efC=150)..." << std::endl;
    PolicyConfig stock_cfg;
    stock_cfg.m_base = 16;
    stock_cfg.ef_c_base = 150;
    AdaptiveHNSWIndex stock_idx(dim, SpaceType::L2, false, stock_cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n_samples; ++i) {
        stock_idx.insert(&ds.data[i * dim]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double stock_build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Query Stock HNSW
    std::vector<std::vector<Candidate>> stock_results(n_queries);
    auto q0 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < n_queries; ++q) {
        stock_results[q] = stock_idx.search(&ds.queries[q * dim], 10, ef_search);
    }
    auto q1 = std::chrono::high_resolution_clock::now();
    double stock_query_ms = std::chrono::duration<double, std::milli>(q1 - q0).count();
    double stock_qps = (n_queries / stock_query_ms) * 1000.0;
    float stock_recall10 = compute_recall(stock_results, ds.gt_100, 10);
    size_t stock_edges = stock_idx.get_total_edges();

    // 2. Benchmark AdaptiveVec
    std::cout << "[2/2] Building AdaptiveVec Index (Dynamic M: 8-22, efC: 40-180)..." << std::endl;
    PolicyConfig adapt_cfg;
    adapt_cfg.type = PolicyType::Continuous;
    adapt_cfg.m_base = 12;
    adapt_cfg.m_min = 6;
    adapt_cfg.m_max = 22;
    adapt_cfg.ef_c_base = 100;
    adapt_cfg.ef_c_min = 35;
    adapt_cfg.ef_c_max = 180;
    adapt_cfg.sensitivity = 0.5f;
    AdaptiveHNSWIndex adapt_idx(dim, SpaceType::L2, true, adapt_cfg);

    // Calibrate baseline statistics with dataset sample
    adapt_idx.calibrate(ds.data.data(), std::min((size_t)300, ds.n_samples));

    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n_samples; ++i) {
        adapt_idx.insert(&ds.data[i * dim]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double adapt_build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Query AdaptiveVec
    std::vector<std::vector<Candidate>> adapt_results(n_queries);
    q0 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < n_queries; ++q) {
        adapt_results[q] = adapt_idx.search(&ds.queries[q * dim], 10, ef_search);
    }
    q1 = std::chrono::high_resolution_clock::now();
    double adapt_query_ms = std::chrono::duration<double, std::milli>(q1 - q0).count();
    double adapt_qps = (n_queries / adapt_query_ms) * 1000.0;
    float adapt_recall10 = compute_recall(adapt_results, ds.gt_100, 10);
    size_t adapt_edges = adapt_idx.get_total_edges();

    double edge_savings = 100.0 * (1.0 - (double)adapt_edges / (double)stock_edges);
    double build_speedup = 100.0 * (1.0 - adapt_build_ms / stock_build_ms);

    // Results Table
    std::cout << "\n=================================================================" << std::endl;
    std::cout << "                        BENCHMARK RESULTS                        " << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << std::left << std::setw(25) << "Metric" 
              << std::setw(20) << "Stock HNSW" 
              << std::setw(20) << "AdaptiveVec" 
              << "Delta / Advantage" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    
    std::cout << std::left << std::setw(25) << "Build Time (ms)" 
              << std::setw(20) << stock_build_ms 
              << std::setw(20) << adapt_build_ms 
              << (build_speedup >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << build_speedup << "% Faster" << std::endl;

    std::cout << std::left << std::setw(25) << "Total Edges" 
              << std::setw(20) << stock_edges 
              << std::setw(20) << adapt_edges 
              << "-" << std::fixed << std::setprecision(1) << edge_savings << "% Memory Saved" << std::endl;

    std::cout << std::left << std::setw(25) << "Avg Edges/Node" 
              << std::setw(20) << (double)stock_edges / n_samples 
              << std::setw(20) << (double)adapt_edges / n_samples 
              << "-" << std::fixed << std::setprecision(2) << ((double)stock_edges - (double)adapt_edges) / n_samples << " edges" << std::endl;

    std::cout << std::left << std::setw(25) << "Recall@10" 
              << std::setw(20) << std::setprecision(4) << stock_recall10 
              << std::setw(20) << std::setprecision(4) << adapt_recall10 
              << (adapt_recall10 >= stock_recall10 ? "+" : "") << (adapt_recall10 - stock_recall10) * 100.0 << "% parity" << std::endl;

    std::cout << std::left << std::setw(25) << "Query QPS" 
              << std::setw(20) << std::setprecision(1) << stock_qps 
              << std::setw(20) << std::setprecision(1) << adapt_qps 
              << ((adapt_qps >= stock_qps) ? "+" : "") << ((adapt_qps - stock_qps) / stock_qps) * 100.0 << "%" << std::endl;
    
    std::cout << "=================================================================\n" << std::endl;

    return 0;
}
