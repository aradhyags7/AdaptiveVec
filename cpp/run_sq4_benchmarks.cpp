/**
 * run_sq4_benchmarks.cpp
 * Milestone 1 (Frontier 3): Ultra-Edge 4-Bit Sub-Byte Scalar Quantization (SQ4) Benchmark Harness
 *
 * Compares:
 * 1. Full-Precision FP32 Vectors (512 bytes/vec)
 * 2. Step 6 SQ8 Quantized Vectors (128 bytes/vec)
 * 3. Milestone 1 SQ4 Sub-Byte Nibble-Packed Vectors (64 bytes/vec)
 *
 * Evaluates on SIFT-100K:
 * - Vector RAM, Graph Edge RAM, and Total Index Memory
 * - Recall@10 across varying efSearch
 * - Query Throughput (QPS)
 * - Latency Percentiles (p50, p95, p99)
 * - Two-Stage Re-ranking factor sensitivity
 */

#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace adaptivevec;

struct CompressionTrialRecord {
    std::string mode;
    int ef_search;
    int rerank_factor;
    double vector_ram_mb;
    double edge_ram_mb;
    double total_ram_mb;
    double build_time_s;
    double recall_10;
    double qps;
    double mean_dist_evals;
    double p50_us;
    double p95_us;
    double p99_us;
};

float evaluate_recall(
    const std::vector<std::vector<Candidate>>& results,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    int k = 10
) {
    if (results.empty() || ground_truth.empty()) return 0.0f;
    size_t total_matches = 0;
    size_t n = std::min(results.size(), ground_truth.size());
    for (size_t q = 0; q < n; ++q) {
        std::unordered_set<uint32_t> gt_set;
        for (int i = 0; i < k && i < (int)ground_truth[q].size(); ++i) {
            gt_set.insert(ground_truth[q][i]);
        }
        for (int i = 0; i < k && i < (int)results[q].size(); ++i) {
            if (gt_set.find(results[q][i].id) != gt_set.end()) {
                total_matches++;
            }
        }
    }
    return (float)total_matches / (float)(n * k);
}

int main(int argc, char** argv) {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "  AdaptiveVec Milestone 1: Ultra-Edge 4-Bit Sub-Byte Quantization (SQ4)   " << std::endl;
    std::cout << "==========================================================================\n" << std::endl;

    std::string dataset_path = "data/sift_subset_100k.fvecs";
    std::string query_path = "data/sift_query.fvecs";
    std::string gt_path = "data/sift_subset_groundtruth.ivecs";

    std::vector<float> data;
    std::vector<float> queries;
    std::vector<std::vector<uint32_t>> ground_truth;
    size_t n_vectors = 0, dim = 0, n_queries = 0, q_dim = 0;

    std::cout << "[*] Loading SIFT-100K dataset from " << dataset_path << "..." << std::endl;
    if (!load_fvecs(dataset_path, data, n_vectors, dim, 100000)) {
        std::cerr << "[-] Failed to load dataset." << std::endl;
        return 1;
    }
    std::cout << "[+] Loaded " << n_vectors << " vectors (dim " << dim << ")." << std::endl;

    std::cout << "[*] Loading test queries from " << query_path << "..." << std::endl;
    if (!load_fvecs(query_path, queries, n_queries, q_dim, 10000)) {
        std::cerr << "[-] Failed to load queries." << std::endl;
        return 1;
    }
    std::cout << "[+] Loaded " << n_queries << " query vectors." << std::endl;

    std::cout << "[*] Loading ground truth from " << gt_path << "..." << std::endl;
    if (!load_ivecs(gt_path, ground_truth, n_queries)) {
        std::cerr << "[-] Failed to load ground truth." << std::endl;
        return 1;
    }
    std::cout << "[+] Loaded exact ground truth for " << ground_truth.size() << " queries.\n" << std::endl;

    // Base AdaptiveVec policy (Step 5: Dynamic M, Layer Scaling, Hubness, Ada-ef)
    PolicyConfig base_policy;
    base_policy.enable_dynamic_m_efc = true;
    base_policy.enable_layer_scaling = true;
    base_policy.enable_hubness_regulation = true;
    base_policy.enable_ada_ef = true;
    base_policy.ada_ef_patience = 6;
    base_policy.ada_ef_epsilon = 1e-4f;
    base_policy.hubness_mu = 0.15f;

    struct ModeConfig {
        std::string name;
        bool sq8;
        bool sq4;
        int default_rerank;
    };

    std::vector<ModeConfig> modes = {
        {"FP32 (Uncompressed)", false, false, 1},
        {"SQ8 (8-Bit Asymmetric)", true, false, 3},
        {"SQ4 (4-Bit Sub-Byte Nibble)", false, true, 3}
    };

    std::vector<int> ef_sweep = {24, 32, 48, 64, 96, 128};
    std::vector<CompressionTrialRecord> all_records;

    for (const auto& mode : modes) {
        std::cout << "==========================================================================" << std::endl;
        std::cout << ">>> Mode: " << mode.name << " <<<" << std::endl;
        std::cout << "==========================================================================" << std::endl;

        PolicyConfig pol = base_policy;
        pol.enable_sq8 = mode.sq8;
        pol.enable_sq4 = mode.sq4;
        pol.sq8_rerank_factor = mode.default_rerank;
        pol.sq4_rerank_factor = mode.default_rerank;

        AdaptiveHNSWIndex index(dim, SpaceType::L2, true, pol);

        if (mode.sq8 || mode.sq4) {
            index.train_quantizer(data.data(), std::min((size_t)10000, n_vectors));
        }

        std::cout << "[*] Building index via parallel batch_insert..." << std::flush;
        auto b_start = std::chrono::high_resolution_clock::now();
        index.batch_insert(data.data(), n_vectors);
        auto b_end = std::chrono::high_resolution_clock::now();
        double build_s = std::chrono::duration<double>(b_end - b_start).count();

        double vec_ram_mb = (double)index.get_vector_memory_bytes() / (1024.0 * 1024.0);
        double edge_ram_mb = (double)index.get_edge_memory_bytes() / (1024.0 * 1024.0);
        double total_ram_mb = (double)index.get_total_memory_bytes() / (1024.0 * 1024.0);

        std::cout << " Done in " << std::fixed << std::setprecision(2) << build_s << " s." << std::endl;
        std::cout << "[+] Vector RAM: " << std::setw(6) << std::setprecision(2) << vec_ram_mb << " MB"
                  << " | Edge RAM: " << std::setw(5) << std::setprecision(2) << edge_ram_mb << " MB"
                  << " | Total RAM: " << std::setw(6) << std::setprecision(2) << total_ram_mb << " MB"
                  << " | Build Throughput: " << std::setprecision(0) << ((double)n_vectors / build_s) << " vec/s" << std::endl;

        std::cout << "\n  --- efSearch Query Sweep ---" << std::endl;
        for (int ef : ef_sweep) {
            index.total_dist_computations = 0;
            std::vector<double> latencies_us(n_queries);
            std::vector<std::vector<Candidate>> results(n_queries);

            auto q_start = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(dynamic, 32)
            for (size_t q = 0; q < n_queries; ++q) {
                auto l_start = std::chrono::high_resolution_clock::now();
                results[q] = index.search(&queries[q * dim], 10, ef);
                auto l_end = std::chrono::high_resolution_clock::now();
                latencies_us[q] = std::chrono::duration<double, std::micro>(l_end - l_start).count();
            }
            auto q_end = std::chrono::high_resolution_clock::now();

            double total_q_s = std::chrono::duration<double>(q_end - q_start).count();
            double qps = (double)n_queries / total_q_s;
            double recall = evaluate_recall(results, ground_truth, 10);
            double mean_dist = (double)index.total_dist_computations / (double)n_queries;

            std::sort(latencies_us.begin(), latencies_us.end());
            double p50 = latencies_us[(size_t)(0.50 * n_queries)];
            double p95 = latencies_us[(size_t)(0.95 * n_queries)];
            double p99 = latencies_us[(size_t)(0.99 * n_queries)];

            all_records.push_back({
                mode.name, ef, mode.default_rerank,
                vec_ram_mb, edge_ram_mb, total_ram_mb, build_s,
                recall, qps, mean_dist, p50, p95, p99
            });

            std::cout << "  ef=" << std::setw(3) << ef 
                      << " | Recall@10: " << std::fixed << std::setprecision(4) << recall 
                      << " | QPS: " << std::setw(7) << std::setprecision(1) << qps 
                      << " | Evals/q: " << std::setw(6) << std::setprecision(1) << mean_dist 
                      << " | p50: " << std::setw(5) << std::setprecision(1) << p50 << " us"
                      << " | p95: " << std::setw(5) << std::setprecision(1) << p95 << " us" << std::endl;
        }

        // If SQ4, run Re-ranking Factor Sensitivity ablation at ef=64
        if (mode.sq4) {
            std::cout << "\n  --- SQ4 Re-ranking Factor Sensitivity Ablation (ef=64) ---" << std::endl;
            std::vector<int> factors = {1, 2, 3, 4, 5};
            for (int f : factors) {
                index.policy.sq4_rerank_factor = f;
                index.total_dist_computations = 0;
                std::vector<double> latencies_us(n_queries);
                std::vector<std::vector<Candidate>> results(n_queries);

                auto q_start = std::chrono::high_resolution_clock::now();
                #pragma omp parallel for schedule(dynamic, 32)
                for (size_t q = 0; q < n_queries; ++q) {
                    auto l_start = std::chrono::high_resolution_clock::now();
                    results[q] = index.search(&queries[q * dim], 10, 64);
                    auto l_end = std::chrono::high_resolution_clock::now();
                    latencies_us[q] = std::chrono::duration<double, std::micro>(l_end - l_start).count();
                }
                auto q_end = std::chrono::high_resolution_clock::now();

                double total_q_s = std::chrono::duration<double>(q_end - q_start).count();
                double qps = (double)n_queries / total_q_s;
                double recall = evaluate_recall(results, ground_truth, 10);
                double mean_dist = (double)index.total_dist_computations / (double)n_queries;

                std::sort(latencies_us.begin(), latencies_us.end());
                double p50 = latencies_us[(size_t)(0.50 * n_queries)];
                double p95 = latencies_us[(size_t)(0.95 * n_queries)];
                double p99 = latencies_us[(size_t)(0.99 * n_queries)];

                all_records.push_back({
                    "SQ4_Rerank_Sweep", 64, f,
                    vec_ram_mb, edge_ram_mb, total_ram_mb, build_s,
                    recall, qps, mean_dist, p50, p95, p99
                });

                std::cout << "  Factor=" << f << "x"
                          << " | Recall@10: " << std::fixed << std::setprecision(4) << recall 
                          << " | QPS: " << std::setw(7) << std::setprecision(1) << qps 
                          << " | Evals/q: " << std::setw(6) << std::setprecision(1) << mean_dist 
                          << " | p50: " << std::setw(5) << std::setprecision(1) << p50 << " us"
                          << " | p95: " << std::setw(5) << std::setprecision(1) << p95 << " us" << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // Save JSON Results
    std::ofstream out("experiments/sq4_results.json");
    if (out.is_open()) {
        out << "[\n";
        for (size_t i = 0; i < all_records.size(); ++i) {
            const auto& r = all_records[i];
            out << "  {\n"
                << "    \"mode\": \"" << r.mode << "\",\n"
                << "    \"ef_search\": " << r.ef_search << ",\n"
                << "    \"rerank_factor\": " << r.rerank_factor << ",\n"
                << "    \"vector_ram_mb\": " << r.vector_ram_mb << ",\n"
                << "    \"edge_ram_mb\": " << r.edge_ram_mb << ",\n"
                << "    \"total_ram_mb\": " << r.total_ram_mb << ",\n"
                << "    \"build_time_s\": " << r.build_time_s << ",\n"
                << "    \"recall_10\": " << r.recall_10 << ",\n"
                << "    \"qps\": " << r.qps << ",\n"
                << "    \"mean_dist_evals\": " << r.mean_dist_evals << ",\n"
                << "    \"p50_us\": " << r.p50_us << ",\n"
                << "    \"p95_us\": " << r.p95_us << ",\n"
                << "    \"p99_us\": " << r.p99_us << "\n"
                << "  }" << (i + 1 < all_records.size() ? ",\n" : "\n");
        }
        out << "]\n";
        std::cout << "[+] Wrote comprehensive benchmark results to experiments/sq4_results.json" << std::endl;
    }

    return 0;
}
