/**
 * run_dynamic_ef_benchmarks.cpp
 * Frontier 1: Query-Time Manifold-Aware Dynamic efSearch Benchmark Harness
 * 
 * Compares Static efSearch vs. Dynamic Manifold-Aware efSearch on SIFT-100K.
 * Evaluates Recall@10, QPS, and Mean Distance Evaluations.
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

struct QueryBenchmarkRecord {
    std::string mode;
    int base_ef;
    float kappa;
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
    std::cout << "   AdaptiveVec Frontier 1: Query-Time Dynamic efSearch Benchmark Harness  " << std::endl;
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
    std::cout << "[+] Loaded exact ground truth for " << ground_truth.size() << " queries." << std::endl;

    // Build AdaptiveVec Index (Step 5 Configuration: Dynamic M, Layer Scaling, Hubness, Stagnation Exit)
    PolicyConfig policy;
    policy.enable_dynamic_m_efc = true;
    policy.enable_layer_scaling = true;
    policy.enable_hubness_regulation = true;
    policy.enable_ada_ef = true;
    policy.ada_ef_patience = 6;
    policy.ada_ef_epsilon = 1e-4f;
    policy.hubness_mu = 0.15f;

    std::cout << "\n[*] Building AdaptiveVec index with multi-threaded insertion (OpenMP)..." << std::endl;
    AdaptiveHNSWIndex index(dim, SpaceType::L2, true, policy);

    auto t_start = std::chrono::high_resolution_clock::now();
    index.batch_insert(data.data(), n_vectors);
    auto t_end = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "[+] Index built in " << std::fixed << std::setprecision(2) << build_s << " s (" 
              << (double)n_vectors / build_s << " vectors/s)" << std::endl;
    std::cout << "[+] Total graph edges: " << index.get_total_edges() 
              << " (Layer 0: " << index.get_layer0_edges() 
              << ", Upper: " << index.get_upper_layer_edges() << ")" << std::endl;

    std::vector<QueryBenchmarkRecord> records;

    // Phase 1: Static efSearch Baseline Sweep
    std::cout << "\n--- [Phase 1: Static efSearch Sweep] ---" << std::endl;
    std::vector<int> static_efs = {24, 32, 48, 64, 80, 96, 128};

    for (int ef : static_efs) {
        index.policy.enable_dynamic_ef_search = false;
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

        records.push_back({"Static", ef, 0.0f, recall, qps, mean_dist, p50, p95, p99});

        std::cout << "  Static ef=" << std::setw(3) << ef 
                  << " | Recall@10: " << std::fixed << std::setprecision(4) << recall 
                  << " | QPS: " << std::setw(7) << std::setprecision(1) << qps 
                  << " | Evals/q: " << std::setw(6) << std::setprecision(1) << mean_dist 
                  << " | p50: " << std::setw(5) << std::setprecision(1) << p50 << " us"
                  << " | p95: " << std::setw(5) << std::setprecision(1) << p95 << " us" << std::endl;
    }

    // Phase 2: Dynamic Manifold-Aware efSearch Sweep
    std::cout << "\n--- [Phase 2: Frontier 1 Dynamic Manifold-Aware efSearch Sweep] ---" << std::endl;
    std::vector<int> dynamic_base_efs = {32, 48, 64, 80};
    std::vector<float> kappas = {0.30f, 0.45f, 0.60f};

    for (int base_ef : dynamic_base_efs) {
        for (float kappa : kappas) {
            index.policy.enable_dynamic_ef_search = true;
            index.policy.dynamic_ef_factor = kappa;
            index.policy.dynamic_ef_min = std::max(12, (int)(base_ef * 0.35f));
            index.policy.dynamic_ef_max = std::min(160, (int)(base_ef * 1.80f));
            index.total_dist_computations = 0;

            std::vector<double> latencies_us(n_queries);
            std::vector<std::vector<Candidate>> results(n_queries);

            auto q_start = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(dynamic, 32)
            for (size_t q = 0; q < n_queries; ++q) {
                auto l_start = std::chrono::high_resolution_clock::now();
                results[q] = index.search(&queries[q * dim], 10, base_ef);
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

            records.push_back({"Dynamic", base_ef, kappa, recall, qps, mean_dist, p50, p95, p99});

            std::cout << "  Dynamic base=" << std::setw(2) << base_ef 
                      << " (k=" << std::fixed << std::setprecision(2) << kappa << ")"
                      << " | Recall@10: " << std::fixed << std::setprecision(4) << recall 
                      << " | QPS: " << std::setw(7) << std::setprecision(1) << qps 
                      << " | Evals/q: " << std::setw(6) << std::setprecision(1) << mean_dist 
                      << " | p50: " << std::setw(5) << std::setprecision(1) << p50 << " us"
                      << " | p95: " << std::setw(5) << std::setprecision(1) << p95 << " us" << std::endl;
        }
    }

    // Save JSON Results
    std::ofstream out("experiments/dynamic_ef_results.json");
    if (out.is_open()) {
        out << "[\n";
        for (size_t i = 0; i < records.size(); ++i) {
            const auto& r = records[i];
            out << "  {\n"
                << "    \"mode\": \"" << r.mode << "\",\n"
                << "    \"base_ef\": " << r.base_ef << ",\n"
                << "    \"kappa\": " << r.kappa << ",\n"
                << "    \"recall_10\": " << r.recall_10 << ",\n"
                << "    \"qps\": " << r.qps << ",\n"
                << "    \"mean_dist_evals\": " << r.mean_dist_evals << ",\n"
                << "    \"p50_us\": " << r.p50_us << ",\n"
                << "    \"p95_us\": " << r.p95_us << ",\n"
                << "    \"p99_us\": " << r.p99_us << "\n"
                << "  }" << (i + 1 < records.size() ? ",\n" : "\n");
        }
        out << "]\n";
        std::cout << "\n[+] Wrote benchmark results to experiments/dynamic_ef_results.json" << std::endl;
    }

    return 0;
}
