/**
 * run_multithread_benchmarks.cpp
 * Benchmarks multi-core scaling for AdaptiveVec:
 * 1. Batch query search scaling (1, 2, 4, 6, 8, 12 threads)
 * 2. Concurrent index construction scaling (1, 2, 4, 8, 12 threads)
 * 
 * Hardware: Intel Core 5 210H (8 physical cores, 12 logical threads, AVX2/FMA)
 */

#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <unordered_set>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace adaptivevec;

struct QueryScalingRecord {
    int threads;
    double time_ms;
    double qps;
    double speedup;
    double efficiency_pct;
    double recall_at_10;
};

struct BuildScalingRecord {
    int threads;
    double build_time_s;
    double build_rate_vps;
    double speedup;
    double efficiency_pct;
    size_t total_edges;
};

float compute_recall(
    const std::vector<std::vector<Candidate>>& results,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    int k
) {
    if (results.empty() || ground_truth.empty()) return 0.0f;
    size_t total_matches = 0;
    size_t total_queries = results.size();

    for (size_t q = 0; q < total_queries; ++q) {
        if (q >= ground_truth.size()) break;
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
    return (float)total_matches / (float)(total_queries * k);
}

int main(int argc, char** argv) {
    std::cout << "=================================================================================\n";
    std::cout << "   AdaptiveVec Multi-Threaded Scaling Benchmark (Intel Core 5 210H, 12 Threads)  \n";
    std::cout << "=================================================================================\n\n";

    #if defined(_OPENMP)
    std::cout << "[*] OpenMP detected. Maximum hardware concurrency: " << omp_get_max_threads() << " threads.\n";
    #else
    std::cout << "[*] OpenMP not detected. Using std::thread fallback.\n";
    #endif

    // 1. Load SIFT-100K
    std::vector<float> data;
    std::vector<float> queries;
    std::vector<std::vector<uint32_t>> ground_truth;
    size_t n_samples = 0, dim = 0;
    size_t n_queries = 0, q_dim = 0;

    std::string data_path = "data/sift_subset_100k.fvecs";
    std::string query_path = "data/sift_query.fvecs";
    std::string gt_path = "data/sift_subset_groundtruth.ivecs";

    std::cout << "[*] Loading SIFT-100K dataset from " << data_path << " ...\n";
    if (!load_fvecs(data_path, data, n_samples, dim, 100000)) {
        std::cerr << "[-] Error loading SIFT-100K data!\n";
        return 1;
    }
    load_fvecs(query_path, queries, n_queries, q_dim, 10000);
    load_ivecs(gt_path, ground_truth, n_queries);

    std::cout << "[+] Loaded " << n_samples << " base vectors (dim " << dim << "), "
              << n_queries << " queries with ground truth.\n\n";

    // 2. Build Index once for Query Scaling Evaluation
    PolicyConfig policy;
    policy.m_base = 16;
    policy.m_min = 8;
    policy.m_max = 24;
    policy.ef_c_base = 120;
    policy.ef_c_min = 40;
    policy.ef_c_max = 220;
    policy.hubness_mu = 0.15f;
    policy.ada_ef_patience = 6;
    policy.ada_ef_epsilon = 1e-4f;
    policy.enable_ada_ef = true;

    std::cout << "[*] Building SIFT-100K index for query scaling benchmarks ... " << std::flush;
    auto b_start = std::chrono::high_resolution_clock::now();
    AdaptiveHNSWIndex index(dim, SpaceType::L2, true, policy);
    for (size_t i = 0; i < n_samples; ++i) {
        index.insert(&data[i * dim]);
    }
    auto b_end = std::chrono::high_resolution_clock::now();
    double b_time = std::chrono::duration<double>(b_end - b_start).count();
    std::cout << "Done! (" << std::fixed << std::setprecision(2) << b_time << "s, "
              << index.get_total_edges() << " edges)\n\n";

    // 3. Warm-up Query Phase
    std::cout << "[*] Warming up CPU caches with 1,000 queries ... " << std::flush;
    auto warmup_res = index.batch_search(queries.data(), 1000, 10, 64, 4);
    std::cout << "Done.\n\n";

    // 4. Query Throughput Scaling Sweep across Thread Counts
    std::vector<int> query_threads = {1, 2, 4, 6, 8, 12};
    std::vector<QueryScalingRecord> query_results;
    double baseline_qps = 0.0;

    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << " PART 1: QUERY THROUGHPUT SCALING ON SIFT-100K (10,000 QUERIES, EF=64, K=10)\n";
    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << " Threads | Wall Time | Query Throughput | Speedup  | Core Efficiency | Recall@10 \n";
    std::cout << "---------+-----------+------------------+----------+-----------------+-----------\n";

    for (int th : query_threads) {
        // Run 3 timing trials and average
        constexpr int TRIALS = 3;
        double total_ms = 0.0;
        std::vector<std::vector<Candidate>> last_res;

        for (int tr = 0; tr < TRIALS; ++tr) {
            auto q_t0 = std::chrono::high_resolution_clock::now();
            last_res = index.batch_search(queries.data(), n_queries, 10, 64, th);
            auto q_t1 = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(q_t1 - q_t0).count();
        }

        double avg_ms = total_ms / TRIALS;
        double qps = (double)n_queries / (avg_ms / 1000.0);
        if (th == 1) baseline_qps = qps;

        double speedup = qps / baseline_qps;
        double efficiency = (speedup / (double)th) * 100.0;
        double recall = compute_recall(last_res, ground_truth, 10);

        query_results.push_back({th, avg_ms, qps, speedup, efficiency, recall});

        std::cout << " " << std::setw(7) << th << " |"
                  << " " << std::setw(7) << std::fixed << std::setprecision(1) << avg_ms << " ms |"
                  << " " << std::setw(11) << std::fixed << std::setprecision(1) << qps << " QPS |"
                  << " " << std::setw(7) << std::fixed << std::setprecision(2) << speedup << "x |"
                  << " " << std::setw(13) << std::fixed << std::setprecision(1) << efficiency << "% |"
                  << " " << std::setw(9) << std::fixed << std::setprecision(4) << recall << "\n";
    }
    std::cout << "---------------------------------------------------------------------------------\n\n";

    // 5. Index Construction Scaling Sweep across Thread Counts (50,000 vectors)
    size_t build_bench_n = 50000;
    std::vector<int> build_threads = {1, 2, 4, 8, 12};
    std::vector<BuildScalingRecord> build_results;
    double baseline_build_time = 0.0;

    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << " PART 2: CONCURRENT INDEX CONSTRUCTION SCALING (50,000 SIFT VECTORS)\n";
    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << " Threads | Build Time | Insertion Rate | Speedup  | Core Efficiency | Total Edges\n";
    std::cout << "---------+------------+----------------+----------+-----------------+------------\n";

    for (int th : build_threads) {
        AdaptiveHNSWIndex build_index(dim, SpaceType::L2, true, policy);
        
        auto start = std::chrono::high_resolution_clock::now();
        build_index.batch_insert(data.data(), build_bench_n, th);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_s = std::chrono::duration<double>(end - start).count();
        double vps = (double)build_bench_n / elapsed_s;
        if (th == 1) baseline_build_time = elapsed_s;

        double speedup = baseline_build_time / elapsed_s;
        double efficiency = (speedup / (double)th) * 100.0;
        size_t edges = build_index.get_total_edges();

        build_results.push_back({th, elapsed_s, vps, speedup, efficiency, edges});

        std::cout << " " << std::setw(7) << th << " |"
                  << " " << std::setw(8) << std::fixed << std::setprecision(2) << elapsed_s << " s |"
                  << " " << std::setw(11) << std::fixed << std::setprecision(0) << vps << " v/s |"
                  << " " << std::setw(7) << std::fixed << std::setprecision(2) << speedup << "x |"
                  << " " << std::setw(13) << std::fixed << std::setprecision(1) << efficiency << "% |"
                  << " " << std::setw(10) << edges << "\n";
    }
    std::cout << "---------------------------------------------------------------------------------\n\n";

    // 6. Write JSON Results
    std::string json_path = "experiments/multithread_scaling_results.json";
    std::ofstream out(json_path);
    if (out.is_open()) {
        out << "{\n";
        out << "  \"hardware\": {\n";
        out << "    \"cpu\": \"Intel Core 5 210H\",\n";
        out << "    \"cores\": 8,\n";
        out << "    \"threads\": 12,\n";
        out << "    \"p_cores\": 4,\n";
        out << "    \"e_cores\": 4,\n";
        out << "    \"simd\": \"AVX2 / FMA\",\n";
        out << "    \"ram_gb\": 16\n";
        out << "  },\n";
        out << "  \"query_scaling\": [\n";
        for (size_t i = 0; i < query_results.size(); ++i) {
            const auto& r = query_results[i];
            out << "    {\n";
            out << "      \"threads\": " << r.threads << ",\n";
            out << "      \"latency_ms\": " << std::fixed << std::setprecision(2) << r.time_ms << ",\n";
            out << "      \"qps\": " << std::fixed << std::setprecision(1) << r.qps << ",\n";
            out << "      \"speedup\": " << std::fixed << std::setprecision(2) << r.speedup << ",\n";
            out << "      \"core_efficiency_pct\": " << std::fixed << std::setprecision(1) << r.efficiency_pct << ",\n";
            out << "      \"recall_at_10\": " << std::fixed << std::setprecision(4) << r.recall_at_10 << "\n";
            out << "    }" << (i + 1 < query_results.size() ? "," : "") << "\n";
        }
        out << "  ],\n";
        out << "  \"build_scaling\": [\n";
        for (size_t i = 0; i < build_results.size(); ++i) {
            const auto& r = build_results[i];
            out << "    {\n";
            out << "      \"threads\": " << r.threads << ",\n";
            out << "      \"build_time_sec\": " << std::fixed << std::setprecision(2) << r.build_time_s << ",\n";
            out << "      \"vectors_per_sec\": " << std::fixed << std::setprecision(1) << r.build_rate_vps << ",\n";
            out << "      \"speedup\": " << std::fixed << std::setprecision(2) << r.speedup << ",\n";
            out << "      \"core_efficiency_pct\": " << std::fixed << std::setprecision(1) << r.efficiency_pct << ",\n";
            out << "      \"total_edges\": " << r.total_edges << "\n";
            out << "    }" << (i + 1 < build_results.size() ? "," : "") << "\n";
        }
        out << "  ]\n";
        out << "}\n";
        std::cout << "[+] Saved empirical scaling results to: " << json_path << "\n";
    }

    return 0;
}
