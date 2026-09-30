/**
 * run_scale_benchmarks.cpp
 * Large-Scale Evaluation Harness for AdaptiveVec (100K to 1 Million Vectors)
 * Benchmarks Baseline HNSW vs. AdaptiveVec across:
 * - 100,000 vectors
 * - 250,000 vectors
 * - 500,000 vectors
 * - 1,000,000 vectors
 * 
 * Target Hardware: Intel Core 5 210H (8 Cores, 12 Threads, AVX2/FMA, 16GB RAM)
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

struct ScaleBenchmarkResult {
    size_t n_vectors;
    // Baseline metrics
    double base_build_time_s;
    double base_build_rate_vps;
    size_t base_edges;
    double base_ram_mb;
    double base_qps;
    double base_recall;
    double base_dist_per_q;
    // AdaptiveVec metrics
    double adapt_build_time_s;
    double adapt_build_rate_vps;
    size_t adapt_edges;
    double adapt_ram_mb;
    double adapt_qps;
    double adapt_recall;
    double adapt_dist_per_q;
    // Comparisons
    double edge_reduction_pct;
    double qps_speedup;
    double build_speedup;
    double ram_savings_pct;
};

float compute_recall(
    const std::vector<std::vector<Candidate>>& results,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    int k
) {
    if (results.empty() || ground_truth.empty()) return 0.0f;
    size_t total_matches = 0;
    size_t count = std::min(results.size(), ground_truth.size());
    for (size_t q = 0; q < count; ++q) {
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
    return (float)total_matches / (float)(count * k);
}

void write_json_results(const std::vector<ScaleBenchmarkResult>& results, const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) return;

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
    out << "  \"scale_benchmarks\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"n_vectors\": " << r.n_vectors << ",\n";
        out << "      \"baseline\": {\n";
        out << "        \"build_time_s\": " << std::fixed << std::setprecision(2) << r.base_build_time_s << ",\n";
        out << "        \"build_rate_vps\": " << std::fixed << std::setprecision(1) << r.base_build_rate_vps << ",\n";
        out << "        \"edges\": " << r.base_edges << ",\n";
        out << "        \"ram_mb\": " << std::fixed << std::setprecision(1) << r.base_ram_mb << ",\n";
        out << "        \"qps\": " << std::fixed << std::setprecision(1) << r.base_qps << ",\n";
        out << "        \"recall_at_10\": " << std::fixed << std::setprecision(4) << r.base_recall << ",\n";
        out << "        \"dist_evals_per_q\": " << std::fixed << std::setprecision(1) << r.base_dist_per_q << "\n";
        out << "      },\n";
        out << "      \"adaptivevec\": {\n";
        out << "        \"build_time_s\": " << std::fixed << std::setprecision(2) << r.adapt_build_time_s << ",\n";
        out << "        \"build_rate_vps\": " << std::fixed << std::setprecision(1) << r.adapt_build_rate_vps << ",\n";
        out << "        \"edges\": " << r.adapt_edges << ",\n";
        out << "        \"ram_mb\": " << std::fixed << std::setprecision(1) << r.adapt_ram_mb << ",\n";
        out << "        \"qps\": " << std::fixed << std::setprecision(1) << r.adapt_qps << ",\n";
        out << "        \"recall_at_10\": " << std::fixed << std::setprecision(4) << r.adapt_recall << ",\n";
        out << "        \"dist_evals_per_q\": " << std::fixed << std::setprecision(1) << r.adapt_dist_per_q << "\n";
        out << "      },\n";
        out << "      \"comparisons\": {\n";
        out << "        \"edge_reduction_pct\": " << std::fixed << std::setprecision(2) << r.edge_reduction_pct << ",\n";
        out << "        \"qps_speedup\": " << std::fixed << std::setprecision(2) << r.qps_speedup << ",\n";
        out << "        \"build_speedup\": " << std::fixed << std::setprecision(2) << r.build_speedup << ",\n";
        out << "        \"ram_savings_pct\": " << std::fixed << std::setprecision(2) << r.ram_savings_pct << "\n";
        out << "      }\n";
        out << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

int main(int argc, char** argv) {
    std::cout << "========================================================================================\n";
    std::cout << "   AdaptiveVec Large-Scale Multi-Threaded Benchmark (100K to 1M Vectors, 12 Threads)   \n";
    std::cout << "========================================================================================\n\n";

    #if defined(_OPENMP)
    int n_threads = omp_get_max_threads();
    std::cout << "[*] OpenMP enabled. Hardware concurrency: " << n_threads << " worker threads.\n";
    #else
    int n_threads = 1;
    std::cout << "[*] OpenMP not detected. Running single-threaded.\n";
    #endif

    // 1. Load Query Vectors
    std::vector<float> queries;
    size_t n_queries = 0, q_dim = 0;
    std::string query_path = "data/sift_query.fvecs";
    std::cout << "[*] Loading query vectors from " << query_path << " ... " << std::flush;
    if (!load_fvecs(query_path, queries, n_queries, q_dim, 10000)) {
        std::cerr << "[-] Error loading query vectors!\n";
        return 1;
    }
    std::cout << "Done (" << n_queries << " queries, dim " << q_dim << ")\n";

    // 2. Define Scales to Benchmark
    struct ScaleConfig {
        size_t n;
        std::string name;
        std::string gt_file;
        size_t eval_queries;
    };

    std::vector<ScaleConfig> scales = {
        {100000,  "SIFT-100K", "data/sift_subset_groundtruth.ivecs", 10000},
        {250000,  "SIFT-250K", "data/sift_groundtruth_250k.ivecs",   1000},
        {500000,  "SIFT-500K", "data/sift_groundtruth_500k.ivecs",   1000},
        {1000000, "SIFT-1M",   "data/sift_groundtruth_1m.ivecs",     10000}
    };

    std::string base_data_file = "data/sift_base_1m.fvecs";
    std::string json_output = "experiments/dataset_scale_results.json";
    std::vector<ScaleBenchmarkResult> scale_results;

    // Sweeps across each scale
    for (const auto& sc : scales) {
        std::cout << "\n========================================================================================\n";
        std::cout << " RUNNING SCALE: " << sc.name << " (" << sc.n << " Vectors, " << sc.eval_queries << " Queries)\n";
        std::cout << "========================================================================================\n";

        // Load dataset up to sc.n
        std::vector<float> data;
        size_t n_samples = 0, dim = 0;
        std::cout << "[*] Loading " << sc.n << " vectors from " << base_data_file << " ... " << std::flush;
        if (!load_fvecs(base_data_file, data, n_samples, dim, sc.n)) {
            std::cerr << "[-] Error loading data vectors!\n";
            continue;
        }
        std::cout << "Done! (" << n_samples << " vectors loaded)\n";

        // Load ground truth
        std::vector<std::vector<uint32_t>> gt;
        std::cout << "[*] Loading ground truth from " << sc.gt_file << " ... " << std::flush;
        load_ivecs(sc.gt_file, gt, sc.eval_queries);
        std::cout << "Done! (" << gt.size() << " ground truth rows)\n";

        size_t test_q_count = std::min(sc.eval_queries, gt.size());

        // -------------------------------------------------------------
        // A. Baseline HNSW (M=16, efC=200, uniform)
        // -------------------------------------------------------------
        PolicyConfig base_policy;
        base_policy.m_base = 16;
        base_policy.ef_c_base = 200;
        base_policy.enable_dynamic_m_efc = false;
        base_policy.enable_layer_scaling = false;
        base_policy.enable_hubness_regulation = false;
        base_policy.enable_ada_ef = false;

        std::cout << "\n  [*] Building Baseline HNSW (M=16, efC=200, 12 threads) ... " << std::flush;
        auto t0_base = std::chrono::high_resolution_clock::now();
        AdaptiveHNSWIndex base_index(dim, SpaceType::L2, false, base_policy);
        base_index.batch_insert(data.data(), n_samples, n_threads);
        auto t1_base = std::chrono::high_resolution_clock::now();
        double base_build_s = std::chrono::duration<double>(t1_base - t0_base).count();
        double base_vps = (double)n_samples / base_build_s;
        size_t base_edges = base_index.get_total_edges();
        double base_ram = (double)base_index.get_total_memory_bytes() / (1024.0 * 1024.0);
        std::cout << "Done! (" << std::fixed << std::setprecision(1) << base_build_s << "s, "
                  << std::setprecision(0) << base_vps << " v/s, "
                  << base_edges << " edges, "
                  << std::setprecision(1) << base_ram << " MB)\n";

        // Baseline Query Evaluation
        std::cout << "  [*] Searching Baseline HNSW (" << test_q_count << " queries, ef=64, k=10) ... " << std::flush;
        base_index.total_dist_computations = 0;
        auto q0_base = std::chrono::high_resolution_clock::now();
        auto base_res = base_index.batch_search(queries.data(), test_q_count, 10, 64, n_threads);
        auto q1_base = std::chrono::high_resolution_clock::now();
        double base_q_ms = std::chrono::duration<double, std::milli>(q1_base - q0_base).count();
        double base_qps = (double)test_q_count / (base_q_ms / 1000.0);
        double base_recall = compute_recall(base_res, gt, 10);
        double base_dist_per_q = (double)base_index.total_dist_computations.load() / (double)test_q_count;
        std::cout << "Done! (" << std::fixed << std::setprecision(1) << base_qps << " QPS, Recall: "
                  << std::setprecision(4) << base_recall << ", Avg Evals: "
                  << std::setprecision(1) << base_dist_per_q << ")\n";

        // -------------------------------------------------------------
        // B. AdaptiveVec (M∈[8,24], efC∈[40,220], λ=0.75, μ=0.15, p=6)
        // -------------------------------------------------------------
        PolicyConfig adapt_policy;
        adapt_policy.m_base = 16;
        adapt_policy.m_min = 8;
        adapt_policy.m_max = 24;
        adapt_policy.ef_c_base = 120;
        adapt_policy.ef_c_min = 40;
        adapt_policy.ef_c_max = 220;
        adapt_policy.lambda_layer = 0.75f;
        adapt_policy.hubness_mu = 0.15f;
        adapt_policy.ada_ef_patience = 6;
        adapt_policy.enable_dynamic_m_efc = true;
        adapt_policy.enable_layer_scaling = true;
        adapt_policy.enable_hubness_regulation = true;
        adapt_policy.enable_ada_ef = true;

        std::cout << "\n  [*] Building AdaptiveVec (Dynamic Manifold, 12 threads) ... " << std::flush;
        auto t0_adapt = std::chrono::high_resolution_clock::now();
        AdaptiveHNSWIndex adapt_index(dim, SpaceType::L2, true, adapt_policy);
        adapt_index.batch_insert(data.data(), n_samples, n_threads);
        auto t1_adapt = std::chrono::high_resolution_clock::now();
        double adapt_build_s = std::chrono::duration<double>(t1_adapt - t0_adapt).count();
        double adapt_vps = (double)n_samples / adapt_build_s;
        size_t adapt_edges = adapt_index.get_total_edges();
        double adapt_ram = (double)adapt_index.get_total_memory_bytes() / (1024.0 * 1024.0);
        std::cout << "Done! (" << std::fixed << std::setprecision(1) << adapt_build_s << "s, "
                  << std::setprecision(0) << adapt_vps << " v/s, "
                  << adapt_edges << " edges, "
                  << std::setprecision(1) << adapt_ram << " MB)\n";

        // AdaptiveVec Query Evaluation
        std::cout << "  [*] Searching AdaptiveVec (" << test_q_count << " queries, ef=64, k=10, p=6) ... " << std::flush;
        adapt_index.total_dist_computations = 0;
        auto q0_adapt = std::chrono::high_resolution_clock::now();
        auto adapt_res = adapt_index.batch_search(queries.data(), test_q_count, 10, 64, n_threads);
        auto q1_adapt = std::chrono::high_resolution_clock::now();
        double adapt_q_ms = std::chrono::duration<double, std::milli>(q1_adapt - q0_adapt).count();
        double adapt_qps = (double)test_q_count / (adapt_q_ms / 1000.0);
        double adapt_recall = compute_recall(adapt_res, gt, 10);
        double adapt_dist_per_q = (double)adapt_index.total_dist_computations.load() / (double)test_q_count;
        std::cout << "Done! (" << std::fixed << std::setprecision(1) << adapt_qps << " QPS, Recall: "
                  << std::setprecision(4) << adapt_recall << ", Avg Evals: "
                  << std::setprecision(1) << adapt_dist_per_q << ")\n";

        // Comparisons
        double edge_diff = ((double)base_edges - (double)adapt_edges) / (double)base_edges * 100.0;
        double qps_sp = adapt_qps / base_qps;
        double b_sp = base_build_s / adapt_build_s;
        double ram_diff = (base_ram - adapt_ram) / base_ram * 100.0;

        ScaleBenchmarkResult rec;
        rec.n_vectors = sc.n;
        rec.base_build_time_s = base_build_s;
        rec.base_build_rate_vps = base_vps;
        rec.base_edges = base_edges;
        rec.base_ram_mb = base_ram;
        rec.base_qps = base_qps;
        rec.base_recall = base_recall;
        rec.base_dist_per_q = base_dist_per_q;

        rec.adapt_build_time_s = adapt_build_s;
        rec.adapt_build_rate_vps = adapt_vps;
        rec.adapt_edges = adapt_edges;
        rec.adapt_ram_mb = adapt_ram;
        rec.adapt_qps = adapt_qps;
        rec.adapt_recall = adapt_recall;
        rec.adapt_dist_per_q = adapt_dist_per_q;

        rec.edge_reduction_pct = edge_diff;
        rec.qps_speedup = qps_sp;
        rec.build_speedup = b_sp;
        rec.ram_savings_pct = ram_diff;

        scale_results.push_back(rec);

        // Print Summary Row
        std::cout << "\n----------------------------------------------------------------------------------------\n";
        std::cout << " SUMMARY FOR " << sc.name << ":\n";
        std::cout << "  • Graph Edges:    " << base_edges << " (Baseline) -> " << adapt_edges << " (AdaptiveVec, -" << std::fixed << std::setprecision(1) << edge_diff << "% edges)\n";
        std::cout << "  • Build Time:     " << base_build_s << "s -> " << adapt_build_s << "s (" << std::setprecision(2) << b_sp << "x build speedup)\n";
        std::cout << "  • Throughput:     " << base_qps << " QPS -> " << adapt_qps << " QPS (" << std::setprecision(2) << qps_sp << "x query speedup)\n";
        std::cout << "  • Recall@10:      " << std::setprecision(4) << base_recall << " -> " << adapt_recall << "\n";
        std::cout << "  • Distance Evals: " << std::setprecision(1) << base_dist_per_q << " -> " << adapt_dist_per_q << "/query\n";
        std::cout << "----------------------------------------------------------------------------------------\n";

        // Incrementally update JSON results
        write_json_results(scale_results, json_output);
        std::cout << "[+] Incremental results safely flushed to " << json_output << "\n";
    }

    std::cout << "\n[+] ALL LARGE-SCALE BENCHMARKS COMPLETED SUCCESSFULLY!\n";
    return 0;
}
