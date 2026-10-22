/**
 * run_transformer_benchmarks.cpp
 * Frontier 4: High-Dimensional Transformer Embeddings (384D) Benchmark Harness
 * 
 * Evaluates AdaptiveVec vs. Baseline HNSW on genuine 384-dimensional sentence transformer
 * embeddings (sentence-transformers/all-MiniLM-L6-v2) across 5 distinct semantic sub-manifolds:
 * SciFact (scientific claims), NFCorpus (medical/nutrition), ArguAna (adversarial debate),
 * Webmasters (web engineering), and Android (system engineering).
 * 
 * Measures:
 * 1. Severe Hubness Mitigation in R^384 (In-Degree Variance, Max In-Degree, Bottleneck Reduction)
 * 2. Ultra-Edge 4-Bit Sub-Byte Quantization (SQ4) compression and AVX2 SIMD QPS
 * 3. Frontier 1 Dynamic efSearch Pareto optimization in high-dimensional spaces
 * 4. Multi-Domain Clustering and Manifold-Aware Degree Allocation M(x)
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
#include <numeric>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace adaptivevec;

struct TransformerTrialRecord {
    std::string config_name;
    std::string quant_mode;
    int ef_search;
    int rerank_factor;
    double vector_ram_mb;
    double edge_ram_mb;
    double total_ram_mb;
    double build_time_s;
    size_t total_edges;
    size_t layer0_edges;
    double recall_10;
    double qps;
    double mean_dist_evals;
    double p50_us;
    double p95_us;
    double p99_us;
    
    // Hubness metrics
    double in_degree_mean;
    double in_degree_var;
    uint32_t in_degree_max;
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

void compute_hubness_stats(
    const std::vector<uint32_t>& in_degrees,
    double& mean_deg,
    double& var_deg,
    uint32_t& max_deg
) {
    if (in_degrees.empty()) {
        mean_deg = 0.0;
        var_deg = 0.0;
        max_deg = 0;
        return;
    }
    max_deg = 0;
    double sum = 0.0;
    for (uint32_t d : in_degrees) {
        sum += d;
        if (d > max_deg) max_deg = d;
    }
    mean_deg = sum / in_degrees.size();
    double sq_diff_sum = 0.0;
    for (uint32_t d : in_degrees) {
        double diff = (double)d - mean_deg;
        sq_diff_sum += diff * diff;
    }
    var_deg = sq_diff_sum / in_degrees.size();
}

int main(int argc, char** argv) {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "  Frontier 4: High-Dimensional Transformer Embeddings (384D) Benchmark   " << std::endl;
    std::cout << "  MiniLM-50K-MultiDomain (SciFact + NFCorpus + ArguAna + Webmasters + Android)" << std::endl;
    std::cout << "==========================================================================\n" << std::endl;

    std::string base_path = "data/transformer_384d/transformer_384d_base.fvecs";
    std::string query_path = "data/transformer_384d/transformer_384d_query.fvecs";
    std::string gt_path = "data/transformer_384d/transformer_384d_groundtruth.ivecs";

    std::vector<float> base_vectors;
    size_t n_base = 0, dim_base = 0;
    std::cout << "Loading base vectors from: " << base_path << "..." << std::endl;
    if (!load_fvecs(base_path, base_vectors, n_base, dim_base)) {
        std::cerr << "Failed to load base vectors!" << std::endl;
        return 1;
    }
    std::cout << "Loaded base: " << n_base << " vectors, " << dim_base << " dimensions.\n" << std::endl;

    std::vector<float> query_vectors;
    size_t n_queries = 0, dim_queries = 0;
    std::cout << "Loading query vectors from: " << query_path << "..." << std::endl;
    if (!load_fvecs(query_path, query_vectors, n_queries, dim_queries)) {
        std::cerr << "Failed to load query vectors!" << std::endl;
        return 1;
    }
    std::cout << "Loaded queries: " << n_queries << " queries, " << dim_queries << " dimensions.\n" << std::endl;

    std::vector<std::vector<uint32_t>> ground_truth;
    std::cout << "Loading ground truth from: " << gt_path << "..." << std::endl;
    if (!load_ivecs(gt_path, ground_truth)) {
        std::cerr << "Failed to load ground truth!" << std::endl;
        return 1;
    }
    std::cout << "Loaded ground truth for " << ground_truth.size() << " queries.\n" << std::endl;

    std::vector<TransformerTrialRecord> all_records;

    // Define test sweeps
    std::vector<int> ef_search_values = {32, 64, 128};
    const int K_NEIGHBORS = 10;

    // ------------------------------------------------------------------------
    // Experiment 1: Stock HNSW Baseline (FP32, M=16, efC=120)
    // ------------------------------------------------------------------------
    {
        std::cout << "--------------------------------------------------------------------------" << std::endl;
        std::cout << ">>> Building Stock HNSW Baseline (FP32, M=16, efC=120)..." << std::endl;
        PolicyConfig base_policy;
        base_policy.m_base = 16;
        base_policy.ef_c_base = 120;
        base_policy.enable_dynamic_m_efc = false;
        base_policy.enable_layer_scaling = false;
        base_policy.enable_hubness_regulation = false;
        base_policy.enable_ada_ef = false;

        AdaptiveHNSWIndex base_idx(dim_base, SpaceType::L2, false, base_policy);

        auto t0_build = std::chrono::high_resolution_clock::now();
        base_idx.batch_insert(base_vectors.data(), n_base);
        auto t1_build = std::chrono::high_resolution_clock::now();
        double build_time = std::chrono::duration<double>(t1_build - t0_build).count();

        double v_ram = (double)base_idx.get_vector_memory_bytes() / (1024.0 * 1024.0);
        double e_ram = (double)base_idx.get_edge_memory_bytes() / (1024.0 * 1024.0);
        double tot_ram = (double)base_idx.get_total_memory_bytes() / (1024.0 * 1024.0);
        size_t total_edges = base_idx.get_total_edges();
        size_t l0_edges = base_idx.get_layer0_edges();

        double mean_deg, var_deg;
        uint32_t max_deg;
        compute_hubness_stats(base_idx.in_degrees, mean_deg, var_deg, max_deg);

        std::cout << "  Build Time: " << std::fixed << std::setprecision(2) << build_time << " s (" 
                  << std::setprecision(0) << ((double)n_base / build_time) << " vec/s)" << std::endl;
        std::cout << "  Total Edges: " << total_edges << " (Layer 0: " << l0_edges << ")" << std::endl;
        std::cout << "  Total RAM: " << tot_ram << " MB (Vector: " << v_ram << " MB, Edge: " << e_ram << " MB)" << std::endl;
        std::cout << "  Hubness Metrics -> Mean Deg: " << mean_deg << ", Var Deg: " << var_deg << ", Max Deg: " << max_deg << std::endl;

        for (int ef : ef_search_values) {
            base_idx.total_dist_computations = 0;
            std::vector<double> latencies_us(n_queries);
            std::vector<std::vector<Candidate>> results(n_queries);

            auto t0_q = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(dynamic, 32)
            for (size_t q = 0; q < n_queries; ++q) {
                auto t_start = std::chrono::high_resolution_clock::now();
                results[q] = base_idx.search(&query_vectors[q * dim_base], K_NEIGHBORS, ef);
                auto t_end = std::chrono::high_resolution_clock::now();
                latencies_us[q] = std::chrono::duration<double, std::micro>(t_end - t_start).count();
            }
            auto t1_q = std::chrono::high_resolution_clock::now();
            double total_time_s = std::chrono::duration<double>(t1_q - t0_q).count();

            std::sort(latencies_us.begin(), latencies_us.end());
            double p50 = latencies_us[(size_t)(0.50 * n_queries)];
            double p95 = latencies_us[(size_t)(0.95 * n_queries)];
            double p99 = latencies_us[(size_t)(0.99 * n_queries)];
            double qps = (double)n_queries / total_time_s;
            float recall = evaluate_recall(results, ground_truth, K_NEIGHBORS);
            double mean_evals = (double)base_idx.total_dist_computations.load() / (double)n_queries;

            std::cout << "    ef=" << std::setw(3) << ef << " | Recall@10: " << std::fixed << std::setprecision(4) << recall
                      << " | QPS: " << std::setw(7) << std::setprecision(1) << qps
                      << " | Evals/q: " << std::setw(6) << mean_evals
                      << " | Latency p50: " << p50 << " us, p95: " << p95 << " us" << std::endl;

            all_records.push_back({
                "Stock HNSW (FP32)", "FP32", ef, 1,
                v_ram, e_ram, tot_ram, build_time, total_edges, l0_edges,
                recall, qps, mean_evals, p50, p95, p99,
                mean_deg, var_deg, max_deg
            });
        }
    }

    // ------------------------------------------------------------------------
    // Experiment 2: AdaptiveVec (FP32 - Topological Adaptations + Stagnation Exit)
    // ------------------------------------------------------------------------
    {
        std::cout << "\n--------------------------------------------------------------------------" << std::endl;
        std::cout << ">>> Building AdaptiveVec (FP32: Dynamic M(x) + Layer Scaling + Hubness Reg)..." << std::endl;
        PolicyConfig ada_policy;
        ada_policy.m_base = 16;
        ada_policy.m_min = 8;
        ada_policy.m_max = 24;
        ada_policy.ef_c_base = 120;
        ada_policy.ef_c_min = 40;
        ada_policy.ef_c_max = 220;
        ada_policy.enable_dynamic_m_efc = true;
        ada_policy.enable_layer_scaling = true;
        ada_policy.enable_hubness_regulation = true;
        ada_policy.hubness_mu = 0.15f;
        ada_policy.enable_ada_ef = true; // Stagnation early exit

        AdaptiveHNSWIndex ada_idx(dim_base, SpaceType::L2, true, ada_policy);

        auto t0_build = std::chrono::high_resolution_clock::now();
        ada_idx.batch_insert(base_vectors.data(), n_base);
        auto t1_build = std::chrono::high_resolution_clock::now();
        double build_time = std::chrono::duration<double>(t1_build - t0_build).count();

        double v_ram = (double)ada_idx.get_vector_memory_bytes() / (1024.0 * 1024.0);
        double e_ram = (double)ada_idx.get_edge_memory_bytes() / (1024.0 * 1024.0);
        double tot_ram = (double)ada_idx.get_total_memory_bytes() / (1024.0 * 1024.0);
        size_t total_edges = ada_idx.get_total_edges();
        size_t l0_edges = ada_idx.get_layer0_edges();

        double mean_deg, var_deg;
        uint32_t max_deg;
        compute_hubness_stats(ada_idx.in_degrees, mean_deg, var_deg, max_deg);

        std::cout << "  Build Time: " << std::fixed << std::setprecision(2) << build_time << " s ("
                  << std::setprecision(0) << ((double)n_base / build_time) << " vec/s)" << std::endl;
        std::cout << "  Total Edges: " << total_edges << " (Layer 0: " << l0_edges << ")" << std::endl;
        std::cout << "  Total RAM: " << tot_ram << " MB (Vector: " << v_ram << " MB, Edge: " << e_ram << " MB)" << std::endl;
        std::cout << "  Hubness Metrics -> Mean Deg: " << mean_deg << ", Var Deg: " << var_deg << ", Max Deg: " << max_deg << std::endl;

        for (int ef : ef_search_values) {
            ada_idx.total_dist_computations = 0;
            std::vector<double> latencies_us(n_queries);
            std::vector<std::vector<Candidate>> results(n_queries);

            auto t0_q = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(dynamic, 32)
            for (size_t q = 0; q < n_queries; ++q) {
                auto t_start = std::chrono::high_resolution_clock::now();
                results[q] = ada_idx.search(&query_vectors[q * dim_base], K_NEIGHBORS, ef);
                auto t_end = std::chrono::high_resolution_clock::now();
                latencies_us[q] = std::chrono::duration<double, std::micro>(t_end - t_start).count();
            }
            auto t1_q = std::chrono::high_resolution_clock::now();
            double total_time_s = std::chrono::duration<double>(t1_q - t0_q).count();

            std::sort(latencies_us.begin(), latencies_us.end());
            double p50 = latencies_us[(size_t)(0.50 * n_queries)];
            double p95 = latencies_us[(size_t)(0.95 * n_queries)];
            double p99 = latencies_us[(size_t)(0.99 * n_queries)];
            double qps = (double)n_queries / total_time_s;
            float recall = evaluate_recall(results, ground_truth, K_NEIGHBORS);
            double mean_evals = (double)ada_idx.total_dist_computations.load() / (double)n_queries;

            std::cout << "    ef=" << std::setw(3) << ef << " | Recall@10: " << std::fixed << std::setprecision(4) << recall
                      << " | QPS: " << std::setw(7) << std::setprecision(1) << qps
                      << " | Evals/q: " << std::setw(6) << mean_evals
                      << " | Latency p50: " << p50 << " us, p95: " << p95 << " us" << std::endl;

            all_records.push_back({
                "AdaptiveVec (FP32)", "FP32", ef, 1,
                v_ram, e_ram, tot_ram, build_time, total_edges, l0_edges,
                recall, qps, mean_evals, p50, p95, p99,
                mean_deg, var_deg, max_deg
            });
        }

        // --------------------------------------------------------------------
        // Experiment 3: AdaptiveVec + Frontier 1 Dynamic efSearch (kappa = 0.4, 0.6)
        // --------------------------------------------------------------------
        std::cout << "\n>>> Evaluating Frontier 1 Dynamic efSearch on AdaptiveVec Index..." << std::endl;
        std::vector<float> kappas = {0.40f, 0.60f};
        for (float kappa : kappas) {
            ada_idx.policy.enable_dynamic_ef_search = true;
            ada_idx.policy.dynamic_ef_factor = kappa;
            ada_idx.total_dist_computations = 0;

            std::vector<double> latencies_us(n_queries);
            std::vector<std::vector<Candidate>> results(n_queries);

            auto t0_q = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(dynamic, 32)
            for (size_t q = 0; q < n_queries; ++q) {
                auto t_start = std::chrono::high_resolution_clock::now();
                results[q] = ada_idx.search(&query_vectors[q * dim_base], K_NEIGHBORS, 64);
                auto t_end = std::chrono::high_resolution_clock::now();
                latencies_us[q] = std::chrono::duration<double, std::micro>(t_end - t_start).count();
            }
            auto t1_q = std::chrono::high_resolution_clock::now();
            double total_time_s = std::chrono::duration<double>(t1_q - t0_q).count();

            std::sort(latencies_us.begin(), latencies_us.end());
            double p50 = latencies_us[(size_t)(0.50 * n_queries)];
            double p95 = latencies_us[(size_t)(0.95 * n_queries)];
            double p99 = latencies_us[(size_t)(0.99 * n_queries)];
            double qps = (double)n_queries / total_time_s;
            float recall = evaluate_recall(results, ground_truth, K_NEIGHBORS);
            double mean_evals = (double)ada_idx.total_dist_computations.load() / (double)n_queries;

            std::cout << "    Dynamic ef (kappa=" << kappa << ") | Recall@10: " << std::fixed << std::setprecision(4) << recall
                      << " | QPS: " << std::setw(7) << std::setprecision(1) << qps
                      << " | Evals/q: " << std::setw(6) << mean_evals
                      << " | Latency p50: " << p50 << " us" << std::endl;

            all_records.push_back({
                "AdaptiveVec + Dynamic ef (" + std::to_string((int)(kappa * 100)) + "%)", "FP32", 64, 1,
                v_ram, e_ram, tot_ram, build_time, total_edges, l0_edges,
                recall, qps, mean_evals, p50, p95, p99,
                mean_deg, var_deg, max_deg
            });
        }
        ada_idx.policy.enable_dynamic_ef_search = false;
    }

    // ------------------------------------------------------------------------
    // Experiment 4: AdaptiveVec + Milestone 1 (Ultra-Edge 4-Bit SQ4 Sub-Byte)
    // ------------------------------------------------------------------------
    {
        std::cout << "\n--------------------------------------------------------------------------" << std::endl;
        std::cout << ">>> Building AdaptiveVec + Milestone 1 (4-Bit Sub-Byte SQ4 Quantizer)..." << std::endl;
        PolicyConfig sq4_policy;
        sq4_policy.m_base = 16;
        sq4_policy.m_min = 8;
        sq4_policy.m_max = 24;
        sq4_policy.ef_c_base = 120;
        sq4_policy.ef_c_min = 40;
        sq4_policy.ef_c_max = 220;
        sq4_policy.enable_dynamic_m_efc = true;
        sq4_policy.enable_layer_scaling = true;
        sq4_policy.enable_hubness_regulation = true;
        sq4_policy.enable_ada_ef = false; // Pure quantization evaluation
        sq4_policy.enable_sq4 = true;
        sq4_policy.sq4_rerank_factor = 2; // R = 2 * k = 20

        AdaptiveHNSWIndex sq4_idx(dim_base, SpaceType::L2, true, sq4_policy);

        std::cout << "  Training 4-Bit Sub-Byte Scalar Quantizer on 384D vectors..." << std::endl;
        sq4_idx.train_quantizer(base_vectors.data(), std::min((size_t)10000, n_base));

        auto t0_build = std::chrono::high_resolution_clock::now();
        sq4_idx.batch_insert(base_vectors.data(), n_base);
        auto t1_build = std::chrono::high_resolution_clock::now();
        double build_time = std::chrono::duration<double>(t1_build - t0_build).count();

        double v_ram = (double)sq4_idx.get_vector_memory_bytes() / (1024.0 * 1024.0);
        double e_ram = (double)sq4_idx.get_edge_memory_bytes() / (1024.0 * 1024.0);
        double tot_ram = (double)sq4_idx.get_total_memory_bytes() / (1024.0 * 1024.0);
        size_t total_edges = sq4_idx.get_total_edges();
        size_t l0_edges = sq4_idx.get_layer0_edges();

        double mean_deg, var_deg;
        uint32_t max_deg;
        compute_hubness_stats(sq4_idx.in_degrees, mean_deg, var_deg, max_deg);

        std::cout << "  Build Time: " << std::fixed << std::setprecision(2) << build_time << " s ("
                  << std::setprecision(0) << ((double)n_base / build_time) << " vec/s)" << std::endl;
        std::cout << "  Total Edges: " << total_edges << " (Layer 0: " << l0_edges << ")" << std::endl;
        std::cout << "  Total RAM: " << tot_ram << " MB (Vector: " << v_ram << " MB, Edge: " << e_ram << " MB)" << std::endl;
        std::cout << "  Hubness Metrics -> Mean Deg: " << mean_deg << ", Var Deg: " << var_deg << ", Max Deg: " << max_deg << std::endl;

        for (int ef : ef_search_values) {
            sq4_idx.total_dist_computations = 0;
            std::vector<double> latencies_us(n_queries);
            std::vector<std::vector<Candidate>> results(n_queries);

            auto t0_q = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(dynamic, 32)
            for (size_t q = 0; q < n_queries; ++q) {
                auto t_start = std::chrono::high_resolution_clock::now();
                results[q] = sq4_idx.search(&query_vectors[q * dim_base], K_NEIGHBORS, ef);
                auto t_end = std::chrono::high_resolution_clock::now();
                latencies_us[q] = std::chrono::duration<double, std::micro>(t_end - t_start).count();
            }
            auto t1_q = std::chrono::high_resolution_clock::now();
            double total_time_s = std::chrono::duration<double>(t1_q - t0_q).count();

            std::sort(latencies_us.begin(), latencies_us.end());
            double p50 = latencies_us[(size_t)(0.50 * n_queries)];
            double p95 = latencies_us[(size_t)(0.95 * n_queries)];
            double p99 = latencies_us[(size_t)(0.99 * n_queries)];
            double qps = (double)n_queries / total_time_s;
            float recall = evaluate_recall(results, ground_truth, K_NEIGHBORS);
            double mean_evals = (double)sq4_idx.total_dist_computations.load() / (double)n_queries;

            std::cout << "    ef=" << std::setw(3) << ef << " | Recall@10: " << std::fixed << std::setprecision(4) << recall
                      << " | QPS: " << std::setw(7) << std::setprecision(1) << qps
                      << " | Evals/q: " << std::setw(6) << mean_evals
                      << " | Latency p50: " << p50 << " us" << std::endl;

            all_records.push_back({
                "AdaptiveVec (SQ4 Sub-Byte)", "SQ4", ef, 2,
                v_ram, e_ram, tot_ram, build_time, total_edges, l0_edges,
                recall, qps, mean_evals, p50, p95, p99,
                mean_deg, var_deg, max_deg
            });
        }
    }

    // ------------------------------------------------------------------------
    // Export Results to JSON
    // ------------------------------------------------------------------------
    std::string out_json_path = "experiments/transformer_384d_results.json";
    std::cout << "\n--------------------------------------------------------------------------" << std::endl;
    std::cout << "Exporting benchmark results to: " << out_json_path << "..." << std::endl;

    std::ofstream ofs(out_json_path);
    ofs << "{\n";
    ofs << "  \"benchmark\": \"Frontier 4: High-Dimensional Transformer Embeddings (384D)\",\n";
    ofs << "  \"dataset\": \"MiniLM-50K-MultiDomain (SciFact + NFCorpus + ArguAna + Webmasters + Android)\",\n";
    ofs << "  \"n_base\": " << n_base << ",\n";
    ofs << "  \"n_queries\": " << n_queries << ",\n";
    ofs << "  \"dim\": " << dim_base << ",\n";
    ofs << "  \"records\": [\n";

    for (size_t i = 0; i < all_records.size(); ++i) {
        const auto& r = all_records[i];
        ofs << "    {\n";
        ofs << "      \"config_name\": \"" << r.config_name << "\",\n";
        ofs << "      \"quant_mode\": \"" << r.quant_mode << "\",\n";
        ofs << "      \"ef_search\": " << r.ef_search << ",\n";
        ofs << "      \"rerank_factor\": " << r.rerank_factor << ",\n";
        ofs << "      \"vector_ram_mb\": " << std::fixed << std::setprecision(2) << r.vector_ram_mb << ",\n";
        ofs << "      \"edge_ram_mb\": " << r.edge_ram_mb << ",\n";
        ofs << "      \"total_ram_mb\": " << r.total_ram_mb << ",\n";
        ofs << "      \"build_time_s\": " << r.build_time_s << ",\n";
        ofs << "      \"total_edges\": " << r.total_edges << ",\n";
        ofs << "      \"layer0_edges\": " << r.layer0_edges << ",\n";
        ofs << "      \"recall_10\": " << std::setprecision(4) << r.recall_10 << ",\n";
        ofs << "      \"qps\": " << std::setprecision(1) << r.qps << ",\n";
        ofs << "      \"mean_dist_evals\": " << r.mean_dist_evals << ",\n";
        ofs << "      \"p50_us\": " << r.p50_us << ",\n";
        ofs << "      \"p95_us\": " << r.p95_us << ",\n";
        ofs << "      \"p99_us\": " << r.p99_us << ",\n";
        ofs << "      \"in_degree_mean\": " << r.in_degree_mean << ",\n";
        ofs << "      \"in_degree_var\": " << r.in_degree_var << ",\n";
        ofs << "      \"in_degree_max\": " << r.in_degree_max << "\n";
        ofs << "    }" << (i + 1 < all_records.size() ? "," : "") << "\n";
    }

    ofs << "  ]\n";
    ofs << "}\n";
    ofs.close();

    std::cout << "[SUCCESS] Benchmark completed successfully! Results written to " << out_json_path << std::endl;
    return 0;
}
