#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <numeric>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace adaptivevec;

struct SchemeResult {
    std::string scheme_id;
    std::string name;
    std::string dataset;
    size_t n_vectors = 0;
    size_t dim = 0;
    
    // Connectivity
    size_t total_edges = 0;
    size_t layer0_edges = 0;
    size_t upper_layer_edges = 0;
    double mean_degree_l0 = 0.0;
    double in_degree_var = 0.0;
    uint32_t max_in_degree = 0;
    double memory_mb = 0.0;
    double build_time_s = 0.0;
    
    // Strict RNG & Fallback Stats
    uint64_t total_heuristic_calls = 0;
    uint64_t fallback_invocations = 0;
    double fallback_invocation_rate_pct = 0.0;
    uint64_t fallback_edges_added = 0;
    
    // Search Performance (efSearch = 64)
    double recall_at_1 = 0.0;
    double recall_at_10 = 0.0;
    double recall_at_100 = 0.0;
    double qps = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double mean_upper_dist_evals = 0.0;
    double mean_layer0_dist_evals = 0.0;
    double mean_total_dist_evals = 0.0;
};

SchemeResult run_scheme(
    const std::string& scheme_id,
    const std::string& name,
    const std::string& dataset_name,
    const std::vector<float>& base_data,
    size_t n,
    size_t dim,
    const std::vector<float>& queries,
    size_t nq,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    PolicyConfig policy,
    bool is_adaptive,
    SpaceType space = SpaceType::L2,
    int ef_search = 64
) {
    SchemeResult res;
    res.scheme_id = scheme_id;
    res.name = name;
    res.dataset = dataset_name;
    res.n_vectors = n;
    res.dim = dim;

    std::cout << "\n>>> [" << scheme_id << "] " << name << " on " << dataset_name << " (N=" << n << ")..." << std::endl;

    AdaptiveHNSWIndex index(dim, space, is_adaptive, policy);
    index.reserve(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        index.insert(&base_data[i * dim]);
        if ((i + 1) % 25000 == 0) {
            std::cout << "  Inserted " << (i + 1) << " / " << n << std::endl << std::flush;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    res.build_time_s = std::chrono::duration<double>(t1 - t0).count();
    res.total_edges = index.get_total_edges();
    res.layer0_edges = index.get_layer0_edges();
    res.upper_layer_edges = (res.total_edges >= res.layer0_edges) ? (res.total_edges - res.layer0_edges) : 0;
    res.mean_degree_l0 = (double)res.layer0_edges / (double)n;
    res.memory_mb = (double)index.get_total_memory_bytes() / (1024.0 * 1024.0);

    // Fallback retention counters
    res.total_heuristic_calls = index.heuristic_calls.load();
    res.fallback_invocations = index.fallback_retention_invocations.load();
    res.fallback_edges_added = index.fallback_edges_added.load();
    res.fallback_invocation_rate_pct = (res.total_heuristic_calls > 0)
        ? (100.0 * (double)res.fallback_invocations / (double)res.total_heuristic_calls) : 0.0;

    // Structural in-degree metrics
    double sum_in = 0, sum_in_sq = 0;
    uint32_t max_in = 0;
    for (size_t u = 0; u < n; ++u) {
        uint32_t in_d = (u < index.in_degrees.size()) ? index.in_degrees[u] : 0;
        sum_in += in_d;
        sum_in_sq += (double)in_d * in_d;
        if (in_d > max_in) max_in = in_d;
    }
    res.max_in_degree = max_in;
    double mean_in = sum_in / (double)n;
    res.in_degree_var = (sum_in_sq / (double)n) - (mean_in * mean_in);

    // Search Evaluation
    std::vector<double> r1(nq, 0.0), r10(nq, 0.0), r100(nq, 0.0);
    std::vector<double> latencies_ms(nq, 0.0);
    uint64_t total_upper_evals = 0, total_layer0_evals = 0;

    auto s0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_upper_evals, total_layer0_evals)
    for (size_t q = 0; q < nq; ++q) {
        auto q_start = std::chrono::high_resolution_clock::now();
        QuerySearchStats stats;
        auto results = index.search(&queries[q * dim], 100, ef_search, stats);
        auto q_end = std::chrono::high_resolution_clock::now();
        latencies_ms[q] = std::chrono::duration<double, std::milli>(q_end - q_start).count();
        total_upper_evals += stats.upper_layer_dist_evals;
        total_layer0_evals += stats.layer0_dist_evals;

        std::unordered_set<uint32_t> gt1, gt10, gt100;
        if (q < ground_truth.size()) {
            for (size_t k = 0; k < 1 && k < ground_truth[q].size(); ++k) gt1.insert(ground_truth[q][k]);
            for (size_t k = 0; k < 10 && k < ground_truth[q].size(); ++k) gt10.insert(ground_truth[q][k]);
            for (size_t k = 0; k < 100 && k < ground_truth[q].size(); ++k) gt100.insert(ground_truth[q][k]);
        }

        size_t m1 = 0, m10 = 0, m100 = 0;
        for (size_t k = 0; k < results.size(); ++k) {
            tableint cid = results[k].id;
            if (k < 1 && gt1.count(cid)) m1++;
            if (k < 10 && gt10.count(cid)) m10++;
            if (k < 100 && gt100.count(cid)) m100++;
        }
        r1[q] = (double)m1 / 1.0;
        r10[q] = (double)m10 / 10.0;
        r100[q] = (double)m100 / 100.0;
    }
    auto s1 = std::chrono::high_resolution_clock::now();

    double total_search_s = std::chrono::duration<double>(s1 - s0).count();
    res.qps = (double)nq / total_search_s;
    res.recall_at_1 = std::accumulate(r1.begin(), r1.end(), 0.0) / (double)nq;
    res.recall_at_10 = std::accumulate(r10.begin(), r10.end(), 0.0) / (double)nq;
    res.recall_at_100 = std::accumulate(r100.begin(), r100.end(), 0.0) / (double)nq;
    res.mean_upper_dist_evals = (double)total_upper_evals / (double)nq;
    res.mean_layer0_dist_evals = (double)total_layer0_evals / (double)nq;
    res.mean_total_dist_evals = (double)(total_upper_evals + total_layer0_evals) / (double)nq;

    std::sort(latencies_ms.begin(), latencies_ms.end());
    res.p50_latency_ms = latencies_ms[nq / 2];
    res.p95_latency_ms = latencies_ms[(size_t)(nq * 0.95)];
    res.p99_latency_ms = latencies_ms[(size_t)(nq * 0.99)];

    std::cout << "  Results: Recall@10=" << std::fixed << std::setprecision(2) << res.recall_at_10 * 100.0
              << "%, Recall@1=" << res.recall_at_1 * 100.0 << "%, QPS=" << std::setprecision(1) << res.qps
              << ", MeanDist=" << res.mean_total_dist_evals << " (L0: " << res.mean_layer0_dist_evals << ", Upper: " << res.mean_upper_dist_evals << ")"
              << ", Edges=" << res.total_edges << " (L0: " << res.layer0_edges << ", Up: " << res.upper_layer_edges << ")"
              << ", Build=" << std::setprecision(2) << res.build_time_s << "s"
              << ", FallbackRate=" << std::setprecision(1) << res.fallback_invocation_rate_pct << "%" << std::endl;

    return res;
}

int main() {
    std::cout << "=============================================================" << std::endl;
    std::cout << "  AdaptiveVec Phase 5.2: Layer Allocation Suite & Ablations  " << std::endl;
    std::cout << "=============================================================" << std::endl;

    // Load SIFT-100K
    std::string sift_base_path = "data/sift_subset_100k.fvecs";
    std::string sift_query_path = "data/sift_query.fvecs";
    std::string sift_gt_path = "data/sift_subset_groundtruth.ivecs";

    size_t sift_n = 0, sift_dim = 0;
    std::vector<float> sift_data;
    load_fvecs(sift_base_path, sift_data, sift_n, sift_dim);
    size_t sift_nq = 0, sift_qdim = 0;
    std::vector<float> sift_queries;
    load_fvecs(sift_query_path, sift_queries, sift_nq, sift_qdim);
    std::vector<std::vector<uint32_t>> sift_gt;
    load_ivecs(sift_gt_path, sift_gt);

    std::cout << "Loaded SIFT-100K: N=" << sift_n << ", D=" << sift_dim << ", NQ=" << sift_nq << std::endl;

    // Load Transformer-384D
    std::string tf_base_path = "data/transformer_384d/transformer_384d_base.fvecs";
    std::string tf_query_path = "data/transformer_384d/transformer_384d_query.fvecs";
    std::string tf_gt_path = "data/transformer_384d/transformer_384d_groundtruth.ivecs";

    size_t tf_n = 0, tf_dim = 0;
    std::vector<float> tf_data;
    load_fvecs(tf_base_path, tf_data, tf_n, tf_dim);
    size_t tf_nq = 0, tf_qdim = 0;
    std::vector<float> tf_queries;
    load_fvecs(tf_query_path, tf_queries, tf_nq, tf_qdim);
    std::vector<std::vector<uint32_t>> tf_gt;
    load_ivecs(tf_gt_path, tf_gt);

    std::cout << "Loaded Transformer-384D: N=" << tf_n << ", D=" << tf_dim << ", NQ=" << tf_nq << std::endl;

    std::vector<SchemeResult> all_results;

    // -------------------------------------------------------------
    // Evaluation on SIFT-100K
    // -------------------------------------------------------------
    std::cout << "\n=============================================================" << std::endl;
    std::cout << "  Evaluating 6 Allocation Schemes on SIFT-100K                " << std::endl;
    std::cout << "=============================================================" << std::endl;

    // Scheme 1: Canonical HNSW Baseline (Standard Retention)
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.ef_c_base = 120;
        cfg.enable_layer_scaling = false;
        cfg.enable_hubness_regulation = false;
        cfg.keep_pruned_connections = true;
        all_results.push_back(run_scheme(
            "SIFT-Scheme1-Canonical-Standard", "Canonical HNSW (Uniform M=16, Standard Retention)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, false
        ));
    }

    // Scheme 2: Architecture B (Monolithic Coupled Scaling)
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.ef_c_base = 120;
        cfg.m_min = 8;
        cfg.m_max = 24;
        cfg.ef_c_min = 40;
        cfg.ef_c_max = 220;
        cfg.alpha_lid = 0.5f;
        cfg.beta_density = 0.5f;
        cfg.enable_layer_scaling = true;
        cfg.lambda_layer = 0.75f;
        cfg.m_min_layer = 4;
        cfg.enable_hubness_regulation = true;
        cfg.hubness_mu = 0.15f;
        cfg.use_streaming_welford = true;
        cfg.keep_pruned_connections = true;
        all_results.push_back(run_scheme(
            "SIFT-Scheme2-ArchB-Monolithic", "Architecture B (Monolithic Coupled Scaling, Standard Retention)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, true
        ));
    }

    // Scheme 3: Pure Layer-0 Decoupled (Standard Retention)
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.m_base_upper = 16;
        cfg.ef_c_base = 120;
        cfg.m_min = 8;
        cfg.m_max = 24;
        cfg.ef_c_min = 40;
        cfg.ef_c_max = 220;
        cfg.alpha_lid = 0.5f;
        cfg.beta_density = 0.5f;
        cfg.enable_layer_scaling = false;
        cfg.enable_layer_decoupled = true;
        cfg.beta_layer0 = 1.0f;
        cfg.beta_upper = 0.0f; // Fixed upper layers
        cfg.enable_hubness_regulation = true;
        cfg.hubness_mu = 0.15f;
        cfg.use_streaming_welford = true;
        cfg.keep_pruned_connections = true;
        all_results.push_back(run_scheme(
            "SIFT-Scheme3-Decoupled-L0-Standard", "Pure Layer-0 Decoupled (Fixed Upper M=16, Standard Retention)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, true
        ));
    }

    // Scheme 4: Pure Layer-0 Decoupled + Strict RNG
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.m_base_upper = 16;
        cfg.ef_c_base = 120;
        cfg.m_min = 8;
        cfg.m_max = 24;
        cfg.ef_c_min = 40;
        cfg.ef_c_max = 220;
        cfg.alpha_lid = 0.5f;
        cfg.beta_density = 0.5f;
        cfg.enable_layer_scaling = false;
        cfg.enable_layer_decoupled = true;
        cfg.beta_layer0 = 1.0f;
        cfg.beta_upper = 0.0f; // Fixed upper layers
        cfg.enable_hubness_regulation = true;
        cfg.hubness_mu = 0.15f;
        cfg.use_streaming_welford = true;
        cfg.keep_pruned_connections = false; // Strict RNG
        all_results.push_back(run_scheme(
            "SIFT-Scheme4-Decoupled-L0-StrictRNG", "Pure Layer-0 Decoupled + Strict RNG (Fixed Upper M=16, keep_pruned=false)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, true
        ));
    }

    // Scheme 5: Boosted Upper Decoupled + Strict RNG
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.m_base_upper = 24; // Boosted upper layers
        cfg.ef_c_base = 120;
        cfg.m_min = 8;
        cfg.m_max = 24;
        cfg.ef_c_min = 40;
        cfg.ef_c_max = 220;
        cfg.alpha_lid = 0.5f;
        cfg.beta_density = 0.5f;
        cfg.enable_layer_scaling = false;
        cfg.enable_layer_decoupled = true;
        cfg.beta_layer0 = 1.0f;
        cfg.beta_upper = 0.0f;
        cfg.enable_hubness_regulation = true;
        cfg.hubness_mu = 0.15f;
        cfg.use_streaming_welford = true;
        cfg.keep_pruned_connections = false;
        all_results.push_back(run_scheme(
            "SIFT-Scheme5-Boosted-Upper-StrictRNG", "Boosted Upper Decoupled + Strict RNG (Boosted Upper M=24, keep_pruned=false)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, true
        ));
    }

    // Scheme 6: Matched Budget Control (M_base calibrated so total edges match Canonical within +-1%)
    {
        PolicyConfig cfg;
        cfg.m_base = 18; // Calibrated M_base to match canonical edge budget under adaptive allocation
        cfg.m_base_upper = 16;
        cfg.ef_c_base = 120;
        cfg.m_min = 10;
        cfg.m_max = 26;
        cfg.ef_c_min = 40;
        cfg.ef_c_max = 220;
        cfg.alpha_lid = 0.5f;
        cfg.beta_density = 0.5f;
        cfg.enable_layer_scaling = false;
        cfg.enable_layer_decoupled = true;
        cfg.beta_layer0 = 1.0f;
        cfg.beta_upper = 0.0f;
        cfg.enable_hubness_regulation = true;
        cfg.hubness_mu = 0.15f;
        cfg.use_streaming_welford = true;
        cfg.keep_pruned_connections = true;
        all_results.push_back(run_scheme(
            "SIFT-Scheme6-Matched-Budget", "Matched Edge Budget Control (M_base=18, Decoupled L0)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, true
        ));
    }

    // -------------------------------------------------------------
    // Evaluation on Transformer-384D
    // -------------------------------------------------------------
    std::cout << "\n=============================================================" << std::endl;
    std::cout << "  Evaluating Core Schemes on Transformer-384D                 " << std::endl;
    std::cout << "=============================================================" << std::endl;

    // TF-1: Canonical HNSW Baseline
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.ef_c_base = 120;
        cfg.enable_layer_scaling = false;
        cfg.enable_hubness_regulation = false;
        cfg.keep_pruned_connections = true;
        all_results.push_back(run_scheme(
            "TF-Scheme1-Canonical-Standard", "Canonical HNSW on Transformer-384D",
            "Transformer-384D", tf_data, tf_n, tf_dim, tf_queries, tf_nq, tf_gt,
            cfg, false
        ));
    }

    // TF-2: Architecture B (Monolithic Coupled)
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.ef_c_base = 120;
        cfg.m_min = 8;
        cfg.m_max = 24;
        cfg.ef_c_min = 40;
        cfg.ef_c_max = 220;
        cfg.alpha_lid = 0.5f;
        cfg.beta_density = 0.5f;
        cfg.enable_layer_scaling = true;
        cfg.lambda_layer = 0.75f;
        cfg.m_min_layer = 4;
        cfg.enable_hubness_regulation = true;
        cfg.hubness_mu = 0.15f;
        cfg.use_streaming_welford = true;
        cfg.keep_pruned_connections = true;
        all_results.push_back(run_scheme(
            "TF-Scheme2-ArchB-Monolithic", "Architecture B on Transformer-384D",
            "Transformer-384D", tf_data, tf_n, tf_dim, tf_queries, tf_nq, tf_gt,
            cfg, true
        ));
    }

    // TF-3: Pure Layer-0 Decoupled + Strict RNG
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.m_base_upper = 16;
        cfg.ef_c_base = 120;
        cfg.m_min = 8;
        cfg.m_max = 24;
        cfg.ef_c_min = 40;
        cfg.ef_c_max = 220;
        cfg.alpha_lid = 0.5f;
        cfg.beta_density = 0.5f;
        cfg.enable_layer_scaling = false;
        cfg.enable_layer_decoupled = true;
        cfg.beta_layer0 = 1.0f;
        cfg.beta_upper = 0.0f;
        cfg.enable_hubness_regulation = true;
        cfg.hubness_mu = 0.15f;
        cfg.use_streaming_welford = true;
        cfg.keep_pruned_connections = false; // Strict RNG
        all_results.push_back(run_scheme(
            "TF-Scheme4-Decoupled-L0-StrictRNG", "Pure Layer-0 Decoupled + Strict RNG on Transformer-384D",
            "Transformer-384D", tf_data, tf_n, tf_dim, tf_queries, tf_nq, tf_gt,
            cfg, true
        ));
    }

    // Export Complete JSON Deliverables
    // 1. experiments/phase_5_2_layer_allocation_ablation.json
    {
        std::string out_path = "experiments/phase_5_2_layer_allocation_ablation.json";
        std::ofstream out(out_path);
        out << "{\n";
        out << "  \"experiment_name\": \"Phase 5.2 Layer-Decoupled Allocation Ablation Suite\",\n";
        out << "  \"timestamp\": \"2026-11-20T13:20:00Z\",\n";
        out << "  \"results\": [\n";
        for (size_t i = 0; i < all_results.size(); ++i) {
            const auto& r = all_results[i];
            out << "    {\n";
            out << "      \"scheme_id\": \"" << r.scheme_id << "\",\n";
            out << "      \"name\": \"" << r.name << "\",\n";
            out << "      \"dataset\": \"" << r.dataset << "\",\n";
            out << "      \"total_edges\": " << r.total_edges << ",\n";
            out << "      \"layer0_edges\": " << r.layer0_edges << ",\n";
            out << "      \"upper_layer_edges\": " << r.upper_layer_edges << ",\n";
            out << "      \"mean_degree_l0\": " << r.mean_degree_l0 << ",\n";
            out << "      \"in_degree_var\": " << r.in_degree_var << ",\n";
            out << "      \"max_in_degree\": " << r.max_in_degree << ",\n";
            out << "      \"build_time_s\": " << r.build_time_s << ",\n";
            out << "      \"memory_mb\": " << r.memory_mb << ",\n";
            out << "      \"total_heuristic_calls\": " << r.total_heuristic_calls << ",\n";
            out << "      \"fallback_invocations\": " << r.fallback_invocations << ",\n";
            out << "      \"fallback_invocation_rate_pct\": " << r.fallback_invocation_rate_pct << ",\n";
            out << "      \"fallback_edges_added\": " << r.fallback_edges_added << ",\n";
            out << "      \"recall_at_1\": " << r.recall_at_1 << ",\n";
            out << "      \"recall_at_10\": " << r.recall_at_10 << ",\n";
            out << "      \"recall_at_100\": " << r.recall_at_100 << ",\n";
            out << "      \"qps\": " << r.qps << ",\n";
            out << "      \"p50_latency_ms\": " << r.p50_latency_ms << ",\n";
            out << "      \"p95_latency_ms\": " << r.p95_latency_ms << ",\n";
            out << "      \"p99_latency_ms\": " << r.p99_latency_ms << ",\n";
            out << "      \"mean_upper_dist_evals\": " << r.mean_upper_dist_evals << ",\n";
            out << "      \"mean_layer0_dist_evals\": " << r.mean_layer0_dist_evals << ",\n";
            out << "      \"mean_total_dist_evals\": " << r.mean_total_dist_evals << "\n";
            out << "    }" << (i + 1 < all_results.size() ? "," : "") << "\n";
        }
        out << "  ]\n";
        out << "}\n";
        out.close();
        std::cout << "\n>>> Saved experiments/phase_5_2_layer_allocation_ablation.json" << std::endl;
    }

    // 2. experiments/phase_5_2_matched_budget_results.json
    {
        std::string out_path = "experiments/phase_5_2_matched_budget_results.json";
        std::ofstream out(out_path);
        out << "{\n";
        out << "  \"experiment_name\": \"Phase 5.2 Matched Edge Budget Comparisons\",\n";
        out << "  \"timestamp\": \"2026-11-20T13:20:00Z\",\n";
        out << "  \"comparisons\": [\n";
        // Pair Canonical vs Scheme 6 Matched Budget
        out << "    {\n";
        out << "      \"baseline\": \"Canonical HNSW\",\n";
        out << "      \"baseline_edges\": " << all_results[0].total_edges << ",\n";
        out << "      \"baseline_recall_at_10\": " << all_results[0].recall_at_10 << ",\n";
        out << "      \"baseline_qps\": " << all_results[0].qps << ",\n";
        out << "      \"baseline_dist_evals\": " << all_results[0].mean_total_dist_evals << ",\n";
        out << "      \"matched_scheme\": \"Scheme 6 (Matched Budget Decoupled L0)\",\n";
        out << "      \"matched_edges\": " << all_results[5].total_edges << ",\n";
        out << "      \"edge_delta_pct\": " << (100.0 * ((double)all_results[5].total_edges - (double)all_results[0].total_edges) / (double)all_results[0].total_edges) << ",\n";
        out << "      \"matched_recall_at_10\": " << all_results[5].recall_at_10 << ",\n";
        out << "      \"recall_delta_pct\": " << ((all_results[5].recall_at_10 - all_results[0].recall_at_10) * 100.0) << ",\n";
        out << "      \"matched_qps\": " << all_results[5].qps << ",\n";
        out << "      \"qps_speedup_pct\": " << (100.0 * (all_results[5].qps - all_results[0].qps) / all_results[0].qps) << ",\n";
        out << "      \"matched_dist_evals\": " << all_results[5].mean_total_dist_evals << ",\n";
        out << "      \"dist_evals_reduction_pct\": " << (100.0 * (all_results[0].mean_total_dist_evals - all_results[5].mean_total_dist_evals) / all_results[0].mean_total_dist_evals) << "\n";
        out << "    }\n";
        out << "  ]\n";
        out << "}\n";
        out.close();
        std::cout << ">>> Saved experiments/phase_5_2_matched_budget_results.json" << std::endl;
    }

    // 3. experiments/phase_5_2_graph_statistics.json
    {
        std::string out_path = "experiments/phase_5_2_graph_statistics.json";
        std::ofstream out(out_path);
        out << "{\n";
        out << "  \"experiment_name\": \"Phase 5.2 Comprehensive Graph Topography Statistics\",\n";
        out << "  \"timestamp\": \"2026-11-20T13:20:00Z\",\n";
        out << "  \"schemes\": [\n";
        for (size_t i = 0; i < all_results.size(); ++i) {
            const auto& r = all_results[i];
            out << "    {\n";
            out << "      \"scheme_id\": \"" << r.scheme_id << "\",\n";
            out << "      \"dataset\": \"" << r.dataset << "\",\n";
            out << "      \"total_edges\": " << r.total_edges << ",\n";
            out << "      \"layer0_edges\": " << r.layer0_edges << ",\n";
            out << "      \"upper_layer_edges\": " << r.upper_layer_edges << ",\n";
            out << "      \"mean_degree_l0\": " << r.mean_degree_l0 << ",\n";
            out << "      \"in_degree_var\": " << r.in_degree_var << ",\n";
            out << "      \"max_in_degree\": " << r.max_in_degree << ",\n";
            out << "      \"fallback_invocation_rate_pct\": " << r.fallback_invocation_rate_pct << ",\n";
            out << "      \"memory_mb\": " << r.memory_mb << "\n";
            out << "    }" << (i + 1 < all_results.size() ? "," : "") << "\n";
        }
        out << "  ]\n";
        out << "}\n";
        out.close();
        std::cout << ">>> Saved experiments/phase_5_2_graph_statistics.json" << std::endl;
    }

    return 0;
}
