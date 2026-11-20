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
#include <queue>
#include <numeric>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace adaptivevec;

struct StrictRNGAblationResult {
    std::string exp_id;
    std::string name;
    std::string dataset;
    size_t n_vectors = 0;
    size_t dim = 0;
    
    // Graph Connectivity
    size_t total_edges = 0;
    size_t layer0_edges = 0;
    size_t upper_layer_edges = 0;
    double mean_degree_l0 = 0.0;
    double in_degree_var = 0.0;
    uint32_t max_in_degree = 0;
    
    // Build Performance
    double build_time_s = 0.0;
    double memory_mb = 0.0;
    
    // Strict RNG & Fallback Retention Stats
    uint64_t total_heuristic_calls = 0;
    uint64_t fallback_invocations = 0;
    double fallback_invocation_rate_pct = 0.0;
    uint64_t fallback_edges_added = 0;
    
    // Connectivity & Graph Topology
    size_t wcc_components = 1;
    double wcc_largest_pct = 100.0;
    bool no_duplicate_edges = true;
    bool no_self_loops = true;

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

// Compute Weakly Connected Components on Layer 0
std::pair<size_t, size_t> compute_wcc(const std::vector<std::vector<tableint>>& adj, size_t n) {
    std::vector<std::vector<tableint>> undir(n);
    for (size_t u = 0; u < n; ++u) {
        if (u < adj.size()) {
            for (tableint v : adj[u]) {
                if (v < n) {
                    undir[u].push_back(v);
                    undir[v].push_back(u);
                }
            }
        }
    }
    std::vector<bool> visited(n, false);
    size_t num_comp = 0;
    size_t max_comp = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!visited[i]) {
            num_comp++;
            size_t c_size = 0;
            std::queue<tableint> q;
            q.push(i);
            visited[i] = true;
            while (!q.empty()) {
                tableint u = q.front();
                q.pop();
                c_size++;
                for (tableint v : undir[u]) {
                    if (!visited[v]) {
                        visited[v] = true;
                        q.push(v);
                    }
                }
            }
            if (c_size > max_comp) max_comp = c_size;
        }
    }
    return {num_comp, max_comp};
}

StrictRNGAblationResult run_strict_rng_experiment(
    const std::string& exp_id,
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
    StrictRNGAblationResult res;
    res.exp_id = exp_id;
    res.name = name;
    res.dataset = dataset_name;
    res.n_vectors = n;
    res.dim = dim;

    std::cout << "\n>>> [" << exp_id << "] " << name << " on " << dataset_name << " (N=" << n << ")..." << std::endl;

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

    // Structural checks and in-degree variance
    bool has_dup = false;
    bool has_self_loop = false;
    double sum_in = 0, sum_in_sq = 0;
    uint32_t max_in = 0;

    for (size_t u = 0; u < n; ++u) {
        if (u < index.graphs[0].size()) {
            const auto& adj = index.graphs[0][u];
            std::unordered_set<tableint> seen;
            for (tableint v : adj) {
                if (v == u) has_self_loop = true;
                if (seen.find(v) != seen.end()) has_dup = true;
                seen.insert(v);
            }
        }
        uint32_t in_d = (u < index.in_degrees.size()) ? index.in_degrees[u] : 0;
        sum_in += in_d;
        sum_in_sq += (double)in_d * in_d;
        if (in_d > max_in) max_in = in_d;
    }
    res.no_duplicate_edges = !has_dup;
    res.no_self_loops = !has_self_loop;
    res.max_in_degree = max_in;
    double mean_in = sum_in / (double)n;
    res.in_degree_var = (sum_in_sq / (double)n) - (mean_in * mean_in);

    // Connected components on Layer 0
    auto [wcc_count, max_wcc] = compute_wcc(index.graphs[0], n);
    res.wcc_components = wcc_count;
    res.wcc_largest_pct = 100.0 * (double)max_wcc / (double)n;

    // Search Evaluation across Queries
    std::vector<double> r1(nq, 0.0), r10(nq, 0.0), r100(nq, 0.0);
    std::vector<double> latencies_ms(nq, 0.0);
    uint64_t total_upper_evals = 0;
    uint64_t total_layer0_evals = 0;

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
            if (k < 1 && gt1.find(cid) != gt1.end()) m1++;
            if (k < 10 && gt10.find(cid) != gt10.end()) m10++;
            if (k < 100 && gt100.find(cid) != gt100.end()) m100++;
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
              << ", FallbackRate=" << std::setprecision(1) << res.fallback_invocation_rate_pct << "%"
              << ", WCC=" << res.wcc_components << " (Max: " << res.wcc_largest_pct << "%)" << std::endl;

    return res;
}

int main() {
    std::cout << "=============================================================" << std::endl;
    std::cout << "  AdaptiveVec Phase 5.2: Strict RNG & Fallback Ablation Suite " << std::endl;
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

    std::cout << "Dataset: SIFT-100K (N=" << sift_n << ", D=" << sift_dim << ", NQ=" << sift_nq << ")" << std::endl;

    std::vector<StrictRNGAblationResult> results;

    // 1. Canonical HNSW with standard retention
    {
        PolicyConfig cfg;
        cfg.keep_pruned_connections = true;
        cfg.strict_rng_layer0_only = false;
        cfg.strict_rng_upper_layers_only = false;
        cfg.min_degree_safeguard = 0;
        cfg.enable_hubness_regulation = false;
        results.push_back(run_strict_rng_experiment(
            "Exp1-Canonical-Standard", "Canonical HNSW (Standard Retention)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, false
        ));
    }

    // 2. Architecture B with standard retention
    {
        PolicyConfig cfg;
        cfg.keep_pruned_connections = true;
        cfg.strict_rng_layer0_only = false;
        cfg.strict_rng_upper_layers_only = false;
        cfg.min_degree_safeguard = 0;
        cfg.enable_hubness_regulation = true;
        results.push_back(run_strict_rng_experiment(
            "Exp2-ArchB-Standard", "Architecture B (Adaptive M/efC, Standard Retention)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, true
        ));
    }

    // 3. Canonical HNSW with Strict RNG (keep_pruned = false)
    {
        PolicyConfig cfg;
        cfg.keep_pruned_connections = false;
        cfg.strict_rng_layer0_only = false;
        cfg.strict_rng_upper_layers_only = false;
        cfg.min_degree_safeguard = 0;
        cfg.enable_hubness_regulation = false;
        results.push_back(run_strict_rng_experiment(
            "Exp3-Canonical-StrictRNG", "Canonical HNSW + Strict RNG (keep_pruned=false)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, false
        ));
    }

    // 4. Architecture B with Strict RNG (keep_pruned = false)
    {
        PolicyConfig cfg;
        cfg.keep_pruned_connections = false;
        cfg.strict_rng_layer0_only = false;
        cfg.strict_rng_upper_layers_only = false;
        cfg.min_degree_safeguard = 0;
        cfg.enable_hubness_regulation = true;
        results.push_back(run_strict_rng_experiment(
            "Exp4-ArchB-StrictRNG", "Architecture B + Strict RNG (keep_pruned=false)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, true
        ));
    }

    // 5. Strict RNG Layer 0 only (Upper layers retain pruned fallback)
    {
        PolicyConfig cfg;
        cfg.keep_pruned_connections = false;
        cfg.strict_rng_layer0_only = true;
        cfg.strict_rng_upper_layers_only = false;
        cfg.min_degree_safeguard = 0;
        cfg.enable_hubness_regulation = true;
        results.push_back(run_strict_rng_experiment(
            "Exp5-ArchB-StrictRNG-L0Only", "Architecture B + Strict RNG (Layer 0 Only)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, true
        ));
    }

    // 6. Strict RNG Upper layers only (Layer 0 retains pruned fallback)
    {
        PolicyConfig cfg;
        cfg.keep_pruned_connections = false;
        cfg.strict_rng_layer0_only = false;
        cfg.strict_rng_upper_layers_only = true;
        cfg.min_degree_safeguard = 0;
        cfg.enable_hubness_regulation = true;
        results.push_back(run_strict_rng_experiment(
            "Exp6-ArchB-StrictRNG-UpperOnly", "Architecture B + Strict RNG (Upper Layers Only)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, true
        ));
    }

    // 7. Strict RNG with minimum-degree safeguard (M_min = 6)
    {
        PolicyConfig cfg;
        cfg.keep_pruned_connections = false;
        cfg.strict_rng_layer0_only = false;
        cfg.strict_rng_upper_layers_only = false;
        cfg.min_degree_safeguard = 6;
        cfg.enable_hubness_regulation = true;
        results.push_back(run_strict_rng_experiment(
            "Exp7-ArchB-StrictRNG-Safeguard6", "Architecture B + Strict RNG (Min-Degree Safeguard M_min=6)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, true
        ));
    }

    // 8. Geometry-Conditioned Pruning (GCP) with standard retention
    {
        PolicyConfig cfg;
        cfg.keep_pruned_connections = true;
        cfg.strict_rng_layer0_only = false;
        cfg.strict_rng_upper_layers_only = false;
        cfg.min_degree_safeguard = 0;
        cfg.enable_hubness_regulation = true;
        cfg.enable_geometry_pruning = true;
        cfg.pruning_alpha_base = 1.0f;
        cfg.pruning_lid_weight = 0.15f;
        cfg.pruning_density_weight = 0.10f;
        results.push_back(run_strict_rng_experiment(
            "Exp8-GCP-Standard", "Geometry-Conditioned Pruning (Adaptive Alpha + Standard Retention)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
            cfg, true
        ));
    }

    // Export JSON
    std::string out_path = "experiments/phase_5_2_strict_rng_ablation.json";
    std::ofstream out(out_path);
    out << "{\n";
    out << "  \"experiment_name\": \"Phase 5.2 Strict RNG and Fallback Retention Ablation\",\n";
    out << "  \"timestamp\": \"2026-11-20T13:00:00Z\",\n";
    out << "  \"dataset\": \"SIFT-100K\",\n";
    out << "  \"num_vectors\": " << sift_n << ",\n";
    out << "  \"dimension\": " << sift_dim << ",\n";
    out << "  \"num_queries\": " << sift_nq << ",\n";
    out << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"exp_id\": \"" << r.exp_id << "\",\n";
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
        out << "      \"wcc_components\": " << r.wcc_components << ",\n";
        out << "      \"wcc_largest_pct\": " << r.wcc_largest_pct << ",\n";
        out << "      \"no_duplicate_edges\": " << (r.no_duplicate_edges ? "true" : "false") << ",\n";
        out << "      \"no_self_loops\": " << (r.no_self_loops ? "true" : "false") << ",\n";
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
        out << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    out.close();

    std::cout << "\n>>> Phase 5.2 Strict RNG Ablation JSON written to " << out_path << std::endl;
    return 0;
}
