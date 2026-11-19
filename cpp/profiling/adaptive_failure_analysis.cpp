/**
 * AdaptiveVec Phase 0.5 Isolated Adaptive Policy Failure Diagnostic
 * Evaluates isolated policy components on SIFT-100K to pinpoint
 * exact mechanisms causing recall regression.
 */

#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <unordered_set>
#include <algorithm>

using namespace adaptivevec;

float compute_recall(const std::vector<std::vector<Candidate>>& results, 
                     const std::vector<std::vector<uint32_t>>& ground_truth, 
                     int k = 10) {
    size_t total_matches = 0;
    size_t total_queries = results.size();
    if (total_queries == 0 || k == 0) return 0.0f;
    for (size_t q = 0; q < total_queries; ++q) {
        std::unordered_set<uint32_t> gt_set(ground_truth[q].begin(), ground_truth[q].begin() + k);
        for (int i = 0; i < std::min((int)results[q].size(), k); ++i) {
            if (gt_set.find(results[q][i].id) != gt_set.end()) {
                total_matches++;
            }
        }
    }
    return (float)total_matches / (float)(total_queries * k);
}

struct RunResult {
    std::string name;
    double build_time_s;
    size_t total_edges;
    double edge_delta_pct;
    float recall_10;
    double qps;
};

RunResult evaluate_config(
    const std::string& name,
    const std::vector<float>& data,
    const std::vector<float>& queries,
    const std::vector<std::vector<uint32_t>>& gt,
    size_t n_samples,
    size_t dim,
    size_t n_queries,
    PolicyConfig cfg,
    bool is_adaptive,
    int ef_search,
    size_t baseline_edges
) {
    std::cout << "  -> Testing: " << std::left << std::setw(36) << name << " ... " << std::flush;
    AdaptiveHNSWIndex index(dim, SpaceType::L2, is_adaptive, cfg);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    index.batch_insert(data.data(), n_samples);
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t1 - t0).count();

    size_t total_edges = index.get_total_edges();
    double edge_delta = (baseline_edges > 0)
        ? ((double)total_edges - (double)baseline_edges) / (double)baseline_edges * 100.0
        : 0.0;

    // Search
    auto q0 = std::chrono::high_resolution_clock::now();
    auto results = index.batch_search(queries.data(), n_queries, 10, ef_search);
    auto q1 = std::chrono::high_resolution_clock::now();
    double query_s = std::chrono::duration<double>(q1 - q0).count();
    double qps = (double)n_queries / query_s;
    float recall = compute_recall(results, gt, 10);

    std::cout << "Done! (Recall@10: " << std::fixed << std::setprecision(4) << recall 
              << ", Edges: " << total_edges << " [" << (edge_delta >= 0 ? "+" : "") << std::setprecision(1) << edge_delta << "%], "
              << "QPS: " << std::setprecision(1) << qps << ")" << std::endl;

    return {name, build_s, total_edges, edge_delta, recall, qps};
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "   AdaptiveVec Isolated Policy Failure Diagnostic (SIFT-100K)    " << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    std::vector<float> base_data;
    size_t n_samples = 0, dim = 0;
    load_fvecs("data/sift_subset_100k.fvecs", base_data, n_samples, dim, 100000);

    std::vector<float> queries;
    size_t n_queries = 0, q_dim = 0;
    load_fvecs("data/sift_query.fvecs", queries, n_queries, q_dim, 10000);

    std::vector<std::vector<uint32_t>> gt;
    load_ivecs("data/sift_subset_groundtruth.ivecs", gt, n_queries);

    int ef_search = 64;
    std::vector<RunResult> results;

    // 1. Baseline Canonical HNSW (M=16, efC=200)
    PolicyConfig cfg_base;
    cfg_base.enable_dynamic_m_efc = false;
    cfg_base.enable_layer_scaling = false;
    cfg_base.enable_hubness_regulation = false;
    cfg_base.enable_ada_ef = false;
    cfg_base.m_base = 16;
    cfg_base.ef_c_base = 200;
    auto r_base = evaluate_config("1. Baseline Canonical HNSW", base_data, queries, gt, n_samples, dim, n_queries, cfg_base, false, ef_search, 0);
    results.push_back(r_base);
    size_t base_edges = r_base.total_edges;

    // 2. Isolated: Adaptive M ONLY (efC fixed at 200)
    PolicyConfig cfg_m = cfg_base;
    cfg_m.enable_dynamic_m_efc = true;
    cfg_m.m_base = 16;
    cfg_m.m_min = 8;
    cfg_m.m_max = 24;
    cfg_m.ef_c_base = 200;
    cfg_m.ef_c_min = 200;
    cfg_m.ef_c_max = 200;
    results.push_back(evaluate_config("2. Isolated: Adaptive M Only", base_data, queries, gt, n_samples, dim, n_queries, cfg_m, true, ef_search, base_edges));

    // 3. Isolated: Adaptive efC ONLY (M fixed at 16)
    PolicyConfig cfg_efc = cfg_base;
    cfg_efc.enable_dynamic_m_efc = true;
    cfg_efc.m_base = 16;
    cfg_efc.m_min = 16;
    cfg_efc.m_max = 16;
    cfg_efc.ef_c_base = 120;
    cfg_efc.ef_c_min = 40;
    cfg_efc.ef_c_max = 220;
    results.push_back(evaluate_config("3. Isolated: Adaptive efC Only", base_data, queries, gt, n_samples, dim, n_queries, cfg_efc, true, ef_search, base_edges));

    // 4. Combined: Adaptive M(x) & efC(x)
    PolicyConfig cfg_m_efc = cfg_base;
    cfg_m_efc.enable_dynamic_m_efc = true;
    cfg_m_efc.m_base = 16;
    cfg_m_efc.m_min = 8;
    cfg_m_efc.m_max = 24;
    cfg_m_efc.ef_c_base = 120;
    cfg_m_efc.ef_c_min = 40;
    cfg_m_efc.ef_c_max = 220;
    results.push_back(evaluate_config("4. Adaptive M + efC Combined", base_data, queries, gt, n_samples, dim, n_queries, cfg_m_efc, true, ef_search, base_edges));

    // 5. Isolated: Layer Scaling ONLY (lambda = 0.75, base HNSW)
    PolicyConfig cfg_layer = cfg_base;
    cfg_layer.enable_layer_scaling = true;
    cfg_layer.lambda_layer = 0.75f;
    cfg_layer.m_min_layer = 4;
    results.push_back(evaluate_config("5. Isolated: Layer Scaling Only", base_data, queries, gt, n_samples, dim, n_queries, cfg_layer, true, ef_search, base_edges));

    // 6. Isolated: Hubness Regulation ONLY (mu = 0.15, base HNSW)
    PolicyConfig cfg_hub = cfg_base;
    cfg_hub.enable_hubness_regulation = true;
    cfg_hub.hubness_mu = 0.15f;
    results.push_back(evaluate_config("6. Isolated: Hubness Regulation Only", base_data, queries, gt, n_samples, dim, n_queries, cfg_hub, true, ef_search, base_edges));

    // 7. Base Adaptive Graph (Dynamic M/efC + Layer Scaling + Hubness)
    PolicyConfig cfg_adapt_graph = cfg_m_efc;
    cfg_adapt_graph.enable_layer_scaling = true;
    cfg_adapt_graph.enable_hubness_regulation = true;
    cfg_adapt_graph.hubness_mu = 0.15f;
    results.push_back(evaluate_config("7. Full Adaptive Graph (No Ada-ef)", base_data, queries, gt, n_samples, dim, n_queries, cfg_adapt_graph, true, ef_search, base_edges));

    // 8. Full Adaptive with Ada-ef (patience=6, epsilon=1e-4)
    PolicyConfig cfg_ada_ef = cfg_adapt_graph;
    cfg_ada_ef.enable_ada_ef = true;
    cfg_ada_ef.ada_ef_patience = 6;
    cfg_ada_ef.ada_ef_epsilon = 1e-4f;
    results.push_back(evaluate_config("8. Full Adaptive + Ada-ef (p=6)", base_data, queries, gt, n_samples, dim, n_queries, cfg_ada_ef, true, ef_search, base_edges));

    // Summary Table
    std::cout << "\n==================================================================================================" << std::endl;
    std::cout << "                        ISOLATED ABLATION ROOT-CAUSE SUMMARY TABLE                                " << std::endl;
    std::cout << "==================================================================================================" << std::endl;
    std::cout << std::left << std::setw(38) << "Component Configuration" 
              << std::setw(14) << "Build (s)" 
              << std::setw(15) << "Edges" 
              << std::setw(14) << "Edge Change" 
              << std::setw(14) << "Recall@10" 
              << "QPS" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto& r : results) {
        std::stringstream edge_ch;
        edge_ch << (r.edge_delta_pct >= 0 ? "+" : "") << std::fixed << std::setprecision(2) << r.edge_delta_pct << "%";
        std::cout << std::left << std::setw(38) << r.name 
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.build_time_s 
                  << std::setw(15) << r.total_edges 
                  << std::setw(14) << edge_ch.str() 
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.recall_10 
                  << std::setprecision(1) << r.qps << std::endl;
    }
    std::cout << "==================================================================================================\n" << std::endl;

    return 0;
}
