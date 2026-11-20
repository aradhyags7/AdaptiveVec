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

struct LayerGeometricStats {
    int layer = 0;
    size_t node_count = 0;
    size_t edge_count = 0;
    double mean_degree = 0.0;
    uint32_t max_degree = 0;
    double in_degree_var = 0.0;
    double memory_mb = 0.0;
    
    // Edge Length Distribution (Euclidean)
    double mean_edge_length = 0.0;
    double p50_edge_length = 0.0;
    double p95_edge_length = 0.0;
    double p99_edge_length = 0.0;

    // Angular Diversity / Directional Redundancy
    double mean_cos_similarity = 0.0; // Mean pairwise cosine between outgoing edge vectors
};

struct LayerSearchProfiling {
    int ef_search = 64;
    double recall_at_10 = 0.0;
    double qps = 0.0;
    double mean_upper_dist_evals = 0.0;
    double mean_layer0_dist_evals = 0.0;
    double mean_total_dist_evals = 0.0;
    double upper_eval_fraction_pct = 0.0;
    double mean_upper_hops = 0.0;
    double mean_layer0_hops = 0.0;
    double mean_handover_distance_to_query = 0.0;
};

int main() {
    std::cout << "=============================================================" << std::endl;
    std::cout << "  AdaptiveVec Phase 5.2: Layer Profiler & Search Profiling   " << std::endl;
    std::cout << "=============================================================" << std::endl;

    // Load SIFT-100K
    std::string sift_base_path = "data/sift_subset_100k.fvecs";
    std::string sift_query_path = "data/sift_query.fvecs";
    std::string sift_gt_path = "data/sift_subset_groundtruth.ivecs";

    size_t sift_n = 0, sift_dim = 0;
    std::vector<float> sift_data;
    if (!load_fvecs(sift_base_path, sift_data, sift_n, sift_dim)) {
        std::cerr << "Failed to load SIFT-100K base!" << std::endl;
        return 1;
    }
    size_t sift_nq = 0, sift_qdim = 0;
    std::vector<float> sift_queries;
    load_fvecs(sift_query_path, sift_queries, sift_nq, sift_qdim);
    std::vector<std::vector<uint32_t>> sift_gt;
    load_ivecs(sift_gt_path, sift_gt);

    std::cout << "Loaded SIFT-100K: N=" << sift_n << ", D=" << sift_dim << ", Queries=" << sift_nq << std::endl;

    // Build Architecture B Index
    std::cout << "\n>>> Building Architecture B Index for Layer Characterization..." << std::endl;
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

    AdaptiveHNSWIndex index(sift_dim, SpaceType::L2, true, cfg);
    index.reserve(sift_n);
    for (size_t i = 0; i < sift_n; ++i) {
        index.insert(&sift_data[i * sift_dim]);
    }

    int max_l = index.max_level;
    std::cout << "Index built. Max level: " << max_l << ", Total Edges: " << index.get_total_edges() << std::endl;

    // -------------------------------------------------------------
    // Part 1: Layer-by-Layer Geometric and Structural Characterization
    // -------------------------------------------------------------
    std::vector<LayerGeometricStats> layer_stats;

    for (int l = 0; l <= max_l; ++l) {
        LayerGeometricStats st;
        st.layer = l;

        if (l >= (int)index.graphs.size()) continue;

        size_t n_nodes = 0;
        size_t n_edges = 0;
        uint32_t max_deg = 0;
        std::vector<uint32_t> in_deg(sift_n, 0);
        std::vector<double> edge_lengths;
        std::vector<double> cos_sims;

        for (size_t u = 0; u < sift_n; ++u) {
            if (u >= index.graphs[l].size()) continue;
            const auto& neighbors = index.graphs[l][u];
            if (neighbors.empty() && l > 0 && index.node_levels[u] < l) continue;

            n_nodes++;
            size_t deg = neighbors.size();
            n_edges += deg;
            if (deg > max_deg) max_deg = deg;

            for (tableint v : neighbors) {
                if (v < sift_n) {
                    in_deg[v]++;
                    // Euclidean edge length
                    double dist_sq = 0.0;
                    for (size_t d = 0; d < sift_dim; ++d) {
                        double diff = sift_data[u * sift_dim + d] - sift_data[v * sift_dim + d];
                        dist_sq += diff * diff;
                    }
                    edge_lengths.push_back(std::sqrt(dist_sq));
                }
            }

            // Angular diversity / cosine similarity of outgoing edges (sample up to 5 pairs per node)
            if (neighbors.size() >= 2) {
                size_t p_count = 0;
                for (size_t i = 0; i < neighbors.size() && p_count < 5; ++i) {
                    for (size_t j = i + 1; j < neighbors.size() && p_count < 5; ++j) {
                        tableint v1 = neighbors[i];
                        tableint v2 = neighbors[j];
                        double dot = 0.0, n1_sq = 0.0, n2_sq = 0.0;
                        for (size_t d = 0; d < sift_dim; ++d) {
                            double d1 = sift_data[v1 * sift_dim + d] - sift_data[u * sift_dim + d];
                            double d2 = sift_data[v2 * sift_dim + d] - sift_data[u * sift_dim + d];
                            dot += d1 * d2;
                            n1_sq += d1 * d1;
                            n2_sq += d2 * d2;
                        }
                        double denom = std::sqrt(n1_sq * n2_sq);
                        if (denom > 1e-9) {
                            cos_sims.push_back(dot / denom);
                            p_count++;
                        }
                    }
                }
            }
        }

        st.node_count = n_nodes;
        st.edge_count = n_edges;
        st.max_degree = max_deg;
        st.mean_degree = (n_nodes > 0) ? ((double)n_edges / (double)n_nodes) : 0.0;
        st.memory_mb = (double)(n_edges * sizeof(tableint) + n_nodes * sizeof(std::vector<tableint>)) / (1024.0 * 1024.0);

        // In-degree variance
        double sum_in = 0, sum_in_sq = 0;
        for (size_t u = 0; u < sift_n; ++u) {
            double d = in_deg[u];
            sum_in += d;
            sum_in_sq += d * d;
        }
        double mean_in = (n_nodes > 0) ? (sum_in / (double)n_nodes) : 0.0;
        st.in_degree_var = (n_nodes > 0) ? ((sum_in_sq / (double)n_nodes) - (mean_in * mean_in)) : 0.0;

        // Edge length percentiles
        if (!edge_lengths.empty()) {
            std::sort(edge_lengths.begin(), edge_lengths.end());
            st.mean_edge_length = std::accumulate(edge_lengths.begin(), edge_lengths.end(), 0.0) / edge_lengths.size();
            st.p50_edge_length = edge_lengths[edge_lengths.size() / 2];
            st.p95_edge_length = edge_lengths[(size_t)(edge_lengths.size() * 0.95)];
            st.p99_edge_length = edge_lengths[(size_t)(edge_lengths.size() * 0.99)];
        }

        // Angular diversity
        if (!cos_sims.empty()) {
            st.mean_cos_similarity = std::accumulate(cos_sims.begin(), cos_sims.end(), 0.0) / cos_sims.size();
        }

        layer_stats.push_back(st);

        std::cout << "Layer " << l << ": Nodes=" << st.node_count << ", Edges=" << st.edge_count
                  << ", MeanDeg=" << std::fixed << std::setprecision(1) << st.mean_degree
                  << ", MeanLen=" << std::setprecision(2) << st.mean_edge_length
                  << " (P50=" << st.p50_edge_length << ", P95=" << st.p95_edge_length << ")"
                  << ", CosSim=" << std::setprecision(3) << st.mean_cos_similarity
                  << ", Mem=" << std::setprecision(2) << st.memory_mb << " MB" << std::endl;
    }

    // Export Layer Geometry JSON
    {
        std::string out_path = "experiments/phase_5_2_layer_geometry.json";
        std::ofstream out(out_path);
        out << "{\n";
        out << "  \"experiment_name\": \"Phase 5.2 Layer-by-Layer Geometric Characterization\",\n";
        out << "  \"timestamp\": \"2026-11-20T13:15:00Z\",\n";
        out << "  \"dataset\": \"SIFT-100K\",\n";
        out << "  \"max_level\": " << max_l << ",\n";
        out << "  \"layers\": [\n";
        for (size_t i = 0; i < layer_stats.size(); ++i) {
            const auto& s = layer_stats[i];
            out << "    {\n";
            out << "      \"layer\": " << s.layer << ",\n";
            out << "      \"node_count\": " << s.node_count << ",\n";
            out << "      \"edge_count\": " << s.edge_count << ",\n";
            out << "      \"mean_degree\": " << s.mean_degree << ",\n";
            out << "      \"max_degree\": " << s.max_degree << ",\n";
            out << "      \"in_degree_var\": " << s.in_degree_var << ",\n";
            out << "      \"memory_mb\": " << s.memory_mb << ",\n";
            out << "      \"mean_edge_length\": " << s.mean_edge_length << ",\n";
            out << "      \"p50_edge_length\": " << s.p50_edge_length << ",\n";
            out << "      \"p95_edge_length\": " << s.p95_edge_length << ",\n";
            out << "      \"p99_edge_length\": " << s.p99_edge_length << ",\n";
            out << "      \"mean_cos_similarity\": " << s.mean_cos_similarity << "\n";
            out << "    }" << (i + 1 < layer_stats.size() ? "," : "") << "\n";
        }
        out << "  ]\n";
        out << "}\n";
        out.close();
        std::cout << "\n>>> Saved experiments/phase_5_2_layer_geometry.json" << std::endl;
    }

    // -------------------------------------------------------------
    // Part 2: Search Workload Profiling Across Layers
    // -------------------------------------------------------------
    std::cout << "\n>>> Profiling Search Workload across Layers across multiple efSearch values..." << std::endl;
    std::vector<int> ef_values = {16, 32, 64, 128, 256};
    std::vector<LayerSearchProfiling> prof_results;

    for (int ef : ef_values) {
        std::vector<double> recalls(sift_nq);
        uint64_t total_upper = 0, total_l0 = 0;
        uint64_t total_upper_hops = 0, total_l0_hops = 0;
        double total_handover_dist = 0.0;

        auto t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_upper, total_l0, total_upper_hops, total_l0_hops, total_handover_dist)
        for (size_t q = 0; q < sift_nq; ++q) {
            const float* q_vec = &sift_queries[q * sift_dim];
            QuerySearchStats stats;
            auto res = index.search(q_vec, 10, ef, stats);

            total_upper += stats.upper_layer_dist_evals;
            total_l0 += stats.layer0_dist_evals;
            total_upper_hops += stats.hops; // In HNSW search, upper hops recorded

            // Compute Recall@10
            std::unordered_set<uint32_t> gt_set;
            for (size_t k = 0; k < 10 && k < sift_gt[q].size(); ++k) gt_set.insert(sift_gt[q][k]);
            size_t m = 0;
            for (const auto& c : res) if (gt_set.count(c.id)) m++;
            recalls[q] = (double)m / 10.0;

            // Distance to query at enter_point (Layer 0 entry)
            tableint ep0 = index.enter_point;
            double d_ep = 0.0;
            for (size_t d = 0; d < sift_dim; ++d) {
                double diff = q_vec[d] - sift_data[ep0 * sift_dim + d];
                d_ep += diff * diff;
            }
            total_handover_dist += std::sqrt(d_ep);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(t1 - t0).count();

        LayerSearchProfiling p;
        p.ef_search = ef;
        p.recall_at_10 = std::accumulate(recalls.begin(), recalls.end(), 0.0) / (double)sift_nq;
        p.qps = (double)sift_nq / total_s;
        p.mean_upper_dist_evals = (double)total_upper / (double)sift_nq;
        p.mean_layer0_dist_evals = (double)total_l0 / (double)sift_nq;
        p.mean_total_dist_evals = p.mean_upper_dist_evals + p.mean_layer0_dist_evals;
        p.upper_eval_fraction_pct = (p.mean_total_dist_evals > 0)
            ? (100.0 * p.mean_upper_dist_evals / p.mean_total_dist_evals) : 0.0;
        p.mean_upper_hops = (double)total_upper_hops / (double)sift_nq;
        p.mean_handover_distance_to_query = total_handover_dist / (double)sift_nq;

        prof_results.push_back(p);

        std::cout << "efSearch=" << ef << ": Recall@10=" << std::fixed << std::setprecision(2) << p.recall_at_10 * 100.0
                  << "%, QPS=" << std::setprecision(1) << p.qps
                  << ", TotalDist=" << p.mean_total_dist_evals
                  << " (Upper: " << p.mean_upper_dist_evals << " [" << std::setprecision(2) << p.upper_eval_fraction_pct << "%], L0: " << p.mean_layer0_dist_evals << ")"
                  << std::endl;
    }

    // Export Layer Search Profiling JSON
    {
        std::string out_path = "experiments/phase_5_2_layer_search_profiling.json";
        std::ofstream out(out_path);
        out << "{\n";
        out << "  \"experiment_name\": \"Phase 5.2 Layer Search Workload Profiling\",\n";
        out << "  \"timestamp\": \"2026-11-20T13:15:00Z\",\n";
        out << "  \"dataset\": \"SIFT-100K\",\n";
        out << "  \"profiling\": [\n";
        for (size_t i = 0; i < prof_results.size(); ++i) {
            const auto& p = prof_results[i];
            out << "    {\n";
            out << "      \"ef_search\": " << p.ef_search << ",\n";
            out << "      \"recall_at_10\": " << p.recall_at_10 << ",\n";
            out << "      \"qps\": " << p.qps << ",\n";
            out << "      \"mean_upper_dist_evals\": " << p.mean_upper_dist_evals << ",\n";
            out << "      \"mean_layer0_dist_evals\": " << p.mean_layer0_dist_evals << ",\n";
            out << "      \"mean_total_dist_evals\": " << p.mean_total_dist_evals << ",\n";
            out << "      \"upper_eval_fraction_pct\": " << p.upper_eval_fraction_pct << ",\n";
            out << "      \"mean_upper_hops\": " << p.mean_upper_hops << ",\n";
            out << "      \"mean_handover_distance_to_query\": " << p.mean_handover_distance_to_query << "\n";
            out << "    }" << (i + 1 < prof_results.size() ? "," : "") << "\n";
        }
        out << "  ]\n";
        out << "}\n";
        out.close();
        std::cout << "\n>>> Saved experiments/phase_5_2_layer_search_profiling.json" << std::endl;
    }

    return 0;
}
