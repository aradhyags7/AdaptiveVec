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
#include <queue>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace adaptivevec;

struct DetailedGraphStats {
    std::string model_name;
    size_t total_edges;
    size_t layer0_edges;
    double mean_out_degree;
    double mean_in_degree;
    double in_degree_var;
    uint32_t max_in_degree;
    
    // Quartile Degrees by LID
    double deg_lid_q1;
    double deg_lid_q2;
    double deg_lid_q3;
    double deg_lid_q4;
    
    // Quartile Degrees by Density
    double deg_dens_q1;
    double deg_dens_q2;
    double deg_dens_q3;
    double deg_dens_q4;

    // Edge Length Metrics
    double mean_edge_length;
    double median_edge_length;
    double p90_edge_length;
    double p99_edge_length;

    // Navigability & Reachability
    size_t wcc_components;
    double wcc_largest_pct;
    double ep_reachability_pct;
};

DetailedGraphStats analyze_index(const AdaptiveHNSWIndex& index, const std::string& name, const std::vector<float>& data, size_t dim) {
    DetailedGraphStats st;
    st.model_name = name;
    size_t n = index.num_elements;
    st.total_edges = index.get_total_edges();
    st.layer0_edges = index.get_layer0_edges();
    st.mean_out_degree = (double)st.layer0_edges / (double)n;

    // In-Degree stats
    double sum_in = 0, sum_in_sq = 0;
    uint32_t max_in = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t in_d = (i < index.in_degrees.size()) ? index.in_degrees[i] : 0;
        sum_in += in_d;
        sum_in_sq += (double)in_d * in_d;
        if (in_d > max_in) max_in = in_d;
    }
    st.mean_in_degree = sum_in / (double)n;
    st.in_degree_var = (sum_in_sq / (double)n) - (st.mean_in_degree * st.mean_in_degree);
    st.max_in_degree = max_in;

    // Edge Length Distribution (sampled 100,000 edges)
    std::vector<float> edge_lengths;
    edge_lengths.reserve(100000);
    for (size_t u = 0; u < n && edge_lengths.size() < 100000; ++u) {
        if (u < index.graphs[0].size()) {
            for (tableint v : index.graphs[0][u]) {
                if (edge_lengths.size() >= 100000) break;
                float d = std::sqrt(index.get_node_to_node_distance_sq(u, v));
                edge_lengths.push_back(d);
            }
        }
    }
    if (!edge_lengths.empty()) {
        std::sort(edge_lengths.begin(), edge_lengths.end());
        double sum_len = std::accumulate(edge_lengths.begin(), edge_lengths.end(), 0.0);
        st.mean_edge_length = sum_len / (double)edge_lengths.size();
        st.median_edge_length = edge_lengths[edge_lengths.size() / 2];
        st.p90_edge_length = edge_lengths[(size_t)(edge_lengths.size() * 0.90)];
        st.p99_edge_length = edge_lengths[(size_t)(edge_lengths.size() * 0.99)];
    }

    // Quartile Analysis
    if (index.node_params.size() >= n) {
        std::vector<size_t> lid_idx(n);
        std::iota(lid_idx.begin(), lid_idx.end(), 0);
        std::sort(lid_idx.begin(), lid_idx.end(), [&](size_t a, size_t b) {
            return index.node_params[a].lid < index.node_params[b].lid;
        });

        size_t q_sz = n / 4;
        auto get_deg = [&](size_t s, size_t e) {
            double sum = 0;
            for (size_t i = s; i < e; ++i) {
                size_t u = lid_idx[i];
                if (u < index.graphs[0].size()) sum += index.graphs[0][u].size();
            }
            return sum / (double)(e - s);
        };
        st.deg_lid_q1 = get_deg(0, q_sz);
        st.deg_lid_q2 = get_deg(q_sz, 2 * q_sz);
        st.deg_lid_q3 = get_deg(2 * q_sz, 3 * q_sz);
        st.deg_lid_q4 = get_deg(3 * q_sz, n);

        std::vector<size_t> dens_idx(n);
        std::iota(dens_idx.begin(), dens_idx.end(), 0);
        std::sort(dens_idx.begin(), dens_idx.end(), [&](size_t a, size_t b) {
            return index.node_params[a].density < index.node_params[b].density;
        });
        auto get_dens_deg = [&](size_t s, size_t e) {
            double sum = 0;
            for (size_t i = s; i < e; ++i) {
                size_t u = dens_idx[i];
                if (u < index.graphs[0].size()) sum += index.graphs[0][u].size();
            }
            return sum / (double)(e - s);
        };
        st.deg_dens_q1 = get_dens_deg(0, q_sz);
        st.deg_dens_q2 = get_dens_deg(q_sz, 2 * q_sz);
        st.deg_dens_q3 = get_dens_deg(2 * q_sz, 3 * q_sz);
        st.deg_dens_q4 = get_dens_deg(3 * q_sz, n);
    }

    // Weakly Connected Components
    std::vector<std::vector<tableint>> undir(n);
    for (size_t u = 0; u < n; ++u) {
        if (u < index.graphs[0].size()) {
            for (tableint v : index.graphs[0][u]) {
                if (v < n) {
                    undir[u].push_back(v);
                    undir[v].push_back(u);
                }
            }
        }
    }
    std::vector<bool> vis(n, false);
    size_t wcc_cnt = 0, max_wcc = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!vis[i]) {
            wcc_cnt++;
            size_t sz = 0;
            std::queue<tableint> q;
            q.push(i);
            vis[i] = true;
            while (!q.empty()) {
                tableint u = q.front(); q.pop(); sz++;
                for (tableint v : undir[u]) {
                    if (!vis[v]) { vis[v] = true; q.push(v); }
                }
            }
            if (sz > max_wcc) max_wcc = sz;
        }
    }
    st.wcc_components = wcc_cnt;
    st.wcc_largest_pct = 100.0 * (double)max_wcc / (double)n;

    // Directed reachability from enter_point on Layer 0
    std::vector<bool> ep_vis(n, false);
    std::queue<tableint> ep_q;
    if (index.enter_point < n) {
        ep_vis[index.enter_point] = true;
        ep_q.push(index.enter_point);
    }
    size_t reached = 0;
    while (!ep_q.empty()) {
        tableint u = ep_q.front(); ep_q.pop(); reached++;
        if (u < index.graphs[0].size()) {
            for (tableint v : index.graphs[0][u]) {
                if (v < n && !ep_vis[v]) { ep_vis[v] = true; ep_q.push(v); }
            }
        }
    }
    st.ep_reachability_pct = 100.0 * (double)reached / (double)n;

    return st;
}

int main() {
    std::cout << "=============================================================" << std::endl;
    std::cout << "  AdaptiveVec Phase 5.1: Graph Structural Analysis Runner     " << std::endl;
    std::cout << "=============================================================" << std::endl;

    std::string sift_base_path = "data/sift_subset_100k.fvecs";
    size_t sift_n = 0, sift_dim = 0;
    std::vector<float> sift_data;
    load_fvecs(sift_base_path, sift_data, sift_n, sift_dim);

    // Build Uniform Canonical HNSW
    PolicyConfig cfg_uniform;
    cfg_uniform.m_base = 16;
    cfg_uniform.ef_c_base = 120;
    cfg_uniform.enable_layer_scaling = false;
    cfg_uniform.enable_hubness_regulation = false;

    std::cout << "Analyzing Uniform Canonical HNSW..." << std::endl;
    AdaptiveHNSWIndex idx_uniform(sift_dim, SpaceType::L2, false, cfg_uniform);
    for (size_t i = 0; i < sift_n; ++i) idx_uniform.insert(&sift_data[i * sift_dim]);
    DetailedGraphStats stats_uniform = analyze_index(idx_uniform, "Canonical Uniform HNSW", sift_data, sift_dim);

    // Build Architecture B
    PolicyConfig cfg_b;
    cfg_b.m_base = 16;
    cfg_b.ef_c_base = 120;
    cfg_b.m_min = 8;
    cfg_b.m_max = 24;
    cfg_b.ef_c_min = 40;
    cfg_b.ef_c_max = 220;
    cfg_b.alpha_lid = 0.5f;
    cfg_b.beta_density = 0.5f;
    cfg_b.sensitivity = 1.0f;
    cfg_b.enable_layer_scaling = true;
    cfg_b.lambda_layer = 0.75f;
    cfg_b.m_min_layer = 4;
    cfg_b.enable_hubness_regulation = true;
    cfg_b.hubness_mu = 0.15f;
    cfg_b.use_streaming_welford = true;

    std::cout << "Analyzing Architecture B (Adaptive Geometry)..." << std::endl;
    AdaptiveHNSWIndex idx_b(sift_dim, SpaceType::L2, true, cfg_b);
    for (size_t i = 0; i < sift_n; ++i) idx_b.insert(&sift_data[i * sift_dim]);
    DetailedGraphStats stats_b = analyze_index(idx_b, "Architecture B (Adaptive M/efC)", sift_data, sift_dim);

    // Build Phase 5.1 GCP (Joint Allocation and Pruning)
    PolicyConfig cfg_gcp = cfg_b;
    cfg_gcp.enable_geometry_pruning = true;
    cfg_gcp.pruning_alpha_base = 1.0f;
    cfg_gcp.pruning_lid_weight = 0.12f;
    cfg_gcp.pruning_density_weight = 0.10f;

    std::cout << "Analyzing Phase 5.1 GCP (Joint Allocation & Pruning)..." << std::endl;
    AdaptiveHNSWIndex idx_gcp(sift_dim, SpaceType::L2, true, cfg_gcp);
    for (size_t i = 0; i < sift_n; ++i) idx_gcp.insert(&sift_data[i * sift_dim]);
    DetailedGraphStats stats_gcp = analyze_index(idx_gcp, "Phase 5.1 GCP (Joint)", sift_data, sift_dim);

    // Output JSON
    std::ofstream out_json("experiments/phase_5_1_graph_statistics.json");
    out_json << "{\n";
    out_json << "  \"timestamp\": \"2026-11-20T13:05:00Z\",\n";
    out_json << "  \"models\": [\n";
    
    auto write_model = [&](const DetailedGraphStats& s, bool is_last) {
        out_json << "    {\n";
        out_json << "      \"model_name\": \"" << s.model_name << "\",\n";
        out_json << "      \"total_edges\": " << s.total_edges << ",\n";
        out_json << "      \"layer0_edges\": " << s.layer0_edges << ",\n";
        out_json << "      \"mean_out_degree\": " << s.mean_out_degree << ",\n";
        out_json << "      \"mean_in_degree\": " << s.mean_in_degree << ",\n";
        out_json << "      \"in_degree_var\": " << s.in_degree_var << ",\n";
        out_json << "      \"max_in_degree\": " << s.max_in_degree << ",\n";
        out_json << "      \"edge_lengths\": {\n";
        out_json << "        \"mean\": " << s.mean_edge_length << ",\n";
        out_json << "        \"median\": " << s.median_edge_length << ",\n";
        out_json << "        \"p90\": " << s.p90_edge_length << ",\n";
        out_json << "        \"p99\": " << s.p99_edge_length << "\n";
        out_json << "      },\n";
        out_json << "      \"lid_quartile_degrees\": {\n";
        out_json << "        \"q1_low_lid\": " << s.deg_lid_q1 << ",\n";
        out_json << "        \"q2\": " << s.deg_lid_q2 << ",\n";
        out_json << "        \"q3\": " << s.deg_lid_q3 << ",\n";
        out_json << "        \"q4_high_lid\": " << s.deg_lid_q4 << "\n";
        out_json << "      },\n";
        out_json << "      \"density_quartile_degrees\": {\n";
        out_json << "        \"q1_dense\": " << s.deg_dens_q1 << ",\n";
        out_json << "        \"q2\": " << s.deg_dens_q2 << ",\n";
        out_json << "        \"q3\": " << s.deg_dens_q3 << ",\n";
        out_json << "        \"q4_sparse\": " << s.deg_dens_q4 << "\n";
        out_json << "      },\n";
        out_json << "      \"navigability\": {\n";
        out_json << "        \"wcc_components\": " << s.wcc_components << ",\n";
        out_json << "        \"wcc_largest_pct\": " << s.wcc_largest_pct << ",\n";
        out_json << "        \"ep_reachability_pct\": " << s.ep_reachability_pct << "\n";
        out_json << "      }\n";
        out_json << "    }" << (is_last ? "" : ",") << "\n";
    };

    write_model(stats_uniform, false);
    write_model(stats_b, false);
    write_model(stats_gcp, true);

    out_json << "  ]\n";
    out_json << "}\n";
    out_json.close();

    std::cout << "Saved Graph Statistics to experiments/phase_5_1_graph_statistics.json" << std::endl;
    return 0;
}
