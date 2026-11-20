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

struct NodeGeometryStats {
    size_t id;
    float lid;
    float density;
    float score;
    size_t out_degree;
    uint32_t in_degree;
    float mean_cand_dist;
    float std_cand_dist;
    float mean_directional_redundancy; // mean cos(theta_jl) for positive cos
    float max_directional_redundancy;
    float fraction_redundant_pairs;     // pairs with cos(theta_jl) > 0.5 (angle < 60 deg)
    uint32_t search_visitations;
};

int main() {
    std::cout << "=============================================================" << std::endl;
    std::cout << "  AdaptiveVec Phase 5.1: Stage 2 Geometry Characterization    " << std::endl;
    std::cout << "=============================================================" << std::endl;

    // Load SIFT-100K Dataset
    std::string sift_base_path = "data/sift_subset_100k.fvecs";
    std::string sift_query_path = "data/sift_query.fvecs";
    std::string sift_gt_path = "data/sift_subset_groundtruth.ivecs";

    size_t n = 0, dim = 0;
    std::vector<float> data;
    if (!load_fvecs(sift_base_path, data, n, dim)) {
        std::cerr << "Failed to load SIFT dataset!" << std::endl;
        return 1;
    }
    size_t nq = 0, qdim = 0;
    std::vector<float> queries;
    load_fvecs(sift_query_path, queries, nq, qdim);

    std::cout << "Loaded SIFT-100K: N=" << n << ", Dim=" << dim << ", Queries=" << nq << std::endl;

    // Build Architecture B Index
    PolicyConfig cfg;
    cfg.m_base = 16;
    cfg.ef_c_base = 120;
    cfg.m_min = 8;
    cfg.m_max = 24;
    cfg.ef_c_min = 40;
    cfg.ef_c_max = 220;
    cfg.alpha_lid = 0.5f;
    cfg.beta_density = 0.5f;
    cfg.sensitivity = 1.0f;
    cfg.enable_layer_scaling = true;
    cfg.lambda_layer = 0.75f;
    cfg.m_min_layer = 4;
    cfg.enable_hubness_regulation = true;
    cfg.hubness_mu = 0.15f;
    cfg.use_streaming_welford = true;

    std::cout << "Building Architecture B Index for Geometric Characterization..." << std::endl;
    AdaptiveHNSWIndex index(dim, SpaceType::L2, true, cfg);
    for (size_t i = 0; i < n; ++i) {
        index.insert(&data[i * dim]);
    }
    std::cout << "Index built. Total edges: " << index.get_total_edges() << std::endl;

    // Track search visitation frequency across 10,000 queries
    std::cout << "Tracking search visitation frequencies across queries..." << std::endl;
    std::vector<uint32_t> visit_counts(n, 0);
    #pragma omp parallel
    {
        std::vector<uint32_t> local_visits(n, 0);
        #pragma omp for schedule(dynamic, 64)
        for (size_t q = 0; q < nq; ++q) {
            QuerySearchStats stats;
            index.search(&queries[q * dim], 10, 64, stats);
        }
    }

    // Detailed Geometric Analysis of Sampled Nodes (5,000 nodes)
    std::cout << "Computing directional vectors and angular redundancies..." << std::endl;
    size_t num_samples = 5000;
    std::vector<NodeGeometryStats> sample_stats(num_samples);

    double sum_mean_redundancy = 0;
    double sum_frac_redundant = 0;
    size_t valid_geom_nodes = 0;

    for (size_t s = 0; s < num_samples; ++s) {
        size_t u = (s * 997) % n; // Stride sample
        NodeGeometryStats ng;
        ng.id = u;
        ng.lid = (u < index.node_params.size()) ? index.node_params[u].lid : 0.0f;
        ng.density = (u < index.node_params.size()) ? index.node_params[u].density : 0.0f;
        ng.score = (u < index.node_params.size()) ? index.node_params[u].score : 0.0f;
        ng.out_degree = (u < index.graphs[0].size()) ? index.graphs[0][u].size() : 0;
        ng.in_degree = (u < index.in_degrees.size()) ? index.in_degrees[u] : 0;
        ng.search_visitations = visit_counts[u];

        const auto& neighbors = index.graphs[0][u];
        size_t deg = neighbors.size();

        if (deg >= 2) {
            // Compute candidate distances
            std::vector<float> dists(deg);
            double sum_d = 0;
            for (size_t j = 0; j < deg; ++j) {
                float d = std::sqrt(index.get_node_to_node_distance_sq(u, neighbors[j]));
                dists[j] = d;
                sum_d += d;
            }
            float mean_d = sum_d / deg;
            double sum_d_sq = 0;
            for (float d : dists) sum_d_sq += (d - mean_d) * (d - mean_d);
            ng.mean_cand_dist = mean_d;
            ng.std_cand_dist = std::sqrt(sum_d_sq / deg);

            // Compute normalized directional vectors u_ij = (x_j - x_i) / ||x_j - x_i||
            std::vector<std::vector<float>> u_dirs(deg, std::vector<float>(dim, 0.0f));
            for (size_t j = 0; j < deg; ++j) {
                tableint v = neighbors[j];
                float d = dists[j];
                if (d > 1e-7f) {
                    float inv_d = 1.0f / d;
                    for (size_t d_idx = 0; d_idx < dim; ++d_idx) {
                        u_dirs[j][d_idx] = (data[v * dim + d_idx] - data[u * dim + d_idx]) * inv_d;
                    }
                }
            }

            // Compute pairwise dot products: cos(theta_jl) = u_ij . u_il
            size_t num_pairs = deg * (deg - 1) / 2;
            double sum_cos = 0;
            float max_cos = -1.0f;
            size_t high_redundancy_pairs = 0;

            for (size_t j = 0; j < deg; ++j) {
                for (size_t l = j + 1; l < deg; ++l) {
                    float dot = 0.0f;
                    #pragma omp simd reduction(+:dot)
                    for (size_t d_idx = 0; d_idx < dim; ++d_idx) {
                        dot += u_dirs[j][d_idx] * u_dirs[l][d_idx];
                    }
                    if (dot > max_cos) max_cos = dot;
                    sum_cos += std::max(0.0f, dot); // positive directional alignment
                    if (dot > 0.5f) { // Angle < 60 degrees (HNSW diversity threshold)
                        high_redundancy_pairs++;
                    }
                }
            }

            ng.mean_directional_redundancy = (float)(sum_cos / num_pairs);
            ng.max_directional_redundancy = max_cos;
            ng.fraction_redundant_pairs = (float)high_redundancy_pairs / (float)num_pairs;

            sum_mean_redundancy += ng.mean_directional_redundancy;
            sum_frac_redundant += ng.fraction_redundant_pairs;
            valid_geom_nodes++;
        } else {
            ng.mean_cand_dist = 0;
            ng.std_cand_dist = 0;
            ng.mean_directional_redundancy = 0;
            ng.max_directional_redundancy = 0;
            ng.fraction_redundant_pairs = 0;
        }

        sample_stats[s] = ng;
    }

    double global_mean_redundancy = valid_geom_nodes > 0 ? (sum_mean_redundancy / valid_geom_nodes) : 0;
    double global_mean_frac_redundant = valid_geom_nodes > 0 ? (sum_frac_redundant / valid_geom_nodes) : 0;

    std::cout << "\n>>> Geometric Characterization Summary:" << std::endl;
    std::cout << "  Sampled Nodes: " << num_samples << std::endl;
    std::cout << "  Mean Directional Redundancy cos(theta): " << std::fixed << std::setprecision(4) << global_mean_redundancy << std::endl;
    std::cout << "  Fraction of Redundant Pairs (cos > 0.5, theta < 60 deg): " << (global_mean_frac_redundant * 100.0) << "%" << std::endl;

    // Analyze Correlations
    std::vector<float> lids(num_samples), densities(num_samples), redundancies(num_samples), degrees(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        lids[i] = sample_stats[i].lid;
        densities[i] = sample_stats[i].density;
        redundancies[i] = sample_stats[i].mean_directional_redundancy;
        degrees[i] = (float)sample_stats[i].out_degree;
    }

    auto compute_corr = [](const std::vector<float>& x, const std::vector<float>& y) {
        size_t n_elem = x.size();
        double mx = std::accumulate(x.begin(), x.end(), 0.0) / n_elem;
        double my = std::accumulate(y.begin(), y.end(), 0.0) / n_elem;
        double cov = 0, varx = 0, vary = 0;
        for (size_t i = 0; i < n_elem; ++i) {
            double dx = x[i] - mx;
            double dy = y[i] - my;
            cov += dx * dy;
            varx += dx * dx;
            vary += dy * dy;
        }
        return (varx > 1e-9 && vary > 1e-9) ? (cov / std::sqrt(varx * vary)) : 0.0;
    };

    double corr_lid_red = compute_corr(lids, redundancies);
    double corr_dens_red = compute_corr(densities, redundancies);
    double corr_deg_red = compute_corr(degrees, redundancies);

    std::cout << "  Correlation (LID vs Directional Redundancy): " << corr_lid_red << std::endl;
    std::cout << "  Correlation (Density vs Directional Redundancy): " << corr_dens_red << std::endl;
    std::cout << "  Correlation (Out-Degree vs Directional Redundancy): " << corr_deg_red << std::endl;

    // Output JSON
    std::ofstream out_json("experiments/phase_5_1_neighborhood_geometry.json");
    out_json << "{\n";
    out_json << "  \"timestamp\": \"2026-11-20T12:45:00Z\",\n";
    out_json << "  \"dataset\": \"SIFT-100K\",\n";
    out_json << "  \"num_samples\": " << num_samples << ",\n";
    out_json << "  \"global_mean_directional_redundancy\": " << global_mean_redundancy << ",\n";
    out_json << "  \"fraction_redundant_neighbor_pairs\": " << global_mean_frac_redundant << ",\n";
    out_json << "  \"correlations\": {\n";
    out_json << "    \"lid_vs_directional_redundancy\": " << corr_lid_red << ",\n";
    out_json << "    \"density_vs_directional_redundancy\": " << corr_dens_red << ",\n";
    out_json << "    \"out_degree_vs_directional_redundancy\": " << corr_deg_red << "\n";
    out_json << "  },\n";
    out_json << "  \"sampled_nodes\": [\n";
    for (size_t i = 0; i < std::min((size_t)50, num_samples); ++i) {
        const auto& s = sample_stats[i];
        out_json << "    {\n";
        out_json << "      \"id\": " << s.id << ",\n";
        out_json << "      \"lid\": " << s.lid << ",\n";
        out_json << "      \"density\": " << s.density << ",\n";
        out_json << "      \"score\": " << s.score << ",\n";
        out_json << "      \"out_degree\": " << s.out_degree << ",\n";
        out_json << "      \"in_degree\": " << s.in_degree << ",\n";
        out_json << "      \"mean_directional_redundancy\": " << s.mean_directional_redundancy << ",\n";
        out_json << "      \"fraction_redundant_pairs\": " << s.fraction_redundant_pairs << "\n";
        out_json << "    }" << (i + 1 < std::min((size_t)50, num_samples) ? "," : "") << "\n";
    }
    out_json << "  ]\n";
    out_json << "}\n";
    out_json.close();
    std::cout << "Saved Neighborhood Geometry analysis to experiments/phase_5_1_neighborhood_geometry.json" << std::endl;

    return 0;
}
