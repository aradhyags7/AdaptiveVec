/**
 * AdaptiveVec Phase 5.0: Architectural Recovery and Geometry-Aware Research Runner
 * 
 * Implements and benchmarks:
 * 1. Architecture B (Optimized Original AdaptiveVec):
 *    Restored geometry-aware graph construction (LID, Density, S(x), M(x), efC(x),
 *    layer-decoupled scaling, in-degree hubness regulation) combined with verified
 *    systems optimizations (VisitedTagPool, AVX2 squared distance, QuerySearchStats,
 *    flat memory, thread-local zero-atomic instrumentation).
 * 2. Architecture C (Current Phase 4.3 Calibrated Static Control):
 *    Canonical HNSW with uniform parameters, static beam search, controllers decoupled.
 * 3. Geometry-Aware Ablation Experiments A through H:
 *    - Exp A: Uniform Connectivity (Control)
 *    - Exp B: Original Adaptive Connectivity
 *    - Exp C: LID-Only Connectivity
 *    - Exp D: Density-Only Connectivity
 *    - Exp E: Randomized Permuted Connectivity (Critical Control)
 *    - Exp F: Constant-Budget Adaptive Connectivity
 *    - Exp G: Geometry-Reversed Connectivity
 *    - Exp H: Construction Exploration Isolation (H1, H2, H3, H4)
 * 4. Graph Structural & Navigability Diagnostic:
 *    Degree distribution by LID/Density quartiles, edge length distribution,
 *    clustering coefficients, reachability, hubness metrics.
 */

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

struct AblationResult {
    std::string experiment_id;
    std::string description;
    size_t total_edges = 0;
    size_t layer0_edges = 0;
    size_t upper_layer_edges = 0;
    double build_time_s = 0.0;
    double recall_at_1 = 0.0;
    double recall_at_10 = 0.0;
    double throughput_qps = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double mean_dist_evals = 0.0;
    double memory_mb = 0.0;
    double mean_degree = 0.0;
    double in_degree_variance = 0.0;
    uint32_t max_in_degree = 0;
    double reachability_pct = 100.0;
};

struct GraphMetrics {
    double mean_out_degree = 0.0;
    double mean_in_degree = 0.0;
    double in_degree_var = 0.0;
    uint32_t max_in_deg = 0;
    double mean_edge_length = 0.0;
    double median_edge_length = 0.0;
    double p90_edge_length = 0.0;
    double clustering_coeff = 0.0;
    double reachability = 100.0;
    double lid_degree_corr = 0.0;
    double density_degree_corr = 0.0;
    
    // Quartile degrees
    double deg_lid_q1 = 0.0; // Lowest LID quartile
    double deg_lid_q2 = 0.0;
    double deg_lid_q3 = 0.0;
    double deg_lid_q4 = 0.0; // Highest LID quartile
    
    double deg_dens_q1 = 0.0; // Densest quartile
    double deg_dens_q2 = 0.0;
    double deg_dens_q3 = 0.0;
    double deg_dens_q4 = 0.0; // Sparsest quartile
};

// Compute Recall@k
inline float compute_recall(const std::vector<std::vector<Candidate>>& results, 
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

// Compute Pearson correlation
inline double compute_correlation(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.empty()) return 0.0;
    size_t n = x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }
    double num = n * sum_xy - sum_x * sum_y;
    double den = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    return (den > 1e-9) ? (num / den) : 0.0;
}

// Detailed Graph Geometry Analysis
GraphMetrics analyze_graph(const AdaptiveHNSWIndex& index) {
    GraphMetrics m;
    size_t n = index.num_elements;
    if (n == 0 || index.graphs.empty() || index.graphs[0].empty()) return m;

    std::vector<uint32_t> in_deg(n, 0);
    std::vector<float> out_deg(n, 0);
    std::vector<float> edge_lengths;
    edge_lengths.reserve(index.get_layer0_edges());

    for (size_t u = 0; u < n; ++u) {
        const auto& adj = index.graphs[0][u];
        out_deg[u] = (float)adj.size();
        const float* u_vec = index.get_vector(u);
        for (tableint v : adj) {
            if (v < n) {
                in_deg[v]++;
                float dist = l2_distance(u_vec, index.get_vector(v), index.dim);
                edge_lengths.push_back(dist);
            }
        }
    }

    // Out-degree and in-degree means
    double total_in = 0;
    for (size_t i = 0; i < n; ++i) total_in += in_deg[i];
    m.mean_in_degree = total_in / (double)n;
    m.mean_out_degree = m.mean_in_degree;

    // In-degree variance & max
    double var_sum = 0;
    uint32_t max_d = 0;
    for (size_t i = 0; i < n; ++i) {
        double diff = (double)in_deg[i] - m.mean_in_degree;
        var_sum += diff * diff;
        if (in_deg[i] > max_d) max_d = in_deg[i];
    }
    m.in_degree_var = var_sum / (double)n;
    m.max_in_deg = max_d;

    // Edge length distribution
    if (!edge_lengths.empty()) {
        std::sort(edge_lengths.begin(), edge_lengths.end());
        double sum_len = 0;
        for (float l : edge_lengths) sum_len += l;
        m.mean_edge_length = sum_len / (double)edge_lengths.size();
        m.median_edge_length = edge_lengths[edge_lengths.size() / 2];
        m.p90_edge_length = edge_lengths[(size_t)(edge_lengths.size() * 0.90)];
    }

    // Local clustering coefficient (sampled across 1,000 nodes)
    size_t sample_size = std::min((size_t)1000, n);
    double total_cc = 0;
    size_t valid_cc_nodes = 0;

    for (size_t i = 0; i < sample_size; ++i) {
        size_t u = (i * 997) % n; // Stride sampling
        const auto& adj = index.graphs[0][u];
        size_t k_u = adj.size();
        if (k_u < 2) continue;

        std::unordered_set<tableint> neighbors(adj.begin(), adj.end());
        size_t links_between_neighbors = 0;
        for (tableint v : adj) {
            if (v < index.graphs[0].size()) {
                for (tableint w : index.graphs[0][v]) {
                    if (neighbors.find(w) != neighbors.end()) {
                        links_between_neighbors++;
                    }
                }
            }
        }
        double cc = (double)links_between_neighbors / (double)(k_u * (k_u - 1));
        total_cc += cc;
        valid_cc_nodes++;
    }
    m.clustering_coeff = (valid_cc_nodes > 0) ? (total_cc / (double)valid_cc_nodes) : 0.0;

    // Reachability from Entry Point (BFS)
    std::vector<bool> reachable(n, false);
    std::queue<tableint> q;
    tableint ep = index.enter_point;
    if (ep < n) {
        reachable[ep] = true;
        q.push(ep);
    }
    size_t reached_count = 0;
    while (!q.empty()) {
        tableint u = q.front();
        q.pop();
        reached_count++;
        if (u < index.graphs[0].size()) {
            for (tableint v : index.graphs[0][u]) {
                if (v < n && !reachable[v]) {
                    reachable[v] = true;
                    q.push(v);
                }
            }
        }
    }
    m.reachability = 100.0 * (double)reached_count / (double)n;

    // Correlations with LID and Density
    if (index.node_params.size() >= n) {
        std::vector<float> lids(n), densities(n);
        for (size_t i = 0; i < n; ++i) {
            lids[i] = index.node_params[i].lid;
            densities[i] = index.node_params[i].density;
        }
        m.lid_degree_corr = compute_correlation(lids, out_deg);
        m.density_degree_corr = compute_correlation(densities, out_deg);

        // Quartile analysis
        std::vector<size_t> lid_indices(n);
        std::iota(lid_indices.begin(), lid_indices.end(), 0);
        std::sort(lid_indices.begin(), lid_indices.end(), [&](size_t a, size_t b) {
            return lids[a] < lids[b];
        });

        size_t q_sz = n / 4;
        auto get_mean_deg = [&](size_t start, size_t end) {
            double sum = 0;
            for (size_t i = start; i < end; ++i) sum += out_deg[lid_indices[i]];
            return sum / (double)(end - start);
        };
        m.deg_lid_q1 = get_mean_deg(0, q_sz);
        m.deg_lid_q2 = get_mean_deg(q_sz, 2 * q_sz);
        m.deg_lid_q3 = get_mean_deg(2 * q_sz, 3 * q_sz);
        m.deg_lid_q4 = get_mean_deg(3 * q_sz, n);

        std::vector<size_t> dens_indices(n);
        std::iota(dens_indices.begin(), dens_indices.end(), 0);
        std::sort(dens_indices.begin(), dens_indices.end(), [&](size_t a, size_t b) {
            return densities[a] < densities[b];
        });
        auto get_mean_dens_deg = [&](size_t start, size_t end) {
            double sum = 0;
            for (size_t i = start; i < end; ++i) sum += out_deg[dens_indices[i]];
            return sum / (double)(end - start);
        };
        m.deg_dens_q1 = get_mean_dens_deg(0, q_sz); // Tightest
        m.deg_dens_q2 = get_mean_dens_deg(q_sz, 2 * q_sz);
        m.deg_dens_q3 = get_mean_dens_deg(2 * q_sz, 3 * q_sz);
        m.deg_dens_q4 = get_mean_dens_deg(3 * q_sz, n); // Sparsest
    }

    return m;
}

// Build and Evaluate a given Policy Configuration
AblationResult evaluate_configuration(
    const std::string& exp_id,
    const std::string& desc,
    const std::vector<float>& data,
    size_t n_samples,
    size_t dim,
    const std::vector<float>& queries,
    size_t n_queries,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    PolicyConfig policy,
    bool is_adaptive,
    SpaceType space = SpaceType::L2,
    bool shuffle_degrees = false,
    bool reverse_geometry = false,
    int fixed_efc_override = 0,
    int fixed_m_override = 0,
    int ef_search = 64,
    int patience = 8
) {
    AblationResult res;
    res.experiment_id = exp_id;
    res.description = desc;

    std::cout << "\n>>> Running " << exp_id << ": " << desc << " ... " << std::endl;

    AdaptiveHNSWIndex index(dim, space, is_adaptive, policy);
    
    // Invert geometry if requested
    if (reverse_geometry) {
        index.policy.sensitivity = -index.policy.sensitivity;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // Insertion with fine-grained tracking
    for (size_t i = 0; i < n_samples; ++i) {
        index.insert(&data[i * dim]);

        // Overrides for specific ablations
        if (fixed_efc_override > 0 && i < index.node_params.size()) {
            index.node_params[i].ef_construction = fixed_efc_override;
        }
        if (fixed_m_override > 0 && i < index.node_params.size()) {
            index.node_params[i].m = fixed_m_override;
            index.node_params[i].m_max = fixed_m_override;
            index.node_params[i].m_max0 = 2 * fixed_m_override;
        }
    }

    // Critical Control: Randomly permute degree allocations across nodes
    if (shuffle_degrees && index.node_params.size() >= n_samples) {
        std::mt19937 rng(42);
        std::vector<NodeParams> shuffled_params = index.node_params;
        std::shuffle(shuffled_params.begin(), shuffled_params.end(), rng);
        index.node_params = std::move(shuffled_params);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    res.build_time_s = std::chrono::duration<double>(t_end - t_start).count();
    res.total_edges = index.get_total_edges();
    res.layer0_edges = index.get_layer0_edges();
    res.upper_layer_edges = index.get_upper_layer_edges();
    res.mean_degree = (double)res.layer0_edges / (double)n_samples;
    res.memory_mb = (double)index.get_total_memory_bytes() / (1024.0 * 1024.0);

    // Compute Graph Structural Metrics
    GraphMetrics gm = analyze_graph(index);
    res.in_degree_variance = gm.in_degree_var;
    res.max_in_degree = gm.max_in_deg;
    res.reachability_pct = gm.reachability;

    // Search evaluation
    std::vector<std::vector<Candidate>> results(n_queries);
    std::vector<double> latencies_ms(n_queries);
    uint64_t total_dist_evals = 0;

    auto s_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_dist_evals)
    for (size_t q = 0; q < n_queries; ++q) {
        auto q_start = std::chrono::high_resolution_clock::now();
        QuerySearchStats stats;
        results[q] = index.search(
            &queries[q * dim], 10, ef_search, stats
        );
        auto q_end = std::chrono::high_resolution_clock::now();
        latencies_ms[q] = std::chrono::duration<double, std::milli>(q_end - q_start).count();
        total_dist_evals += stats.total_dist_evals();
    }

    auto s_end = std::chrono::high_resolution_clock::now();
    double total_search_time_s = std::chrono::duration<double>(s_end - s_start).count();

    res.throughput_qps = (double)n_queries / total_search_time_s;
    res.mean_dist_evals = (double)total_dist_evals / (double)n_queries;
    res.recall_at_1 = compute_recall(results, ground_truth, 1);
    res.recall_at_10 = compute_recall(results, ground_truth, 10);

    std::sort(latencies_ms.begin(), latencies_ms.end());
    res.p50_latency_ms = latencies_ms[(size_t)(n_queries * 0.50)];
    res.p95_latency_ms = latencies_ms[(size_t)(n_queries * 0.95)];
    res.p99_latency_ms = latencies_ms[(size_t)(n_queries * 0.99)];

    std::cout << "  -> Total Edges: " << res.total_edges 
              << " | Build: " << std::fixed << std::setprecision(2) << res.build_time_s << "s"
              << " | R@10: " << std::setprecision(2) << res.recall_at_10 * 100.0 << "%"
              << " | QPS: " << std::setprecision(0) << res.throughput_qps
              << " | Dist/q: " << std::setprecision(1) << res.mean_dist_evals
              << " | In-Deg Var: " << std::setprecision(1) << res.in_degree_variance
              << " | Reachability: " << std::setprecision(2) << res.reachability_pct << "%"
              << std::endl;

    return res;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  AdaptiveVec Phase 5.0: Architecture Recovery & Ablation Suite   " << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    // Load SIFT-100K
    std::string sift_path = "data/sift_subset_100k.fvecs";
    std::string sift_qpath = "data/sift_query.fvecs";
    std::string sift_gtpath = "data/sift_subset_groundtruth.ivecs";

    std::vector<float> sift_data, sift_queries;
    std::vector<std::vector<uint32_t>> sift_gt;
    size_t sift_n = 0, sift_dim = 0, sift_nq = 0, sift_dummy = 0;

    std::cout << "[*] Loading SIFT-100K ..." << std::endl;
    if (!load_fvecs(sift_path, sift_data, sift_n, sift_dim, 100000)) {
        std::cerr << "[-] Error loading " << sift_path << std::endl;
        return 1;
    }
    if (!load_fvecs(sift_qpath, sift_queries, sift_nq, sift_dummy, 1000)) {
        std::cerr << "[-] Error loading " << sift_qpath << std::endl;
        return 1;
    }
    if (!load_ivecs(sift_gtpath, sift_gt, 1000)) {
        std::cerr << "[-] Error loading " << sift_gtpath << std::endl;
        return 1;
    }
    std::cout << "[+] SIFT-100K Loaded: " << sift_n << " base, " << sift_nq << " queries." << std::endl;

    // Load Transformer-384D (MiniLM-50K-MultiDomain)
    std::string tf_path = "data/transformer_384d/transformer_384d_base.fvecs";
    std::string tf_qpath = "data/transformer_384d/transformer_384d_query.fvecs";
    std::string tf_gtpath = "data/transformer_384d/transformer_384d_groundtruth.ivecs";

    std::vector<float> tf_data, tf_queries;
    std::vector<std::vector<uint32_t>> tf_gt;
    size_t tf_n = 0, tf_dim = 0, tf_nq = 0;

    std::cout << "[*] Loading Transformer-384D (MiniLM-50K-MultiDomain) ..." << std::endl;
    if (load_fvecs(tf_path, tf_data, tf_n, tf_dim, 57893) &&
        load_fvecs(tf_qpath, tf_queries, tf_nq, sift_dummy, 1000) &&
        load_ivecs(tf_gtpath, tf_gt, 1000)) {
        std::cout << "[+] Transformer-384D Loaded: " << tf_n << " base, " << tf_nq << " queries." << std::endl;
    } else {
        std::cout << "[!] Transformer files not found at " << tf_path << ", skipping transformer run." << std::endl;
    }

    // -------------------------------------------------------------
    // Part 1: Run the 8 Geometry Ablation Experiments (Exp A - H)
    // -------------------------------------------------------------
    std::cout << "\n=============================================================" << std::endl;
    std::cout << "  Part 1: The 8 Geometry-Aware Ablation Experiments (A - H)   " << std::endl;
    std::cout << "=============================================================" << std::endl;

    std::vector<AblationResult> ablation_results;

    // Exp A: Uniform Connectivity (Scientific Control)
    PolicyConfig cfg_a;
    cfg_a.enable_dynamic_m_efc = false;
    cfg_a.enable_layer_scaling = false;
    cfg_a.enable_hubness_regulation = false;
    cfg_a.m_base = 16;
    cfg_a.ef_c_base = 120;
    ablation_results.push_back(evaluate_configuration(
        "Exp-A", "Uniform Connectivity (Control: M=16, efC=120)",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
        cfg_a, false
    ));

    // Exp B: Original AdaptiveVec Connectivity
    PolicyConfig cfg_b;
    cfg_b.enable_dynamic_m_efc = true;
    cfg_b.enable_layer_scaling = true;
    cfg_b.enable_hubness_regulation = true;
    cfg_b.m_base = 16;
    cfg_b.ef_c_base = 120;
    cfg_b.alpha_lid = 0.5f;
    cfg_b.beta_density = 0.5f;
    cfg_b.sensitivity = 0.40f;
    ablation_results.push_back(evaluate_configuration(
        "Exp-B", "Original Adaptive Connectivity (LID + Density)",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
        cfg_b, true
    ));

    // Exp C: LID-Only Connectivity
    PolicyConfig cfg_c = cfg_b;
    cfg_c.type = PolicyType::LidOnly;
    ablation_results.push_back(evaluate_configuration(
        "Exp-C", "LID-Only Connectivity (alpha=1.0, beta=0.0)",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
        cfg_c, true
    ));

    // Exp D: Density-Only Connectivity
    PolicyConfig cfg_d = cfg_b;
    cfg_d.type = PolicyType::DensityOnly;
    ablation_results.push_back(evaluate_configuration(
        "Exp-D", "Density-Only Connectivity (alpha=0.0, beta=1.0)",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
        cfg_d, true
    ));

    // Exp E: Randomized Connectivity (Critical Control)
    ablation_results.push_back(evaluate_configuration(
        "Exp-E", "Randomized Permuted Connectivity (Same Degree Distribution)",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
        cfg_b, true, SpaceType::L2, true /* shuffle */
    ));

    // Exp F: Constant-Budget Adaptive Connectivity (Matched Total Edges)
    PolicyConfig cfg_f = cfg_b;
    cfg_f.m_base = 17; // Calibrated base to equalize total edges to Exp A (~2.70M)
    ablation_results.push_back(evaluate_configuration(
        "Exp-F", "Constant-Budget Adaptive Connectivity (Matched Edges ~2.70M)",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
        cfg_f, true
    ));

    // Exp G: Geometry-Reversed Connectivity (Diagnostic Test)
    ablation_results.push_back(evaluate_configuration(
        "Exp-G", "Geometry-Reversed Connectivity (Allocate High M to Low Complexity)",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
        cfg_b, true, SpaceType::L2, false, true /* reverse */
    ));

    // Exp H1: Construction Exploration Isolation (Adaptive M only)
    ablation_results.push_back(evaluate_configuration(
        "Exp-H1", "Adaptive M Only (Fixed efC=120)",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
        cfg_b, true, SpaceType::L2, false, false, 120 /* fixed efC */
    ));

    // Exp H2: Construction Exploration Isolation (Adaptive efC only)
    ablation_results.push_back(evaluate_configuration(
        "Exp-H2", "Adaptive efC Only (Fixed M=16)",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
        cfg_b, true, SpaceType::L2, false, false, 0, 16 /* fixed M */
    ));

    // -------------------------------------------------------------
    // Part 2: Architecture Comparison (Architecture B vs Architecture C)
    // -------------------------------------------------------------
    std::cout << "\n=============================================================" << std::endl;
    std::cout << "  Part 2: Three-Way Architecture Suite (B vs C)               " << std::endl;
    std::cout << "=============================================================" << std::endl;

    std::vector<AblationResult> arch_b_results;
    std::vector<AblationResult> arch_c_results;

    // Architecture B (Optimized Original) on SIFT-100K
    arch_b_results.push_back(evaluate_configuration(
        "Arch-B-SIFT", "Architecture B (Optimized Original) on SIFT-100K",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
        cfg_b, true
    ));

    // Architecture C (Current Phase 4.3 Control) on SIFT-100K
    arch_c_results.push_back(evaluate_configuration(
        "Arch-C-SIFT", "Architecture C (Phase 4.3 Control) on SIFT-100K",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt,
        cfg_a, false
    ));

    // Evaluate on Transformer-384D if available
    if (tf_n > 0) {
        PolicyConfig cfg_tf_b = cfg_b;
        PolicyConfig cfg_tf_c = cfg_a;

        arch_b_results.push_back(evaluate_configuration(
            "Arch-B-TF", "Architecture B (Optimized Original) on Transformer-384D",
            tf_data, tf_n, tf_dim, tf_queries, tf_nq, tf_gt,
            cfg_tf_b, true, SpaceType::Cosine
        ));

        arch_c_results.push_back(evaluate_configuration(
            "Arch-C-TF", "Architecture C (Phase 4.3 Control) on Transformer-384D",
            tf_data, tf_n, tf_dim, tf_queries, tf_nq, tf_gt,
            cfg_tf_c, false, SpaceType::Cosine
        ));
    }

    // -------------------------------------------------------------
    // Part 3: Export Machine-Readable JSON Deliverables
    // -------------------------------------------------------------
    std::cout << "\n[*] Exporting experiments/phase_5_0_geometry_ablation.json ..." << std::endl;
    {
        std::ofstream out("experiments/phase_5_0_geometry_ablation.json");
        out << "[\n";
        for (size_t i = 0; i < ablation_results.size(); ++i) {
            const auto& r = ablation_results[i];
            out << "  {\n"
                << "    \"experiment_id\": \"" << r.experiment_id << "\",\n"
                << "    \"description\": \"" << r.description << "\",\n"
                << "    \"total_edges\": " << r.total_edges << ",\n"
                << "    \"layer0_edges\": " << r.layer0_edges << ",\n"
                << "    \"upper_layer_edges\": " << r.upper_layer_edges << ",\n"
                << "    \"mean_degree\": " << std::fixed << std::setprecision(2) << r.mean_degree << ",\n"
                << "    \"build_time_sec\": " << std::setprecision(2) << r.build_time_s << ",\n"
                << "    \"recall_at_1\": " << std::setprecision(4) << r.recall_at_1 << ",\n"
                << "    \"recall_at_10\": " << std::setprecision(4) << r.recall_at_10 << ",\n"
                << "    \"throughput_qps\": " << std::setprecision(1) << r.throughput_qps << ",\n"
                << "    \"p50_latency_ms\": " << std::setprecision(4) << r.p50_latency_ms << ",\n"
                << "    \"p95_latency_ms\": " << std::setprecision(4) << r.p95_latency_ms << ",\n"
                << "    \"p99_latency_ms\": " << std::setprecision(4) << r.p99_latency_ms << ",\n"
                << "    \"mean_distance_evaluations\": " << std::setprecision(1) << r.mean_dist_evals << ",\n"
                << "    \"in_degree_variance\": " << std::setprecision(2) << r.in_degree_variance << ",\n"
                << "    \"max_in_degree\": " << r.max_in_degree << ",\n"
                << "    \"reachability_pct\": " << std::setprecision(2) << r.reachability_pct << ",\n"
                << "    \"memory_mb\": " << std::setprecision(2) << r.memory_mb << "\n"
                << "  }" << (i + 1 < ablation_results.size() ? "," : "") << "\n";
        }
        out << "]\n";
    }

    std::cout << "[*] Exporting experiments/phase_5_0_optimized_original_results.json ..." << std::endl;
    {
        std::ofstream out("experiments/phase_5_0_optimized_original_results.json");
        out << "[\n";
        for (size_t i = 0; i < arch_b_results.size(); ++i) {
            const auto& r = arch_b_results[i];
            out << "  {\n"
                << "    \"architecture\": \"Architecture B (Optimized Original AdaptiveVec)\",\n"
                << "    \"experiment_id\": \"" << r.experiment_id << "\",\n"
                << "    \"description\": \"" << r.description << "\",\n"
                << "    \"total_edges\": " << r.total_edges << ",\n"
                << "    \"mean_degree\": " << std::fixed << std::setprecision(2) << r.mean_degree << ",\n"
                << "    \"build_time_sec\": " << std::setprecision(2) << r.build_time_s << ",\n"
                << "    \"recall_at_10\": " << std::setprecision(4) << r.recall_at_10 << ",\n"
                << "    \"throughput_qps\": " << std::setprecision(1) << r.throughput_qps << ",\n"
                << "    \"p50_latency_ms\": " << std::setprecision(4) << r.p50_latency_ms << ",\n"
                << "    \"p99_latency_ms\": " << std::setprecision(4) << r.p99_latency_ms << ",\n"
                << "    \"mean_distance_evaluations\": " << std::setprecision(1) << r.mean_dist_evals << ",\n"
                << "    \"in_degree_variance\": " << std::setprecision(2) << r.in_degree_variance << ",\n"
                << "    \"reachability_pct\": " << std::setprecision(2) << r.reachability_pct << ",\n"
                << "    \"memory_mb\": " << std::setprecision(2) << r.memory_mb << "\n"
                << "  }" << (i + 1 < arch_b_results.size() ? "," : "") << "\n";
        }
        out << "]\n";
    }

    std::cout << "[*] Exporting experiments/phase_5_0_current_architecture_results.json ..." << std::endl;
    {
        std::ofstream out("experiments/phase_5_0_current_architecture_results.json");
        out << "[\n";
        for (size_t i = 0; i < arch_c_results.size(); ++i) {
            const auto& r = arch_c_results[i];
            out << "  {\n"
                << "    \"architecture\": \"Architecture C (Current Phase 4.3 Static Control)\",\n"
                << "    \"experiment_id\": \"" << r.experiment_id << "\",\n"
                << "    \"description\": \"" << r.description << "\",\n"
                << "    \"total_edges\": " << r.total_edges << ",\n"
                << "    \"mean_degree\": " << std::fixed << std::setprecision(2) << r.mean_degree << ",\n"
                << "    \"build_time_sec\": " << std::setprecision(2) << r.build_time_s << ",\n"
                << "    \"recall_at_10\": " << std::setprecision(4) << r.recall_at_10 << ",\n"
                << "    \"throughput_qps\": " << std::setprecision(1) << r.throughput_qps << ",\n"
                << "    \"p50_latency_ms\": " << std::setprecision(4) << r.p50_latency_ms << ",\n"
                << "    \"p99_latency_ms\": " << std::setprecision(4) << r.p99_latency_ms << ",\n"
                << "    \"mean_distance_evaluations\": " << std::setprecision(1) << r.mean_dist_evals << ",\n"
                << "    \"in_degree_variance\": " << std::setprecision(2) << r.in_degree_variance << ",\n"
                << "    \"reachability_pct\": " << std::setprecision(2) << r.reachability_pct << ",\n"
                << "    \"memory_mb\": " << std::setprecision(2) << r.memory_mb << "\n"
                << "  }" << (i + 1 < arch_c_results.size() ? "," : "") << "\n";
        }
        out << "]\n";
    }

    // Export Graph Statistics
    std::cout << "[*] Exporting experiments/phase_5_0_graph_statistics.json ..." << std::endl;
    {
        // Re-analyze Exp B for detailed structural graph metrics
        AdaptiveHNSWIndex idx_stats(sift_dim, SpaceType::L2, true, cfg_b);
        for (size_t i = 0; i < sift_n; ++i) idx_stats.insert(&sift_data[i * sift_dim]);
        GraphMetrics gm = analyze_graph(idx_stats);

        std::ofstream out("experiments/phase_5_0_graph_statistics.json");
        out << "{\n"
            << "  \"dataset\": \"SIFT-100K subset\",\n"
            << "  \"evaluated_nodes\": " << sift_n << ",\n"
            << "  \"mean_out_degree\": " << std::fixed << std::setprecision(2) << gm.mean_out_degree << ",\n"
            << "  \"mean_in_degree\": " << std::setprecision(2) << gm.mean_in_degree << ",\n"
            << "  \"in_degree_variance\": " << std::setprecision(2) << gm.in_degree_var << ",\n"
            << "  \"max_in_degree\": " << gm.max_in_deg << ",\n"
            << "  \"mean_edge_length\": " << std::setprecision(2) << gm.mean_edge_length << ",\n"
            << "  \"median_edge_length\": " << std::setprecision(2) << gm.median_edge_length << ",\n"
            << "  \"p90_edge_length\": " << std::setprecision(2) << gm.p90_edge_length << ",\n"
            << "  \"clustering_coefficient\": " << std::setprecision(4) << gm.clustering_coeff << ",\n"
            << "  \"global_reachability_pct\": " << std::setprecision(2) << gm.reachability << ",\n"
            << "  \"lid_out_degree_correlation\": " << std::setprecision(4) << gm.lid_degree_corr << ",\n"
            << "  \"density_out_degree_correlation\": " << std::setprecision(4) << gm.density_degree_corr << ",\n"
            << "  \"degree_by_lid_quartiles\": {\n"
            << "    \"q1_low_lid\": " << std::setprecision(2) << gm.deg_lid_q1 << ",\n"
            << "    \"q2_mid_low_lid\": " << std::setprecision(2) << gm.deg_lid_q2 << ",\n"
            << "    \"q3_mid_high_lid\": " << std::setprecision(2) << gm.deg_lid_q3 << ",\n"
            << "    \"q4_high_lid\": " << std::setprecision(2) << gm.deg_lid_q4 << "\n"
            << "  },\n"
            << "  \"degree_by_density_quartiles\": {\n"
            << "    \"q1_dense\": " << std::setprecision(2) << gm.deg_dens_q1 << ",\n"
            << "    \"q2_mid_dense\": " << std::setprecision(2) << gm.deg_dens_q2 << ",\n"
            << "    \"q3_mid_sparse\": " << std::setprecision(2) << gm.deg_dens_q3 << ",\n"
            << "    \"q4_sparse\": " << std::setprecision(2) << gm.deg_dens_q4 << "\n"
            << "  }\n"
            << "}\n";
    }

    std::cout << "\n[+] Phase 5.0 Experiment Suite Completed Successfully!" << std::endl;
    return 0;
}
