#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <cmath>

using namespace adaptivevec;

struct DatasetWrapper {
    std::string name;
    std::vector<float> data;
    std::vector<float> queries;
    size_t n_samples = 0;
    size_t n_queries = 0;
    size_t dim = 0;
    SpaceType space = SpaceType::L2;
    std::vector<std::vector<uint32_t>> ground_truth;
};

// Generates the 8-cluster synthetic benchmark dataset, strictly matching benchmark_main.cpp
DatasetWrapper generate_synthetic_multi_cluster(size_t n_samples = 50000, size_t dim = 64, size_t n_queries = 1000) {
    DatasetWrapper ds;
    ds.name = "Synthetic-Multi-Cluster";
    ds.n_samples = n_samples;
    ds.n_queries = n_queries;
    ds.dim = dim;
    ds.space = SpaceType::L2;
    ds.data.resize(n_samples * dim);
    ds.queries.resize(n_queries * dim);

    std::mt19937 rng(42);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> cluster_dist(0, 7);

    // 8 cluster centers
    std::vector<std::vector<float>> centers(8, std::vector<float>(dim));
    for (int c = 0; c < 8; ++c) {
        for (size_t d = 0; d < dim; ++d) {
            centers[c][d] = norm_dist(rng) * 15.0f;
        }
    }

    // Fill data with alternating dense and sparse clusters
    for (size_t i = 0; i < n_samples; ++i) {
        size_t c = cluster_dist(rng);
        float scale = (c % 2 == 0) ? 0.6f : 5.0f;
        for (size_t d = 0; d < dim; ++d) {
            ds.data[i * dim + d] = centers[c][d] + norm_dist(rng) * scale;
        }
    }

    // Fill queries
    for (size_t q = 0; q < n_queries; ++q) {
        size_t c = cluster_dist(rng);
        float scale = (c % 2 == 0) ? 0.6f : 5.0f;
        for (size_t d = 0; d < dim; ++d) {
            ds.queries[q * dim + d] = centers[c][d] + norm_dist(rng) * scale;
        }
    }

    // Ground truth
    std::cout << "[*] Computing ground truth for " << n_queries << " queries on " << n_samples << " vectors..." << std::endl;
    ds.ground_truth.resize(n_queries);
    for (size_t q = 0; q < n_queries; ++q) {
        const float* q_vec = &ds.queries[q * dim];
        std::vector<Candidate> all_dists(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            const float* d_vec = &ds.data[i * dim];
            all_dists[i] = {l2_distance(q_vec, d_vec, dim), (tableint)i};
        }
        std::partial_sort(all_dists.begin(), all_dists.begin() + 100, all_dists.end());
        ds.ground_truth[q].resize(100);
        for (int k = 0; k < 100; ++k) {
            ds.ground_truth[q][k] = all_dists[k].id;
        }
    }

    return ds;
}

DatasetWrapper load_sift_dataset() {
    DatasetWrapper ds;
    ds.name = "SIFT-100K";
    ds.space = SpaceType::L2;

    size_t n = 0, d = 0;
    if (!load_fvecs("data/sift_subset_100k.fvecs", ds.data, n, d, 100000)) {
        std::cerr << "[-] Error: Failed to load data/sift_subset_100k.fvecs" << std::endl;
        return ds;
    }
    ds.n_samples = n;
    ds.dim = d;

    size_t q_n = 0, q_d = 0;
    load_fvecs("data/sift_query.fvecs", ds.queries, q_n, q_d, 10000);
    ds.n_queries = q_n;

    load_ivecs("data/sift_subset_groundtruth.ivecs", ds.ground_truth, ds.n_queries);
    std::cout << "[+] Loaded SIFT-100K: " << ds.n_samples << " vectors, " << ds.n_queries << " queries with ground truth." << std::endl;
    return ds;
}

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

double measure_reachability(const AdaptiveHNSWIndex& index) {
    if (index.graphs.empty() || index.graphs[0].empty() || index.num_elements == 0) return 0.0;
    std::unordered_set<tableint> visited;
    std::queue<tableint> q;
    q.push(index.enter_point);
    visited.insert(index.enter_point);
    while (!q.empty()) {
        tableint curr = q.front();
        q.pop();
        if (curr < index.graphs[0].size()) {
            for (tableint neighbor : index.graphs[0][curr]) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }
    }
    return (double)visited.size() / (double)index.num_elements * 100.0;
}

// ---------------------------------------------------------------------------
// 1. Hubness Penalty Weight (mu) Sweep on Synthetic-Multi-Cluster
// ---------------------------------------------------------------------------
void run_hubness_sweep(const DatasetWrapper& ds) {
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "HUBNESS PENALTY WEIGHT (mu) SWEEP - Synthetic-Multi-Cluster" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::vector<float> mu_values = {0.0f, 0.05f, 0.10f, 0.15f, 0.20f, 0.30f};
    int k = 10;
    int ef_search = 50;

    std::ofstream out("experiments/hubness_sweep_results.json");
    out << "{\n";
    out << "  \"experiment\": \"hubness_penalty_weight_sweep\",\n";
    out << "  \"dataset\": \"Synthetic-Multi-Cluster\",\n";
    out << "  \"n_samples\": " << ds.n_samples << ",\n";
    out << "  \"dim\": " << ds.dim << ",\n";
    out << "  \"n_queries\": " << ds.n_queries << ",\n";
    out << "  \"k\": " << k << ",\n";
    out << "  \"mu_values_tested\": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],\n";
    out << "  \"results\": [\n";

    std::cout << std::right << std::setw(6) << "mu" << " | "
              << std::setw(10) << "Recall@10" << " | "
              << std::setw(8) << "QPS" << " | "
              << std::setw(11) << "Reachable%" << " | "
              << std::setw(10) << "Edges" << " | "
              << std::setw(10) << "Deg Median" << " | "
              << std::setw(9) << "Build (s)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    for (size_t m_idx = 0; m_idx < mu_values.size(); ++m_idx) {
        float mu = mu_values[m_idx];

        PolicyConfig cfg;
        cfg.enable_dynamic_m_efc = true;
        cfg.sensitivity = 0.4f;
        cfg.enable_layer_scaling = true;
        cfg.lambda_layer = 0.75f;
        cfg.m_min_layer = 4;
        cfg.enable_hubness_regulation = (mu > 0.0f);
        cfg.hubness_mu = mu;
        cfg.enable_ada_ef = false; // Steps 1-4 only

        AdaptiveHNSWIndex index(ds.dim, ds.space, true, cfg);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < ds.n_samples; ++i) {
            index.insert(&ds.data[i * ds.dim]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double build_time = std::chrono::duration<double>(t1 - t0).count();

        double reachability = measure_reachability(index);

        // Degrees
        std::vector<int> l0_degrees;
        for (size_t i = 0; i < ds.n_samples; ++i) {
            if (i < index.graphs[0].size()) {
                l0_degrees.push_back((int)index.graphs[0][i].size());
            }
        }
        std::sort(l0_degrees.begin(), l0_degrees.end());
        int deg_min = l0_degrees.empty() ? 0 : l0_degrees.front();
        int deg_max = l0_degrees.empty() ? 0 : l0_degrees.back();
        double deg_median = l0_degrees.empty() ? 0.0 : (double)l0_degrees[l0_degrees.size() / 2];
        double deg_sum = 0.0;
        for (int d : l0_degrees) deg_sum += d;
        double deg_mean = l0_degrees.empty() ? 0.0 : deg_sum / l0_degrees.size();

        // Query
        uint64_t comps_before = index.total_dist_computations;
        std::vector<std::vector<Candidate>> results(ds.n_queries);
        auto q0 = std::chrono::high_resolution_clock::now();
        for (size_t q = 0; q < ds.n_queries; ++q) {
            results[q] = index.search(&ds.queries[q * ds.dim], k, ef_search);
        }
        auto q1 = std::chrono::high_resolution_clock::now();
        double query_time = std::chrono::duration<double>(q1 - q0).count();

        double qps = (query_time > 0.0) ? ((double)ds.n_queries / query_time) : 0.0;
        double recall = compute_recall(results, ds.ground_truth, k);
        double dist_evals = (double)(index.total_dist_computations - comps_before) / (double)ds.n_queries;
        size_t total_edges = index.get_total_edges();
        double ram_mb = (double)index.get_total_memory_bytes() / (1024.0 * 1024.0);

        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(6) << mu << " | "
                  << std::setprecision(4) << std::setw(10) << recall << " | "
                  << std::setprecision(1) << std::setw(8) << qps << " | "
                  << std::setprecision(2) << std::setw(10) << reachability << "% | "
                  << std::setw(10) << total_edges << " | "
                  << std::setprecision(1) << std::setw(10) << deg_median << " | "
                  << std::setprecision(1) << std::setw(9) << build_time << std::endl;

        out << "    {\n"
            << "      \"mu\": " << std::fixed << std::setprecision(2) << mu << ",\n"
            << "      \"recall_at_10\": " << std::setprecision(4) << recall << ",\n"
            << "      \"qps\": " << std::setprecision(1) << qps << ",\n"
            << "      \"build_time_sec\": " << std::setprecision(2) << build_time << ",\n"
            << "      \"total_edges\": " << total_edges << ",\n"
            << "      \"memory_mb\": " << std::setprecision(2) << ram_mb << ",\n"
            << "      \"reachability_pct\": " << std::setprecision(2) << reachability << ",\n"
            << "      \"dist_evals_per_query\": " << std::setprecision(1) << dist_evals << ",\n"
            << "      \"degree_min\": " << deg_min << ",\n"
            << "      \"degree_max\": " << deg_max << ",\n"
            << "      \"degree_median\": " << std::setprecision(1) << deg_median << ",\n"
            << "      \"degree_mean\": " << std::setprecision(2) << deg_mean << "\n"
            << "    }" << (m_idx + 1 < mu_values.size() ? "," : "") << "\n";
    }

    out << "  ]\n}\n";
    std::cout << "[+] Saved hubness sweep to experiments/hubness_sweep_results.json" << std::endl;
}

// ---------------------------------------------------------------------------
// 2. Stagnation Early-Exit Patience (p) Sweep on SIFT-100K
// ---------------------------------------------------------------------------
void run_patience_sweep(const DatasetWrapper& ds) {
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "STAGNATION PATIENCE (p) SWEEP - SIFT-100K" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::vector<int> patience_values = {3, 4, 5, 6, 8, 10};
    int k = 10;
    int ef_search = 50;

    // Build index ONCE with Step 4 topology (Dynamic M + Layer Scaling + Hubness)
    std::cout << "[*] Building AdaptiveVec index on SIFT-100K..." << std::endl;
    PolicyConfig cfg;
    cfg.enable_dynamic_m_efc = true;
    cfg.sensitivity = 0.4f;
    cfg.enable_layer_scaling = true;
    cfg.lambda_layer = 0.75f;
    cfg.m_min_layer = 4;
    cfg.enable_hubness_regulation = true;
    cfg.hubness_mu = 0.15f;
    cfg.enable_ada_ef = true; // We will modulate patience during search

    AdaptiveHNSWIndex index(ds.dim, ds.space, true, cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ds.n_samples; ++i) {
        index.insert(&ds.data[i * ds.dim]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(t1 - t0).count();
    size_t total_edges = index.get_total_edges();
    double ram_mb = (double)index.get_total_memory_bytes() / (1024.0 * 1024.0);

    std::cout << "[+] Build complete in " << std::fixed << std::setprecision(1) << build_time << "s ("
              << total_edges << " edges, " << ram_mb << " MB)" << std::endl;

    std::ofstream out("experiments/patience_sweep_results.json");
    out << "{\n";
    out << "  \"experiment\": \"stagnation_patience_sweep\",\n";
    out << "  \"dataset\": \"SIFT-100K\",\n";
    out << "  \"n_samples\": " << ds.n_samples << ",\n";
    out << "  \"dim\": " << ds.dim << ",\n";
    out << "  \"n_queries\": " << ds.n_queries << ",\n";
    out << "  \"k\": " << k << ",\n";
    out << "  \"ef_search\": " << ef_search << ",\n";
    out << "  \"build_time_sec\": " << std::fixed << std::setprecision(2) << build_time << ",\n";
    out << "  \"index_edges\": " << total_edges << ",\n";
    out << "  \"index_memory_mb\": " << std::setprecision(2) << ram_mb << ",\n";
    out << "  \"patience_values_tested\": [3, 4, 5, 6, 8, 10],\n";
    out << "  \"results\": [\n";

    std::cout << std::right << std::setw(10) << "Config" << " | "
              << std::setw(10) << "Recall@10" << " | "
              << std::setw(8) << "QPS" << " | "
              << std::setw(10) << "Dist/q" << " | "
              << std::setw(8) << "p50 (ms)" << " | "
              << std::setw(8) << "p95 (ms)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    struct TestConfig {
        std::string name;
        bool use_exit;
        int patience;
    };

    std::vector<TestConfig> test_configs = {{"no_exit", false, 0}};
    for (int p : patience_values) {
        test_configs.push_back({"p=" + std::to_string(p), true, p});
    }

    for (size_t t_idx = 0; t_idx < test_configs.size(); ++t_idx) {
        const auto& tc = test_configs[t_idx];
        index.policy.enable_ada_ef = tc.use_exit;
        index.policy.ada_ef_patience = tc.patience;

        std::vector<double> latencies_ms(ds.n_queries);
        std::vector<std::vector<Candidate>> results(ds.n_queries);
        uint64_t comps_before = index.total_dist_computations;

        auto q0 = std::chrono::high_resolution_clock::now();
        for (size_t q = 0; q < ds.n_queries; ++q) {
            auto l0 = std::chrono::high_resolution_clock::now();
            results[q] = index.search(&ds.queries[q * ds.dim], k, ef_search);
            auto l1 = std::chrono::high_resolution_clock::now();
            latencies_ms[q] = std::chrono::duration<double, std::milli>(l1 - l0).count();
        }
        auto q1 = std::chrono::high_resolution_clock::now();
        double query_time = std::chrono::duration<double>(q1 - q0).count();

        std::sort(latencies_ms.begin(), latencies_ms.end());
        double p50 = latencies_ms[(size_t)(ds.n_queries * 0.50)];
        double p95 = latencies_ms[(size_t)(ds.n_queries * 0.95)];
        double p99 = latencies_ms[(size_t)(ds.n_queries * 0.99)];

        double qps = (query_time > 0.0) ? ((double)ds.n_queries / query_time) : 0.0;
        double recall = compute_recall(results, ds.ground_truth, k);
        double dist_evals = (double)(index.total_dist_computations - comps_before) / (double)ds.n_queries;

        std::cout << std::setw(10) << tc.name << " | "
                  << std::fixed << std::setprecision(4) << std::setw(10) << recall << " | "
                  << std::setprecision(1) << std::setw(8) << qps << " | "
                  << std::setprecision(1) << std::setw(10) << dist_evals << " | "
                  << std::setprecision(3) << std::setw(8) << p50 << " | "
                  << std::setprecision(3) << std::setw(8) << p95 << std::endl;

        out << "    {\n"
            << "      \"config\": \"" << tc.name << "\",\n"
            << "      \"early_exit\": " << (tc.use_exit ? "true" : "false") << ",\n"
            << "      \"patience\": " << tc.patience << ",\n"
            << "      \"recall_at_10\": " << std::setprecision(4) << recall << ",\n"
            << "      \"qps\": " << std::setprecision(1) << qps << ",\n"
            << "      \"dist_evals_per_query\": " << std::setprecision(1) << dist_evals << ",\n"
            << "      \"latency_p50_ms\": " << std::setprecision(3) << p50 << ",\n"
            << "      \"latency_p95_ms\": " << std::setprecision(3) << p95 << ",\n"
            << "      \"latency_p99_ms\": " << std::setprecision(3) << p99 << "\n"
            << "    }" << (t_idx + 1 < test_configs.size() ? "," : "") << "\n";
    }

    out << "  ]\n}\n";
    std::cout << "[+] Saved patience sweep to experiments/patience_sweep_results.json" << std::endl;
}

// ---------------------------------------------------------------------------
// 3. Policy Sensitivity Factor (gamma) Sweep on SIFT-100K
// ---------------------------------------------------------------------------
void run_sensitivity_sweep(const DatasetWrapper& ds) {
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "POLICY SENSITIVITY (gamma) SWEEP - SIFT-100K" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::vector<float> gamma_values = {0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
    int k = 10;
    int ef_search = 50;

    std::ofstream out("experiments/sensitivity_sweep_results.json");
    out << "{\n";
    out << "  \"experiment\": \"policy_sensitivity_sweep\",\n";
    out << "  \"dataset\": \"SIFT-100K\",\n";
    out << "  \"n_samples\": " << ds.n_samples << ",\n";
    out << "  \"dim\": " << ds.dim << ",\n";
    out << "  \"n_queries\": " << ds.n_queries << ",\n";
    out << "  \"k\": " << k << ",\n";
    out << "  \"gamma_values_tested\": [0.2, 0.4, 0.6, 0.8, 1.0],\n";
    out << "  \"results\": [\n";

    std::cout << std::right << std::setw(6) << "gamma" << " | "
              << std::setw(10) << "Recall@10" << " | "
              << std::setw(8) << "QPS" << " | "
              << std::setw(10) << "Edges" << " | "
              << std::setw(8) << "M mean" << " | "
              << std::setw(8) << "Build(s)" << std::endl;
    std::cout << std::string(62, '-') << std::endl;

    for (size_t g_idx = 0; g_idx < gamma_values.size(); ++g_idx) {
        float gamma = gamma_values[g_idx];

        PolicyConfig cfg;
        cfg.enable_dynamic_m_efc = true;
        cfg.sensitivity = gamma;
        cfg.enable_layer_scaling = false; // Steps 1-2 only
        cfg.enable_hubness_regulation = false;
        cfg.enable_ada_ef = false;

        AdaptiveHNSWIndex index(ds.dim, ds.space, true, cfg);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < ds.n_samples; ++i) {
            index.insert(&ds.data[i * ds.dim]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double build_time = std::chrono::duration<double>(t1 - t0).count();

        // M stats
        std::vector<int> m_vals;
        double m_sum = 0.0;
        for (const auto& np : index.node_params) {
            m_vals.push_back(np.m);
            m_sum += np.m;
        }
        double m_mean = m_vals.empty() ? 0.0 : m_sum / m_vals.size();
        double m_sq_diff = 0.0;
        for (int m : m_vals) m_sq_diff += (m - m_mean) * (m - m_mean);
        double m_std = m_vals.empty() ? 0.0 : std::sqrt(m_sq_diff / m_vals.size());

        // Query
        uint64_t comps_before = index.total_dist_computations;
        std::vector<std::vector<Candidate>> results(ds.n_queries);
        auto q0 = std::chrono::high_resolution_clock::now();
        for (size_t q = 0; q < ds.n_queries; ++q) {
            results[q] = index.search(&ds.queries[q * ds.dim], k, ef_search);
        }
        auto q1 = std::chrono::high_resolution_clock::now();
        double query_time = std::chrono::duration<double>(q1 - q0).count();

        double qps = (query_time > 0.0) ? ((double)ds.n_queries / query_time) : 0.0;
        double recall = compute_recall(results, ds.ground_truth, k);
        size_t total_edges = index.get_total_edges();
        double ram_mb = (double)index.get_total_memory_bytes() / (1024.0 * 1024.0);

        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(6) << gamma << " | "
                  << std::setprecision(4) << std::setw(10) << recall << " | "
                  << std::setprecision(1) << std::setw(8) << qps << " | "
                  << std::setw(10) << total_edges << " | "
                  << std::setprecision(2) << std::setw(8) << m_mean << " | "
                  << std::setprecision(1) << std::setw(8) << build_time << std::endl;

        out << "    {\n"
            << "      \"gamma\": " << std::fixed << std::setprecision(2) << gamma << ",\n"
            << "      \"recall_at_10\": " << std::setprecision(4) << recall << ",\n"
            << "      \"qps\": " << std::setprecision(1) << qps << ",\n"
            << "      \"build_time_sec\": " << std::setprecision(2) << build_time << ",\n"
            << "      \"total_edges\": " << total_edges << ",\n"
            << "      \"memory_mb\": " << std::setprecision(2) << ram_mb << ",\n"
            << "      \"m_distribution\": {\n"
            << "        \"mean\": " << std::setprecision(2) << m_mean << ",\n"
            << "        \"std\": " << std::setprecision(2) << m_std << "\n"
            << "      }\n"
            << "    }" << (g_idx + 1 < gamma_values.size() ? "," : "") << "\n";
    }

    out << "  ]\n}\n";
    std::cout << "[+] Saved sensitivity sweep to experiments/sensitivity_sweep_results.json" << std::endl;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "AdaptiveVec Scientific Rigor & Reviewer-Proofing Parameter Sweeps" << std::endl;
    std::cout << "Native C++ AVX2 Benchmark Engine" << std::endl;
    std::cout << "======================================================================" << std::endl;

    // 1. Synthetic-Multi-Cluster Hubness Sweep
    DatasetWrapper synth_ds = generate_synthetic_multi_cluster(50000, 64, 1000);
    run_hubness_sweep(synth_ds);

    // 2. SIFT-100K Patience and Sensitivity Sweeps
    DatasetWrapper sift_ds = load_sift_dataset();
    if (sift_ds.n_samples > 0) {
        run_patience_sweep(sift_ds);
        run_sensitivity_sweep(sift_ds);
    } else {
        std::cerr << "[-] Skipping SIFT-100K sweeps: dataset files not found." << std::endl;
    }

    std::cout << "\n[+] ALL SWEEPS COMPLETE!" << std::endl;
    return 0;
}
