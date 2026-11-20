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

struct PermutationTrial {
    std::string mode; // "M_only", "efC_only", "joint"
    int seed;
    double score_m_correlation = 0.0;
    size_t total_edges = 0;
    double in_degree_var = 0.0;
    double recall_at_10 = 0.0;
    double qps = 0.0;
    double mean_dist_evals = 0.0;
    double jaccard_edge_overlap = 0.0;
};

struct SyntheticBoundaryResult {
    std::string config_name;
    double overall_recall_at_10 = 0.0;
    double dense_cluster_recall = 0.0;
    double sparse_boundary_recall = 0.0;
    double mean_dist_evals = 0.0;
    size_t total_edges = 0;
};

// Compute Pearson correlation
double compute_correlation(const std::vector<float>& x, const std::vector<float>& y) {
    size_t n = x.size();
    if (n == 0) return 0.0;
    double mx = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double my = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double cov = 0, varx = 0, vary = 0;
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mx;
        double dy = y[i] - my;
        cov += dx * dy;
        varx += dx * dx;
        vary += dy * dy;
    }
    return (varx > 1e-9 && vary > 1e-9) ? (cov / std::sqrt(varx * vary)) : 0.0;
}

int main() {
    std::cout << "=============================================================" << std::endl;
    std::cout << " AdaptiveVec Phase 5.2: Stage 2 Randomization Isolation Suite" << std::endl;
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

    // Search evaluation helper
    auto eval_index = [&](const AdaptiveHNSWIndex& idx, int ef_search = 64) {
        std::vector<double> recalls(sift_nq);
        std::vector<double> latencies(sift_nq);
        uint64_t total_dist_evals = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_dist_evals)
        for (size_t q = 0; q < sift_nq; ++q) {
            auto q_start = std::chrono::high_resolution_clock::now();
            QuerySearchStats stats;
            auto res = idx.search(&sift_queries[q * sift_dim], 10, ef_search, stats);
            auto q_end = std::chrono::high_resolution_clock::now();
            latencies[q] = std::chrono::duration<double, std::milli>(q_end - q_start).count();
            total_dist_evals += stats.total_dist_evals();

            std::unordered_set<uint32_t> gt_set;
            for (size_t k = 0; k < 10 && k < sift_gt[q].size(); ++k) {
                gt_set.insert(sift_gt[q][k]);
            }
            size_t match = 0;
            for (const auto& c : res) {
                if (gt_set.find(c.id) != gt_set.end()) match++;
            }
            recalls[q] = (double)match / 10.0;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_time_s = std::chrono::duration<double>(t1 - t0).count();
        double qps = (double)sift_nq / total_time_s;
        double mean_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / (double)sift_nq;
        double mean_dist = (double)total_dist_evals / (double)sift_nq;

        return std::make_tuple(mean_recall, qps, mean_dist, recalls);
    };

    // 1. Build Reference Architecture B Index
    std::cout << "\n>>> Building Reference Architecture B Index..." << std::endl;
    PolicyConfig cfg_arch_b;
    cfg_arch_b.m_base = 16;
    cfg_arch_b.ef_c_base = 120;
    cfg_arch_b.m_min = 8;
    cfg_arch_b.m_max = 24;
    cfg_arch_b.ef_c_min = 40;
    cfg_arch_b.ef_c_max = 220;
    cfg_arch_b.alpha_lid = 0.5f;
    cfg_arch_b.beta_density = 0.5f;
    cfg_arch_b.sensitivity = 1.0f;
    cfg_arch_b.enable_layer_scaling = true;
    cfg_arch_b.lambda_layer = 0.75f;
    cfg_arch_b.m_min_layer = 4;
    cfg_arch_b.enable_hubness_regulation = true;
    cfg_arch_b.hubness_mu = 0.15f;
    cfg_arch_b.use_streaming_welford = true;

    AdaptiveHNSWIndex arch_b_index(sift_dim, SpaceType::L2, true, cfg_arch_b);
    arch_b_index.reserve(sift_n);
    for (size_t i = 0; i < sift_n; ++i) {
        arch_b_index.insert(&sift_data[i * sift_dim]);
    }
    auto [arch_b_rec, arch_b_qps, arch_b_dist, arch_b_per_query_rec] = eval_index(arch_b_index);
    size_t arch_b_total_edges = arch_b_index.get_total_edges();
    std::cout << "Arch B: Recall@10=" << arch_b_rec * 100.0 << "%, QPS=" << arch_b_qps
              << ", Dist=" << arch_b_dist << ", Edges=" << arch_b_total_edges << std::endl;

    // Collect Arch B edge set for Jaccard overlap
    std::unordered_set<uint64_t> arch_b_edges;
    for (size_t u = 0; u < sift_n; ++u) {
        if (u < arch_b_index.graphs[0].size()) {
            for (tableint v : arch_b_index.graphs[0][u]) {
                arch_b_edges.insert(((uint64_t)u << 32) | (uint64_t)v);
            }
        }
    }

    // Extract original NodeParams
    std::vector<NodeParams> original_params = arch_b_index.node_params;
    std::vector<float> orig_scores(sift_n);
    std::vector<float> orig_m(sift_n);
    for (size_t i = 0; i < sift_n; ++i) {
        orig_scores[i] = original_params[i].score;
        orig_m[i] = (float)original_params[i].m;
    }
    double arch_b_corr = compute_correlation(orig_scores, orig_m);
    std::cout << "Arch B Correlation S(x) vs M: " << arch_b_corr << std::endl;

    // 2. Build Canonical HNSW Index
    std::cout << "\n>>> Building Canonical HNSW Baseline..." << std::endl;
    PolicyConfig cfg_canonical;
    cfg_canonical.m_base = 16;
    cfg_canonical.ef_c_base = 120;
    cfg_canonical.enable_layer_scaling = false;
    cfg_canonical.enable_hubness_regulation = false;

    AdaptiveHNSWIndex canonical_index(sift_dim, SpaceType::L2, false, cfg_canonical);
    canonical_index.reserve(sift_n);
    for (size_t i = 0; i < sift_n; ++i) {
        canonical_index.insert(&sift_data[i * sift_dim]);
    }
    auto [canon_rec, canon_qps, canon_dist, canon_per_query_rec] = eval_index(canonical_index);
    std::cout << "Canonical: Recall@10=" << canon_rec * 100.0 << "%, QPS=" << canon_qps
              << ", Dist=" << canon_dist << ", Edges=" << canonical_index.get_total_edges() << std::endl;

    // 3. Multi-Seed Permutations
    std::vector<int> seeds = {42, 101, 777, 999, 2024};
    std::vector<PermutationTrial> trials;

    // Store Seed 2024 per-query recall for boundary deprivation deep-dive
    std::vector<double> seed_2024_per_query_rec;
    std::vector<NodeParams> seed_2024_params;

    for (int seed : seeds) {
        std::cout << "\n--- Evaluating Permutation Seed " << seed << " ---" << std::endl;

        // A. M-only permutation (shuffle M, keep efC = 120 fixed)
        {
            std::vector<NodeParams> m_only_params = original_params;
            std::vector<int> m_vals(sift_n);
            for (size_t i = 0; i < sift_n; ++i) m_vals[i] = original_params[i].m;
            std::mt19937 rng(seed);
            std::shuffle(m_vals.begin(), m_vals.end(), rng);

            for (size_t i = 0; i < sift_n; ++i) {
                m_only_params[i].m = m_vals[i];
                m_only_params[i].m_max = m_vals[i];
                m_only_params[i].m_max0 = 2 * m_vals[i];
                m_only_params[i].ef_construction = 120; // fixed efC
            }

            AdaptiveHNSWIndex idx(sift_dim, SpaceType::L2, false, cfg_arch_b);
            idx.reserve(sift_n);
            idx.preassigned_node_params = m_only_params;
            for (size_t i = 0; i < sift_n; ++i) idx.insert(&sift_data[i * sift_dim]);

            auto [rec, qps, dist, pq_rec] = eval_index(idx);
            size_t edges = idx.get_total_edges();

            // Overlap & correlation
            std::vector<float> p_m(sift_n);
            for (size_t i = 0; i < sift_n; ++i) p_m[i] = (float)m_only_params[i].m;
            double corr = compute_correlation(orig_scores, p_m);

            size_t overlap = 0;
            for (size_t u = 0; u < sift_n; ++u) {
                if (u < idx.graphs[0].size()) {
                    for (tableint v : idx.graphs[0][u]) {
                        if (arch_b_edges.count(((uint64_t)u << 32) | (uint64_t)v)) overlap++;
                    }
                }
            }
            double jaccard = (double)overlap / (double)(arch_b_edges.size() + idx.get_layer0_edges() - overlap);

            // In-degree var
            double s_in = 0, s_in2 = 0;
            for (size_t i = 0; i < sift_n; ++i) {
                double d = (i < idx.in_degrees.size()) ? (double)idx.in_degrees[i] : 0.0;
                s_in += d; s_in2 += d * d;
            }
            double mean_in = s_in / (double)sift_n;
            double in_var = (s_in2 / (double)sift_n) - (mean_in * mean_in);

            trials.push_back({"M_only", seed, corr, edges, in_var, rec, qps, dist, jaccard});
            std::cout << "  [M-only, Seed " << seed << "] Rec=" << rec * 100.0 << "%, QPS=" << qps
                      << ", Dist=" << dist << ", Corr=" << corr << ", Jaccard=" << jaccard << std::endl;
        }

        // B. efC-only permutation (shuffle efC, keep M = 16 fixed)
        {
            std::vector<NodeParams> efc_only_params = original_params;
            std::vector<int> efc_vals(sift_n);
            for (size_t i = 0; i < sift_n; ++i) efc_vals[i] = original_params[i].ef_construction;
            std::mt19937 rng(seed + 1000);
            std::shuffle(efc_vals.begin(), efc_vals.end(), rng);

            for (size_t i = 0; i < sift_n; ++i) {
                efc_only_params[i].m = 16; // fixed M
                efc_only_params[i].m_max = 16;
                efc_only_params[i].m_max0 = 32;
                efc_only_params[i].ef_construction = efc_vals[i];
            }

            AdaptiveHNSWIndex idx(sift_dim, SpaceType::L2, false, cfg_arch_b);
            idx.reserve(sift_n);
            idx.preassigned_node_params = efc_only_params;
            for (size_t i = 0; i < sift_n; ++i) idx.insert(&sift_data[i * sift_dim]);

            auto [rec, qps, dist, pq_rec] = eval_index(idx);
            size_t edges = idx.get_total_edges();

            size_t overlap = 0;
            for (size_t u = 0; u < sift_n; ++u) {
                if (u < idx.graphs[0].size()) {
                    for (tableint v : idx.graphs[0][u]) {
                        if (arch_b_edges.count(((uint64_t)u << 32) | (uint64_t)v)) overlap++;
                    }
                }
            }
            double jaccard = (double)overlap / (double)(arch_b_edges.size() + idx.get_layer0_edges() - overlap);

            double s_in = 0, s_in2 = 0;
            for (size_t i = 0; i < sift_n; ++i) {
                double d = (i < idx.in_degrees.size()) ? (double)idx.in_degrees[i] : 0.0;
                s_in += d; s_in2 += d * d;
            }
            double mean_in = s_in / (double)sift_n;
            double in_var = (s_in2 / (double)sift_n) - (mean_in * mean_in);

            trials.push_back({"efC_only", seed, 0.0, edges, in_var, rec, qps, dist, jaccard});
            std::cout << "  [efC-only, Seed " << seed << "] Rec=" << rec * 100.0 << "%, QPS=" << qps
                      << ", Dist=" << dist << ", Jaccard=" << jaccard << std::endl;
        }

        // C. Joint (M + efC) permutation
        {
            std::vector<NodeParams> joint_params = original_params;
            std::mt19937 rng(seed);
            std::shuffle(joint_params.begin(), joint_params.end(), rng);

            AdaptiveHNSWIndex idx(sift_dim, SpaceType::L2, false, cfg_arch_b);
            idx.reserve(sift_n);
            idx.preassigned_node_params = joint_params;
            for (size_t i = 0; i < sift_n; ++i) idx.insert(&sift_data[i * sift_dim]);

            auto [rec, qps, dist, pq_rec] = eval_index(idx);
            size_t edges = idx.get_total_edges();

            std::vector<float> p_m(sift_n);
            for (size_t i = 0; i < sift_n; ++i) p_m[i] = (float)joint_params[i].m;
            double corr = compute_correlation(orig_scores, p_m);

            size_t overlap = 0;
            for (size_t u = 0; u < sift_n; ++u) {
                if (u < idx.graphs[0].size()) {
                    for (tableint v : idx.graphs[0][u]) {
                        if (arch_b_edges.count(((uint64_t)u << 32) | (uint64_t)v)) overlap++;
                    }
                }
            }
            double jaccard = (double)overlap / (double)(arch_b_edges.size() + idx.get_layer0_edges() - overlap);

            double s_in = 0, s_in2 = 0;
            for (size_t i = 0; i < sift_n; ++i) {
                double d = (i < idx.in_degrees.size()) ? (double)idx.in_degrees[i] : 0.0;
                s_in += d; s_in2 += d * d;
            }
            double mean_in = s_in / (double)sift_n;
            double in_var = (s_in2 / (double)sift_n) - (mean_in * mean_in);

            trials.push_back({"joint", seed, corr, edges, in_var, rec, qps, dist, jaccard});
            std::cout << "  [Joint, Seed " << seed << "] Rec=" << rec * 100.0 << "%, QPS=" << qps
                      << ", Dist=" << dist << ", Corr=" << corr << ", Jaccard=" << jaccard << std::endl;

            if (seed == 2024) {
                seed_2024_per_query_rec = pq_rec;
                seed_2024_params = joint_params;
            }
        }
    }

    // 4. Seed 2024 Boundary Deprivation Investigation
    std::cout << "\n>>> Investigating Seed 2024 Boundary Deprivation Anomaly..." << std::endl;
    // Stratify queries into dense-cluster queries vs sparse-boundary queries
    // A query is boundary if its 1-NN base vector has high score S(x) (high sparsity / high LID)
    std::vector<double> arch_b_boundary_rec, arch_b_dense_rec;
    std::vector<double> seed_2024_boundary_rec, seed_2024_dense_rec;

    for (size_t q = 0; q < sift_nq; ++q) {
        if (sift_gt[q].empty()) continue;
        uint32_t nn_id = sift_gt[q][0];
        if (nn_id >= original_params.size()) continue;

        float s_val = original_params[nn_id].score;
        bool is_boundary = (s_val > 0.0f); // above mean score

        if (is_boundary) {
            arch_b_boundary_rec.push_back(arch_b_per_query_rec[q]);
            if (!seed_2024_per_query_rec.empty()) seed_2024_boundary_rec.push_back(seed_2024_per_query_rec[q]);
        } else {
            arch_b_dense_rec.push_back(arch_b_per_query_rec[q]);
            if (!seed_2024_per_query_rec.empty()) seed_2024_dense_rec.push_back(seed_2024_per_query_rec[q]);
        }
    }

    double arch_b_mean_bound = std::accumulate(arch_b_boundary_rec.begin(), arch_b_boundary_rec.end(), 0.0) / std::max((size_t)1, arch_b_boundary_rec.size());
    double arch_b_mean_dense = std::accumulate(arch_b_dense_rec.begin(), arch_b_dense_rec.end(), 0.0) / std::max((size_t)1, arch_b_dense_rec.size());
    double s2024_mean_bound = std::accumulate(seed_2024_boundary_rec.begin(), seed_2024_boundary_rec.end(), 0.0) / std::max((size_t)1, seed_2024_boundary_rec.size());
    double s2024_mean_dense = std::accumulate(seed_2024_dense_rec.begin(), seed_2024_dense_rec.end(), 0.0) / std::max((size_t)1, seed_2024_dense_rec.size());

    std::cout << "  Arch B Recall: Dense=" << arch_b_mean_dense * 100.0 << "%, Boundary=" << arch_b_mean_bound * 100.0 << "%" << std::endl;
    std::cout << "  Seed 2024 Recall: Dense=" << s2024_mean_dense * 100.0 << "%, Boundary=" << s2024_mean_bound * 100.0 << "%" << std::endl;
    std::cout << "  Delta on Boundary Queries (Arch B - Seed 2024): +" << (arch_b_mean_bound - s2024_mean_bound) * 100.0 << "%" << std::endl;

    // 5. Synthetic Heterogeneous Clustered Mixture Experiment
    std::cout << "\n>>> Generating Synthetic Heterogeneous Clustered Dataset (5,000 vectors, 16D)..." << std::endl;
    const size_t syn_n = 5000;
    const size_t syn_dim = 16;
    const size_t syn_nq = 500;
    std::vector<float> syn_data(syn_n * syn_dim);
    std::vector<float> syn_queries(syn_nq * syn_dim);
    std::vector<uint8_t> syn_is_boundary(syn_n, 0);
    std::vector<uint8_t> syn_q_is_boundary(syn_nq, 0);

    // 4,000 vectors in 4 tight dense Gaussian clusters (std = 0.05)
    // 1,000 vectors in a diffuse sparse boundary/halo (std = 0.40)
    std::mt19937 syn_rng(42);
    std::normal_distribution<float> norm_cluster(0.0f, 0.05f);
    std::normal_distribution<float> norm_halo(0.0f, 0.40f);

    std::vector<std::vector<float>> cluster_centers = {
        {-1.0f, -1.0f, -1.0f, -1.0f,  0.0f, 0.0f, 0.0f, 0.0f,  1.0f, 1.0f, 1.0f, 1.0f,  0.0f, 0.0f, 0.0f, 0.0f},
        { 1.0f,  1.0f, -1.0f, -1.0f,  0.0f, 0.0f, 0.0f, 0.0f, -1.0f,-1.0f, 1.0f, 1.0f,  0.0f, 0.0f, 0.0f, 0.0f},
        { 0.0f,  0.0f,  1.0f,  1.0f,  1.0f, 1.0f,-1.0f,-1.0f,  0.0f, 0.0f, 0.0f, 0.0f,  1.0f, 1.0f,-1.0f,-1.0f},
        {-0.5f,  0.5f, -0.5f,  0.5f, -0.5f, 0.5f,-0.5f, 0.5f, -0.5f, 0.5f,-0.5f, 0.5f, -0.5f, 0.5f,-0.5f, 0.5f}
    };

    for (size_t i = 0; i < syn_n; ++i) {
        if (i < 4000) {
            int c = i % 4;
            for (size_t d = 0; d < syn_dim; ++d) {
                syn_data[i * syn_dim + d] = cluster_centers[c][d] + norm_cluster(syn_rng);
            }
            syn_is_boundary[i] = 0;
        } else {
            // Halo / boundary node
            for (size_t d = 0; d < syn_dim; ++d) {
                syn_data[i * syn_dim + d] = norm_halo(syn_rng);
            }
            syn_is_boundary[i] = 1;
        }
    }

    for (size_t q = 0; q < syn_nq; ++q) {
        if (q < 350) {
            int c = q % 4;
            for (size_t d = 0; d < syn_dim; ++d) {
                syn_queries[q * syn_dim + d] = cluster_centers[c][d] + norm_cluster(syn_rng);
            }
            syn_q_is_boundary[q] = 0;
        } else {
            for (size_t d = 0; d < syn_dim; ++d) {
                syn_queries[q * syn_dim + d] = norm_halo(syn_rng);
            }
            syn_q_is_boundary[q] = 1;
        }
    }

    // Compute exact ground truth for synthetic dataset
    std::vector<std::vector<uint32_t>> syn_gt(syn_nq, std::vector<uint32_t>(10));
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t q = 0; q < syn_nq; ++q) {
        std::vector<std::pair<float, uint32_t>> dists(syn_n);
        for (size_t i = 0; i < syn_n; ++i) {
            float d2 = 0.0f;
            for (size_t d = 0; d < syn_dim; ++d) {
                float diff = syn_queries[q * syn_dim + d] - syn_data[i * syn_dim + d];
                d2 += diff * diff;
            }
            dists[i] = {d2, (uint32_t)i};
        }
        std::partial_sort(dists.begin(), dists.begin() + 10, dists.end());
        for (size_t k = 0; k < 10; ++k) syn_gt[q][k] = dists[k].second;
    }

    auto eval_syn = [&](const AdaptiveHNSWIndex& idx, const std::string& name) {
        SyntheticBoundaryResult res;
        res.config_name = name;
        res.total_edges = idx.get_total_edges();

        std::vector<double> overall_rec(syn_nq), dense_rec, bound_rec;
        uint64_t total_evals = 0;

        for (size_t q = 0; q < syn_nq; ++q) {
            QuerySearchStats stats;
            auto top = idx.search(&syn_queries[q * syn_dim], 10, 32, stats);
            total_evals += stats.total_dist_evals();

            std::unordered_set<uint32_t> gt(syn_gt[q].begin(), syn_gt[q].end());
            size_t m = 0;
            for (const auto& c : top) if (gt.count(c.id)) m++;
            double r = (double)m / 10.0;
            overall_rec[q] = r;
            if (syn_q_is_boundary[q]) bound_rec.push_back(r);
            else dense_rec.push_back(r);
        }
        res.overall_recall_at_10 = std::accumulate(overall_rec.begin(), overall_rec.end(), 0.0) / (double)syn_nq;
        res.dense_cluster_recall = std::accumulate(dense_rec.begin(), dense_rec.end(), 0.0) / (double)dense_rec.size();
        res.sparse_boundary_recall = std::accumulate(bound_rec.begin(), bound_rec.end(), 0.0) / (double)bound_rec.size();
        res.mean_dist_evals = (double)total_evals / (double)syn_nq;

        std::cout << "  [" << name << "] Overall=" << res.overall_recall_at_10 * 100.0 << "%"
                  << ", Dense=" << res.dense_cluster_recall * 100.0 << "%"
                  << ", Boundary=" << res.sparse_boundary_recall * 100.0 << "%"
                  << ", Edges=" << res.total_edges << std::endl;
        return res;
    };

    std::vector<SyntheticBoundaryResult> syn_results;

    // Synthetic Test 1: Canonical HNSW (M=16, efC=100)
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.ef_c_base = 100;
        cfg.enable_layer_scaling = false;
        AdaptiveHNSWIndex idx(syn_dim, SpaceType::L2, false, cfg);
        for (size_t i = 0; i < syn_n; ++i) idx.insert(&syn_data[i * syn_dim]);
        syn_results.push_back(eval_syn(idx, "Canonical HNSW"));
    }

    // Synthetic Test 2: Architecture B (Geometry-Aware)
    std::vector<NodeParams> syn_arch_b_params;
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.ef_c_base = 100;
        cfg.m_min = 8;
        cfg.m_max = 24;
        cfg.ef_c_min = 40;
        cfg.ef_c_max = 200;
        cfg.alpha_lid = 0.5f;
        cfg.beta_density = 0.5f;
        cfg.enable_layer_scaling = true;
        cfg.lambda_layer = 0.75f;
        cfg.m_min_layer = 4;
        cfg.use_streaming_welford = true;
        AdaptiveHNSWIndex idx(syn_dim, SpaceType::L2, true, cfg);
        for (size_t i = 0; i < syn_n; ++i) idx.insert(&syn_data[i * syn_dim]);
        syn_arch_b_params = idx.node_params;
        syn_results.push_back(eval_syn(idx, "Architecture B (Adaptive)"));
    }

    // Synthetic Test 3: Permuted Controls (M-only permuted, efC-only permuted, joint permuted)
    {
        std::vector<NodeParams> p_params = syn_arch_b_params;
        std::mt19937 rng(2024);
        std::shuffle(p_params.begin(), p_params.end(), rng);
        PolicyConfig cfg;
        AdaptiveHNSWIndex idx(syn_dim, SpaceType::L2, false, cfg);
        idx.preassigned_node_params = p_params;
        for (size_t i = 0; i < syn_n; ++i) idx.insert(&syn_data[i * syn_dim]);
        syn_results.push_back(eval_syn(idx, "Randomized Permuted (Seed 2024)"));
    }

    // Export Complete Results JSON
    std::string out_path = "experiments/phase_5_2_randomization_isolation.json";
    std::ofstream out(out_path);
    out << "{\n";
    out << "  \"experiment_name\": \"Phase 5.2 Randomization Isolation and Multi-Dataset Validation\",\n";
    out << "  \"timestamp\": \"2026-11-20T13:10:00Z\",\n";
    out << "  \"sift100k_reference\": {\n";
    out << "    \"arch_b\": {\n";
    out << "      \"recall_at_10\": " << arch_b_rec << ",\n";
    out << "      \"qps\": " << arch_b_qps << ",\n";
    out << "      \"mean_dist_evals\": " << arch_b_dist << ",\n";
    out << "      \"total_edges\": " << arch_b_total_edges << ",\n";
    out << "      \"correlation_score_vs_m\": " << arch_b_corr << ",\n";
    out << "      \"boundary_queries_recall\": " << arch_b_mean_bound << ",\n";
    out << "      \"dense_queries_recall\": " << arch_b_mean_dense << "\n";
    out << "    },\n";
    out << "    \"canonical_hnsw\": {\n";
    out << "      \"recall_at_10\": " << canon_rec << ",\n";
    out << "      \"qps\": " << canon_qps << ",\n";
    out << "      \"mean_dist_evals\": " << canon_dist << ",\n";
    out << "      \"total_edges\": " << canonical_index.get_total_edges() << "\n";
    out << "    },\n";
    out << "    \"seed_2024_boundary_deprivation\": {\n";
    out << "      \"boundary_queries_recall\": " << s2024_mean_bound << ",\n";
    out << "      \"dense_queries_recall\": " << s2024_mean_dense << ",\n";
    out << "      \"boundary_recall_delta_pct\": " << (arch_b_mean_bound - s2024_mean_bound) * 100.0 << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"permutation_trials\": [\n";
    for (size_t i = 0; i < trials.size(); ++i) {
        const auto& t = trials[i];
        out << "    {\n";
        out << "      \"mode\": \"" << t.mode << "\",\n";
        out << "      \"seed\": " << t.seed << ",\n";
        out << "      \"score_m_correlation\": " << t.score_m_correlation << ",\n";
        out << "      \"total_edges\": " << t.total_edges << ",\n";
        out << "      \"in_degree_var\": " << t.in_degree_var << ",\n";
        out << "      \"recall_at_10\": " << t.recall_at_10 << ",\n";
        out << "      \"qps\": " << t.qps << ",\n";
        out << "      \"mean_dist_evals\": " << t.mean_dist_evals << ",\n";
        out << "      \"jaccard_edge_overlap\": " << t.jaccard_edge_overlap << "\n";
        out << "    }" << (i + 1 < trials.size() ? "," : "") << "\n";
    }
    out << "  ],\n";
    out << "  \"synthetic_heterogeneous_experiment\": [\n";
    for (size_t i = 0; i < syn_results.size(); ++i) {
        const auto& sr = syn_results[i];
        out << "    {\n";
        out << "      \"config_name\": \"" << sr.config_name << "\",\n";
        out << "      \"overall_recall_at_10\": " << sr.overall_recall_at_10 << ",\n";
        out << "      \"dense_cluster_recall\": " << sr.dense_cluster_recall << ",\n";
        out << "      \"sparse_boundary_recall\": " << sr.sparse_boundary_recall << ",\n";
        out << "      \"mean_dist_evals\": " << sr.mean_dist_evals << ",\n";
        out << "      \"total_edges\": " << sr.total_edges << "\n";
        out << "    }" << (i + 1 < syn_results.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    out.close();

    std::cout << "\n>>> Phase 5.2 Randomization Isolation JSON written to " << out_path << std::endl;
    return 0;
}
