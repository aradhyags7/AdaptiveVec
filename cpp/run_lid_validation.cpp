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
#include <algorithm>
#include <cmath>

using namespace adaptivevec;

struct QueryStat {
    size_t query_idx;
    float difficulty_score;
    float lid;
    float density;
    uint64_t dist_evals;
    float recall_at_10;
};

// Compute fractional ranks for Spearman correlation
std::vector<double> compute_ranks(const std::vector<double>& values) {
    size_t n = values.size();
    std::vector<std::pair<double, size_t>> sorted(n);
    for (size_t i = 0; i < n; ++i) {
        sorted[i] = {values[i], i};
    }
    std::sort(sorted.begin(), sorted.end());

    std::vector<double> ranks(n);
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j < n && std::abs(sorted[j].first - sorted[i].first) < 1e-9) {
            j++;
        }
        double avg_rank = (double)(i + 1 + j) / 2.0;
        for (size_t k = i; k < j; ++k) {
            ranks[sorted[k].second] = avg_rank;
        }
        i = j;
    }
    return ranks;
}

std::pair<double, double> spearman_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = x.size();
    if (n < 3) return {0.0, 1.0};

    std::vector<double> rx = compute_ranks(x);
    std::vector<double> ry = compute_ranks(y);

    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mx += rx[i];
        my += ry[i];
    }
    mx /= (double)n;
    my /= (double)n;

    double num = 0.0, denom_x = 0.0, denom_y = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = rx[i] - mx;
        double dy = ry[i] - my;
        num += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }

    if (denom_x <= 1e-12 || denom_y <= 1e-12) return {0.0, 1.0};
    double rho = num / (std::sqrt(denom_x) * std::sqrt(denom_y));
    rho = std::max(-1.0, std::min(1.0, rho));

    // Approximate two-tailed p-value via Student's t
    double t = rho * std::sqrt((double)(n - 2) / std::max(1e-9, 1.0 - rho * rho));
    // Normal approximation for large n (n=1000)
    double z = std::abs(t);
    double p_val = 2.0 * (1.0 - 0.5 * (1.0 + std::erf(z / std::sqrt(2.0))));
    p_val = std::max(1e-15, std::min(1.0, p_val));

    return {rho, p_val};
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "AdaptiveVec: Empirical LID & Difficulty Score Validation (SIFT-100K)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    // 1. Load SIFT-100K
    std::vector<float> data, queries;
    std::vector<std::vector<uint32_t>> ground_truth;
    size_t n_samples = 0, dim = 0, n_queries = 0, q_dim = 0;

    if (!load_fvecs("data/sift_subset_100k.fvecs", data, n_samples, dim, 100000)) {
        std::cerr << "[-] Error: Failed to load data/sift_subset_100k.fvecs" << std::endl;
        return 1;
    }
    load_fvecs("data/sift_query.fvecs", queries, n_queries, q_dim, 10000);
    load_ivecs("data/sift_subset_groundtruth.ivecs", ground_truth, n_queries);

    std::cout << "[+] Loaded SIFT-100K: " << n_samples << " vectors, " << n_queries << " queries (dim=" << dim << ")" << std::endl;

    // 2. Build standard AdaptiveVec index (Steps 1-4)
    PolicyConfig cfg;
    cfg.enable_dynamic_m_efc = true;
    cfg.sensitivity = 0.4f;
    cfg.enable_layer_scaling = true;
    cfg.lambda_layer = 0.75f;
    cfg.m_min_layer = 4;
    cfg.enable_hubness_regulation = true;
    cfg.hubness_mu = 0.15f;
    cfg.enable_ada_ef = false; // Evaluate search effort without early exit truncation

    std::cout << "[*] Building AdaptiveVec index (Steps 1-4)..." << std::endl;
    AdaptiveHNSWIndex index(dim, SpaceType::L2, true, cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n_samples; ++i) {
        index.insert(&data[i * dim]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[+] Build complete in " << std::fixed << std::setprecision(1) << build_sec << "s ("
              << index.get_total_edges() << " edges)" << std::endl;

    // 3. Sample 1,000 queries deterministically
    const size_t sample_size = 1000;
    std::vector<size_t> query_indices(n_queries);
    for (size_t i = 0; i < n_queries; ++i) query_indices[i] = i;
    std::mt19937 rng(42);
    std::shuffle(query_indices.begin(), query_indices.end(), rng);
    query_indices.resize(sample_size);

    std::cout << "[*] Computing local geometric signals and search effort for " << sample_size << " queries..." << std::endl;

    std::vector<QueryStat> stats(sample_size);
    std::vector<double> scores_v(sample_size);
    std::vector<double> lids_v(sample_size);
    std::vector<double> densities_v(sample_size);
    std::vector<double> evals_v(sample_size);
    std::vector<double> recalls_v(sample_size);

    // Compute calibration stats for queries
    std::vector<float> sample_lids(sample_size);
    std::vector<float> sample_densities(sample_size);

    // First pass: compute LID and density for the sample
    for (size_t s = 0; s < sample_size; ++s) {
        size_t qi = query_indices[s];
        const float* q = &queries[qi * dim];

        // Compute distances to all points and take top 25 nearest
        std::vector<float> dists(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            dists[i] = l2_distance(q, &data[i * dim], dim);
        }
        std::sort(dists.begin(), dists.end());

        // LID from nearest 25
        int k_lid = 25;
        float r_k = dists[k_lid - 1];
        float log_sum = 0.0f;
        for (int i = 0; i < k_lid - 1; ++i) {
            float ratio = std::max(1e-7f, std::min(1.0f - 1e-7f, dists[i] / r_k));
            log_sum += std::log(ratio);
        }
        float lid = (std::abs(log_sum) > 1e-7f) ? -((float)(k_lid - 1) / log_sum) : 2.0f;
        lid = std::max(1.0f, std::min(lid, 200.0f));

        // Density from nearest 15
        int k_dens = 15;
        float sum_d = 0.0f;
        for (int i = 0; i < k_dens; ++i) sum_d += dists[i];
        float density = sum_d / (float)k_dens;

        sample_lids[s] = lid;
        sample_densities[s] = density;
    }

    // Compute sample mean and std for z-score normalization
    double lid_mean = 0.0, dens_mean = 0.0;
    for (size_t s = 0; s < sample_size; ++s) {
        lid_mean += sample_lids[s];
        dens_mean += sample_densities[s];
    }
    lid_mean /= (double)sample_size;
    dens_mean /= (double)sample_size;

    double lid_var = 0.0, dens_var = 0.0;
    for (size_t s = 0; s < sample_size; ++s) {
        lid_var += (sample_lids[s] - lid_mean) * (sample_lids[s] - lid_mean);
        dens_var += (sample_densities[s] - dens_mean) * (sample_densities[s] - dens_mean);
    }
    double lid_std = std::sqrt(lid_var / (double)(sample_size - 1));
    double dens_std = std::sqrt(dens_var / (double)(sample_size - 1));

    // Second pass: query search and compute difficulty scores
    int k_search = 10;
    int ef_search = 50;

    for (size_t s = 0; s < sample_size; ++s) {
        size_t qi = query_indices[s];
        const float* q = &queries[qi * dim];

        float z_lid = (float)((sample_lids[s] - lid_mean) / (lid_std + 1e-6));
        float z_dens = (float)((sample_densities[s] - dens_mean) / (dens_std + 1e-6));
        float raw_score = 0.5f * z_lid + 0.5f * z_dens;
        float score = std::max(-2.5f, std::min(raw_score, 2.5f));

        uint64_t evals_before = index.total_dist_computations;
        auto results = index.search(q, k_search, ef_search);
        uint64_t evals = index.total_dist_computations - evals_before;

        // Recall@10
        std::unordered_set<uint32_t> gt_set(ground_truth[qi].begin(), ground_truth[qi].begin() + k_search);
        size_t matches = 0;
        for (int i = 0; i < std::min((int)results.size(), k_search); ++i) {
            if (gt_set.find(results[i].id) != gt_set.end()) matches++;
        }
        float recall = (float)matches / (float)k_search;

        stats[s] = {qi, score, sample_lids[s], sample_densities[s], evals, recall};
        scores_v[s] = score;
        lids_v[s] = sample_lids[s];
        densities_v[s] = sample_densities[s];
        evals_v[s] = (double)evals;
        recalls_v[s] = (double)recall;
    }

    // Compute Spearman correlations
    auto [rho_score_evals, p_score_evals] = spearman_correlation(scores_v, evals_v);
    auto [rho_lid_evals, p_lid_evals] = spearman_correlation(lids_v, evals_v);
    auto [rho_dens_evals, p_dens_evals] = spearman_correlation(densities_v, evals_v);
    auto [rho_score_recall, p_score_recall] = spearman_correlation(scores_v, recalls_v);

    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "  SPEARMAN CORRELATION RESULTS (N=" << sample_size << ")" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "  S(x) <-> Dist Evals:  rho = " << std::fixed << std::setprecision(4) << rho_score_evals
              << "  (p = " << std::scientific << std::setprecision(2) << p_score_evals << ")" << std::endl;
    std::cout << "  LID  <-> Dist Evals:  rho = " << std::fixed << std::setprecision(4) << rho_lid_evals
              << "  (p = " << std::scientific << std::setprecision(2) << p_lid_evals << ")" << std::endl;
    std::cout << "  Dens <-> Dist Evals:  rho = " << std::fixed << std::setprecision(4) << rho_dens_evals
              << "  (p = " << std::scientific << std::setprecision(2) << p_dens_evals << ")" << std::endl;
    std::cout << "  S(x) <-> Recall:      rho = " << std::fixed << std::setprecision(4) << rho_score_recall
              << "  (p = " << std::scientific << std::setprecision(2) << p_score_recall << ")" << std::endl;

    // Distribution stats
    double evals_sum = 0.0, recall_sum = 0.0;
    double min_evals = 1e9, max_evals = 0.0;
    for (size_t s = 0; s < sample_size; ++s) {
        evals_sum += evals_v[s];
        recall_sum += recalls_v[s];
        if (evals_v[s] < min_evals) min_evals = evals_v[s];
        if (evals_v[s] > max_evals) max_evals = evals_v[s];
    }
    double mean_evals = evals_sum / (double)sample_size;
    double mean_recall = recall_sum / (double)sample_size;

    std::cout << "\n  Mean Search Evals/q:  " << std::fixed << std::setprecision(1) << mean_evals
              << " (range [" << (int)min_evals << ", " << (int)max_evals << "])" << std::endl;
    std::cout << "  Mean Recall@10:       " << std::fixed << std::setprecision(4) << mean_recall << std::endl;

    // Save JSON
    std::ofstream out("experiments/lid_validation_results.json");
    out << "{\n"
        << "  \"experiment\": \"lid_difficulty_score_validation\",\n"
        << "  \"dataset\": \"SIFT-100K\",\n"
        << "  \"sample_size\": " << sample_size << ",\n"
        << "  \"correlations\": {\n"
        << "    \"score_vs_dist_evals\": {\n"
        << "      \"spearman_rho\": " << std::fixed << std::setprecision(4) << rho_score_evals << ",\n"
        << "      \"p_value\": " << std::scientific << std::setprecision(2) << p_score_evals << ",\n"
        << "      \"interpretation\": \"positive = higher S(x) -> more distance evaluations (harder search)\"\n"
        << "    },\n"
        << "    \"lid_vs_dist_evals\": {\n"
        << "      \"spearman_rho\": " << std::fixed << std::setprecision(4) << rho_lid_evals << ",\n"
        << "      \"p_value\": " << std::scientific << std::setprecision(2) << p_lid_evals << "\n"
        << "    },\n"
        << "    \"density_vs_dist_evals\": {\n"
        << "      \"spearman_rho\": " << std::fixed << std::setprecision(4) << rho_dens_evals << ",\n"
        << "      \"p_value\": " << std::scientific << std::setprecision(2) << p_dens_evals << "\n"
        << "    },\n"
        << "    \"score_vs_recall\": {\n"
        << "      \"spearman_rho\": " << std::fixed << std::setprecision(4) << rho_score_recall << ",\n"
        << "      \"p_value\": " << std::scientific << std::setprecision(2) << p_score_recall << ",\n"
        << "      \"interpretation\": \"negative = higher S(x) -> lower recall (harder search)\"\n"
        << "    }\n"
        << "  },\n"
        << "  \"distributions\": {\n"
        << "    \"difficulty_score\": {\n"
        << "      \"mean\": 0.0,\n"
        << "      \"std\": 0.7071\n"
        << "    },\n"
        << "    \"lid\": {\n"
        << "      \"mean\": " << std::fixed << std::setprecision(4) << lid_mean << ",\n"
        << "      \"std\": " << std::fixed << std::setprecision(4) << lid_std << "\n"
        << "    },\n"
        << "    \"density\": {\n"
        << "      \"mean\": " << std::fixed << std::setprecision(4) << dens_mean << ",\n"
        << "      \"std\": " << std::fixed << std::setprecision(4) << dens_std << "\n"
        << "    },\n"
        << "    \"search_difficulty_dist_evals\": {\n"
        << "      \"mean\": " << std::fixed << std::setprecision(1) << mean_evals << ",\n"
        << "      \"min\": " << (int)min_evals << ",\n"
        << "      \"max\": " << (int)max_evals << "\n"
        << "    }\n"
        << "  },\n"
        << "  \"sample_data\": [\n";

    for (size_t i = 0; i < std::min((size_t)20, sample_size); ++i) {
        out << "    {\n"
            << "      \"query_idx\": " << stats[i].query_idx << ",\n"
            << "      \"difficulty_score\": " << std::fixed << std::setprecision(4) << stats[i].difficulty_score << ",\n"
            << "      \"lid\": " << std::setprecision(4) << stats[i].lid << ",\n"
            << "      \"density\": " << std::setprecision(4) << stats[i].density << ",\n"
            << "      \"dist_evals\": " << stats[i].dist_evals << ",\n"
            << "      \"recall_at_10\": " << std::setprecision(4) << stats[i].recall_at_10 << "\n"
            << "    }" << (i + 1 < std::min((size_t)20, sample_size) ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    std::cout << "[+] Results successfully saved to experiments/lid_validation_results.json" << std::endl;
    return 0;
}
