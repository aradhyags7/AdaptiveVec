#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>

using namespace adaptivevec;

std::vector<float> load_fvecs(const std::string& path, size_t max_vectors, size_t& dim, size_t& count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    int d = 0;
    in.read((char*)&d, sizeof(int));
    dim = d;
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t record_size = sizeof(int) + dim * sizeof(float);
    size_t total_vectors = file_size / record_size;
    count = std::min(total_vectors, max_vectors);

    std::vector<float> data(count * dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < count; ++i) {
        int cur_d = 0;
        in.read((char*)&cur_d, sizeof(int));
        in.read((char*)&data[i * dim], dim * sizeof(float));
    }
    return data;
}

double pearson_corr(const std::vector<float>& x, const std::vector<float>& y) {
    size_t n = x.size();
    if (n < 2) return 0.0;
    double mx = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double my = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double num = 0.0, dx2 = 0.0, dy2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mx;
        double dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    double den = std::sqrt(dx2 * dy2);
    return (den > 1e-9) ? (num / den) : 0.0;
}

std::vector<float> compute_ranks(const std::vector<float>& vals) {
    size_t n = vals.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&vals](size_t a, size_t b) {
        return vals[a] < vals[b];
    });
    std::vector<float> ranks(n);
    for (size_t r = 0; r < n; ++r) {
        ranks[idx[r]] = static_cast<float>(r + 1);
    }
    return ranks;
}

double spearman_corr(const std::vector<float>& x, const std::vector<float>& y) {
    auto rx = compute_ranks(x);
    auto ry = compute_ranks(y);
    return pearson_corr(rx, ry);
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << " AdaptiveVec Phase 2: Stage 3 Quantization Sensitivity Audit\n";
    std::cout << "=================================================================\n\n";

    size_t dim = 0, n_base = 0;
    std::vector<float> base_data = load_fvecs("data/sift_subset_100k.fvecs", 20000, dim, n_base);
    if (base_data.empty()) {
        std::cerr << "[-] Failed to load dataset.\n";
        return 1;
    }
    std::cout << "[+] Loaded SIFT sample: N=" << n_base << ", Dim=" << dim << "\n";

    // Build Canonical HNSW Index to extract clean reference topology
    PolicyConfig cfg;
    cfg.m_base = 16;
    cfg.ef_c_base = 120;
    cfg.enable_dynamic_m_efc = false;
    cfg.enable_layer_scaling = false;
    cfg.enable_hubness_regulation = false;
    cfg.enable_ada_ef = false;
    cfg.enable_bridge_aware = false;

    AdaptiveHNSWIndex index(dim, SpaceType::L2, false, cfg);
    std::cout << "[*] Building reference graph...\n";
    index.batch_insert(base_data.data(), n_base, 12);

    // Train SQ8 and SQ4 quantizers
    ScalarQuantizer quantizer_sq8(dim);
    ScalarQuantizer4Bit quantizer_sq4(dim);
    quantizer_sq8.train(base_data.data(), n_base);
    quantizer_sq4.train(base_data.data(), n_base);

    // Encode all vectors in sample
    std::vector<uint8_t> codes_sq8(n_base * dim);
    size_t packed_dim = (dim + 1) / 2;
    std::vector<uint8_t> codes_sq4(n_base * packed_dim);
    for (size_t i = 0; i < n_base; ++i) {
        quantizer_sq8.encode(&base_data[i * dim], &codes_sq8[i * dim]);
        quantizer_sq4.encode(&base_data[i * dim], &codes_sq4[i * packed_dim]);
    }

    std::vector<float> lids(n_base);
    std::vector<float> densities(n_base);
    std::vector<float> recon_err_sq8(n_base);
    std::vector<float> recon_err_sq4(n_base);
    std::vector<float> dist_margins(n_base);
    std::vector<float> in_degs(n_base);
    std::vector<float> rank_distort_sq8(n_base);
    std::vector<float> rank_distort_sq4(n_base);

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < n_base; ++i) {
        const float* u_vec = &base_data[i * dim];

        // 1. Normalized reconstruction error
        float norm_u_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) norm_u_sq += u_vec[d] * u_vec[d];
        float norm_u = std::max(1e-7f, std::sqrt(norm_u_sq));

        std::vector<float> dec_sq8(dim);
        quantizer_sq8.decode(&codes_sq8[i * dim], dec_sq8.data());
        float err_sq8 = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float diff = u_vec[d] - dec_sq8[d];
            err_sq8 += diff * diff;
        }
        recon_err_sq8[i] = std::sqrt(err_sq8) / norm_u;

        std::vector<float> dec_sq4(dim);
        quantizer_sq4.decode(&codes_sq4[i * packed_dim], dec_sq4.data());
        float err_sq4 = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float diff = u_vec[d] - dec_sq4[d];
            err_sq4 += diff * diff;
        }
        recon_err_sq4[i] = std::sqrt(err_sq4) / norm_u;

        // Extract layer-0 neighbors
        std::vector<tableint> neighbors;
        if (!index.graphs.empty() && i < index.graphs[0].size()) {
            neighbors = index.graphs[0][i];
        }
        in_degs[i] = (i < index.in_degrees.size()) ? static_cast<float>(index.in_degrees[i]) : 0.0f;

        size_t k = neighbors.size();
        if (k < 3) {
            lids[i] = 4.0f;
            densities[i] = 1.0f;
            dist_margins[i] = 0.5f;
            rank_distort_sq8[i] = 0.0f;
            rank_distort_sq4[i] = 0.0f;
            continue;
        }

        // True distances and quantized distances to neighbors
        std::vector<float> d_true(k);
        std::vector<float> d_sq8(k);
        std::vector<float> d_sq4(k);

        for (size_t j = 0; j < k; ++j) {
            tableint nb = neighbors[j];
            d_true[j] = l2_sq_distance(u_vec, &base_data[nb * dim], dim);
            d_sq8[j] = quantizer_sq8.asymmetric_l2_sq_distance(u_vec, &codes_sq8[nb * dim]);
            d_sq4[j] = quantizer_sq4.asymmetric_l2_sq_distance(u_vec, &codes_sq4[nb * packed_dim]);
        }

        // LID & Density
        std::vector<dist_t> true_euclid(k);
        for (size_t j = 0; j < k; ++j) true_euclid[j] = std::sqrt(d_true[j]);
        lids[i] = estimate_mle_lid(true_euclid, std::min((int)k, 15));
        densities[i] = estimate_local_density(true_euclid);

        // Distance margin
        std::vector<float> sorted_dt = d_true;
        std::sort(sorted_dt.begin(), sorted_dt.end());
        float d_min = std::max(1e-7f, sorted_dt.front());
        float d_max = sorted_dt.back();
        dist_margins[i] = (d_max - d_min) / d_min;

        // Ranking distortion: Kendall pair inversion ratio
        size_t total_pairs = 0;
        size_t inv_sq8 = 0;
        size_t inv_sq4 = 0;

        for (size_t a = 0; a < k; ++a) {
            for (size_t b = a + 1; b < k; ++b) {
                total_pairs++;
                bool true_order = d_true[a] < d_true[b];
                bool sq8_order = d_sq8[a] < d_sq8[b];
                bool sq4_order = d_sq4[a] < d_sq4[b];

                if (true_order != sq8_order) inv_sq8++;
                if (true_order != sq4_order) inv_sq4++;
            }
        }

        rank_distort_sq8[i] = (total_pairs > 0) ? (static_cast<float>(inv_sq8) / total_pairs) : 0.0f;
        rank_distort_sq4[i] = (total_pairs > 0) ? (static_cast<float>(inv_sq4) / total_pairs) : 0.0f;
    }

    std::cout << "\n--- Signal vs Ranking Distortion Correlation Matrix ---\n";
    std::cout << std::left << std::setw(32) << "Candidate Sensitivity Signal"
              << std::setw(18) << "Pearson r (SQ4)"
              << std::setw(18) << "Spearman rho (SQ4)"
              << std::setw(18) << "Pearson r (SQ8)"
              << "Spearman rho (SQ8)\n";
    std::cout << std::string(100, '-') << "\n";

    auto print_row = [&](const std::string& name, const std::vector<float>& sig) {
        double p_sq4 = pearson_corr(sig, rank_distort_sq4);
        double s_sq4 = spearman_corr(sig, rank_distort_sq4);
        double p_sq8 = pearson_corr(sig, rank_distort_sq8);
        double s_sq8 = spearman_corr(sig, rank_distort_sq8);
        std::cout << std::left << std::setw(32) << name
                  << std::fixed << std::setprecision(4)
                  << std::setw(18) << p_sq4
                  << std::setw(18) << s_sq4
                  << std::setw(18) << p_sq8
                  << s_sq8 << "\n";
    };

    print_row("1. Local Intrinsic Dim (LID)", lids);
    print_row("2. Neighborhood Density", densities);
    print_row("3. Normalized Recon Error", recon_err_sq4);
    print_row("4. Distance Margin (Inverse)", dist_margins);
    print_row("5. Routing In-Degree (Hubness)", in_degs);

    // Print summary stats of distortion
    double mean_sq4_dist = std::accumulate(rank_distort_sq4.begin(), rank_distort_sq4.end(), 0.0) / n_base;
    double mean_sq8_dist = std::accumulate(rank_distort_sq8.begin(), rank_distort_sq8.end(), 0.0) / n_base;
    std::cout << "\n--- Mean Ranking Distortion (Pair Inversion Ratio) ---\n";
    std::cout << "    - SQ4 Mean Neighbor Inversion: " << (mean_sq4_dist * 100.0) << "%\n";
    std::cout << "    - SQ8 Mean Neighbor Inversion: " << (mean_sq8_dist * 100.0) << "%\n";

    std::cout << "\n[+] Sensitivity audit complete.\n";
    return 0;
}
