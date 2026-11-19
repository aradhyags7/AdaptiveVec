#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <iomanip>
#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <sstream>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <psapi.h>
#endif

using namespace adaptivevec;

struct MemSnap {
    double working_set_mb = 0.0;
    double peak_working_set_mb = 0.0;
    double private_mb = 0.0;
};

MemSnap get_mem_snapshot() {
    MemSnap s;
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        s.working_set_mb = pmc.WorkingSetSize / (1024.0 * 1024.0);
        s.peak_working_set_mb = pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
        s.private_mb = pmc.PrivateUsage / (1024.0 * 1024.0);
    }
#endif
    return s;
}

std::vector<float> load_fvecs(const std::string& path, size_t max_vectors, size_t& dim, size_t& count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[-] Failed to open fvecs: " << path << std::endl;
        return {};
    }
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

std::vector<std::vector<uint32_t>> load_ivecs(const std::string& path, size_t max_records) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[-] Failed to open ivecs: " << path << std::endl;
        return {};
    }
    std::vector<std::vector<uint32_t>> records;
    while (in && records.size() < max_records) {
        int k = 0;
        in.read((char*)&k, sizeof(int));
        if (!in) break;
        std::vector<uint32_t> rec(k);
        in.read((char*)rec.data(), k * sizeof(uint32_t));
        records.push_back(std::move(rec));
    }
    return records;
}

// Compute exact brute-force ground truth using parallel L2 distance
std::vector<std::vector<uint32_t>> compute_ground_truth(
    const std::vector<float>& base, size_t n_base,
    const std::vector<float>& queries, size_t n_queries,
    size_t dim, int top_k = 100
) {
    std::vector<std::vector<uint32_t>> gt(n_queries);
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t q = 0; q < n_queries; ++q) {
        const float* q_vec = &queries[q * dim];
        std::vector<std::pair<float, uint32_t>> dists(n_base);
        for (size_t i = 0; i < n_base; ++i) {
            const float* b_vec = &base[i * dim];
            float sum = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float diff = q_vec[d] - b_vec[d];
                sum += diff * diff;
            }
            dists[i] = {sum, (uint32_t)i};
        }
        std::partial_sort(dists.begin(), dists.begin() + top_k, dists.end());
        gt[q].resize(top_k);
        for (int k = 0; k < top_k; ++k) {
            gt[q][k] = dists[k].second;
        }
    }
    return gt;
}

// Synthetic-Multi-Cluster Generator
void generate_synthetic_multicluster(
    size_t n_samples, size_t dim, size_t n_queries,
    std::vector<float>& out_base,
    std::vector<float>& out_queries,
    std::vector<std::vector<uint32_t>>& out_gt,
    uint64_t seed = 42
) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> std_norm(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    const int num_clusters = 8;
    std::vector<std::vector<float>> centroids(num_clusters, std::vector<float>(dim, 0.0f));

    // Position 8 centroids in 64D
    for (int c = 0; c < num_clusters; ++c) {
        float angle = 2.0f * 3.1415926535f * c / num_clusters;
        centroids[c][0] = 20.0f * std::cos(angle);
        centroids[c][1] = 20.0f * std::sin(angle);
        for (size_t d = 2; d < dim; ++d) {
            centroids[c][d] = std_norm(rng) * 2.0f;
        }
    }

    size_t n_bridge = n_samples / 10; // 10% bridge transit nodes
    size_t n_cluster_points = n_samples - n_bridge;
    out_base.resize(n_samples * dim);

    // 1. Cluster points (tight clusters around centroids)
    for (size_t i = 0; i < n_cluster_points; ++i) {
        int c = i % num_clusters;
        for (size_t d = 0; d < dim; ++d) {
            out_base[i * dim + d] = centroids[c][d] + std_norm(rng) * 0.4f;
        }
    }

    // 2. Bridge corridor points (connecting centroid c to (c+1)%num_clusters)
    for (size_t i = 0; i < n_bridge; ++i) {
        int c1 = i % num_clusters;
        int c2 = (c1 + 1) % num_clusters;
        float alpha = uniform(rng); // interpolation along the bridge line
        size_t idx = n_cluster_points + i;
        for (size_t d = 0; d < dim; ++d) {
            float interp = (1.0f - alpha) * centroids[c1][d] + alpha * centroids[c2][d];
            out_base[idx * dim + d] = interp + std_norm(rng) * 0.15f;
        }
    }

    // 3. Queries: half from clusters, half from bridge regions
    out_queries.resize(n_queries * dim);
    for (size_t q = 0; q < n_queries; ++q) {
        if (q % 2 == 0) {
            int c = q % num_clusters;
            for (size_t d = 0; d < dim; ++d) {
                out_queries[q * dim + d] = centroids[c][d] + std_norm(rng) * 0.5f;
            }
        } else {
            int c1 = (q / 2) % num_clusters;
            int c2 = (c1 + 1) % num_clusters;
            float alpha = uniform(rng);
            for (size_t d = 0; d < dim; ++d) {
                float interp = (1.0f - alpha) * centroids[c1][d] + alpha * centroids[c2][d];
                out_queries[q * dim + d] = interp + std_norm(rng) * 0.2f;
            }
        }
    }

    // Compute ground truth
    std::cout << "[*] Computing exact brute-force ground truth for Synthetic-Multi-Cluster..." << std::endl;
    out_gt = compute_ground_truth(out_base, n_samples, out_queries, n_queries, dim, 100);
}

struct EvalResult {
    std::string config_name;
    double build_time_s = 0.0;
    size_t total_edges = 0;
    size_t layer0_edges = 0;
    size_t upper_edges = 0;
    double steady_state_ws_mb = 0.0;
    double private_mb = 0.0;
    double peak_ws_mb = 0.0;
    double owned_capacity_mb = 0.0;
    float avg_bridge_score = 0.0f;
    float bridge_preservation_ratio = 1.0f;
    
    struct SweepPoint {
        int ef = 0;
        double recall10 = 0.0;
        double mt_qps = 0.0;
        double st_qps = 0.0;
        double lat_mean_us = 0.0;
        double lat_p50_us = 0.0;
        double lat_p95_us = 0.0;
        double lat_p99_us = 0.0;
    };
    std::vector<SweepPoint> sweep;
};

EvalResult evaluate_config(
    const std::string& name,
    bool is_adaptive,
    PolicyConfig policy,
    const std::vector<float>& base_data,
    size_t n_base,
    size_t dim,
    const std::vector<float>& query_data,
    size_t n_queries,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    const std::vector<int>& ef_list,
    int num_threads = 12
) {
    EvalResult res;
    res.config_name = name;

    std::cout << "\n=======================================================" << std::endl;
    std::cout << " Evaluating Config: " << name << std::endl;
    std::cout << "=======================================================" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    AdaptiveHNSWIndex index(dim, SpaceType::L2, is_adaptive, policy);
    index.batch_insert(base_data.data(), n_base, num_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    res.build_time_s = std::chrono::duration<double>(t1 - t0).count();

    res.total_edges = index.get_total_edges();
    res.layer0_edges = index.get_layer0_edges();
    res.upper_edges = index.get_upper_layer_edges();

    MemoryBreakdown mb = index.get_memory_breakdown();
    MemSnap snap = get_mem_snapshot();
    res.steady_state_ws_mb = snap.working_set_mb;
    res.private_mb = snap.private_mb;
    res.peak_ws_mb = snap.peak_working_set_mb;
    res.owned_capacity_mb = mb.total_owned_bytes / (1024.0 * 1024.0);
    res.avg_bridge_score = index.get_average_bridge_score();
    res.bridge_preservation_ratio = index.get_bridge_preservation_ratio(0.3f);

    std::cout << "[-] Build Time: " << res.build_time_s << " s | Edges: " << res.total_edges
              << " | AvgBridge: " << res.avg_bridge_score << " | BridgePreserve: " << res.bridge_preservation_ratio
              << " | WS: " << res.steady_state_ws_mb << " MB | Private: " << res.private_mb << " MB" << std::endl;

    for (int ef : ef_list) {
        // Multi-threaded search
        auto q_start = std::chrono::high_resolution_clock::now();
        auto batch_res = index.batch_search(query_data.data(), n_queries, 10, ef, num_threads);
        auto q_end = std::chrono::high_resolution_clock::now();
        double mt_dur = std::chrono::duration<double>(q_end - q_start).count();
        double mt_qps = n_queries / mt_dur;

        // Recall@10
        size_t total_hits = 0;
        for (size_t q = 0; q < n_queries; ++q) {
            std::set<uint32_t> gt_set(ground_truth[q].begin(), ground_truth[q].begin() + std::min((size_t)10, ground_truth[q].size()));
            for (int k = 0; k < 10 && k < (int)batch_res[q].size(); ++k) {
                if (gt_set.count(batch_res[q][k].id)) total_hits++;
            }
        }
        double recall = (double)total_hits / (n_queries * 10);

        // Single-threaded search for exact tail latencies
        std::vector<double> latencies_us(n_queries);
        for (size_t q = 0; q < n_queries; ++q) {
            auto l_start = std::chrono::high_resolution_clock::now();
            auto single_res = index.search(&query_data[q * dim], 10, ef);
            auto l_end = std::chrono::high_resolution_clock::now();
            latencies_us[q] = std::chrono::duration<double, std::micro>(l_end - l_start).count();
        }
        std::sort(latencies_us.begin(), latencies_us.end());
        double sum_lat = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
        double mean_lat = sum_lat / n_queries;
        double st_qps = 1e6 / mean_lat;
        double p50 = latencies_us[n_queries * 50 / 100];
        double p95 = latencies_us[n_queries * 95 / 100];
        double p99 = latencies_us[n_queries * 99 / 100];

        std::cout << "    ef=" << ef << " | Recall@10: " << std::fixed << std::setprecision(4) << (recall * 100.0) << "%"
                  << " | MT-QPS: " << std::setprecision(1) << mt_qps
                  << " | Lat(us) mean: " << mean_lat << " p50: " << p50 << " p95: " << p95 << " p99: " << p99 << std::endl;

        EvalResult::SweepPoint pt;
        pt.ef = ef;
        pt.recall10 = recall;
        pt.mt_qps = mt_qps;
        pt.st_qps = st_qps;
        pt.lat_mean_us = mean_lat;
        pt.lat_p50_us = p50;
        pt.lat_p95_us = p95;
        pt.lat_p99_us = p99;
        res.sweep.push_back(pt);
    }

    return res;
}

void output_json_report(const std::string& dataset_name, size_t n_base, size_t dim, size_t n_queries,
                        const std::vector<EvalResult>& results, const std::string& out_path) {
    std::ofstream out(out_path);
    out << "{\n";
    out << "  \"dataset\": \"" << dataset_name << "\",\n";
    out << "  \"base_count\": " << n_base << ",\n";
    out << "  \"dim\": " << dim << ",\n";
    out << "  \"query_count\": " << n_queries << ",\n";
    out << "  \"configurations\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"name\": \"" << r.config_name << "\",\n";
        out << "      \"build_time_s\": " << r.build_time_s << ",\n";
        out << "      \"total_edges\": " << r.total_edges << ",\n";
        out << "      \"layer0_edges\": " << r.layer0_edges << ",\n";
        out << "      \"upper_edges\": " << r.upper_edges << ",\n";
        out << "      \"steady_state_ws_mb\": " << r.steady_state_ws_mb << ",\n";
        out << "      \"private_mb\": " << r.private_mb << ",\n";
        out << "      \"peak_ws_mb\": " << r.peak_ws_mb << ",\n";
        out << "      \"owned_capacity_mb\": " << r.owned_capacity_mb << ",\n";
        out << "      \"avg_bridge_score\": " << r.avg_bridge_score << ",\n";
        out << "      \"bridge_preservation_ratio\": " << r.bridge_preservation_ratio << ",\n";
        out << "      \"sweep\": [\n";
        for (size_t j = 0; j < r.sweep.size(); ++j) {
            const auto& pt = r.sweep[j];
            out << "        {\n";
            out << "          \"ef\": " << pt.ef << ",\n";
            out << "          \"recall10\": " << pt.recall10 << ",\n";
            out << "          \"mt_qps\": " << pt.mt_qps << ",\n";
            out << "          \"st_qps\": " << pt.st_qps << ",\n";
            out << "          \"lat_mean_us\": " << pt.lat_mean_us << ",\n";
            out << "          \"lat_p50_us\": " << pt.lat_p50_us << ",\n";
            out << "          \"lat_p95_us\": " << pt.lat_p95_us << ",\n";
            out << "          \"lat_p99_us\": " << pt.lat_p99_us << "\n";
            out << "        }" << (j + 1 < r.sweep.size() ? "," : "") << "\n";
        }
        out << "      ]\n";
        out << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    out.close();
    std::cout << "[+] Saved ablation results to " << out_path << std::endl;
}

int main(int argc, char** argv) {
    std::string dataset = "synthetic";
    if (argc > 1) {
        dataset = argv[1];
    }

    std::vector<float> base_data;
    std::vector<float> query_data;
    std::vector<std::vector<uint32_t>> gt;
    size_t dim = 0, n_base = 0, n_queries = 0;

    if (dataset == "synthetic") {
        dim = 64;
        n_base = 50000;
        n_queries = 1000;
        std::cout << "[*] Generating Synthetic-Multi-Cluster (64D, 50K base, 1K queries)..." << std::endl;
        generate_synthetic_multicluster(n_base, dim, n_queries, base_data, query_data, gt, 1337);
    } else if (dataset == "sift100k") {
        std::cout << "[*] Loading SIFT-100K..." << std::endl;
        base_data = load_fvecs("data/sift_subset_100k.fvecs", 100000, dim, n_base);
        size_t q_dim = 0;
        query_data = load_fvecs("data/sift_query.fvecs", 10000, q_dim, n_queries);
        gt = load_ivecs("data/sift_subset_groundtruth.ivecs", n_queries);
        std::cout << "[-] Loaded SIFT-100K: " << n_base << " base, " << n_queries << " queries, dim: " << dim << std::endl;
    } else if (dataset == "transformer384d") {
        std::cout << "[*] Loading Transformer-384D..." << std::endl;
        base_data = load_fvecs("data/transformer_384d/transformer_384d_base.fvecs", 60000, dim, n_base);
        size_t q_dim = 0;
        query_data = load_fvecs("data/transformer_384d/transformer_384d_query.fvecs", 1000, q_dim, n_queries);
        gt = load_ivecs("data/transformer_384d/transformer_384d_groundtruth.ivecs", n_queries);
        std::cout << "[-] Loaded Transformer-384D: " << n_base << " base, " << n_queries << " queries, dim: " << dim << std::endl;
    } else {
        std::cerr << "[-] Unknown dataset: " << dataset << std::endl;
        return 1;
    }

    std::vector<int> ef_list = {16, 32, 64, 128, 256};
    std::vector<EvalResult> results;

    // Config 1: Canonical HNSW (Control)
    {
        PolicyConfig p;
        p.enable_dynamic_m_efc = false;
        p.enable_layer_scaling = false;
        p.enable_hubness_regulation = false;
        p.enable_ada_ef = false;
        p.enable_bridge_aware = false;
        results.push_back(evaluate_config("Canonical HNSW", false, p, base_data, n_base, dim, query_data, n_queries, gt, ef_list));
    }

    // Config 2: Adaptive connectivity without hubness regulation (mu = 0)
    {
        PolicyConfig p;
        p.enable_dynamic_m_efc = true;
        p.enable_layer_scaling = true;
        p.enable_hubness_regulation = false;
        p.enable_ada_ef = true;
        p.enable_bridge_aware = false;
        results.push_back(evaluate_config("Adaptive without Hubness Regulation", true, p, base_data, n_base, dim, query_data, n_queries, gt, ef_list));
    }

    // Config 3: Original hubness regulation (mu = 0.15, bridge = false)
    {
        PolicyConfig p;
        p.enable_dynamic_m_efc = true;
        p.enable_layer_scaling = true;
        p.enable_hubness_regulation = true;
        p.hubness_mu = 0.15f;
        p.enable_ada_ef = true;
        p.enable_bridge_aware = false;
        results.push_back(evaluate_config("Original Hubness Regulation (mu=0.15)", true, p, base_data, n_base, dim, query_data, n_queries, gt, ef_list));
    }

    // Config 4: Calibrated hubness regulation (mu = 0.03, bridge = false)
    {
        PolicyConfig p;
        p.enable_dynamic_m_efc = true;
        p.enable_layer_scaling = true;
        p.enable_hubness_regulation = true;
        p.hubness_mu = 0.03f;
        p.enable_ada_ef = true;
        p.enable_bridge_aware = false;
        results.push_back(evaluate_config("Calibrated Hubness Regulation (mu=0.03)", true, p, base_data, n_base, dim, query_data, n_queries, gt, ef_list));
    }

    // Config 5: Bridge-aware hubness regulation (mu = 0.15, bridge = true, alpha = 1.0)
    {
        PolicyConfig p;
        p.enable_dynamic_m_efc = true;
        p.enable_layer_scaling = true;
        p.enable_hubness_regulation = true;
        p.hubness_mu = 0.15f;
        p.enable_ada_ef = true;
        p.enable_bridge_aware = true;
        p.bridge_alpha = 1.0f;
        p.bridge_estimator = 0; // Local clustering coefficient
        results.push_back(evaluate_config("Bridge-Aware Hubness Regulation (mu=0.15, B)", true, p, base_data, n_base, dim, query_data, n_queries, gt, ef_list));
    }

    // Config 6: Bridge-aware calibrated hubness regulation (mu = 0.03, bridge = true, alpha = 1.0)
    {
        PolicyConfig p;
        p.enable_dynamic_m_efc = true;
        p.enable_layer_scaling = true;
        p.enable_hubness_regulation = true;
        p.hubness_mu = 0.03f;
        p.enable_ada_ef = true;
        p.enable_bridge_aware = true;
        p.bridge_alpha = 1.0f;
        p.bridge_estimator = 0;
        results.push_back(evaluate_config("Bridge-Aware Calibrated (mu=0.03, B)", true, p, base_data, n_base, dim, query_data, n_queries, gt, ef_list));
    }

    std::string out_file = "experiments/phase_1_ablation_" + dataset + ".json";
    output_json_report(dataset, n_base, dim, n_queries, results, out_file);

    return 0;
}
