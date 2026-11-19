#include "adaptive_hnsw.hpp"
#include "hnswlib/hnswlib.h"
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
        std::cerr << "[-] Failed to open: " << path << std::endl;
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
        std::cerr << "[-] Failed to open: " << path << std::endl;
        return {};
    }
    std::vector<std::vector<uint32_t>> gt;
    while (in.peek() != EOF && gt.size() < max_records) {
        int k = 0;
        in.read((char*)&k, sizeof(int));
        if (in.gcount() == 0) break;
        std::vector<uint32_t> record(k);
        in.read((char*)record.data(), k * sizeof(uint32_t));
        gt.push_back(record);
    }
    return gt;
}

float compute_recall(const std::vector<std::vector<uint32_t>>& results,
                     const std::vector<std::vector<uint32_t>>& gt,
                     int k = 10) {
    size_t total_matches = 0;
    size_t n_queries = results.size();
    for (size_t q = 0; q < n_queries; ++q) {
        std::set<uint32_t> gt_set(gt[q].begin(), gt[q].begin() + k);
        size_t matches = 0;
        for (int i = 0; i < std::min((int)results[q].size(), k); ++i) {
            if (gt_set.find(results[q][i]) != gt_set.end()) {
                matches++;
            }
        }
        total_matches += matches;
    }
    return (float)total_matches / (float)(n_queries * k);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <canonical|adaptive|hnswlib>" << std::endl;
        return 1;
    }
    std::string engine = argv[1];

    std::cout << "=================================================================\n";
    std::cout << "  AdaptiveVec Phase 1: Isolated SIFT-500K Benchmark [" << engine << "]\n";
    std::cout << "=================================================================\n\n";

    MemSnap m_init = get_mem_snapshot();
    std::cout << "[*] Process Startup Memory -> WorkingSet: " << m_init.working_set_mb 
              << " MB, Private: " << m_init.private_mb << " MB\n";

    std::string base_file = "data/sift_base_1m.fvecs";
    std::string query_file = "data/sift_query.fvecs";
    std::string gt_file = "data/sift_groundtruth_500k.ivecs";

    size_t dim = 0, n_samples = 0, n_queries = 0;
    auto base_data = load_fvecs(base_file, 500000, dim, n_samples);
    auto query_data = load_fvecs(query_file, 1000, dim, n_queries);
    auto gt = load_ivecs(gt_file, 1000);

    MemSnap m_loaded = get_mem_snapshot();
    double dataset_raw_mb = (n_samples * dim * sizeof(float)) / (1024.0 * 1024.0);
    std::cout << "[+] Dataset Loaded: N=" << n_samples << ", D=" << dim << " (" << dataset_raw_mb << " MB)\n";
    std::cout << "[*] Memory After Load -> WorkingSet: " << m_loaded.working_set_mb 
              << " MB, Private: " << m_loaded.private_mb << " MB\n\n";

    std::vector<int> ef_search_vals = {16, 32, 64, 128, 256};
    double build_time_s = 0.0;
    size_t total_edges = 0;
    double allocated_capacity_mb = 0.0;
    double resident_fp32_mb = 0.0;
    double graph_adjacency_mb = 0.0;
    double metadata_mb = 0.0;

    std::vector<std::vector<uint32_t>> search_results;

    if (engine == "canonical" || engine == "adaptive" || engine == "bridge_aware") {
        PolicyConfig cfg;
        bool is_adaptive = (engine != "canonical");
        if (engine == "bridge_aware") {
            cfg.enable_dynamic_m_efc = true;
            cfg.enable_layer_scaling = true;
            cfg.enable_hubness_regulation = true;
            cfg.enable_ada_ef = true;
            cfg.hubness_mu = 0.03f;
            cfg.ada_ef_patience = 12;
            cfg.enable_bridge_aware = true;
            cfg.bridge_alpha = 1.0f;
            cfg.m_base = 16;
            cfg.ef_c_base = 120;
        } else if (is_adaptive) {
            cfg.enable_dynamic_m_efc = true;
            cfg.enable_layer_scaling = true;
            cfg.enable_hubness_regulation = true;
            cfg.enable_ada_ef = true;
            cfg.hubness_mu = 0.03f;
            cfg.ada_ef_patience = 12;
            cfg.m_base = 16;
            cfg.ef_c_base = 120;
        } else {
            cfg.enable_dynamic_m_efc = false;
            cfg.enable_layer_scaling = false;
            cfg.enable_hubness_regulation = false;
            cfg.enable_ada_ef = false;
            cfg.m_base = 16;
            cfg.ef_c_base = 200;
        }

        std::cout << "[*] Constructing AdaptiveVec (" << engine << ") ...\n";
        AdaptiveHNSWIndex index(dim, SpaceType::L2, is_adaptive, cfg);

        auto t0 = std::chrono::high_resolution_clock::now();
        index.batch_insert(base_data.data(), n_samples);
        auto t1 = std::chrono::high_resolution_clock::now();
        build_time_s = std::chrono::duration<double>(t1 - t0).count();
        total_edges = index.get_total_edges();

        auto mb = index.get_memory_breakdown();
        allocated_capacity_mb = mb.total_owned_bytes / (1024.0 * 1024.0);
        resident_fp32_mb = mb.resident_fp32_bytes / (1024.0 * 1024.0);
        graph_adjacency_mb = mb.graph_adjacency_bytes / (1024.0 * 1024.0);
        metadata_mb = mb.per_node_metadata_bytes / (1024.0 * 1024.0);

        // Free caller's base_data buffer now to measure pure steady-state index memory
        base_data.clear();
        base_data.shrink_to_fit();

        MemSnap m_post_build = get_mem_snapshot();
        std::cout << "[+] Build Time: " << build_time_s << " s\n";
        std::cout << "[+] Total Edges: " << total_edges << "\n";
        std::cout << "[+] Owned Allocated Capacity: " << allocated_capacity_mb << " MB\n";
        std::cout << "    - Resident FP32 Vectors: " << resident_fp32_mb << " MB\n";
        std::cout << "    - Graph Adjacency: " << graph_adjacency_mb << " MB\n";
        std::cout << "    - Per-Node Metadata: " << metadata_mb << " MB\n";
        std::cout << "[*] Memory After Build (caller buffer freed) -> WorkingSet: " << m_post_build.working_set_mb 
                  << " MB, PeakWorkingSet: " << m_post_build.peak_working_set_mb 
                  << " MB, Private: " << m_post_build.private_mb << " MB\n\n";

        // Query evaluations
        std::ofstream out("experiments/sift500k_isolated_" + engine + ".json");
        out << "{\n";
        out << "  \"engine\": \"adaptivevec_" << engine << "\",\n";
        out << "  \"n_samples\": " << n_samples << ",\n";
        out << "  \"dim\": " << dim << ",\n";
        out << "  \"build_time_sec\": " << build_time_s << ",\n";
        out << "  \"total_edges\": " << total_edges << ",\n";
        out << "  \"initial_working_set_mb\": " << m_init.working_set_mb << ",\n";
        out << "  \"initial_private_mb\": " << m_init.private_mb << ",\n";
        out << "  \"dataset_loaded_working_set_mb\": " << m_loaded.working_set_mb << ",\n";
        out << "  \"peak_construction_working_set_mb\": " << m_post_build.peak_working_set_mb << ",\n";
        out << "  \"steady_state_working_set_mb\": " << m_post_build.working_set_mb << ",\n";
        out << "  \"steady_state_private_mb\": " << m_post_build.private_mb << ",\n";
        out << "  \"allocated_capacity_mb\": " << allocated_capacity_mb << ",\n";
        out << "  \"resident_fp32_mb\": " << resident_fp32_mb << ",\n";
        out << "  \"graph_adjacency_mb\": " << graph_adjacency_mb << ",\n";
        out << "  \"per_node_metadata_mb\": " << metadata_mb << ",\n";
        out << "  \"evaluations\": [\n";

        for (size_t i = 0; i < ef_search_vals.size(); ++i) {
            int ef = ef_search_vals[i];
            // Warmup
            index.batch_search(query_data.data(), 100, 10, ef);

            // Single query latencies
            std::vector<double> latencies_us(n_queries);
            std::vector<std::vector<uint32_t>> curr_results(n_queries);

            auto q0 = std::chrono::high_resolution_clock::now();
            auto cand_results = index.batch_search(query_data.data(), n_queries, 10, ef);
            auto q1 = std::chrono::high_resolution_clock::now();
            double total_q_s = std::chrono::duration<double>(q1 - q0).count();
            double qps = (double)n_queries / total_q_s;

            for (size_t q = 0; q < n_queries; ++q) {
                for (const auto& c : cand_results[q]) {
                    curr_results[q].push_back(c.id);
                }
            }
            float recall = compute_recall(curr_results, gt, 10);

            // Measure single-threaded individual latencies for p50/p95/p99
            for (size_t q = 0; q < n_queries; ++q) {
                auto l0 = std::chrono::high_resolution_clock::now();
                auto res = index.search(&query_data[q * dim], 10, ef);
                auto l1 = std::chrono::high_resolution_clock::now();
                latencies_us[q] = std::chrono::duration<double, std::micro>(l1 - l0).count();
            }
            std::sort(latencies_us.begin(), latencies_us.end());
            double p50 = latencies_us[(size_t)(n_queries * 0.50)];
            double p95 = latencies_us[(size_t)(n_queries * 0.95)];
            double p99 = latencies_us[(size_t)(n_queries * 0.99)];
            double mean_lat = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / n_queries;

            std::cout << "  ef=" << std::setw(3) << ef 
                      << " -> Recall@10: " << std::fixed << std::setprecision(4) << recall 
                      << ", MT-QPS: " << std::setprecision(1) << qps 
                      << ", Latency (μs): mean=" << mean_lat << ", p50=" << p50 << ", p95=" << p95 << ", p99=" << p99 << "\n";

            out << "    {\n";
            out << "      \"ef_search\": " << ef << ",\n";
            out << "      \"recall_at_10\": " << recall << ",\n";
            out << "      \"qps\": " << qps << ",\n";
            out << "      \"latency_mean_us\": " << mean_lat << ",\n";
            out << "      \"latency_p50_us\": " << p50 << ",\n";
            out << "      \"latency_p95_us\": " << p95 << ",\n";
            out << "      \"latency_p99_us\": " << p99 << "\n";
            out << "    }" << (i + 1 < ef_search_vals.size() ? "," : "") << "\n";
        }
        out << "  ]\n}\n";
        out.close();

    } else if (engine == "hnswlib") {
        std::cout << "[*] Constructing hnswlib (M=16, efC=200) ...\n";
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> alg_hnsw(&space, n_samples, 16, 200);

        auto t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t i = 0; i < n_samples; ++i) {
            alg_hnsw.addPoint(&base_data[i * dim], i);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        build_time_s = std::chrono::duration<double>(t1 - t0).count();

        // Free caller's base_data buffer
        base_data.clear();
        base_data.shrink_to_fit();

        MemSnap m_post_build = get_mem_snapshot();
        std::cout << "[+] Build Time: " << build_time_s << " s\n";
        std::cout << "[*] Memory After Build (caller buffer freed) -> WorkingSet: " << m_post_build.working_set_mb 
                  << " MB, PeakWorkingSet: " << m_post_build.peak_working_set_mb 
                  << " MB, Private: " << m_post_build.private_mb << " MB\n\n";

        std::ofstream out("experiments/sift500k_isolated_hnswlib.json");
        out << "{\n";
        out << "  \"engine\": \"hnswlib\",\n";
        out << "  \"n_samples\": " << n_samples << ",\n";
        out << "  \"dim\": " << dim << ",\n";
        out << "  \"build_time_sec\": " << build_time_s << ",\n";
        out << "  \"initial_working_set_mb\": " << m_init.working_set_mb << ",\n";
        out << "  \"initial_private_mb\": " << m_init.private_mb << ",\n";
        out << "  \"dataset_loaded_working_set_mb\": " << m_loaded.working_set_mb << ",\n";
        out << "  \"peak_construction_working_set_mb\": " << m_post_build.peak_working_set_mb << ",\n";
        out << "  \"steady_state_working_set_mb\": " << m_post_build.working_set_mb << ",\n";
        out << "  \"steady_state_private_mb\": " << m_post_build.private_mb << ",\n";
        out << "  \"evaluations\": [\n";

        for (size_t i = 0; i < ef_search_vals.size(); ++i) {
            int ef = ef_search_vals[i];
            alg_hnsw.setEf(ef);

            // Multithreaded search
            std::vector<std::vector<uint32_t>> curr_results(n_queries);
            auto q0 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(dynamic, 32)
            for (size_t q = 0; q < n_queries; ++q) {
                auto res = alg_hnsw.searchKnn(&query_data[q * dim], 10);
                std::vector<uint32_t> topk;
                while (!res.empty()) {
                    topk.push_back(res.top().second);
                    res.pop();
                }
                std::reverse(topk.begin(), topk.end());
                curr_results[q] = topk;
            }
            auto q1 = std::chrono::high_resolution_clock::now();
            double total_q_s = std::chrono::duration<double>(q1 - q0).count();
            double qps = (double)n_queries / total_q_s;
            float recall = compute_recall(curr_results, gt, 10);

            // Latencies
            std::vector<double> latencies_us(n_queries);
            for (size_t q = 0; q < n_queries; ++q) {
                auto l0 = std::chrono::high_resolution_clock::now();
                auto res = alg_hnsw.searchKnn(&query_data[q * dim], 10);
                auto l1 = std::chrono::high_resolution_clock::now();
                latencies_us[q] = std::chrono::duration<double, std::micro>(l1 - l0).count();
            }
            std::sort(latencies_us.begin(), latencies_us.end());
            double p50 = latencies_us[(size_t)(n_queries * 0.50)];
            double p95 = latencies_us[(size_t)(n_queries * 0.95)];
            double p99 = latencies_us[(size_t)(n_queries * 0.99)];
            double mean_lat = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / n_queries;

            std::cout << "  ef=" << std::setw(3) << ef 
                      << " -> Recall@10: " << std::fixed << std::setprecision(4) << recall 
                      << ", MT-QPS: " << std::setprecision(1) << qps 
                      << ", Latency (μs): mean=" << mean_lat << ", p50=" << p50 << ", p95=" << p95 << ", p99=" << p99 << "\n";

            out << "    {\n";
            out << "      \"ef_search\": " << ef << ",\n";
            out << "      \"recall_at_10\": " << recall << ",\n";
            out << "      \"qps\": " << qps << ",\n";
            out << "      \"latency_mean_us\": " << mean_lat << ",\n";
            out << "      \"latency_p50_us\": " << p50 << ",\n";
            out << "      \"latency_p95_us\": " << p95 << ",\n";
            out << "      \"latency_p99_us\": " << p99 << "\n";
            out << "    }" << (i + 1 < ef_search_vals.size() ? "," : "") << "\n";
        }
        out << "  ]\n}\n";
        out.close();
    } else {
        std::cerr << "[-] Unknown engine: " << engine << std::endl;
        return 1;
    }

    std::cout << "\n[+] Isolated benchmark for " << engine << " completed successfully.\n";
    return 0;
}
