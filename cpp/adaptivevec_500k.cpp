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

using namespace adaptivevec;

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

float compute_recall(const std::vector<std::vector<Candidate>>& results,
                     const std::vector<std::vector<uint32_t>>& gt,
                     int k = 10) {
    size_t total_matches = 0;
    size_t n_queries = results.size();
    for (size_t q = 0; q < n_queries; ++q) {
        std::set<uint32_t> gt_set(gt[q].begin(), gt[q].begin() + k);
        size_t matches = 0;
        for (int i = 0; i < std::min((int)results[q].size(), k); ++i) {
            if (gt_set.find(results[q][i].id) != gt_set.end()) {
                matches++;
            }
        }
        total_matches += matches;
    }
    return (float)total_matches / (float)(n_queries * k);
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "    AdaptiveVec Phase 0.5: SIFT-500K Benchmark Evaluation       \n";
    std::cout << "=================================================================\n\n";

    std::string base_file = "data/sift_base_1m.fvecs";
    std::string query_file = "data/sift_query.fvecs";
    std::string gt_file = "data/sift_groundtruth_500k.ivecs";

    size_t dim = 0, n_samples = 0, n_queries = 0;
    std::cout << "[*] Loading first 500,000 vectors from " << base_file << " ...\n";
    auto base_data = load_fvecs(base_file, 500000, dim, n_samples);
    std::cout << "[+] Loaded " << n_samples << " base vectors (dim=" << dim << ")\n";

    auto query_data = load_fvecs(query_file, 1000, dim, n_queries);
    std::cout << "[+] Loaded " << n_queries << " query vectors\n";

    auto gt = load_ivecs(gt_file, 1000);
    std::cout << "[+] Loaded " << gt.size() << " ground truth records\n\n";

    std::vector<int> ef_search_vals = {16, 32, 64, 128, 256};

    // -------------------------------------------------------------
    // 1. AdaptiveVec Canonical Baseline
    // -------------------------------------------------------------
    std::cout << "[*] Building AdaptiveVec Canonical Baseline (M=16, efC=200) ...\n";
    PolicyConfig cfg_base;
    cfg_base.enable_dynamic_m_efc = false;
    cfg_base.enable_layer_scaling = false;
    cfg_base.enable_hubness_regulation = false;
    cfg_base.enable_ada_ef = false;
    cfg_base.m_base = 16;
    cfg_base.ef_c_base = 200;

    AdaptiveHNSWIndex idx_base(dim, SpaceType::L2, false, cfg_base);
    auto t0 = std::chrono::high_resolution_clock::now();
    idx_base.batch_insert(base_data.data(), n_samples);
    auto t1 = std::chrono::high_resolution_clock::now();
    double base_build_time = std::chrono::duration<double>(t1 - t0).count();
    auto mem_base = idx_base.get_memory_breakdown();
    std::cout << "[+] Baseline Build Time: " << base_build_time << " s\n";
    std::cout << "[+] Total Edges: " << idx_base.get_total_edges() 
              << ", Working Set: " << (mem_base.process_working_set_bytes / (1024.0 * 1024.0)) << " MB\n";

    std::ofstream out_base("experiments/adaptivevec_baseline_500k.json");
    out_base << "{\n  \"engine\": \"adaptivevec_baseline\",\n";
    out_base << "  \"n_samples\": " << n_samples << ",\n";
    out_base << "  \"dim\": " << dim << ",\n";
    out_base << "  \"build_time_sec\": " << base_build_time << ",\n";
    out_base << "  \"working_set_mb\": " << (mem_base.process_working_set_bytes / (1024.0 * 1024.0)) << ",\n";
    out_base << "  \"total_edges\": " << idx_base.get_total_edges() << ",\n";
    out_base << "  \"evaluations\": [\n";

    for (size_t i = 0; i < ef_search_vals.size(); ++i) {
        int ef = ef_search_vals[i];
        // Warmup
        idx_base.batch_search(query_data.data(), 100, 10, ef);

        auto q0 = std::chrono::high_resolution_clock::now();
        auto results = idx_base.batch_search(query_data.data(), n_queries, 10, ef);
        auto q1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(q1 - q0).count();
        double qps = (double)n_queries / elapsed;
        float recall = compute_recall(results, gt, 10);
        std::cout << "  ef=" << std::setw(3) << ef << " -> Recall@10: " 
                  << std::fixed << std::setprecision(4) << recall 
                  << ", QPS: " << std::setprecision(1) << qps << "\n";

        out_base << "    {\"ef_search\": " << ef << ", \"recall_at_10\": " << recall << ", \"qps\": " << qps << "}";
        if (i + 1 < ef_search_vals.size()) out_base << ",";
        out_base << "\n";
    }
    out_base << "  ]\n}\n";
    out_base.close();

    // -------------------------------------------------------------
    // 2. AdaptiveVec Calibrated Adaptive Architecture
    // -------------------------------------------------------------
    std::cout << "\n[*] Building AdaptiveVec Calibrated Adaptive (M=16, efC=120, mu=0.03, patience=12) ...\n";
    PolicyConfig cfg_adapt;
    cfg_adapt.enable_dynamic_m_efc = true;
    cfg_adapt.enable_layer_scaling = true;
    cfg_adapt.enable_hubness_regulation = true;
    cfg_adapt.enable_ada_ef = true;
    cfg_adapt.hubness_mu = 0.03f;
    cfg_adapt.ada_ef_patience = 12;
    cfg_adapt.m_base = 16;
    cfg_adapt.ef_c_base = 120;

    AdaptiveHNSWIndex idx_adapt(dim, SpaceType::L2, true, cfg_adapt);
    t0 = std::chrono::high_resolution_clock::now();
    idx_adapt.batch_insert(base_data.data(), n_samples);
    t1 = std::chrono::high_resolution_clock::now();
    double adapt_build_time = std::chrono::duration<double>(t1 - t0).count();
    auto mem_adapt = idx_adapt.get_memory_breakdown();
    double edge_red = (1.0 - (double)idx_adapt.get_total_edges() / (double)idx_base.get_total_edges()) * 100.0;
    std::cout << "[+] Adaptive Build Time: " << adapt_build_time << " s\n";
    std::cout << "[+] Total Edges: " << idx_adapt.get_total_edges() << " (-" << edge_red << "%)\n";
    std::cout << "[+] Working Set: " << (mem_adapt.process_working_set_bytes / (1024.0 * 1024.0)) << " MB\n";

    std::ofstream out_adapt("experiments/adaptivevec_adaptive_500k.json");
    out_adapt << "{\n  \"engine\": \"adaptivevec_adaptive\",\n";
    out_adapt << "  \"n_samples\": " << n_samples << ",\n";
    out_adapt << "  \"dim\": " << dim << ",\n";
    out_adapt << "  \"build_time_sec\": " << adapt_build_time << ",\n";
    out_adapt << "  \"working_set_mb\": " << (mem_adapt.process_working_set_bytes / (1024.0 * 1024.0)) << ",\n";
    out_adapt << "  \"total_edges\": " << idx_adapt.get_total_edges() << ",\n";
    out_adapt << "  \"edge_reduction_pct\": " << edge_red << ",\n";
    out_adapt << "  \"evaluations\": [\n";

    for (size_t i = 0; i < ef_search_vals.size(); ++i) {
        int ef = ef_search_vals[i];
        // Warmup
        idx_adapt.batch_search(query_data.data(), 100, 10, ef);

        auto q0 = std::chrono::high_resolution_clock::now();
        auto results = idx_adapt.batch_search(query_data.data(), n_queries, 10, ef);
        auto q1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(q1 - q0).count();
        double qps = (double)n_queries / elapsed;
        float recall = compute_recall(results, gt, 10);
        std::cout << "  ef=" << std::setw(3) << ef << " -> Recall@10: " 
                  << std::fixed << std::setprecision(4) << recall 
                  << ", QPS: " << std::setprecision(1) << qps << "\n";

        out_adapt << "    {\"ef_search\": " << ef << ", \"recall_at_10\": " << recall << ", \"qps\": " << qps << "}";
        if (i + 1 < ef_search_vals.size()) out_adapt << ",";
        out_adapt << "\n";
    }
    out_adapt << "  ]\n}\n";
    out_adapt.close();

    std::cout << "\n[+] SIFT-500K benchmark complete. Results saved to experiments/.\n";
    return 0;
}
