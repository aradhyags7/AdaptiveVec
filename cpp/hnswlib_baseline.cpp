/**
 * Official hnswlib (nmslib/hnswlib) Baseline Runner
 * Evaluates canonical HNSW under identical flags (-O3 -mavx2 -mfma -fopenmp).
 */

#include "hnswlib/hnswlib.h"
#include "dataset_loader.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <unordered_set>
#include <algorithm>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <psapi.h>
#endif

using namespace adaptivevec;

float compute_recall(const std::vector<std::vector<hnswlib::labeltype>>& results, 
                     const std::vector<std::vector<uint32_t>>& ground_truth, 
                     int k = 10) {
    size_t total_matches = 0;
    size_t total_queries = results.size();
    if (total_queries == 0 || k == 0) return 0.0f;
    for (size_t q = 0; q < total_queries; ++q) {
        std::unordered_set<uint32_t> gt_set(ground_truth[q].begin(), ground_truth[q].begin() + k);
        for (int i = 0; i < std::min((int)results[q].size(), k); ++i) {
            if (gt_set.find((uint32_t)results[q][i]) != gt_set.end()) {
                total_matches++;
            }
        }
    }
    return (float)total_matches / (float)(total_queries * k);
}

int main(int argc, char** argv) {
    std::cout << "=================================================================" << std::endl;
    std::cout << "         Official hnswlib (nmslib/hnswlib) Baseline Runner       " << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    std::string dataset_path = "data/sift_subset_100k.fvecs";
    std::string query_path = "data/sift_query.fvecs";
    std::string gt_path = "data/sift_subset_groundtruth.ivecs";
    std::string out_json = "experiments/hnswlib_baseline_results.json";

    if (argc > 1) dataset_path = argv[1];
    if (argc > 2) query_path = argv[2];
    if (argc > 3) gt_path = argv[3];

    std::vector<float> data;
    size_t n_samples = 0, dim = 0;
    load_fvecs(dataset_path, data, n_samples, dim, 100000);

    std::vector<float> queries;
    size_t n_queries = 0, q_dim = 0;
    load_fvecs(query_path, queries, n_queries, q_dim, 10000);

    std::vector<std::vector<uint32_t>> ground_truth;
    load_ivecs(gt_path, ground_truth, n_queries);

    std::cout << "[+] Loaded " << n_samples << " vectors (dim=" << dim << "), " 
              << n_queries << " queries with ground truth." << std::endl;

    int M = 16;
    int efConstruction = 200;

    std::cout << "[*] Initializing hnswlib::L2Space and HierarchicalNSW (M=" << M << ", efC=" << efConstruction << ") ..." << std::endl;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> alg_hnsw(&space, n_samples, M, efConstruction);

    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_samples; ++i) {
        alg_hnsw.addPoint(&data[i * dim], (hnswlib::labeltype)i);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_time_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[+] Build complete in " << std::fixed << std::setprecision(2) << build_time_s << " s." << std::endl;

    size_t working_set_bytes = 0;
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        working_set_bytes = pmc.WorkingSetSize;
    }
#endif
    double ram_mb = (double)working_set_bytes / (1024.0 * 1024.0);

    std::vector<int> ef_search_vals = {16, 32, 64, 128, 256};
    std::ofstream out(out_json);
    out << "{\n  \"engine\": \"hnswlib\",\n  \"n_samples\": " << n_samples 
        << ",\n  \"dim\": " << dim 
        << ",\n  \"build_time_sec\": " << build_time_s 
        << ",\n  \"working_set_mb\": " << ram_mb 
        << ",\n  \"evaluations\": [\n";

    std::cout << "\n" << std::left << std::setw(12) << "efSearch" 
              << std::setw(15) << "Recall@10" 
              << std::setw(15) << "QPS" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    for (size_t idx = 0; idx < ef_search_vals.size(); ++idx) {
        int ef = ef_search_vals[idx];
        alg_hnsw.setEf(ef);

        std::vector<std::vector<hnswlib::labeltype>> results(n_queries);
        auto q0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(static)
        for (size_t q = 0; q < n_queries; ++q) {
            auto pq = alg_hnsw.searchKnn(&queries[q * dim], 10);
            std::vector<hnswlib::labeltype> res;
            while (!pq.empty()) {
                res.push_back(pq.top().second);
                pq.pop();
            }
            std::reverse(res.begin(), res.end());
            results[q] = std::move(res);
        }
        auto q1 = std::chrono::high_resolution_clock::now();
        double query_time_s = std::chrono::duration<double>(q1 - q0).count();
        double qps = (double)n_queries / query_time_s;
        float recall10 = compute_recall(results, ground_truth, 10);

        std::cout << std::left << std::setw(12) << ef 
                  << std::setw(15) << std::fixed << std::setprecision(4) << recall10 
                  << std::setw(15) << std::fixed << std::setprecision(1) << qps << std::endl;

        out << "    {\"ef_search\": " << ef << ", \"recall_at_10\": " << recall10 << ", \"qps\": " << qps << "}"
            << (idx + 1 < ef_search_vals.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::cout << "\n[+] Results saved to: " << out_json << std::endl;
    return 0;
}
