#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <psapi.h>
#endif

using namespace adaptivevec;

struct ProcMemSnap {
    double working_set_mb = 0.0;
    double peak_working_set_mb = 0.0;
    double private_mb = 0.0;
};

inline ProcMemSnap get_proc_mem_snap() {
    ProcMemSnap snap;
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        snap.working_set_mb = pmc.WorkingSetSize / (1024.0 * 1024.0);
        snap.peak_working_set_mb = pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
        snap.private_mb = pmc.PrivateUsage / (1024.0 * 1024.0);
    }
#endif
    return snap;
}

int main(int argc, char** argv) {
    std::string precision = "fp32";
    std::string output_file = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--precision" && i + 1 < argc) {
            precision = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    ProcMemSnap mem_initial = get_proc_mem_snap();

    std::cout << "=================================================================\n";
    std::cout << "  Fresh Process Precision Benchmark: " << precision << "\n";
    std::cout << "  Initial Process Working Set: " << mem_initial.working_set_mb << " MB, Private: " << mem_initial.private_mb << " MB\n";
    std::cout << "=================================================================\n";

    // Load SIFT-100K
    std::vector<float> sift_data, sift_queries;
    size_t sift_n = 0, sift_dim = 0, sift_qn = 0, sift_qdim = 0;
    if (!load_fvecs("data/sift_subset_100k.fvecs", sift_data, sift_n, sift_dim, 100000)) {
        std::cerr << "Failed to load SIFT data!\n";
        return 1;
    }
    if (!load_fvecs("data/sift_query.fvecs", sift_queries, sift_qn, sift_qdim, 100)) {
        std::cerr << "Failed to load SIFT queries!\n";
        return 1;
    }
    ProcMemSnap mem_after_load = get_proc_mem_snap();
    double dataset_input_mb = (sift_data.size() * sizeof(float)) / (1024.0 * 1024.0);

    // Compute ground truth on 100 queries
    std::vector<std::vector<tableint>> gt(sift_qn, std::vector<tableint>(10));
    #pragma omp parallel for schedule(dynamic, 8)
    for (int q = 0; q < (int)sift_qn; ++q) {
        std::vector<std::pair<float, tableint>> dists;
        dists.reserve(sift_n);
        const float* q_ptr = &sift_queries[q * sift_dim];
        for (size_t i = 0; i < sift_n; ++i) {
            float d = l2_distance(q_ptr, &sift_data[i * sift_dim], sift_dim);
            dists.push_back({d, static_cast<tableint>(i)});
        }
        std::partial_sort(dists.begin(), dists.begin() + 10, dists.end());
        for (size_t j = 0; j < 10; ++j) {
            gt[q][j] = dists[j].second;
        }
    }

    // Configure Policy
    PolicyConfig cfg;
    cfg.m_base = 16;
    cfg.ef_c_base = 64;
    cfg.keep_raw_vectors = true;
    cfg.enable_sq8 = (precision == "sq8");
    cfg.enable_sq4 = (precision == "sq4");

    AdaptiveHNSWIndex index(sift_dim, SpaceType::L2, false, cfg);
    index.reserve(sift_n);

    auto t_build0 = std::chrono::high_resolution_clock::now();
    index.batch_insert(sift_data.data(), sift_n);
    auto t_build1 = std::chrono::high_resolution_clock::now();
    double build_sec = std::chrono::duration<double>(t_build1 - t_build0).count();

    if (precision == "sq8") {
        index.quantize_vectors(true, false);
        index.drop_raw_vectors();
    } else if (precision == "sq4") {
        index.quantize_vectors(false, true);
        index.drop_raw_vectors();
    }

    // Drop input dataset to simulate standalone deployed service memory
    sift_data.clear();
    sift_data.shrink_to_fit();

    // Query Evaluation
    size_t hits = 0;
    auto t_q0 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < sift_qn; ++q) {
        auto res = index.search(&sift_queries[q * sift_dim], 10, 64);
        std::unordered_set<tableint> gt_set(gt[q].begin(), gt[q].end());
        for (const auto& r : res) {
            if (gt_set.count(r.id)) hits++;
        }
    }
    auto t_q1 = std::chrono::high_resolution_clock::now();
    double query_sec = std::chrono::duration<double>(t_q1 - t_q0).count();
    double recall = static_cast<double>(hits) / (sift_qn * 10.0);
    double qps = sift_qn / std::max(1e-6, query_sec);

    MemoryBreakdown mb = index.get_memory_breakdown();
    ProcMemSnap mem_final = get_proc_mem_snap();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << ">>> Results for Precision: " << precision << "\n";
    std::cout << "  - Build Time: " << build_sec << " s\n";
    std::cout << "  - Recall@10: " << (recall * 100.0) << " %\n";
    std::cout << "  - QPS: " << qps << "\n";
    std::cout << "  - Dataset Input MB: " << dataset_input_mb << " MB\n";
    std::cout << "  - Logical Compressed Payload: " << (mb.logical_compressed_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  - Allocated Quantized Capacity: " << (mb.allocated_quantized_capacity_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  - Resident FP32 Vectors: " << (mb.resident_fp32_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  - Graph Adjacency: " << (mb.graph_adjacency_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  - Total Index Owned Memory: " << (mb.total_owned_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  - Process Private Committed: " << mem_final.private_mb << " MB\n";
    std::cout << "  - Process Working Set: " << mem_final.working_set_mb << " MB\n";

    if (!output_file.empty()) {
        std::ofstream out(output_file);
        out << std::fixed << std::setprecision(4);
        out << "{\n"
            << "  \"precision\": \"" << precision << "\",\n"
            << "  \"initial_working_set_mb\": " << mem_initial.working_set_mb << ",\n"
            << "  \"initial_private_mb\": " << mem_initial.private_mb << ",\n"
            << "  \"dataset_input_mb\": " << dataset_input_mb << ",\n"
            << "  \"logical_compressed_bytes\": " << mb.logical_compressed_bytes << ",\n"
            << "  \"logical_compressed_mb\": " << (mb.logical_compressed_bytes / (1024.0 * 1024.0)) << ",\n"
            << "  \"allocated_quantized_capacity_bytes\": " << mb.allocated_quantized_capacity_bytes << ",\n"
            << "  \"allocated_quantized_capacity_mb\": " << (mb.allocated_quantized_capacity_bytes / (1024.0 * 1024.0)) << ",\n"
            << "  \"resident_fp32_bytes\": " << mb.resident_fp32_bytes << ",\n"
            << "  \"resident_fp32_mb\": " << (mb.resident_fp32_bytes / (1024.0 * 1024.0)) << ",\n"
            << "  \"graph_adjacency_bytes\": " << mb.graph_adjacency_bytes << ",\n"
            << "  \"graph_adjacency_mb\": " << (mb.graph_adjacency_bytes / (1024.0 * 1024.0)) << ",\n"
            << "  \"per_node_metadata_bytes\": " << mb.per_node_metadata_bytes << ",\n"
            << "  \"per_node_metadata_mb\": " << (mb.per_node_metadata_bytes / (1024.0 * 1024.0)) << ",\n"
            << "  \"total_index_owned_bytes\": " << mb.total_owned_bytes << ",\n"
            << "  \"total_index_owned_mb\": " << (mb.total_owned_bytes / (1024.0 * 1024.0)) << ",\n"
            << "  \"process_private_committed_mb\": " << mem_final.private_mb << ",\n"
            << "  \"process_working_set_mb\": " << mem_final.working_set_mb << ",\n"
            << "  \"peak_working_set_mb\": " << mem_final.peak_working_set_mb << ",\n"
            << "  \"recall_at_10\": " << recall << ",\n"
            << "  \"qps\": " << qps << ",\n"
            << "  \"build_time_sec\": " << build_sec << "\n"
            << "}\n";
    }

    return 0;
}
