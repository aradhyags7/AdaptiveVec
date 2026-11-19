/**
 * AdaptiveVec Paper Benchmark Runner
 * Executes the Phase 0 ablation suite and macro-benchmarks.
 * Compiles with: g++ -O3 -mavx2 -mfma -fopenmp -std=c++17 cpp/benchmark_main.cpp -o cpp/benchmark_runner.exe
 */

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
#include <cassert>

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

struct BenchmarkRecord {
    std::string dataset_name;
    std::string config_name;
    size_t n_samples = 0;
    size_t dim = 0;
    size_t n_queries = 0;
    double build_time_s = 0.0;
    size_t total_edges = 0;
    size_t layer0_edges = 0;
    size_t upper_layer_edges = 0;
    double edge_change_pct = 0.0;
    double build_speedup_pct = 0.0;
    double ram_mb = 0.0;
    double recall_10 = 0.0;
    double qps = 0.0;
    double avg_dist_evals = 0.0;
    uint64_t graph_fingerprint = 0;
    MemoryBreakdown memory_breakdown;
};

// Generates the 8-cluster synthetic benchmark dataset, strictly labeled "Synthetic-Multi-Cluster"
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

    // Compute ground truth for queries
    std::cout << "[*] Computing exact brute-force ground truth for " << n_queries << " queries on " << n_samples << " vectors..." << std::endl;
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

BenchmarkRecord run_query_benchmark_on_existing_graph(
    const DatasetWrapper& ds,
    const std::string& config_name,
    const AdaptiveHNSWIndex& index,
    int ef_search,
    int k,
    size_t baseline_edges,
    double build_time_s,
    double baseline_build_time_s
) {
    std::cout << "  -> Querying (Reusing Graph): " << config_name << " ... " << std::flush;
    
    size_t total_edges = index.get_total_edges();
    size_t layer0_edges = index.get_layer0_edges();
    size_t upper_edges = index.get_upper_layer_edges();
    MemoryBreakdown mb = index.get_memory_breakdown();
    double ram_mb = (double)mb.total_owned_bytes / (1024.0 * 1024.0);
    uint64_t fp = index.get_graph_fingerprint();

    // Query Phase
    uint64_t dist_comps_before = index.total_dist_computations;
    std::vector<std::vector<Candidate>> results(ds.n_queries);
    
    auto q0 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < ds.n_queries; ++q) {
        results[q] = index.search(&ds.queries[q * ds.dim], k, ef_search);
    }
    auto q1 = std::chrono::high_resolution_clock::now();
    double query_time_s = std::chrono::duration<double>(q1 - q0).count();
    
    uint64_t dist_comps_during = index.total_dist_computations - dist_comps_before;
    double avg_dist_evals = (ds.n_queries > 0) ? (double)dist_comps_during / ds.n_queries : 0.0;
    double qps = (query_time_s > 0.0) ? ((double)ds.n_queries / query_time_s) : 0.0;
    double recall10 = compute_recall(results, ds.ground_truth, k);
    
    double edge_change_pct = (baseline_edges > 0) 
        ? ((double)total_edges - (double)baseline_edges) / (double)baseline_edges * 100.0 
        : 0.0;
    double build_speedup_pct = (baseline_build_time_s > 0.0)
        ? (baseline_build_time_s - build_time_s) / baseline_build_time_s * 100.0
        : 0.0;
        
    std::cout << "Done! (Fingerprint: 0x" << std::hex << fp << std::dec << ", "
              << "Edges: " << total_edges << ", "
              << "RAM: " << std::fixed << std::setprecision(1) << ram_mb << " MB, "
              << "Recall@10: " << std::setprecision(4) << recall10 << ", "
              << "QPS: " << std::setprecision(1) << qps << ")" << std::endl;
              
    return {
        ds.name,
        config_name,
        ds.n_samples,
        ds.dim,
        ds.n_queries,
        build_time_s,
        total_edges,
        layer0_edges,
        upper_edges,
        edge_change_pct,
        build_speedup_pct,
        ram_mb,
        recall10,
        qps,
        avg_dist_evals,
        fp,
        mb
    };
}

void write_json(const std::string& path, const std::vector<BenchmarkRecord>& records) {
    std::ofstream out(path);
    if (!out.is_open()) return;
    
    out << "{\n  \"benchmark_results\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        out << "    {\n";
        out << "      \"dataset\": \"" << r.dataset_name << "\",\n";
        out << "      \"configuration\": \"" << r.config_name << "\",\n";
        out << "      \"n_samples\": " << r.n_samples << ",\n";
        out << "      \"dim\": " << r.dim << ",\n";
        out << "      \"n_queries\": " << r.n_queries << ",\n";
        out << "      \"build_time_sec\": " << std::fixed << std::setprecision(4) << r.build_time_s << ",\n";
        out << "      \"total_edges\": " << r.total_edges << ",\n";
        out << "      \"layer0_edges\": " << r.layer0_edges << ",\n";
        out << "      \"upper_layer_edges\": " << r.upper_layer_edges << ",\n";
        out << "      \"edge_change_pct\": " << std::setprecision(2) << r.edge_change_pct << ",\n";
        out << "      \"build_speedup_pct\": " << std::setprecision(2) << r.build_speedup_pct << ",\n";
        out << "      \"memory_mb\": " << std::setprecision(2) << r.ram_mb << ",\n";
        out << "      \"recall_at_10\": " << std::setprecision(4) << r.recall_10 << ",\n";
        out << "      \"qps\": " << std::setprecision(1) << r.qps << ",\n";
        out << "      \"distance_evaluations_per_query\": " << std::setprecision(1) << r.avg_dist_evals << ",\n";
        std::stringstream fp_ss;
        fp_ss << "0x" << std::hex << r.graph_fingerprint;
        out << "      \"graph_fingerprint\": \"" << fp_ss.str() << "\",\n";
        out << "      \"memory_breakdown\": {\n";
        out << "        \"logical_compressed_bytes\": " << r.memory_breakdown.logical_compressed_bytes << ",\n";
        out << "        \"allocated_quantized_capacity_bytes\": " << r.memory_breakdown.allocated_quantized_capacity_bytes << ",\n";
        out << "        \"resident_fp32_bytes\": " << r.memory_breakdown.resident_fp32_bytes << ",\n";
        out << "        \"graph_adjacency_bytes\": " << r.memory_breakdown.graph_adjacency_bytes << ",\n";
        out << "        \"per_node_metadata_bytes\": " << r.memory_breakdown.per_node_metadata_bytes << ",\n";
        out << "        \"quantization_params_bytes\": " << r.memory_breakdown.quantization_params_bytes << ",\n";
        out << "        \"synchronization_bytes\": " << r.memory_breakdown.synchronization_bytes << ",\n";
        out << "        \"allocator_overhead_bytes\": " << r.memory_breakdown.allocator_overhead_bytes << ",\n";
        out << "        \"total_owned_bytes\": " << r.memory_breakdown.total_owned_bytes << ",\n";
        out << "        \"process_working_set_bytes\": " << r.memory_breakdown.process_working_set_bytes << ",\n";
        out << "        \"peak_process_working_set_bytes\": " << r.memory_breakdown.peak_process_working_set_bytes << "\n";
        out << "      }\n";
        out << "    }" << (i + 1 < records.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::cout << "[+] Wrote machine-readable JSON: " << path << std::endl;
}

void write_csv(const std::string& path, const std::vector<BenchmarkRecord>& records) {
    std::ofstream out(path);
    if (!out.is_open()) return;
    
    out << "dataset,configuration,n_samples,dim,n_queries,build_time_sec,total_edges,edge_change_pct,build_speedup_pct,memory_mb,recall_at_10,qps,distance_evals,fingerprint,resident_fp32_mb,quantized_mb,working_set_mb\n";
    for (const auto& r : records) {
        std::stringstream fp_ss;
        fp_ss << "0x" << std::hex << r.graph_fingerprint;
        double fp32_mb = (double)r.memory_breakdown.resident_fp32_bytes / (1024.0 * 1024.0);
        double q_mb = (double)r.memory_breakdown.allocated_quantized_capacity_bytes / (1024.0 * 1024.0);
        double ws_mb = (double)r.memory_breakdown.process_working_set_bytes / (1024.0 * 1024.0);

        out << "\"" << r.dataset_name << "\","
            << "\"" << r.config_name << "\","
            << r.n_samples << ","
            << r.dim << ","
            << r.n_queries << ","
            << std::fixed << std::setprecision(4) << r.build_time_s << ","
            << r.total_edges << ","
            << std::setprecision(2) << r.edge_change_pct << ","
            << std::setprecision(2) << r.build_speedup_pct << ","
            << std::setprecision(2) << r.ram_mb << ","
            << std::setprecision(4) << r.recall_10 << ","
            << std::setprecision(1) << r.qps << ","
            << std::setprecision(1) << r.avg_dist_evals << ","
            << "\"" << fp_ss.str() << "\","
            << std::setprecision(2) << fp32_mb << ","
            << std::setprecision(2) << q_mb << ","
            << std::setprecision(2) << ws_mb << "\n";
    }
    std::cout << "[+] Wrote machine-readable CSV: " << path << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=================================================================================" << std::endl;
    std::cout << "     AdaptiveVec Research Benchmark Suite: Phase 0 Rigorous Ablations            " << std::endl;
    std::cout << "=================================================================================\n" << std::endl;

    std::string dataset_path = "";
    std::string query_path = "";
    std::string gt_path = "";
    std::string dataset_label = "";
    size_t max_samples = 100000;
    size_t max_queries = 10000;
    int ef_search = 64;
    int k = 10;
    bool force_synthetic = false;
    std::string json_out = "benchmark_results.json";
    std::string csv_out = "benchmark_results.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) dataset_path = argv[++i];
        else if (arg == "--queries" && i + 1 < argc) query_path = argv[++i];
        else if (arg == "--gt" && i + 1 < argc) gt_path = argv[++i];
        else if (arg == "--label" && i + 1 < argc) dataset_label = argv[++i];
        else if (arg == "--max_samples" && i + 1 < argc) max_samples = std::stoul(argv[++i]);
        else if (arg == "--max_queries" && i + 1 < argc) max_queries = std::stoul(argv[++i]);
        else if (arg == "--ef_search" && i + 1 < argc) ef_search = std::stoi(argv[++i]);
        else if (arg == "--synthetic") force_synthetic = true;
        else if (arg == "--out" && i + 1 < argc) json_out = argv[++i];
        else if (arg == "--csv" && i + 1 < argc) csv_out = argv[++i];
    }

    DatasetWrapper ds;

    if (!force_synthetic && !dataset_path.empty() && std::ifstream(dataset_path).good()) {
        std::cout << "[*] Loading real dataset: " << dataset_path << std::endl;
        size_t n = 0, d = 0;
        if (!load_fvecs(dataset_path, ds.data, n, d, max_samples)) {
            std::cerr << "[-] Error: Failed to load fvecs dataset from " << dataset_path << std::endl;
            return 1;
        }
        ds.n_samples = n;
        ds.dim = d;
        ds.name = dataset_label.empty() ? "SIFT-100K subset" : dataset_label;
        ds.space = SpaceType::L2;

        // Load queries
        if (!query_path.empty() && std::ifstream(query_path).good()) {
            size_t q_n = 0, q_d = 0;
            load_fvecs(query_path, ds.queries, q_n, q_d, max_queries);
            ds.n_queries = q_n;
        }

        // Load ground truth
        if (!gt_path.empty() && std::ifstream(gt_path).good()) {
            load_ivecs(gt_path, ds.ground_truth, ds.n_queries);
        }
        std::cout << "[+] Loaded " << ds.n_samples << " vectors (dim " << ds.dim << "), " 
                  << ds.n_queries << " queries with ground truth." << std::endl;
    } else if (!force_synthetic && std::ifstream("data/sift_subset_100k.fvecs").good()) {
        std::cout << "[*] Discovered cached real SIFT dataset in data/ ..." << std::endl;
        size_t n = 0, d = 0;
        load_fvecs("data/sift_subset_100k.fvecs", ds.data, n, d, max_samples);
        ds.n_samples = n;
        ds.dim = d;
        ds.name = "SIFT-100K subset";
        ds.space = SpaceType::L2;

        size_t q_n = 0, q_d = 0;
        load_fvecs("data/sift_query.fvecs", ds.queries, q_n, q_d, max_queries);
        ds.n_queries = q_n;

        load_ivecs("data/sift_subset_groundtruth.ivecs", ds.ground_truth, ds.n_queries);
        std::cout << "[+] Loaded " << ds.n_samples << " vectors (dim " << ds.dim << "), " 
                  << ds.n_queries << " queries with ground truth." << std::endl;
    } else {
        std::cout << "[*] Running with Synthetic benchmark dataset: strictly labeled 'Synthetic-Multi-Cluster'..." << std::endl;
        ds = generate_synthetic_multi_cluster(50000, 64, 1000);
    }

    std::cout << "\n[*] Beginning Ablation Matrix on " << ds.name 
              << " (N=" << ds.n_samples << ", D=" << ds.dim << ", Q=" << ds.n_queries << "):\n" << std::endl;

    std::vector<BenchmarkRecord> records;
    size_t base_edges = 0;
    double base_build_time = 0.0;

    // =========================================================================
    // STEP 1: Baseline HNSW (Fixed M=16, efC=200)
    // =========================================================================
    std::cout << ">>> [1/7] 1. Baseline HNSW (Fixed M=16) <<<" << std::endl;
    PolicyConfig cfg1;
    cfg1.enable_dynamic_m_efc = false;
    cfg1.enable_layer_scaling = false;
    cfg1.enable_hubness_regulation = false;
    cfg1.enable_ada_ef = false;
    cfg1.enable_sq8 = false;
    cfg1.m_base = 16;
    cfg1.ef_c_base = 200;
    
    AdaptiveHNSWIndex index1(ds.dim, ds.space, false, cfg1);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ds.n_samples; ++i) {
        index1.insert(&ds.data[i * ds.dim]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double build1_s = std::chrono::duration<double>(t1 - t0).count();
    base_edges = index1.get_total_edges();
    base_build_time = build1_s;
    records.push_back(run_query_benchmark_on_existing_graph(ds, "1. Baseline HNSW (Fixed M=16)", index1, ef_search, k, base_edges, build1_s, base_build_time));

    // =========================================================================
    // STEP 2: + Dynamic M(x) & efC(x)
    // =========================================================================
    std::cout << "\n>>> [2/7] 2. + Dynamic M(x) & efC(x) <<<" << std::endl;
    PolicyConfig cfg2;
    cfg2.enable_dynamic_m_efc = true;
    cfg2.enable_layer_scaling = false;
    cfg2.enable_hubness_regulation = false;
    cfg2.enable_ada_ef = false;
    cfg2.enable_sq8 = false;
    cfg2.m_base = 16;
    cfg2.m_min = 8;
    cfg2.m_max = 24;
    cfg2.ef_c_base = 120;
    cfg2.ef_c_min = 40;
    cfg2.ef_c_max = 220;
    cfg2.sensitivity = 0.4f;
    
    AdaptiveHNSWIndex index2(ds.dim, ds.space, true, cfg2);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ds.n_samples; ++i) {
        index2.insert(&ds.data[i * ds.dim]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double build2_s = std::chrono::duration<double>(t1 - t0).count();
    records.push_back(run_query_benchmark_on_existing_graph(ds, "2. + Dynamic M(x) & efC(x)", index2, ef_search, k, base_edges, build2_s, base_build_time));

    // =========================================================================
    // STEP 3: + Layer-Decoupled Scaling
    // =========================================================================
    std::cout << "\n>>> [3/7] 3. + Layer-Decoupled Scaling <<<" << std::endl;
    PolicyConfig cfg3 = cfg2;
    cfg3.enable_layer_scaling = true;
    cfg3.lambda_layer = 0.75f;
    cfg3.m_min_layer = 4;
    
    AdaptiveHNSWIndex index3(ds.dim, ds.space, true, cfg3);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ds.n_samples; ++i) {
        index3.insert(&ds.data[i * ds.dim]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double build3_s = std::chrono::duration<double>(t1 - t0).count();
    records.push_back(run_query_benchmark_on_existing_graph(ds, "3. + Layer-Decoupled Scaling", index3, ef_search, k, base_edges, build3_s, base_build_time));

    // =========================================================================
    // STEP 4: + Hubness Regulation (Base Adaptive Graph)
    // =========================================================================
    std::cout << "\n>>> [4/7] 4. + Hubness Regulation (mu=0.15) <<<" << std::endl;
    PolicyConfig cfg4 = cfg3;
    cfg4.enable_hubness_regulation = true;
    cfg4.hubness_mu = 0.15f;
    
    AdaptiveHNSWIndex index4(ds.dim, ds.space, true, cfg4);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ds.n_samples; ++i) {
        index4.insert(&ds.data[i * ds.dim]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double build4_s = std::chrono::duration<double>(t1 - t0).count();
    uint64_t fp4 = index4.get_graph_fingerprint();
    records.push_back(run_query_benchmark_on_existing_graph(ds, "4. + Hubness Regulation (mu=0.15)", index4, ef_search, k, base_edges, build4_s, base_build_time));

    // =========================================================================
    // STEP 5: + Ada-ef Stagnation Exit (Reusing Step 4 Graph)
    // =========================================================================
    std::cout << "\n>>> [5/7] 5. + Ada-ef Stagnation Exit (REUSING STEP 4 GRAPH) <<<" << std::endl;
    index4.policy.enable_ada_ef = true;
    index4.policy.ada_ef_patience = 6;
    index4.policy.ada_ef_epsilon = 1e-4f;
    uint64_t fp5 = index4.get_graph_fingerprint();
    if (fp5 != fp4) {
        std::cerr << "[-] CRITICAL ERROR: Graph fingerprint changed for query-only Ada-ef modification!" << std::endl;
        return 1;
    }
    std::cout << "  [✓] Verified Graph Invariance: Fingerprint 0x" << std::hex << fp5 << std::dec << " matches Step 4 exactly." << std::endl;
    records.push_back(run_query_benchmark_on_existing_graph(ds, "5. + Ada-ef Stagnation Exit", index4, ef_search, k, base_edges, build4_s, base_build_time));

    // =========================================================================
    // STEP 6: + Asymmetric INT8 SQ8 (Config B: SQ8 + FP32 Re-rank, REUSING STEP 4 GRAPH)
    // =========================================================================
    std::cout << "\n>>> [6/7] 6. + Asymmetric INT8 SQ8 (Config B: SQ8 + FP32 Re-rank) <<<" << std::endl;
    index4.quantize_vectors(true, false);
    index4.policy.enable_sq8 = true;
    index4.policy.enable_rerank = true;
    uint64_t fp6 = index4.get_graph_fingerprint();
    if (fp6 != fp4) {
        std::cerr << "[-] CRITICAL ERROR: Graph fingerprint changed during vector quantization!" << std::endl;
        return 1;
    }
    std::cout << "  [✓] Verified Graph Invariance: Fingerprint 0x" << std::hex << fp6 << std::dec << " matches Step 4 exactly." << std::endl;
    records.push_back(run_query_benchmark_on_existing_graph(ds, "6. + Asymmetric INT8 SQ8 (Config B)", index4, ef_search, k, base_edges, build4_s, base_build_time));

    // =========================================================================
    // STEP 7: + Asymmetric INT8 SQ8 (Config A: Pure Quantized, No Resident FP32)
    // =========================================================================
    std::cout << "\n>>> [7/7] 7. + Asymmetric INT8 SQ8 (Config A: Pure Quantized, No FP32) <<<" << std::endl;
    index4.drop_raw_vectors();
    uint64_t fp7 = index4.get_graph_fingerprint();
    if (fp7 != fp4) {
        std::cerr << "[-] CRITICAL ERROR: Graph fingerprint changed during raw vector drop!" << std::endl;
        return 1;
    }
    std::cout << "  [✓] Verified Graph Invariance: Fingerprint 0x" << std::hex << fp7 << std::dec << " matches Step 4 exactly." << std::endl;
    records.push_back(run_query_benchmark_on_existing_graph(ds, "7. + Asymmetric INT8 SQ8 (Config A)", index4, ef_search, k, base_edges, build4_s, base_build_time));

    // =========================================================================
    // Output Summary Tables
    // =========================================================================
    std::cout << "\n================================================================================================================================================" << std::endl;
    std::cout << "                                  ACTUAL MEASURED STEPWISE ABLATION MATRIX (PHASE 0 AUDITED)                                                    " << std::endl;
    std::cout << "================================================================================================================================================" << std::endl;
    std::cout << std::left << std::setw(38) << "Ablation Configuration" 
              << std::setw(14) << "Graph Edges" 
              << std::setw(12) << "Build Time" 
              << std::setw(12) << "Total RAM"
              << std::setw(12) << "FP32 RAM"
              << std::setw(12) << "Recall@10" 
              << std::setw(12) << "QPS" 
              << "Dist Evals/q" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto& r : records) {
        std::stringstream edges_str, build_str;
        edges_str << r.total_edges << " (" << (r.edge_change_pct >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << r.edge_change_pct << "%)";
        build_str << std::fixed << std::setprecision(1) << r.build_time_s << "s";
        double fp32_mb = (double)r.memory_breakdown.resident_fp32_bytes / (1024.0 * 1024.0);
        
        std::cout << std::left << std::setw(38) << r.config_name
                  << std::setw(14) << edges_str.str()
                  << std::setw(12) << build_str.str()
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.ram_mb
                  << std::setw(12) << std::fixed << std::setprecision(1) << fp32_mb
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.recall_10
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.qps
                  << std::fixed << std::setprecision(1) << r.avg_dist_evals << std::endl;
    }
    std::cout << "================================================================================================================================================\n" << std::endl;

    write_json(json_out, records);
    write_csv(csv_out, records);

    return 0;
}
