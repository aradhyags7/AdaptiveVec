#include "adaptive_hnsw.hpp"
#include "../third_party/hnswlib/hnswlib/hnswlib.h"
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
#include <random>
#include <map>
#include <set>
#include <sstream>
#include <omp.h>

using namespace adaptivevec;

struct BenchmarkQueryMetrics {
    int ef = 0;
    float recall1 = 0.0f;
    float recall10 = 0.0f;
    float recall100 = 0.0f;
    double mt_qps = 0.0;
    double st_qps = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double avg_hops = 0.0;
};

struct StrategyEvaluationResult {
    std::string dataset_name;
    std::string strategy_name;
    double budget_ratio = 0.0;
    size_t n_fp32 = 0;
    size_t n_sq8 = 0;
    size_t n_sq4 = 0;
    double vector_storage_bytes = 0.0;
    double vector_savings_pct = 0.0;
    double total_index_ram_mb = 0.0;
    std::vector<BenchmarkQueryMetrics> sweeps;
};

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

std::vector<std::vector<uint32_t>> load_ivecs(const std::string& path, size_t max_records) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    std::vector<std::vector<uint32_t>> records;
    size_t count = 0;
    while (in.peek() != EOF && count < max_records) {
        int k = 0;
        in.read((char*)&k, sizeof(int));
        if (k <= 0 || k > 10000) break;
        std::vector<uint32_t> vec(k);
        in.read((char*)vec.data(), k * sizeof(uint32_t));
        records.push_back(vec);
        count++;
    }
    return records;
}

void generate_synthetic_data(
    size_t n_base, size_t n_queries, size_t dim,
    std::vector<float>& base_data,
    std::vector<float>& query_data,
    std::vector<std::vector<uint32_t>>& gt
) {
    std::mt19937 rng(1337);
    base_data.resize(n_base * dim);
    query_data.resize(n_queries * dim);
    gt.resize(n_queries);

    size_t n_clusters = 8;
    std::vector<std::vector<float>> centroids(n_clusters, std::vector<float>(dim));
    std::normal_distribution<float> cent_dist(0.0f, 10.0f);
    for (auto& c : centroids) {
        for (auto& val : c) val = cent_dist(rng);
    }

    std::vector<float> cluster_stds = {0.2f, 0.4f, 0.8f, 1.2f, 1.8f, 2.5f, 3.5f, 5.0f};
    std::uniform_int_distribution<size_t> cluster_choice(0, n_clusters - 1);
    std::normal_distribution<float> std_norm(0.0f, 1.0f);

    for (size_t i = 0; i < n_base; ++i) {
        size_t c = cluster_choice(rng);
        float sigma = cluster_stds[c];
        for (size_t d = 0; d < dim; ++d) {
            base_data[i * dim + d] = centroids[c][d] + sigma * std_norm(rng);
        }
    }

    for (size_t q = 0; q < n_queries; ++q) {
        size_t c = cluster_choice(rng);
        float sigma = cluster_stds[c];
        for (size_t d = 0; d < dim; ++d) {
            query_data[q * dim + d] = centroids[c][d] + sigma * std_norm(rng);
        }
    }

    // Exact ground truth for synthetic dataset
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t q = 0; q < n_queries; ++q) {
        const float* q_vec = &query_data[q * dim];
        std::vector<std::pair<float, uint32_t>> all_dist(n_base);
        for (size_t i = 0; i < n_base; ++i) {
            all_dist[i] = {l2_sq_distance(q_vec, &base_data[i * dim], dim), (uint32_t)i};
        }
        std::partial_sort(all_dist.begin(), all_dist.begin() + 10, all_dist.end());
        gt[q].resize(10);
        for (size_t k = 0; k < 10; ++k) gt[q][k] = all_dist[k].second;
    }
}

BenchmarkQueryMetrics evaluate_query_performance(
    const AdaptiveHNSWIndex& index,
    const std::vector<float>& query_data,
    size_t n_queries,
    size_t dim,
    const std::vector<std::vector<uint32_t>>& gt,
    int ef
) {
    BenchmarkQueryMetrics m;
    m.ef = ef;

    // 1. Single-threaded latency measurement
    size_t test_queries = std::min(n_queries, (size_t)500);
    std::vector<double> latencies_us(test_queries);
    auto t0_st = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < test_queries; ++q) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = index.search(&query_data[q * dim], 10, ef);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_us[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    auto t1_st = std::chrono::high_resolution_clock::now();
    double st_sec = std::chrono::duration<double>(t1_st - t0_st).count();
    m.st_qps = (st_sec > 0) ? (test_queries / st_sec) : 0.0;

    std::sort(latencies_us.begin(), latencies_us.end());
    m.p50_us = latencies_us[test_queries * 50 / 100];
    m.p95_us = latencies_us[test_queries * 95 / 100];
    m.p99_us = latencies_us[test_queries * 99 / 100];

    // 2. Multi-threaded throughput & Recall measurement (OpenMP)
    uint64_t dist_comp_before = index.get_total_dist_computations();
    std::vector<std::vector<Candidate>> results(n_queries);
    auto t0_mt = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 32)
    for (size_t q = 0; q < n_queries; ++q) {
        results[q] = index.search(&query_data[q * dim], 10, ef);
    }
    auto t1_mt = std::chrono::high_resolution_clock::now();
    double mt_sec = std::chrono::duration<double>(t1_mt - t0_mt).count();
    m.mt_qps = (mt_sec > 0) ? (n_queries / mt_sec) : 0.0;
    uint64_t dist_comp_after = index.get_total_dist_computations();
    m.avg_hops = static_cast<double>(dist_comp_after - dist_comp_before) / n_queries;

    // Recall calculation
    double total_r1 = 0.0, total_r10 = 0.0;
    for (size_t q = 0; q < n_queries; ++q) {
        if (gt[q].empty()) continue;
        const auto& res = results[q];
        
        // Recall@1
        if (!res.empty() && res[0].id == gt[q][0]) total_r1 += 1.0;

        // Recall@10
        int hit10 = 0;
        int target10 = std::min((int)gt[q].size(), 10);
        for (int k = 0; k < std::min((int)res.size(), 10); ++k) {
            for (int g = 0; g < target10; ++g) {
                if (res[k].id == gt[q][g]) { hit10++; break; }
            }
        }
        total_r10 += (double)hit10 / target10;
    }
    m.recall1 = total_r1 / n_queries;
    m.recall10 = total_r10 / n_queries;
    m.recall100 = 0.0f;

    return m;
}

void run_suite_for_dataset(
    const std::string& name,
    const std::vector<float>& base_data,
    size_t n_base,
    const std::vector<float>& query_data,
    size_t n_queries,
    size_t dim,
    const std::vector<std::vector<uint32_t>>& gt,
    std::vector<StrategyEvaluationResult>& all_results
) {
    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  EVALUATING DATASET: " << name << " (N=" << n_base << ", Q=" << n_queries << ", D=" << dim << ")" << std::endl;
    std::cout << "=================================================================" << std::endl;

    PolicyConfig cfg;
    cfg.m_base = 16;
    cfg.ef_c_base = 64;
    cfg.enable_dynamic_m_efc = true;
    cfg.enable_layer_scaling = true;
    cfg.enable_hubness_regulation = true;
    cfg.hubness_mu = 0.003f;
    cfg.enable_bridge_aware = false;
    cfg.enable_ada_ef = false; // Deterministic search sweep
    cfg.enable_sq8 = false;   // Clean FP32 graph construction
    cfg.enable_sq4 = false;
    cfg.keep_raw_vectors = true;

    // External Baseline: hnswlib FP32
    {
        std::cout << "[*] Running External Baseline: hnswlib FP32 (M=16, efConstruction=64)..." << std::endl;
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> alg_hnsw(&space, n_base, 16, 64, 42);
        auto t0_b = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_base; ++i) {
            alg_hnsw.addPoint(&base_data[i * dim], (hnswlib::labeltype)i);
        }
        auto t1_b = std::chrono::high_resolution_clock::now();
        double build_sec = std::chrono::duration<double>(t1_b - t0_b).count();
        std::cout << "    Build completed in " << std::fixed << std::setprecision(2) << build_sec << " s" << std::endl;

        StrategyEvaluationResult res;
        res.dataset_name = name;
        res.strategy_name = "hnswlib_FP32";
        res.budget_ratio = 1.0;
        res.n_fp32 = n_base;
        res.vector_storage_bytes = n_base * dim * sizeof(float);
        res.total_index_ram_mb = (res.vector_storage_bytes + n_base * 16 * sizeof(uint32_t) * 1.5) / (1024.0 * 1024.0);

        for (int ef : {16, 32, 64, 128}) {
            alg_hnsw.setEf(ef);
            BenchmarkQueryMetrics m;
            m.ef = ef;

            auto t0_mt = std::chrono::high_resolution_clock::now();
            std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> results(n_queries);
            #pragma omp parallel for schedule(dynamic, 32)
            for (size_t q = 0; q < n_queries; ++q) {
                results[q] = alg_hnsw.searchKnn(&query_data[q * dim], 10);
            }
            auto t1_mt = std::chrono::high_resolution_clock::now();
            double mt_sec = std::chrono::duration<double>(t1_mt - t0_mt).count();
            m.mt_qps = (mt_sec > 0) ? (n_queries / mt_sec) : 0.0;

            int hits10 = 0;
            for (size_t q = 0; q < n_queries; ++q) {
                auto pq = results[q];
                std::set<uint32_t> gt_set(gt[q].begin(), gt[q].begin() + std::min((size_t)10, gt[q].size()));
                while (!pq.empty()) {
                    if (gt_set.count((uint32_t)pq.top().second)) hits10++;
                    pq.pop();
                }
            }
            m.recall10 = (double)hits10 / (n_queries * 10);
            std::cout << "    ef=" << ef << " | Recall@10: " << std::fixed << std::setprecision(4) << m.recall10
                      << " | MT-QPS: " << std::fixed << std::setprecision(1) << m.mt_qps << std::endl;
            res.sweeps.push_back(m);
        }
        all_results.push_back(res);
    }

    // Build Master AdaptiveVec Graph Index (Pure FP32)
    std::cout << "\n[*] Building Master AdaptiveVec Index (Pure FP32 vectors)...\n";
    AdaptiveHNSWIndex master_index(dim, SpaceType::L2, true, cfg);
    auto t0_ib = std::chrono::high_resolution_clock::now();
    master_index.batch_insert(base_data.data(), n_base, 12);
    auto t1_ib = std::chrono::high_resolution_clock::now();
    std::cout << "    AdaptiveVec Master Graph constructed in "
              << std::chrono::duration<double>(t1_ib - t0_ib).count() << " s (Total Edges: "
              << master_index.get_total_edges() << ")\n";

    // Strategies to Evaluate
    std::vector<std::pair<std::string, AllocationStrategy>> strategies = {
        {"1_Canonical_FP32", AllocationStrategy::UniformFP32},
        {"2_Uniform_SQ8", AllocationStrategy::UniformSQ8},
        {"3_Uniform_SQ4", AllocationStrategy::UniformSQ4},
        {"4_Random_Mixed", AllocationStrategy::Random},
        {"5_ReconError_Mixed", AllocationStrategy::ReconstructionError},
        {"6_Geometry_Mixed", AllocationStrategy::Geometry},
        {"7_Routing_Mixed", AllocationStrategy::RoutingImportance},
        {"8_CombinedSensitivity_Mixed", AllocationStrategy::CombinedSensitivity}
    };

    double test_budget = 0.40; // 40% of FP32 raw vector budget

    for (const auto& [strat_name, strat_enum] : strategies) {
        std::cout << "\n---> Strategy: " << strat_name << " (Target Budget Ratio: " << test_budget << ")\n";
        auto eval_index = master_index.clone();
        eval_index->policy.keep_raw_vectors = false; // pure mixed-storage

        double budget = (strat_enum == AllocationStrategy::UniformFP32) ? 1.0 :
                        ((strat_enum == AllocationStrategy::UniformSQ8) ? 0.25 :
                        ((strat_enum == AllocationStrategy::UniformSQ4) ? 0.125 : test_budget));

        PrecisionDistribution pdist = eval_index->allocate_mixed_precision(strat_enum, budget);
        MemoryBreakdown mb = eval_index->get_memory_breakdown();

        StrategyEvaluationResult res;
        res.dataset_name = name;
        res.strategy_name = strat_name;
        res.budget_ratio = budget;
        res.n_fp32 = pdist.n_fp32;
        res.n_sq8 = pdist.n_sq8;
        res.n_sq4 = pdist.n_sq4;
        res.vector_storage_bytes = mb.resident_fp32_bytes + mb.allocated_quantized_capacity_bytes;
        res.vector_savings_pct = pdist.savings_ratio * 100.0;
        res.total_index_ram_mb = mb.total_owned_bytes / (1024.0 * 1024.0);

        std::cout << "     Distribution: FP32=" << pdist.n_fp32 << " (" << std::fixed << std::setprecision(1) << (pdist.n_fp32 * 100.0 / n_base) << "%), "
                  << "SQ8=" << pdist.n_sq8 << " (" << (pdist.n_sq8 * 100.0 / n_base) << "%), "
                  << "SQ4=" << pdist.n_sq4 << " (" << (pdist.n_sq4 * 100.0 / n_base) << "%)\n";
        std::cout << "     Vector Storage: " << (res.vector_storage_bytes / (1024.0 * 1024.0)) << " MB "
                  << "(-" << res.vector_savings_pct << "% vs FP32) | Total Index RAM: "
                  << res.total_index_ram_mb << " MB\n";

        for (int ef : {16, 32, 64, 128}) {
            BenchmarkQueryMetrics qm = evaluate_query_performance(*eval_index, query_data, n_queries, dim, gt, ef);
            res.sweeps.push_back(qm);
            std::cout << "     ef=" << ef
                      << " | R@10: " << std::fixed << std::setprecision(4) << qm.recall10
                      << " | R@1: " << qm.recall1
                      << " | MT-QPS: " << std::fixed << std::setprecision(1) << qm.mt_qps
                      << " | ST-QPS: " << qm.st_qps
                      << " | Lat(p50/p95/p99): " << qm.p50_us << "/" << qm.p95_us << "/" << qm.p99_us << " us"
                      << " | Hops: " << qm.avg_hops << std::endl;
        }
        all_results.push_back(res);
    }
}

int main() {
    std::cout << "========================================================================" << std::endl;
    std::cout << "  ADAPTIVEVEC PHASE 2.1: FULL BENCHMARK SUITE & RETRIEVAL RECOVERY     " << std::endl;
    std::cout << "========================================================================" << std::endl;

    std::vector<StrategyEvaluationResult> all_results;

    // 1. Synthetic Heterogeneous Manifold (64D)
    {
        size_t n_base = 50000, n_queries = 1000, dim = 64;
        std::vector<float> base_data, query_data;
        std::vector<std::vector<uint32_t>> gt;
        generate_synthetic_data(n_base, n_queries, dim, base_data, query_data, gt);
        run_suite_for_dataset("Synthetic-Multi-Cluster-64D", base_data, n_base, query_data, n_queries, dim, gt, all_results);
    }

    // 2. SIFT-100K (128D)
    {
        size_t dim = 0, n_base = 0, q_dim = 0, n_queries = 0;
        std::vector<float> base_data = load_fvecs("data/sift_subset_100k.fvecs", 100000, dim, n_base);
        std::vector<float> query_data = load_fvecs("data/sift_query.fvecs", 1000, q_dim, n_queries);
        auto gt = load_ivecs("data/sift_subset_groundtruth.ivecs", n_queries);
        if (!base_data.empty() && !query_data.empty() && !gt.empty()) {
            run_suite_for_dataset("SIFT-100K-128D", base_data, n_base, query_data, n_queries, dim, gt, all_results);
        }
    }

    // 3. Transformer-384D (MiniLM-L6-v2)
    {
        size_t dim = 0, n_base = 0, q_dim = 0, n_queries = 0;
        std::vector<float> base_data = load_fvecs("data/transformer_384d/transformer_384d_base.fvecs", 57893, dim, n_base);
        std::vector<float> query_data = load_fvecs("data/transformer_384d/transformer_384d_query.fvecs", 1000, q_dim, n_queries);
        auto gt = load_ivecs("data/transformer_384d/transformer_384d_groundtruth.ivecs", n_queries);
        if (!base_data.empty() && !query_data.empty() && !gt.empty()) {
            run_suite_for_dataset("Transformer-384D", base_data, n_base, query_data, n_queries, dim, gt, all_results);
        }
    }

    // 4. SIFT-500K (128D)
    {
        size_t dim = 0, n_base = 0, q_dim = 0, n_queries = 0;
        std::vector<float> base_data = load_fvecs("data/sift_base_1m.fvecs", 500000, dim, n_base);
        std::vector<float> query_data = load_fvecs("data/sift_query.fvecs", 1000, q_dim, n_queries);
        auto gt = load_ivecs("data/sift_groundtruth_500k.ivecs", n_queries);
        if (!base_data.empty() && !query_data.empty() && !gt.empty()) {
            run_suite_for_dataset("SIFT-500K-128D", base_data, n_base, query_data, n_queries, dim, gt, all_results);
        }
    }

    // Export JSON Artifacts
    std::string json_path = "experiments/phase_2_1_recovered_benchmarks.json";
    std::ofstream out(json_path);
    out << "{\n  \"evaluation_timestamp\": \"2026-11-20T00:36:00Z\",\n";
    out << "  \"hardware\": \"Intel Core 5 210H (AVX2/FMA, 12 Threads)\",\n";
    out << "  \"results\": [\n";

    for (size_t i = 0; i < all_results.size(); ++i) {
        const auto& r = all_results[i];
        out << "    {\n";
        out << "      \"dataset\": \"" << r.dataset_name << "\",\n";
        out << "      \"strategy\": \"" << r.strategy_name << "\",\n";
        out << "      \"budget_ratio\": " << r.budget_ratio << ",\n";
        out << "      \"n_fp32\": " << r.n_fp32 << ",\n";
        out << "      \"n_sq8\": " << r.n_sq8 << ",\n";
        out << "      \"n_sq4\": " << r.n_sq4 << ",\n";
        out << "      \"vector_storage_bytes\": " << r.vector_storage_bytes << ",\n";
        out << "      \"vector_savings_pct\": " << r.vector_savings_pct << ",\n";
        out << "      \"total_index_ram_mb\": " << r.total_index_ram_mb << ",\n";
        out << "      \"sweeps\": [\n";
        for (size_t j = 0; j < r.sweeps.size(); ++j) {
            const auto& sw = r.sweeps[j];
            out << "        {\"ef\": " << sw.ef << ", \"recall1\": " << sw.recall1
                << ", \"recall10\": " << sw.recall10 << ", \"recall100\": " << sw.recall100
                << ", \"mt_qps\": " << sw.mt_qps << ", \"st_qps\": " << sw.st_qps
                << ", \"p50_us\": " << sw.p50_us << ", \"p95_us\": " << sw.p95_us
                << ", \"p99_us\": " << sw.p99_us << ", \"avg_hops\": " << sw.avg_hops << "}"
                << (j + 1 < r.sweeps.size() ? "," : "") << "\n";
        }
        out << "      ]\n";
        out << "    }" << (i + 1 < all_results.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::cout << "\n[+] Benchmark complete! Saved to " << json_path << std::endl;

    return 0;
}
