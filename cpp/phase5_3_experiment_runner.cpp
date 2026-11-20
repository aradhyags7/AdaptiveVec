#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"
#include "budget_allocator.hpp"
#include "hnswlib/hnswlib.h"
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

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <psapi.h>
#endif

using namespace adaptivevec;

struct ProcessMemorySnap {
    double working_set_mb = 0.0;
    double peak_working_set_mb = 0.0;
    double private_mb = 0.0;
};

inline ProcessMemorySnap get_proc_mem() {
    ProcessMemorySnap snap;
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

struct SearchEvalResult {
    int ef_search = 64;
    double recall_at_1 = 0.0;
    double recall_at_10 = 0.0;
    double recall_at_100 = 0.0;
    double dense_query_recall = 0.0;
    double boundary_query_recall = 0.0;
    double qps = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double mean_upper_dist_evals = 0.0;
    double mean_layer0_dist_evals = 0.0;
    double mean_total_dist_evals = 0.0;
};

struct DetailedMemoryReport {
    size_t logical_edges = 0;
    size_t logical_edge_bytes = 0;
    size_t allocated_adj_capacity_bytes = 0;
    size_t unused_adj_capacity_bytes = 0;
    size_t per_node_metadata_bytes = 0;
    size_t vector_storage_bytes = 0;
    size_t search_workspace_bytes = 0;
    size_t total_owned_graph_bytes = 0;
    size_t total_owned_index_bytes = 0;
    double working_set_mb = 0.0;
    double peak_working_set_mb = 0.0;
};

struct ComprehensiveResult {
    std::string experiment_tag;
    std::string scheme_name;
    std::string dataset_name;
    size_t n_vectors = 0;
    size_t dim = 0;
    
    // Budget & Graph Topography
    long long assigned_degree_budget = 0;
    size_t total_edges = 0;
    size_t layer0_edges = 0;
    size_t upper_edges = 0;
    double mean_degree_l0 = 0.0;
    double in_degree_var = 0.0;
    uint32_t max_in_degree = 0;
    double build_time_s = 0.0;
    
    // Heuristic & Fallback
    uint64_t fallback_invocations = 0;
    double fallback_rate_pct = 0.0;
    
    // Memory
    DetailedMemoryReport mem;
    
    // Search Sweeps
    std::vector<SearchEvalResult> sweeps;
    
    // Primary Search (efSearch = 64)
    SearchEvalResult primary_eval;
};

// Evaluate Search on AdaptiveHNSWIndex
SearchEvalResult evaluate_search(
    AdaptiveHNSWIndex& index,
    const std::vector<float>& queries,
    size_t nq,
    size_t dim,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    const std::vector<uint8_t>& is_boundary_query,
    int ef_search
) {
    SearchEvalResult res;
    res.ef_search = ef_search;

    std::vector<double> r1(nq, 0.0), r10(nq, 0.0), r100(nq, 0.0);
    std::vector<double> latencies_ms(nq, 0.0);
    uint64_t total_upper_evals = 0, total_layer0_evals = 0;
    double dense_matches = 0.0, boundary_matches = 0.0;
    size_t n_dense = 0, n_boundary = 0;

    auto s0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_upper_evals, total_layer0_evals, dense_matches, boundary_matches, n_dense, n_boundary)
    for (size_t q = 0; q < nq; ++q) {
        auto q_start = std::chrono::high_resolution_clock::now();
        QuerySearchStats stats;
        auto results = index.search(&queries[q * dim], 100, ef_search, stats);
        auto q_end = std::chrono::high_resolution_clock::now();
        latencies_ms[q] = std::chrono::duration<double, std::milli>(q_end - q_start).count();
        total_upper_evals += stats.upper_layer_dist_evals;
        total_layer0_evals += stats.layer0_dist_evals;

        std::unordered_set<uint32_t> gt1, gt10, gt100;
        if (q < ground_truth.size()) {
            for (size_t k = 0; k < 1 && k < ground_truth[q].size(); ++k) gt1.insert(ground_truth[q][k]);
            for (size_t k = 0; k < 10 && k < ground_truth[q].size(); ++k) gt10.insert(ground_truth[q][k]);
            for (size_t k = 0; k < 100 && k < ground_truth[q].size(); ++k) gt100.insert(ground_truth[q][k]);
        }

        size_t m1 = 0, m10 = 0, m100 = 0;
        for (size_t k = 0; k < results.size(); ++k) {
            tableint cid = results[k].id;
            if (k < 1 && gt1.count(cid)) m1++;
            if (k < 10 && gt10.count(cid)) m10++;
            if (k < 100 && gt100.count(cid)) m100++;
        }
        r1[q] = (double)m1 / 1.0;
        r10[q] = (double)m10 / 10.0;
        r100[q] = (double)m100 / 100.0;

        if (q < is_boundary_query.size() && is_boundary_query[q]) {
            boundary_matches += (double)m10 / 10.0;
            n_boundary++;
        } else {
            dense_matches += (double)m10 / 10.0;
            n_dense++;
        }
    }
    auto s1 = std::chrono::high_resolution_clock::now();

    double total_s = std::chrono::duration<double>(s1 - s0).count();
    res.qps = (double)nq / total_s;
    res.recall_at_1 = std::accumulate(r1.begin(), r1.end(), 0.0) / (double)nq;
    res.recall_at_10 = std::accumulate(r10.begin(), r10.end(), 0.0) / (double)nq;
    res.recall_at_100 = std::accumulate(r100.begin(), r100.end(), 0.0) / (double)nq;
    res.dense_query_recall = (n_dense > 0) ? (dense_matches / (double)n_dense) : res.recall_at_10;
    res.boundary_query_recall = (n_boundary > 0) ? (boundary_matches / (double)n_boundary) : res.recall_at_10;
    res.mean_upper_dist_evals = (double)total_upper_evals / (double)nq;
    res.mean_layer0_dist_evals = (double)total_layer0_evals / (double)nq;
    res.mean_total_dist_evals = (double)(total_upper_evals + total_layer0_evals) / (double)nq;

    std::sort(latencies_ms.begin(), latencies_ms.end());
    res.p50_latency_ms = latencies_ms[nq / 2];
    res.p95_latency_ms = latencies_ms[(size_t)(nq * 0.95)];
    res.p99_latency_ms = latencies_ms[(size_t)(nq * 0.99)];

    return res;
}

// Compute Detailed Memory Report
DetailedMemoryReport compute_memory_report(const AdaptiveHNSWIndex& index, size_t n, size_t dim) {
    DetailedMemoryReport rep;
    rep.logical_edges = index.get_total_edges();
    rep.logical_edge_bytes = rep.logical_edges * sizeof(tableint);
    
    // Adjacency capacity & container headers
    size_t total_adj_cap_entries = 0;
    size_t container_header_bytes = index.graphs.capacity() * sizeof(std::vector<std::vector<tableint>>);
    for (const auto& layer : index.graphs) {
        container_header_bytes += layer.capacity() * sizeof(std::vector<tableint>);
        for (const auto& adj : layer) {
            total_adj_cap_entries += adj.capacity();
        }
    }
    rep.allocated_adj_capacity_bytes = total_adj_cap_entries * sizeof(tableint) + container_header_bytes;
    rep.unused_adj_capacity_bytes = (total_adj_cap_entries >= rep.logical_edges)
        ? ((total_adj_cap_entries - rep.logical_edges) * sizeof(tableint)) : 0;
        
    rep.per_node_metadata_bytes = n * (sizeof(int) + sizeof(NodeParams) + sizeof(uint32_t) + sizeof(uint8_t) + sizeof(float));
    rep.vector_storage_bytes = n * dim * sizeof(float);
    rep.search_workspace_bytes = 64 * 1024; // VisitedPool buffer
    
    rep.total_owned_graph_bytes = rep.allocated_adj_capacity_bytes + rep.per_node_metadata_bytes;
    rep.total_owned_index_bytes = rep.total_owned_graph_bytes + rep.vector_storage_bytes + rep.search_workspace_bytes;
    
    ProcessMemorySnap p = get_proc_mem();
    rep.working_set_mb = p.working_set_mb;
    rep.peak_working_set_mb = p.peak_working_set_mb;
    return rep;
}

// Pre-compute Offline Geometric Signals (MLE LID, Density, Sparsity)
std::vector<GeometrySignal> precompute_geometry_signals(
    const std::vector<float>& data, size_t n, size_t dim, int k_probe = 30
) {
    std::cout << "\n>>> Precomputing Offline Geometry Signals (N=" << n << ", D=" << dim << ", k=" << k_probe << ")..." << std::endl;
    std::vector<GeometrySignal> signals(n);

    // Build lightweight sampling index for fast k-NN estimation
    PolicyConfig sample_cfg;
    sample_cfg.m_base = 12;
    sample_cfg.ef_c_base = 60;
    sample_cfg.keep_pruned_connections = false; // Fast build
    AdaptiveHNSWIndex sample_index(dim, SpaceType::L2, false, sample_cfg);
    sample_index.reserve(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        sample_index.insert(&data[i * dim]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  Interim Probe Index Built in " << std::fixed << std::setprecision(2)
              << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;

    // Estimate signals in parallel
    #pragma omp parallel for schedule(dynamic, 128)
    for (size_t i = 0; i < n; ++i) {
        QuerySearchStats stats;
        auto neighbors = sample_index.search(&data[i * dim], k_probe + 1, k_probe * 2, stats);
        std::vector<float> dists;
        dists.reserve(k_probe);
        for (const auto& nb : neighbors) {
            if (nb.id != (tableint)i) {
                dists.push_back(std::sqrt(nb.dist));
            }
            if (dists.size() >= (size_t)k_probe) break;
        }

        if (dists.size() >= 5) {
            std::sort(dists.begin(), dists.end());
            float r_k = std::max(1e-6f, dists.back());
            double log_sum = 0.0;
            for (size_t j = 0; j < dists.size() - 1; ++j) {
                float r_j = std::max(1e-6f, dists[j]);
                log_sum += std::log(r_k / r_j);
            }
            signals[i].lid = (log_sum > 0.0) ? (float)((dists.size() - 1) / log_sum) : 10.0f;
            
            // Sparsity = ln(mean dist)
            double mean_d = std::accumulate(dists.begin(), dists.end(), 0.0) / (double)dists.size();
            signals[i].sparsity = (float)std::log(std::max(1e-6, mean_d));
            signals[i].density = (float)(1.0 / std::max(1e-6, mean_d));
        } else {
            signals[i].lid = 12.0f;
            signals[i].sparsity = 5.0f;
            signals[i].density = 0.01f;
        }
    }

    BudgetAllocator::standardize_signals(signals, 0.5f, 0.5f);
    std::cout << "  Geometry Signals Precomputed and Standardized." << std::endl;
    return signals;
}

// Run a Native AdaptiveHNSWIndex Experiment
ComprehensiveResult run_adaptivevec_experiment(
    const std::string& experiment_tag,
    const std::string& scheme_name,
    const std::string& dataset_name,
    const std::vector<float>& data,
    size_t n,
    size_t dim,
    const std::vector<float>& queries,
    size_t nq,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    const std::vector<uint8_t>& is_boundary_query,
    PolicyConfig policy,
    bool is_adaptive,
    const std::vector<int>& precomputed_degrees = {}
) {
    ComprehensiveResult cr;
    cr.experiment_tag = experiment_tag;
    cr.scheme_name = scheme_name;
    cr.dataset_name = dataset_name;
    cr.n_vectors = n;
    cr.dim = dim;

    std::cout << "\n>>> [" << experiment_tag << "] " << scheme_name << " on " << dataset_name << " (N=" << n << ")..." << std::endl;

    AdaptiveHNSWIndex index(dim, SpaceType::L2, is_adaptive, policy);
    index.reserve(n);

    // Apply precomputed degrees if provided
    if (!precomputed_degrees.empty()) {
        std::vector<NodeParams> params(n);
        long long budget_sum = 0;
        for (size_t i = 0; i < n; ++i) {
            int m_val = precomputed_degrees[i];
            budget_sum += m_val;
            params[i] = {m_val, m_val, 2 * m_val, policy.ef_c_base, 0.0f, 0.0f, 0.0f};
        }
        index.preassigned_node_params = std::move(params);
        cr.assigned_degree_budget = budget_sum;
    } else {
        cr.assigned_degree_budget = (long long)n * policy.m_base;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        index.insert(&data[i * dim]);
        if ((i + 1) % 50000 == 0 || (i + 1) == n) {
            std::cout << "  Inserted " << (i + 1) << " / " << n << std::endl << std::flush;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    cr.build_time_s = std::chrono::duration<double>(t1 - t0).count();

    cr.total_edges = index.get_total_edges();
    cr.layer0_edges = index.get_layer0_edges();
    cr.upper_edges = (cr.total_edges >= cr.layer0_edges) ? (cr.total_edges - cr.layer0_edges) : 0;
    cr.mean_degree_l0 = (double)cr.layer0_edges / (double)n;

    // In-degree variance
    double sum_in = 0, sum_in_sq = 0;
    uint32_t max_in = 0;
    for (size_t u = 0; u < n; ++u) {
        uint32_t in_d = (u < index.in_degrees.size()) ? index.in_degrees[u] : 0;
        sum_in += in_d;
        sum_in_sq += (double)in_d * in_d;
        if (in_d > max_in) max_in = in_d;
    }
    cr.max_in_degree = max_in;
    double mean_in = sum_in / (double)n;
    cr.in_degree_var = (sum_in_sq / (double)n) - (mean_in * mean_in);

    // Fallback stats
    cr.fallback_invocations = index.fallback_retention_invocations.load();
    uint64_t total_h = index.heuristic_calls.load();
    cr.fallback_rate_pct = (total_h > 0) ? (100.0 * (double)cr.fallback_invocations / (double)total_h) : 0.0;

    // Detailed Memory
    cr.mem = compute_memory_report(index, n, dim);

    // Primary Evaluation (efSearch = 64)
    cr.primary_eval = evaluate_search(index, queries, nq, dim, ground_truth, is_boundary_query, 64);

    // Search Depth Sweeps (16, 32, 64, 128, 256)
    const int sweeps_ef[] = {16, 32, 64, 128, 256};
    for (int ef : sweeps_ef) {
        if (ef == 64) {
            cr.sweeps.push_back(cr.primary_eval);
        } else {
            cr.sweeps.push_back(evaluate_search(index, queries, nq, dim, ground_truth, is_boundary_query, ef));
        }
    }

    std::cout << "  Result: Rec@10=" << std::fixed << std::setprecision(2) << cr.primary_eval.recall_at_10 * 100.0
              << "%, QPS=" << std::setprecision(1) << cr.primary_eval.qps
              << ", Dist=" << cr.primary_eval.mean_total_dist_evals
              << ", Edges=" << cr.total_edges
              << ", GraphMem=" << std::setprecision(2) << (double)cr.mem.total_owned_graph_bytes / (1024.0 * 1024.0) << " MB"
              << ", DenseRec=" << std::setprecision(2) << cr.primary_eval.dense_query_recall * 100.0 << "%"
              << ", BoundRec=" << cr.primary_eval.boundary_query_recall * 100.0 << "%" << std::endl;

    return cr;
}

// Evaluate Official hnswlib Baseline
ComprehensiveResult run_official_hnswlib_baseline(
    const std::string& experiment_tag,
    const std::string& dataset_name,
    const std::vector<float>& data,
    size_t n,
    size_t dim,
    const std::vector<float>& queries,
    size_t nq,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    const std::vector<uint8_t>& is_boundary_query,
    int M = 16,
    int ef_construction = 120
) {
    ComprehensiveResult cr;
    cr.experiment_tag = experiment_tag;
    cr.scheme_name = "Official hnswlib (nmslib/hnswlib v0.8.0)";
    cr.dataset_name = dataset_name;
    cr.n_vectors = n;
    cr.dim = dim;
    cr.assigned_degree_budget = (long long)n * M;

    std::cout << "\n>>> [" << experiment_tag << "] Official hnswlib on " << dataset_name << " (M=" << M << ", efC=" << ef_construction << ")..." << std::endl;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, n, M, ef_construction);

    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < n; ++i) {
        index.addPoint(&data[i * dim], (hnswlib::labeltype)i);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    cr.build_time_s = std::chrono::duration<double>(t1 - t0).count();

    // In hnswlib, exact edge count is obtained by traversing linkLists
    size_t total_edges = 0, layer0_edges = 0;
    for (size_t i = 0; i < n; ++i) {
        hnswlib::linklistsizeint* data_ll0 = (hnswlib::linklistsizeint*)index.get_linklist0(i);
        size_t count0 = index.getListCount(data_ll0);
        layer0_edges += count0;
        total_edges += count0;
        for (int l = 1; l <= index.element_levels_[i]; ++l) {
            hnswlib::linklistsizeint* data_l = (hnswlib::linklistsizeint*)index.get_linklist_at_level(i, l);
            total_edges += index.getListCount(data_l);
        }
    }
    cr.total_edges = total_edges;
    cr.layer0_edges = layer0_edges;
    cr.upper_edges = total_edges - layer0_edges;
    cr.mean_degree_l0 = (double)layer0_edges / (double)n;

    // Memory estimation
    cr.mem.logical_edges = total_edges;
    cr.mem.logical_edge_bytes = total_edges * sizeof(tableint);
    cr.mem.allocated_adj_capacity_bytes = (n * 2 * M + n * M) * sizeof(tableint);
    cr.mem.vector_storage_bytes = n * dim * sizeof(float);
    cr.mem.total_owned_graph_bytes = cr.mem.allocated_adj_capacity_bytes + n * sizeof(int);
    cr.mem.total_owned_index_bytes = cr.mem.total_owned_graph_bytes + cr.mem.vector_storage_bytes;
    ProcessMemorySnap p = get_proc_mem();
    cr.mem.working_set_mb = p.working_set_mb;
    cr.mem.peak_working_set_mb = p.peak_working_set_mb;

    // Sweep evaluations
    const int sweeps_ef[] = {16, 32, 64, 128, 256};
    for (int ef : sweeps_ef) {
        index.setEf(ef);
        SearchEvalResult res;
        res.ef_search = ef;

        std::vector<double> r1(nq, 0.0), r10(nq, 0.0), r100(nq, 0.0);
        std::vector<double> latencies_ms(nq, 0.0);
        double dense_matches = 0.0, boundary_matches = 0.0;
        size_t n_dense = 0, n_boundary = 0;

        auto s0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, 64) reduction(+:dense_matches, boundary_matches, n_dense, n_boundary)
        for (size_t q = 0; q < nq; ++q) {
            auto q_start = std::chrono::high_resolution_clock::now();
            auto q_res = index.searchKnn(&queries[q * dim], 100);
            auto q_end = std::chrono::high_resolution_clock::now();
            latencies_ms[q] = std::chrono::duration<double, std::milli>(q_end - q_start).count();

            std::unordered_set<uint32_t> gt1, gt10, gt100;
            if (q < ground_truth.size()) {
                for (size_t k = 0; k < 1 && k < ground_truth[q].size(); ++k) gt1.insert(ground_truth[q][k]);
                for (size_t k = 0; k < 10 && k < ground_truth[q].size(); ++k) gt10.insert(ground_truth[q][k]);
                for (size_t k = 0; k < 100 && k < ground_truth[q].size(); ++k) gt100.insert(ground_truth[q][k]);
            }

            std::vector<uint32_t> results;
            while (!q_res.empty()) {
                results.push_back(q_res.top().second);
                q_res.pop();
            }
            std::reverse(results.begin(), results.end());

            size_t m1 = 0, m10 = 0, m100 = 0;
            for (size_t k = 0; k < results.size(); ++k) {
                uint32_t cid = results[k];
                if (k < 1 && gt1.count(cid)) m1++;
                if (k < 10 && gt10.count(cid)) m10++;
                if (k < 100 && gt100.count(cid)) m100++;
            }
            r1[q] = (double)m1 / 1.0;
            r10[q] = (double)m10 / 10.0;
            r100[q] = (double)m100 / 100.0;

            if (q < is_boundary_query.size() && is_boundary_query[q]) {
                boundary_matches += (double)m10 / 10.0;
                n_boundary++;
            } else {
                dense_matches += (double)m10 / 10.0;
                n_dense++;
            }
        }
        auto s1 = std::chrono::high_resolution_clock::now();

        double total_s = std::chrono::duration<double>(s1 - s0).count();
        res.qps = (double)nq / total_s;
        res.recall_at_1 = std::accumulate(r1.begin(), r1.end(), 0.0) / (double)nq;
        res.recall_at_10 = std::accumulate(r10.begin(), r10.end(), 0.0) / (double)nq;
        res.recall_at_100 = std::accumulate(r100.begin(), r100.end(), 0.0) / (double)nq;
        res.dense_query_recall = (n_dense > 0) ? (dense_matches / (double)n_dense) : res.recall_at_10;
        res.boundary_query_recall = (n_boundary > 0) ? (boundary_matches / (double)n_boundary) : res.recall_at_10;
        
        // hnswlib does not expose an internal dist evaluation counter without recompiling
        res.mean_total_dist_evals = 0.0;

        std::sort(latencies_ms.begin(), latencies_ms.end());
        res.p50_latency_ms = latencies_ms[nq / 2];
        res.p95_latency_ms = latencies_ms[(size_t)(nq * 0.95)];
        res.p99_latency_ms = latencies_ms[(size_t)(nq * 0.99)];

        cr.sweeps.push_back(res);
        if (ef == 64) cr.primary_eval = res;
    }

    std::cout << "  Result: Rec@10=" << std::fixed << std::setprecision(2) << cr.primary_eval.recall_at_10 * 100.0
              << "%, QPS=" << std::setprecision(1) << cr.primary_eval.qps
              << ", Edges=" << cr.total_edges
              << ", Build=" << cr.build_time_s << "s" << std::endl;

    return cr;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  AdaptiveVec Phase 5.3: Budget-Constrained Geometric Allocation  " << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    // Load SIFT-100K
    std::string sift_base_path = "data/sift_subset_100k.fvecs";
    std::string sift_query_path = "data/sift_query.fvecs";
    std::string sift_gt_path = "data/sift_subset_groundtruth.ivecs";

    size_t sift_n = 0, sift_dim = 0;
    std::vector<float> sift_data;
    load_fvecs(sift_base_path, sift_data, sift_n, sift_dim);
    size_t sift_nq = 0, sift_qdim = 0;
    std::vector<float> sift_queries;
    load_fvecs(sift_query_path, sift_queries, sift_nq, sift_qdim);
    std::vector<std::vector<uint32_t>> sift_gt;
    load_ivecs(sift_gt_path, sift_gt);

    std::cout << "Loaded SIFT-100K: N=" << sift_n << ", D=" << sift_dim << ", NQ=" << sift_nq << std::endl;

    // Pre-determine Boundary vs Dense queries on SIFT-100K (distance to 10th-NN >= 200.0)
    std::vector<uint8_t> sift_is_boundary(sift_nq, 0);
    for (size_t q = 0; q < sift_nq && q < sift_gt.size(); ++q) {
        if (sift_gt[q].size() >= 10) {
            uint32_t nn10 = sift_gt[q][9];
            float d2 = 0.0f;
            for (size_t d = 0; d < sift_dim; ++d) {
                float diff = sift_queries[q * sift_dim + d] - sift_data[nn10 * sift_dim + d];
                d2 += diff * diff;
            }
            if (std::sqrt(d2) >= 200.0f) {
                sift_is_boundary[q] = 1;
            }
        }
    }

    // Precompute SIFT-100K Geometry Signals
    std::vector<GeometrySignal> sift_signals = precompute_geometry_signals(sift_data, sift_n, sift_dim, 30);

    // =============================================================
    // STAGE 1 — INDEPENDENT COMPONENT ABLATIONS (SIFT-100K)
    // =============================================================
    std::cout << "\n=============================================================" << std::endl;
    std::cout << "  STAGE 1: Independent Component Ablations (SIFT-100K)        " << std::endl;
    std::cout << "=============================================================" << std::endl;

    std::vector<ComprehensiveResult> stage1_results;

    // Comp 1: Native Uniform without Fallback (Strict RNG, official hnswlib parity)
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.ef_c_base = 120;
        cfg.keep_pruned_connections = false; // Strict RNG
        cfg.enable_layer_scaling = false;
        cfg.enable_hubness_regulation = false;
        stage1_results.push_back(run_adaptivevec_experiment(
            "Comp1-Native-StrictRNG", "Native Uniform (Strict RNG, keep_pruned=false)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
            cfg, false
        ));
    }

    // Comp 2: Native Uniform with Fallback (Malkov Algorithm 4)
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.ef_c_base = 120;
        cfg.keep_pruned_connections = true; // Fallback loop
        cfg.enable_layer_scaling = false;
        cfg.enable_hubness_regulation = false;
        stage1_results.push_back(run_adaptivevec_experiment(
            "Comp2-Native-Fallback", "Native Uniform with Fallback Retention (keep_pruned=true)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
            cfg, false
        ));
    }

    // Comp 3: Comp 1 + Hubness Regulation (mu = 0.15)
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.ef_c_base = 120;
        cfg.keep_pruned_connections = false;
        cfg.enable_layer_scaling = false;
        cfg.enable_hubness_regulation = true;
        cfg.hubness_mu = 0.15f;
        cfg.use_streaming_welford = true;
        stage1_results.push_back(run_adaptivevec_experiment(
            "Comp3-StrictRNG-Hubness", "Native Strict RNG + Hubness Regulation (mu=0.15)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
            cfg, false
        ));
    }

    // Comp 4: Comp 1 + Adaptive efConstruction (efC in [40, 220])
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.ef_c_base = 120;
        cfg.ef_c_min = 40;
        cfg.ef_c_max = 220;
        cfg.keep_pruned_connections = false;
        cfg.enable_layer_scaling = false;
        cfg.enable_hubness_regulation = false;
        stage1_results.push_back(run_adaptivevec_experiment(
            "Comp4-StrictRNG-Adaptive-efC", "Native Strict RNG + Adaptive efConstruction",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
            cfg, true
        ));
    }

    // Comp 5: Comp 1 + Adaptive Layer-0 Degree (M in [8, 24], Monolithic)
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.m_min = 8;
        cfg.m_max = 24;
        cfg.ef_c_base = 120;
        cfg.keep_pruned_connections = false;
        cfg.enable_layer_scaling = true; // Monolithic
        cfg.lambda_layer = 0.75f;
        cfg.enable_hubness_regulation = false;
        stage1_results.push_back(run_adaptivevec_experiment(
            "Comp5-StrictRNG-Monolithic-AdaptiveM", "Native Strict RNG + Monolithic Adaptive Degree",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
            cfg, true
        ));
    }

    // Comp 6: Comp 1 + Layer Decoupled (beta_u = 0, beta_0 = 1.0)
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.m_min = 8;
        cfg.m_max = 24;
        cfg.ef_c_base = 120;
        cfg.keep_pruned_connections = false;
        cfg.enable_layer_scaling = false;
        cfg.enable_layer_decoupled = true;
        cfg.beta_layer0 = 1.0f;
        cfg.beta_upper = 0.0f;
        cfg.enable_hubness_regulation = false;
        stage1_results.push_back(run_adaptivevec_experiment(
            "Comp6-StrictRNG-Layer-Decoupled", "Native Strict RNG + Layer-Decoupled Adaptive Degree",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
            cfg, true
        ));
    }

    // Comp 7: Full Combined (Decoupled L0 + Strict RNG + Hubness + Adaptive efC)
    {
        PolicyConfig cfg;
        cfg.m_base = 16;
        cfg.m_min = 8;
        cfg.m_max = 24;
        cfg.ef_c_base = 120;
        cfg.ef_c_min = 40;
        cfg.ef_c_max = 220;
        cfg.keep_pruned_connections = false;
        cfg.enable_layer_scaling = false;
        cfg.enable_layer_decoupled = true;
        cfg.beta_layer0 = 1.0f;
        cfg.beta_upper = 0.0f;
        cfg.enable_hubness_regulation = true;
        cfg.hubness_mu = 0.15f;
        cfg.use_streaming_welford = true;
        stage1_results.push_back(run_adaptivevec_experiment(
            "Comp7-Full-Combined-Decoupled", "Full Combined Architecture (Layer-Decoupled + Strict RNG + Hubness)",
            "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
            cfg, true
        ));
    }

    // Export experiments/phase_5_3_component_ablation.json
    {
        std::ofstream out("experiments/phase_5_3_component_ablation.json");
        out << "{\n  \"experiment_name\": \"Phase 5.3 Component Ablation Suite\",\n  \"results\": [\n";
        for (size_t i = 0; i < stage1_results.size(); ++i) {
            const auto& r = stage1_results[i];
            out << "    {\n"
                << "      \"experiment_tag\": \"" << r.experiment_tag << "\",\n"
                << "      \"scheme_name\": \"" << r.scheme_name << "\",\n"
                << "      \"total_edges\": " << r.total_edges << ",\n"
                << "      \"layer0_edges\": " << r.layer0_edges << ",\n"
                << "      \"recall_at_10\": " << r.primary_eval.recall_at_10 << ",\n"
                << "      \"qps\": " << r.primary_eval.qps << ",\n"
                << "      \"mean_dist_evals\": " << r.primary_eval.mean_total_dist_evals << ",\n"
                << "      \"in_degree_var\": " << r.in_degree_var << ",\n"
                << "      \"fallback_rate_pct\": " << r.fallback_rate_pct << "\n"
                << "    }" << (i + 1 < stage1_results.size() ? "," : "") << "\n";
        }
        out << "  ]\n}\n";
        out.close();
        std::cout << "\n>>> Saved experiments/phase_5_3_component_ablation.json" << std::endl;
    }

    // =============================================================
    // STAGE 3 & 6 — BUDGET-NORMALIZED ALLOCATION SUITE (SIFT-100K)
    // =============================================================
    std::cout << "\n=============================================================" << std::endl;
    std::cout << "  STAGE 3 & 6: Budget-Normalized Allocation Suite (SIFT-100K) " << std::endl;
    std::cout << "=============================================================" << std::endl;

    std::vector<ComprehensiveResult> stage3_results;

    // Common decoupled configuration template
    PolicyConfig decoupled_template;
    decoupled_template.m_base = 16;
    decoupled_template.ef_c_base = 120;
    decoupled_template.keep_pruned_connections = false; // Strict RNG
    decoupled_template.enable_layer_scaling = false;
    decoupled_template.enable_layer_decoupled = true;
    decoupled_template.beta_layer0 = 1.0f;
    decoupled_template.beta_upper = 0.0f;
    decoupled_template.enable_hubness_regulation = true;
    decoupled_template.hubness_mu = 0.15f;
    decoupled_template.use_streaming_welford = true;

    // Approach A: Uniform Allocation (B0 = 1.6M exact)
    std::vector<int> alloc_A = BudgetAllocator::allocate_uniform(sift_n, 16);
    stage3_results.push_back(run_adaptivevec_experiment(
        "Approach-A-Uniform", "Approach A: Uniform Allocation (M=16, B0=1.6M)",
        "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
        decoupled_template, false, alloc_A
    ));

    // Approach B: Original AdaptiveVec (Unconstrained Clipped)
    std::vector<int> alloc_B = BudgetAllocator::allocate_original_adaptive(sift_signals, 16, 8, 24, 1.0f);
    stage3_results.push_back(run_adaptivevec_experiment(
        "Approach-B-Original-Clipped", "Approach B: Original AdaptiveVec (Unconstrained Clipped)",
        "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
        decoupled_template, true, alloc_B
    ));

    // Approach C: Normalized Linear Allocation (B0 = 1.6M exact)
    std::vector<int> alloc_C = BudgetAllocator::allocate_normalized_linear(sift_signals, 16, 8, 24, 1.0f);
    stage3_results.push_back(run_adaptivevec_experiment(
        "Approach-C-Normalized-Linear", "Approach C: Normalized Linear Allocation (Exact B0=1.6M)",
        "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
        decoupled_template, true, alloc_C
    ));

    // Approach D: Normalized Exponential Allocation (B0 = 1.6M exact)
    std::vector<int> alloc_D = BudgetAllocator::allocate_normalized_exponential(sift_signals, 16, 8, 24, 0.5f);
    stage3_results.push_back(run_adaptivevec_experiment(
        "Approach-D-Normalized-Exponential", "Approach D: Normalized Exponential Allocation (Exact B0=1.6M)",
        "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
        decoupled_template, true, alloc_D
    ));

    // Approach E: Geometry Quantile Allocation (5 Quintiles, B0 = 1.6M exact)
    std::vector<int> alloc_E = BudgetAllocator::allocate_quantile(sift_signals, 16);
    stage3_results.push_back(run_adaptivevec_experiment(
        "Approach-E-Quantile", "Approach E: Geometry Quantile Allocation (Exact B0=1.6M)",
        "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
        decoupled_template, true, alloc_E
    ));

    // Control 1: Randomized Budget Allocation (Shuffled alloc_D, seed 42)
    std::vector<int> alloc_R = BudgetAllocator::allocate_randomized(alloc_D, 42);
    stage3_results.push_back(run_adaptivevec_experiment(
        "Control-1-Randomized-Budget", "Control 1: Randomized Budget Allocation (Exact B0=1.6M, Seed 42)",
        "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
        decoupled_template, true, alloc_R
    ));

    // Control 2: Reversed Geometry Allocation (Inverted alloc_D)
    std::vector<int> alloc_Rev = BudgetAllocator::allocate_reversed(alloc_D, sift_signals);
    stage3_results.push_back(run_adaptivevec_experiment(
        "Control-2-Reversed-Geometry", "Control 2: Reversed Geometry Allocation (High M to Low Si)",
        "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
        decoupled_template, true, alloc_Rev
    ));

    // Control 3: Official hnswlib (nmslib/hnswlib v0.8.0)
    stage3_results.push_back(run_official_hnswlib_baseline(
        "Control-3-Official-hnswlib", "SIFT-100K",
        sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
        16, 120
    ));

    // Export experiments/phase_5_3_budget_allocation_ablation.json
    {
        std::ofstream out("experiments/phase_5_3_budget_allocation_ablation.json");
        out << "{\n  \"experiment_name\": \"Phase 5.3 Budget Allocation Ablation Suite\",\n  \"results\": [\n";
        for (size_t i = 0; i < stage3_results.size(); ++i) {
            const auto& r = stage3_results[i];
            out << "    {\n"
                << "      \"experiment_tag\": \"" << r.experiment_tag << "\",\n"
                << "      \"scheme_name\": \"" << r.scheme_name << "\",\n"
                << "      \"assigned_degree_budget\": " << r.assigned_degree_budget << ",\n"
                << "      \"total_edges\": " << r.total_edges << ",\n"
                << "      \"layer0_edges\": " << r.layer0_edges << ",\n"
                << "      \"mean_degree_l0\": " << r.mean_degree_l0 << ",\n"
                << "      \"recall_at_10\": " << r.primary_eval.recall_at_10 << ",\n"
                << "      \"dense_query_recall\": " << r.primary_eval.dense_query_recall << ",\n"
                << "      \"boundary_query_recall\": " << r.primary_eval.boundary_query_recall << ",\n"
                << "      \"qps\": " << r.primary_eval.qps << ",\n"
                << "      \"p50_latency_ms\": " << r.primary_eval.p50_latency_ms << ",\n"
                << "      \"p95_latency_ms\": " << r.primary_eval.p95_latency_ms << ",\n"
                << "      \"mean_dist_evals\": " << r.primary_eval.mean_total_dist_evals << ",\n"
                << "      \"in_degree_var\": " << r.in_degree_var << ",\n"
                << "      \"graph_memory_mb\": " << (double)r.mem.total_owned_graph_bytes / (1024.0 * 1024.0) << "\n"
                << "    }" << (i + 1 < stage3_results.size() ? "," : "") << "\n";
        }
        out << "  ]\n}\n";
        out.close();
        std::cout << "\n>>> Saved experiments/phase_5_3_budget_allocation_ablation.json" << std::endl;
    }

    // Export experiments/phase_5_3_matched_recall_results.json
    {
        std::ofstream out("experiments/phase_5_3_matched_recall_results.json");
        out << "{\n  \"experiment_name\": \"Phase 5.3 Matched Recall Search Sweeps\",\n  \"schemes\": [\n";
        for (size_t i = 0; i < stage3_results.size(); ++i) {
            const auto& r = stage3_results[i];
            out << "    {\n"
                << "      \"experiment_tag\": \"" << r.experiment_tag << "\",\n"
                << "      \"sweeps\": [\n";
            for (size_t s = 0; s < r.sweeps.size(); ++s) {
                const auto& sw = r.sweeps[s];
                out << "        {\"ef_search\": " << sw.ef_search
                    << ", \"recall_at_10\": " << sw.recall_at_10
                    << ", \"qps\": " << sw.qps
                    << ", \"mean_dist_evals\": " << sw.mean_total_dist_evals
                    << ", \"p95_ms\": " << sw.p95_latency_ms << "}"
                    << (s + 1 < r.sweeps.size() ? "," : "") << "\n";
            }
            out << "      ]\n    }" << (i + 1 < stage3_results.size() ? "," : "") << "\n";
        }
        out << "  ]\n}\n";
        out.close();
        std::cout << ">>> Saved experiments/phase_5_3_matched_recall_results.json" << std::endl;
    }

    // Export experiments/phase_5_3_matched_memory_results.json
    {
        std::ofstream out("experiments/phase_5_3_matched_memory_results.json");
        out << "{\n  \"experiment_name\": \"Phase 5.3 Matched Memory & Edge Budget Results\",\n  \"comparisons\": [\n";
        // Compare Approach A (Uniform) vs Approach D (Normalized Exponential) at identical budget 1.6M
        const auto& uni = stage3_results[0];
        const auto& exp = stage3_results[3];
        out << "    {\n"
            << "      \"baseline\": \"" << uni.experiment_tag << "\",\n"
            << "      \"baseline_assigned_budget\": " << uni.assigned_degree_budget << ",\n"
            << "      \"baseline_realized_edges\": " << uni.total_edges << ",\n"
            << "      \"baseline_recall_at_10\": " << uni.primary_eval.recall_at_10 << ",\n"
            << "      \"baseline_qps\": " << uni.primary_eval.qps << ",\n"
            << "      \"baseline_dist_evals\": " << uni.primary_eval.mean_total_dist_evals << ",\n"
            << "      \"baseline_graph_bytes\": " << uni.mem.total_owned_graph_bytes << ",\n"
            << "      \"candidate\": \"" << exp.experiment_tag << "\",\n"
            << "      \"candidate_assigned_budget\": " << exp.assigned_degree_budget << ",\n"
            << "      \"candidate_realized_edges\": " << exp.total_edges << ",\n"
            << "      \"candidate_recall_at_10\": " << exp.primary_eval.recall_at_10 << ",\n"
            << "      \"candidate_qps\": " << exp.primary_eval.qps << ",\n"
            << "      \"candidate_dist_evals\": " << exp.primary_eval.mean_total_dist_evals << ",\n"
            << "      \"candidate_graph_bytes\": " << exp.mem.total_owned_graph_bytes << ",\n"
            << "      \"budget_delta_pct\": " << (100.0 * (double)(exp.assigned_degree_budget - uni.assigned_degree_budget) / (double)uni.assigned_degree_budget) << ",\n"
            << "      \"recall_delta_pct\": " << (100.0 * (exp.primary_eval.recall_at_10 - uni.primary_eval.recall_at_10)) << ",\n"
            << "      \"qps_speedup_pct\": " << (100.0 * (exp.primary_eval.qps - uni.primary_eval.qps) / uni.primary_eval.qps) << ",\n"
            << "      \"dist_reduction_pct\": " << (100.0 * (uni.primary_eval.mean_total_dist_evals - exp.primary_eval.mean_total_dist_evals) / uni.primary_eval.mean_total_dist_evals) << ",\n"
            << "      \"graph_memory_delta_pct\": " << (100.0 * (double)((long long)exp.mem.total_owned_graph_bytes - (long long)uni.mem.total_owned_graph_bytes) / (double)uni.mem.total_owned_graph_bytes) << "\n"
            << "    }\n";
        out << "  ]\n}\n";
        out.close();
        std::cout << ">>> Saved experiments/phase_5_3_matched_memory_results.json" << std::endl;
    }

    // =============================================================
    // STAGE 7 — DETAILED MEMORY ACCOUNTING
    // =============================================================
    std::cout << "\n=============================================================" << std::endl;
    std::cout << "  STAGE 7: Detailed Physical Memory Accounting                " << std::endl;
    std::cout << "=============================================================" << std::endl;

    {
        std::ofstream out("experiments/phase_5_3_memory_accounting.json");
        out << "{\n  \"experiment_name\": \"Phase 5.3 Physical Memory Accounting\",\n  \"breakdowns\": [\n";
        for (size_t i = 0; i < stage3_results.size(); ++i) {
            const auto& r = stage3_results[i];
            const auto& m = r.mem;
            out << "    {\n"
                << "      \"scheme\": \"" << r.experiment_tag << "\",\n"
                << "      \"logical_edges\": " << m.logical_edges << ",\n"
                << "      \"logical_edge_bytes\": " << m.logical_edge_bytes << ",\n"
                << "      \"allocated_adj_capacity_bytes\": " << m.allocated_adj_capacity_bytes << ",\n"
                << "      \"unused_adj_capacity_bytes\": " << m.unused_adj_capacity_bytes << ",\n"
                << "      \"per_node_metadata_bytes\": " << m.per_node_metadata_bytes << ",\n"
                << "      \"vector_storage_bytes\": " << m.vector_storage_bytes << ",\n"
                << "      \"total_owned_graph_bytes\": " << m.total_owned_graph_bytes << ",\n"
                << "      \"total_owned_index_bytes\": " << m.total_owned_index_bytes << ",\n"
                << "      \"process_working_set_mb\": " << m.working_set_mb << ",\n"
                << "      \"peak_working_set_mb\": " << m.peak_working_set_mb << "\n"
                << "    }" << (i + 1 < stage3_results.size() ? "," : "") << "\n";
        }
        out << "  ]\n}\n";
        out.close();
        std::cout << ">>> Saved experiments/phase_5_3_memory_accounting.json" << std::endl;
    }

    // =============================================================
    // STAGE 5 — GEOMETRY UTILITY & BOUNDARY VALIDATION
    // =============================================================
    std::cout << "\n=============================================================" << std::endl;
    std::cout << "  STAGE 5: Geometry Feature Utility Analysis                  " << std::endl;
    std::cout << "=============================================================" << std::endl;

    std::vector<ComprehensiveResult> stage5_results;

    // Feature 1: LID-only Allocation
    std::vector<GeometrySignal> signals_lid_only = sift_signals;
    for (auto& s : signals_lid_only) s.combined_score = s.standardized_lid;
    std::vector<int> alloc_lid = BudgetAllocator::allocate_normalized_exponential(signals_lid_only, 16, 8, 24, 0.5f);
    stage5_results.push_back(run_adaptivevec_experiment(
        "Feature-LID-Only", "LID-Only Exponential Allocation (alpha=1.0, beta=0.0)",
        "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
        decoupled_template, true, alloc_lid
    ));

    // Feature 2: Sparsity-only Allocation
    std::vector<GeometrySignal> signals_spars_only = sift_signals;
    for (auto& s : signals_spars_only) s.combined_score = s.standardized_sparsity;
    std::vector<int> alloc_spars = BudgetAllocator::allocate_normalized_exponential(signals_spars_only, 16, 8, 24, 0.5f);
    stage5_results.push_back(run_adaptivevec_experiment(
        "Feature-Sparsity-Only", "Sparsity-Only Exponential Allocation (alpha=0.0, beta=1.0)",
        "SIFT-100K", sift_data, sift_n, sift_dim, sift_queries, sift_nq, sift_gt, sift_is_boundary,
        decoupled_template, true, alloc_spars
    ));

    // Feature 3: Combined (Already computed in stage3_results[3] Approach D)
    stage5_results.push_back(stage3_results[3]);

    // Export experiments/phase_5_3_geometry_utility_analysis.json
    {
        std::ofstream out("experiments/phase_5_3_geometry_utility_analysis.json");
        out << "{\n  \"experiment_name\": \"Phase 5.3 Geometric Feature Utility Analysis\",\n  \"features\": [\n";
        for (size_t i = 0; i < stage5_results.size(); ++i) {
            const auto& r = stage5_results[i];
            out << "    {\n"
                << "      \"feature_config\": \"" << r.experiment_tag << "\",\n"
                << "      \"overall_recall_at_10\": " << r.primary_eval.recall_at_10 << ",\n"
                << "      \"dense_query_recall\": " << r.primary_eval.dense_query_recall << ",\n"
                << "      \"boundary_query_recall\": " << r.primary_eval.boundary_query_recall << ",\n"
                << "      \"boundary_delta_pct\": " << (100.0 * (r.primary_eval.boundary_query_recall - stage3_results[0].primary_eval.boundary_query_recall)) << ",\n"
                << "      \"mean_dist_evals\": " << r.primary_eval.mean_total_dist_evals << ",\n"
                << "      \"qps\": " << r.primary_eval.qps << "\n"
                << "    }" << (i + 1 < stage5_results.size() ? "," : "") << "\n";
        }
        out << "  ]\n}\n";
        out.close();
        std::cout << ">>> Saved experiments/phase_5_3_geometry_utility_analysis.json" << std::endl;
    }

    // =============================================================
    // STAGE 8 — CROSS-DATASET VALIDATION (SIFT-500K & TRANSFORMER-384D)
    // =============================================================
    std::cout << "\n=============================================================" << std::endl;
    std::cout << "  STAGE 8: Cross-Dataset Validation Suite                     " << std::endl;
    std::cout << "=============================================================" << std::endl;

    std::vector<ComprehensiveResult> cross_dataset_results;

    // 8.1 SIFT-500K Scale Validation (N=500,000, D=128)
    std::string sift1m_path = "data/sift_base_1m.fvecs";
    std::string sift500k_gt_path = "data/sift_groundtruth_500k.ivecs";
    size_t s500_n = 0, s500_dim = 0;
    std::vector<float> s500_data;
    load_fvecs(sift1m_path, s500_data, s500_n, s500_dim, 500000);
    std::vector<std::vector<uint32_t>> s500_gt;
    load_ivecs(sift500k_gt_path, s500_gt);

    size_t s500_nq = std::min(sift_nq, s500_gt.size());
    std::cout << "Loaded SIFT-500K: N=" << s500_n << ", D=" << s500_dim << ", NQ=" << s500_nq << std::endl;

    std::vector<uint8_t> s500_is_boundary(s500_nq, 0);

    // SIFT-500K: Uniform Baseline (Official hnswlib parity, Strict RNG)
    {
        PolicyConfig cfg = decoupled_template;
        std::vector<int> alloc_u500 = BudgetAllocator::allocate_uniform(s500_n, 16);
        cross_dataset_results.push_back(run_adaptivevec_experiment(
            "SIFT500K-Uniform-StrictRNG", "Uniform Allocation on SIFT-500K (M=16, B0=8.0M)",
            "SIFT-500K", s500_data, s500_n, s500_dim, sift_queries, s500_nq, s500_gt, s500_is_boundary,
            cfg, false, alloc_u500
        ));
    }

    // SIFT-500K: Budget-Normalized Exponential Allocation
    {
        std::vector<GeometrySignal> s500_signals = precompute_geometry_signals(s500_data, s500_n, s500_dim, 30);
        std::vector<int> alloc_exp500 = BudgetAllocator::allocate_normalized_exponential(s500_signals, 16, 8, 24, 0.5f);
        cross_dataset_results.push_back(run_adaptivevec_experiment(
            "SIFT500K-Normalized-Exponential", "Normalized Exponential on SIFT-500K (Exact B0=8.0M)",
            "SIFT-500K", s500_data, s500_n, s500_dim, sift_queries, s500_nq, s500_gt, s500_is_boundary,
            decoupled_template, true, alloc_exp500
        ));
    }

    // 8.2 Transformer-384D Embedding Corpus (N=57,893, D=384, NQ=1,000)
    std::string tf_base_path = "data/transformer_384d/transformer_384d_base.fvecs";
    std::string tf_query_path = "data/transformer_384d/transformer_384d_query.fvecs";
    std::string tf_gt_path = "data/transformer_384d/transformer_384d_groundtruth.ivecs";

    size_t tf_n = 0, tf_dim = 0;
    std::vector<float> tf_data;
    load_fvecs(tf_base_path, tf_data, tf_n, tf_dim);
    size_t tf_nq = 0, tf_qdim = 0;
    std::vector<float> tf_queries;
    load_fvecs(tf_query_path, tf_queries, tf_nq, tf_qdim);
    std::vector<std::vector<uint32_t>> tf_gt;
    load_ivecs(tf_gt_path, tf_gt);

    std::cout << "Loaded Transformer-384D: N=" << tf_n << ", D=" << tf_dim << ", NQ=" << tf_nq << std::endl;
    std::vector<uint8_t> tf_is_boundary(tf_nq, 0);

    // TF-384D: Uniform Baseline
    {
        PolicyConfig cfg = decoupled_template;
        std::vector<int> tf_alloc_u = BudgetAllocator::allocate_uniform(tf_n, 16);
        cross_dataset_results.push_back(run_adaptivevec_experiment(
            "TF384D-Uniform-StrictRNG", "Uniform Allocation on Transformer-384D (M=16)",
            "Transformer-384D", tf_data, tf_n, tf_dim, tf_queries, tf_nq, tf_gt, tf_is_boundary,
            cfg, false, tf_alloc_u
        ));
    }

    // TF-384D: Budget-Normalized Exponential Allocation
    {
        std::vector<GeometrySignal> tf_signals = precompute_geometry_signals(tf_data, tf_n, tf_dim, 30);
        std::vector<int> tf_alloc_exp = BudgetAllocator::allocate_normalized_exponential(tf_signals, 16, 8, 24, 0.5f);
        cross_dataset_results.push_back(run_adaptivevec_experiment(
            "TF384D-Normalized-Exponential", "Normalized Exponential on Transformer-384D (Exact B0)",
            "Transformer-384D", tf_data, tf_n, tf_dim, tf_queries, tf_nq, tf_gt, tf_is_boundary,
            decoupled_template, true, tf_alloc_exp
        ));
    }

    // Export experiments/phase_5_3_cross_dataset_validation.json
    {
        std::ofstream out("experiments/phase_5_3_cross_dataset_validation.json");
        out << "{\n  \"experiment_name\": \"Phase 5.3 Cross-Dataset Scalability & Validation Suite\",\n  \"results\": [\n";
        for (size_t i = 0; i < cross_dataset_results.size(); ++i) {
            const auto& r = cross_dataset_results[i];
            out << "    {\n"
                << "      \"experiment_tag\": \"" << r.experiment_tag << "\",\n"
                << "      \"scheme_name\": \"" << r.scheme_name << "\",\n"
                << "      \"dataset\": \"" << r.dataset_name << "\",\n"
                << "      \"n_vectors\": " << r.n_vectors << ",\n"
                << "      \"dim\": " << r.dim << ",\n"
                << "      \"assigned_degree_budget\": " << r.assigned_degree_budget << ",\n"
                << "      \"total_edges\": " << r.total_edges << ",\n"
                << "      \"layer0_edges\": " << r.layer0_edges << ",\n"
                << "      \"recall_at_10\": " << r.primary_eval.recall_at_10 << ",\n"
                << "      \"qps\": " << r.primary_eval.qps << ",\n"
                << "      \"mean_dist_evals\": " << r.primary_eval.mean_total_dist_evals << ",\n"
                << "      \"p95_latency_ms\": " << r.primary_eval.p95_latency_ms << ",\n"
                << "      \"build_time_s\": " << r.build_time_s << ",\n"
                << "      \"graph_memory_mb\": " << (double)r.mem.total_owned_graph_bytes / (1024.0 * 1024.0) << "\n"
                << "    }" << (i + 1 < cross_dataset_results.size() ? "," : "") << "\n";
        }
        out << "  ]\n}\n";
        out.close();
        std::cout << "\n>>> Saved experiments/phase_5_3_cross_dataset_validation.json" << std::endl;
    }

    std::cout << "\n=============================================================" << std::endl;
    std::cout << "  PHASE 5.3 BENCHMARK SUITE COMPLETE! ALL ARTIFACTS SAVED.   " << std::endl;
    std::cout << "=============================================================" << std::endl;

    return 0;
}
