/**
 * run_streaming_deletion_benchmarks.cpp
 * Frontier 2: Dynamic Streaming Node Deletions & Localized Graph Healing Benchmark Harness
 * 
 * Compares:
 * 1. Naive Soft Tombstone (Unhealed): Nodes marked deleted without rewiring; severed links and ghost traversal.
 * 2. AdaptiveVec Dynamic Graph Healing (Ours): Predecessors dynamically reconnected to surviving neighbors,
 *    mutual neighbor bridging, entry point migration, and layer-0 hubness in-degree tracking.
 * 3. Compaction: Post-deletion zero-fragmentation memory reclamation and contiguous index repacking.
 * 4. Clean Rebuild Oracle: Scratch index built directly on surviving vector partition (theoretical ceiling).
 * 
 * Measures across churn rates (0%, 10%, 20%, 30%, 40%, 50%):
 * - Recall@10 on surviving vectors against exact ground truth computed strictly on active corpus
 * - Query Throughput (QPS)
 * - Search Latency (p50, p95, p99 in microseconds)
 * - Distance evaluations per query
 * - Online healing throughput (microseconds per deletion)
 * - Memory footprint and compaction recovery
 */

#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <numeric>
#include <random>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace adaptivevec;

struct DeletionRecord {
    std::string mode;
    double churn_pct;
    size_t active_nodes;
    size_t deleted_nodes;
    size_t total_edges;
    double recall_10;
    double qps;
    double mean_dist_evals;
    double p50_us;
    double p95_us;
    double p99_us;
    double total_ram_mb;
    double heal_us_per_op;
};

// Compute exact brute-force ground truth for queries against only the surviving active vectors
std::vector<std::vector<uint32_t>> compute_surviving_ground_truth(
    const float* base_data,
    size_t n_base,
    size_t dim,
    const float* queries,
    size_t n_queries,
    const std::vector<uint8_t>& is_deleted_mask,
    int k = 10
) {
    std::vector<std::vector<uint32_t>> gt(n_queries);

    #if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 16)
    #endif
    for (size_t q = 0; q < n_queries; ++q) {
        const float* q_vec = queries + q * dim;
        // Max-heap storing pairs of (distance, node_id)
        std::priority_queue<std::pair<float, uint32_t>> heap;

        for (uint32_t i = 0; i < n_base; ++i) {
            if (is_deleted_mask[i]) continue;
            float d = l2_distance(q_vec, base_data + i * dim, dim);
            if ((int)heap.size() < k) {
                heap.push({d, i});
            } else if (d < heap.top().first) {
                heap.pop();
                heap.push({d, i});
            }
        }

        std::vector<uint32_t> topk;
        topk.reserve(k);
        while (!heap.empty()) {
            topk.push_back(heap.top().second);
            heap.pop();
        }
        std::reverse(topk.begin(), topk.end());
        gt[q] = std::move(topk);
    }

    return gt;
}

float evaluate_recall_against_gt(
    const std::vector<std::vector<Candidate>>& search_results,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    int k = 10
) {
    if (search_results.empty() || ground_truth.empty()) return 0.0f;
    size_t total_hits = 0;
    size_t total_possible = 0;

    for (size_t q = 0; q < search_results.size(); ++q) {
        std::unordered_set<uint32_t> gt_set(ground_truth[q].begin(), ground_truth[q].end());
        int count = 0;
        for (const auto& cand : search_results[q]) {
            if (count >= k) break;
            if (gt_set.find(cand.id) != gt_set.end()) {
                total_hits++;
            }
            count++;
        }
        total_possible += std::min((size_t)k, ground_truth[q].size());
    }

    return total_possible > 0 ? (float)total_hits / (float)total_possible : 0.0f;
}

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "  AdaptiveVec Frontier 2: Dynamic Streaming Node Deletions & Graph Healing" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::string base_path = "data/sift_subset_100k.fvecs";
    std::string query_path = "data/sift_query.fvecs";

    size_t max_base = 50000;
    size_t max_queries = 1000;
    size_t dim = 128;
    size_t n_base = 0;
    size_t n_queries = 0;

    std::vector<float> base_data;
    std::vector<float> query_data;

    std::cout << "[1/4] Loading SIFT evaluation dataset: " << base_path << std::endl;
    if (!load_fvecs(base_path, base_data, n_base, dim, max_base)) {
        std::cerr << "Failed to load base vectors from " << base_path << std::endl;
        return 1;
    }
    if (!load_fvecs(query_path, query_data, n_queries, dim, max_queries)) {
        std::cerr << "Failed to load query vectors from " << query_path << std::endl;
        return 1;
    }
    std::cout << "      Loaded Base: " << n_base << " vectors (D=" << dim << ")" << std::endl;
    std::cout << "      Loaded Queries: " << n_queries << " vectors" << std::endl;

    // Generate deterministic shuffled permutation of node IDs for streaming deletion
    std::vector<uint32_t> deletion_order(n_base);
    std::iota(deletion_order.begin(), deletion_order.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(deletion_order.begin(), deletion_order.end(), rng);

    PolicyConfig policy_cfg;
    policy_cfg.m_base = 16;
    policy_cfg.ef_c_base = 120;
    policy_cfg.enable_dynamic_m_efc = true;
    policy_cfg.enable_layer_scaling = true;
    policy_cfg.enable_hubness_regulation = true;
    policy_cfg.hubness_mu = 0.15f;
    policy_cfg.enable_ada_ef = true;
    policy_cfg.ada_ef_patience = 6;
    policy_cfg.ada_ef_epsilon = 1e-4f;

    std::cout << "\n[2/4] Constructing initial graph indexes (N=" << n_base << ")..." << std::endl;
    AdaptiveHNSWIndex unhealed_index(dim, SpaceType::L2, true, policy_cfg);
    AdaptiveHNSWIndex healed_index(dim, SpaceType::L2, true, policy_cfg);

    // Build unhealed index
    auto t_start_unhealed = std::chrono::high_resolution_clock::now();
    unhealed_index.batch_insert(base_data.data(), n_base);
    auto t_end_unhealed = std::chrono::high_resolution_clock::now();
    double build_unhealed_s = std::chrono::duration<double>(t_end_unhealed - t_start_unhealed).count();

    // Build healed index
    auto t_start_healed = std::chrono::high_resolution_clock::now();
    healed_index.batch_insert(base_data.data(), n_base);
    auto t_end_healed = std::chrono::high_resolution_clock::now();
    double build_healed_s = std::chrono::duration<double>(t_end_healed - t_start_healed).count();

    std::cout << "      Unhealed Index Built: " << build_unhealed_s << "s | Edges: " 
              << (unhealed_index.graphs[0].size() > 0 ? unhealed_index.total_in_degrees / 2 : 0) << std::endl;
    std::cout << "      Healed Index Built:   " << build_healed_s << "s | Edges: " 
              << (healed_index.graphs[0].size() > 0 ? healed_index.total_in_degrees / 2 : 0) << std::endl;

    std::vector<double> churn_levels = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0};
    std::vector<DeletionRecord> records;

    std::cout << "\n[3/4] Executing Streaming Deletion & Healing Evaluation across Churn Rates..." << std::endl;

    std::vector<uint8_t> is_deleted_mask(n_base, 0);
    size_t current_deleted_count = 0;

    for (double churn : churn_levels) {
        size_t target_deleted = (size_t)std::round(n_base * (churn / 100.0));
        std::vector<uint32_t> newly_deleted;
        for (size_t i = current_deleted_count; i < target_deleted; ++i) {
            newly_deleted.push_back(deletion_order[i]);
            is_deleted_mask[deletion_order[i]] = 1;
        }

        // Apply newly deleted nodes to unhealed index (heal = false)
        for (uint32_t id : newly_deleted) {
            unhealed_index.mark_deleted(id, false);
        }

        // Apply newly deleted nodes to healed index (heal = true) and measure healing latency
        double heal_us_per_op = 0.0;
        if (!newly_deleted.empty()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t id : newly_deleted) {
                healed_index.mark_deleted(id, true);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double total_heal_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            heal_us_per_op = total_heal_us / newly_deleted.size();
        }

        current_deleted_count = target_deleted;
        size_t active_nodes = n_base - current_deleted_count;

        // Compute exact ground truth for surviving vectors
        auto surviving_gt = compute_surviving_ground_truth(
            base_data.data(), n_base, dim, query_data.data(), n_queries, is_deleted_mask, 10
        );

        int ef_search = 64;

        // --- Benchmark Unhealed Index ---
        unhealed_index.total_dist_computations = 0;
        std::vector<double> latencies_unhealed(n_queries);
        std::vector<std::vector<Candidate>> results_unhealed(n_queries);

        auto q_start_u = std::chrono::high_resolution_clock::now();
        for (size_t q = 0; q < n_queries; ++q) {
            auto t0 = std::chrono::high_resolution_clock::now();
            results_unhealed[q] = unhealed_index.search(&query_data[q * dim], 10, ef_search);
            auto t1 = std::chrono::high_resolution_clock::now();
            latencies_unhealed[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        auto q_end_u = std::chrono::high_resolution_clock::now();
        double q_time_u_s = std::chrono::duration<double>(q_end_u - q_start_u).count();
        double qps_u = n_queries / q_time_u_s;
        double evals_u = (double)unhealed_index.total_dist_computations / (double)n_queries;

        std::sort(latencies_unhealed.begin(), latencies_unhealed.end());
        double p50_u = latencies_unhealed[(size_t)(0.50 * n_queries)];
        double p95_u = latencies_unhealed[(size_t)(0.95 * n_queries)];
        double p99_u = latencies_unhealed[(size_t)(0.99 * n_queries)];
        double recall_u = evaluate_recall_against_gt(results_unhealed, surviving_gt, 10);

        size_t total_edges_u = 0;
        for (const auto& layer : unhealed_index.graphs) {
            for (const auto& adj : layer) total_edges_u += adj.size();
        }
        double ram_u = (double)(n_base * dim * 4 + total_edges_u * 4) / (1024.0 * 1024.0);

        records.push_back({
            "Naive Tombstones (Unhealed)",
            churn,
            active_nodes,
            current_deleted_count,
            total_edges_u,
            recall_u,
            qps_u,
            evals_u,
            p50_u,
            p95_u,
            p99_u,
            ram_u,
            0.0
        });

        // --- Benchmark Healed Index ---
        healed_index.total_dist_computations = 0;
        std::vector<double> latencies_healed(n_queries);
        std::vector<std::vector<Candidate>> results_healed(n_queries);

        auto q_start_h = std::chrono::high_resolution_clock::now();
        for (size_t q = 0; q < n_queries; ++q) {
            auto t0 = std::chrono::high_resolution_clock::now();
            results_healed[q] = healed_index.search(&query_data[q * dim], 10, ef_search);
            auto t1 = std::chrono::high_resolution_clock::now();
            latencies_healed[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        auto q_end_h = std::chrono::high_resolution_clock::now();
        double q_time_h_s = std::chrono::duration<double>(q_end_h - q_start_h).count();
        double qps_h = n_queries / q_time_h_s;
        double evals_h = (double)healed_index.total_dist_computations / (double)n_queries;

        std::sort(latencies_healed.begin(), latencies_healed.end());
        double p50_h = latencies_healed[(size_t)(0.50 * n_queries)];
        double p95_h = latencies_healed[(size_t)(0.95 * n_queries)];
        double p99_h = latencies_healed[(size_t)(0.99 * n_queries)];
        double recall_h = evaluate_recall_against_gt(results_healed, surviving_gt, 10);

        size_t total_edges_h = 0;
        for (const auto& layer : healed_index.graphs) {
            for (const auto& adj : layer) total_edges_h += adj.size();
        }
        double ram_h = (double)(n_base * dim * 4 + total_edges_h * 4) / (1024.0 * 1024.0);

        records.push_back({
            "AdaptiveVec Localized Healing",
            churn,
            active_nodes,
            current_deleted_count,
            total_edges_h,
            recall_h,
            qps_h,
            evals_h,
            p50_h,
            p95_h,
            p99_h,
            ram_h,
            heal_us_per_op
        });

        std::cout << "  Churn: " << std::setw(3) << (int)churn << "% (" << std::setw(5) << active_nodes << " active) | "
                  << "Unhealed: R@10=" << std::fixed << std::setprecision(4) << recall_u 
                  << ", QPS=" << std::setw(7) << std::setprecision(1) << qps_u
                  << " | Healed: R@10=" << std::setprecision(4) << recall_h 
                  << " (Δ=" << std::showpos << std::setprecision(4) << (recall_h - recall_u) << std::noshowpos << ")"
                  << ", QPS=" << std::setw(7) << std::setprecision(1) << qps_h
                  << ", Heal=" << std::setprecision(1) << heal_us_per_op << "μs/del" << std::endl;
    }

    // --- Benchmark Mode 3: Compaction at 50% Churn ---
    std::cout << "\n[4/4] Executing Online Compaction on Healed Index at 50% Churn..." << std::endl;
    auto t_comp_start = std::chrono::high_resolution_clock::now();
    healed_index.compact();
    auto t_comp_end = std::chrono::high_resolution_clock::now();
    double comp_time_ms = std::chrono::duration<double, std::milli>(t_comp_end - t_comp_start).count();

    // Map surviving ground truth IDs to compacted IDs
    std::vector<uint32_t> id_map(n_base, (uint32_t)-1);
    uint32_t new_idx = 0;
    for (size_t i = 0; i < n_base; ++i) {
        if (!is_deleted_mask[i]) {
            id_map[i] = new_idx++;
        }
    }

    auto surviving_gt_compact = compute_surviving_ground_truth(
        base_data.data(), n_base, dim, query_data.data(), n_queries, is_deleted_mask, 10
    );
    for (auto& row : surviving_gt_compact) {
        for (auto& val : row) {
            val = id_map[val];
        }
    }

    healed_index.total_dist_computations = 0;
    std::vector<double> latencies_compact(n_queries);
    std::vector<std::vector<Candidate>> results_compact(n_queries);

    auto q_start_c = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < n_queries; ++q) {
        auto t0 = std::chrono::high_resolution_clock::now();
        results_compact[q] = healed_index.search(&query_data[q * dim], 10, 64);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_compact[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    auto q_end_c = std::chrono::high_resolution_clock::now();
    double q_time_c_s = std::chrono::duration<double>(q_end_c - q_start_c).count();
    double qps_c = n_queries / q_time_c_s;
    double evals_c = (double)healed_index.total_dist_computations / (double)n_queries;

    std::sort(latencies_compact.begin(), latencies_compact.end());
    double p50_c = latencies_compact[(size_t)(0.50 * n_queries)];
    double p95_c = latencies_compact[(size_t)(0.95 * n_queries)];
    double p99_c = latencies_compact[(size_t)(0.99 * n_queries)];
    double recall_c = evaluate_recall_against_gt(results_compact, surviving_gt_compact, 10);

    size_t total_edges_c = 0;
    for (const auto& layer : healed_index.graphs) {
        for (const auto& adj : layer) total_edges_c += adj.size();
    }
    double ram_c = (double)(healed_index.num_elements * dim * 4 + total_edges_c * 4) / (1024.0 * 1024.0);

    records.push_back({
        "AdaptiveVec Post-Compaction",
        50.0,
        healed_index.num_elements,
        0,
        total_edges_c,
        recall_c,
        qps_c,
        evals_c,
        p50_c,
        p95_c,
        p99_c,
        ram_c,
        comp_time_ms
    });

    std::cout << "      Compaction completed in: " << comp_time_ms << " ms" << std::endl;
    std::cout << "      Compacted RAM: " << ram_c << " MB (vs Unhealed " << records[records.size()-3].total_ram_mb << " MB)" << std::endl;
    std::cout << "      Compacted Recall@10: " << recall_c << " | QPS: " << qps_c << std::endl;

    // --- Benchmark Mode 4: Clean Rebuilt Baseline Control (50% Surviving) ---
    std::cout << "\n      Building Clean Rebuild Oracle on surviving 50% vectors..." << std::endl;
    std::vector<float> surviving_data;
    surviving_data.reserve(healed_index.num_elements * dim);
    for (size_t i = 0; i < n_base; ++i) {
        if (!is_deleted_mask[i]) {
            surviving_data.insert(surviving_data.end(), base_data.data() + i * dim, base_data.data() + (i + 1) * dim);
        }
    }

    AdaptiveHNSWIndex oracle_index(dim, SpaceType::L2, true, policy_cfg);
    auto t_or_start = std::chrono::high_resolution_clock::now();
    oracle_index.batch_insert(surviving_data.data(), surviving_data.size() / dim);
    auto t_or_end = std::chrono::high_resolution_clock::now();
    double oracle_build_s = std::chrono::duration<double>(t_or_end - t_or_start).count();

    oracle_index.total_dist_computations = 0;
    std::vector<double> latencies_oracle(n_queries);
    std::vector<std::vector<Candidate>> results_oracle(n_queries);

    auto q_start_or = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < n_queries; ++q) {
        auto t0 = std::chrono::high_resolution_clock::now();
        results_oracle[q] = oracle_index.search(&query_data[q * dim], 10, 64);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_oracle[q] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    auto q_end_or = std::chrono::high_resolution_clock::now();
    double q_time_or_s = std::chrono::duration<double>(q_end_or - q_start_or).count();
    double qps_or = n_queries / q_time_or_s;
    double evals_or = (double)oracle_index.total_dist_computations / (double)n_queries;

    std::sort(latencies_oracle.begin(), latencies_oracle.end());
    double p50_or = latencies_oracle[(size_t)(0.50 * n_queries)];
    double p95_or = latencies_oracle[(size_t)(0.95 * n_queries)];
    double p99_or = latencies_oracle[(size_t)(0.99 * n_queries)];
    double recall_or = evaluate_recall_against_gt(results_oracle, surviving_gt_compact, 10);

    size_t total_edges_or = 0;
    for (const auto& layer : oracle_index.graphs) {
        for (const auto& adj : layer) total_edges_or += adj.size();
    }
    double ram_or = (double)(oracle_index.num_elements * dim * 4 + total_edges_or * 4) / (1024.0 * 1024.0);

    records.push_back({
        "Clean Scratch Rebuild (Oracle)",
        50.0,
        oracle_index.num_elements,
        0,
        total_edges_or,
        recall_or,
        qps_or,
        evals_or,
        p50_or,
        p95_or,
        p99_or,
        ram_or,
        oracle_build_s * 1000.0
    });

    std::cout << "      Clean Rebuild Oracle Recall@10: " << recall_or << " | QPS: " << qps_or << std::endl;

    // Export to JSON
    std::string out_json = "experiments/streaming_deletion_results.json";
    std::ofstream jf(out_json);
    jf << "{\n";
    jf << "  \"benchmark\": \"Frontier 2: Dynamic Streaming Node Deletions & Localized Graph Healing\",\n";
    jf << "  \"dataset\": \"SIFT-50K-Streaming\",\n";
    jf << "  \"n_base_initial\": " << n_base << ",\n";
    jf << "  \"n_queries\": " << n_queries << ",\n";
    jf << "  \"dim\": " << dim << ",\n";
    jf << "  \"records\": [\n";

    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        jf << "    {\n";
        jf << "      \"mode\": \"" << r.mode << "\",\n";
        jf << "      \"churn_pct\": " << r.churn_pct << ",\n";
        jf << "      \"active_nodes\": " << r.active_nodes << ",\n";
        jf << "      \"deleted_nodes\": " << r.deleted_nodes << ",\n";
        jf << "      \"total_edges\": " << r.total_edges << ",\n";
        jf << "      \"recall_10\": " << std::fixed << std::setprecision(4) << r.recall_10 << ",\n";
        jf << "      \"qps\": " << std::fixed << std::setprecision(1) << r.qps << ",\n";
        jf << "      \"mean_dist_evals\": " << std::fixed << std::setprecision(1) << r.mean_dist_evals << ",\n";
        jf << "      \"p50_us\": " << std::fixed << std::setprecision(1) << r.p50_us << ",\n";
        jf << "      \"p95_us\": " << std::fixed << std::setprecision(1) << r.p95_us << ",\n";
        jf << "      \"p99_us\": " << std::fixed << std::setprecision(1) << r.p99_us << ",\n";
        jf << "      \"total_ram_mb\": " << std::fixed << std::setprecision(2) << r.total_ram_mb << ",\n";
        jf << "      \"heal_us_per_op\": " << std::fixed << std::setprecision(1) << r.heal_us_per_op << "\n";
        jf << "    }" << (i + 1 < records.size() ? "," : "") << "\n";
    }
    jf << "  ]\n}\n";
    jf.close();

    std::cout << "\nSuccessfully wrote benchmark records to: " << out_json << std::endl;
    return 0;
}
