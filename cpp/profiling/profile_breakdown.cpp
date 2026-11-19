/**
 * AdaptiveVec Phase 0.5 Native Search Profiler & Bottleneck Analyzer
 * Measures microsecond breakdown of distance calculations, visited set lookups,
 * priority queue operations, and graph adjacency traversal.
 */

#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <algorithm>

using namespace adaptivevec;

struct ProfileStats {
    double total_query_us = 0.0;
    double dist_calc_us = 0.0;
    double visited_set_us = 0.0;
    double heap_ops_us = 0.0;
    double graph_traversal_us = 0.0;
    uint64_t dist_evals = 0;
    uint64_t visited_checks = 0;
    uint64_t visited_inserts = 0;
    uint64_t heap_pushes = 0;
    uint64_t heap_pops = 0;
};

// Instrumented Search Layer to measure exact CPU time in each sub-system
std::vector<Candidate> instrumented_search_layer(
    const AdaptiveHNSWIndex& index,
    const float* query,
    const std::vector<tableint>& enter_points,
    int ef,
    int lc,
    ProfileStats& stats
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    auto t_v0 = std::chrono::high_resolution_clock::now();
    std::unordered_set<tableint> visited(enter_points.begin(), enter_points.end());
    stats.visited_inserts += enter_points.size();
    auto t_v1 = std::chrono::high_resolution_clock::now();
    stats.visited_set_us += std::chrono::duration<double, std::micro>(t_v1 - t_v0).count();

    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
    std::priority_queue<Candidate, std::vector<Candidate>, std::less<Candidate>> w_furthest;
    
    for (tableint ep : enter_points) {
        auto t_d0 = std::chrono::high_resolution_clock::now();
        dist_t d = index.get_distance(query, index.get_vector(ep));
        stats.dist_evals++;
        auto t_d1 = std::chrono::high_resolution_clock::now();
        stats.dist_calc_us += std::chrono::duration<double, std::micro>(t_d1 - t_d0).count();

        auto t_h0 = std::chrono::high_resolution_clock::now();
        candidates.push({d, ep});
        w_furthest.push({d, ep});
        stats.heap_pushes += 2;
        auto t_h1 = std::chrono::high_resolution_clock::now();
        stats.heap_ops_us += std::chrono::duration<double, std::micro>(t_h1 - t_h0).count();
    }
    
    while (!candidates.empty()) {
        auto t_h0 = std::chrono::high_resolution_clock::now();
        Candidate curr = candidates.top();
        candidates.pop();
        stats.heap_pops++;
        dist_t furthest_d = w_furthest.top().dist;
        auto t_h1 = std::chrono::high_resolution_clock::now();
        stats.heap_ops_us += std::chrono::duration<double, std::micro>(t_h1 - t_h0).count();
        
        if (curr.dist > furthest_d) break;
        
        auto t_g0 = std::chrono::high_resolution_clock::now();
        const auto& neighbors = index.graphs[lc][curr.id];
        auto t_g1 = std::chrono::high_resolution_clock::now();
        stats.graph_traversal_us += std::chrono::duration<double, std::micro>(t_g1 - t_g0).count();

        for (size_t n_idx = 0; n_idx < neighbors.size(); ++n_idx) {
            tableint neighbor = neighbors[n_idx];
            
            t_v0 = std::chrono::high_resolution_clock::now();
            stats.visited_checks++;
            bool not_visited = (visited.find(neighbor) == visited.end());
            if (not_visited) {
                visited.insert(neighbor);
                stats.visited_inserts++;
            }
            t_v1 = std::chrono::high_resolution_clock::now();
            stats.visited_set_us += std::chrono::duration<double, std::micro>(t_v1 - t_v0).count();

            if (not_visited) {
                auto t_d0 = std::chrono::high_resolution_clock::now();
                dist_t d = index.get_distance(query, index.get_vector(neighbor));
                stats.dist_evals++;
                auto t_d1 = std::chrono::high_resolution_clock::now();
                stats.dist_calc_us += std::chrono::duration<double, std::micro>(t_d1 - t_d0).count();

                t_h0 = std::chrono::high_resolution_clock::now();
                furthest_d = w_furthest.top().dist;
                if (d < furthest_d || (int)w_furthest.size() < ef) {
                    candidates.push({d, neighbor});
                    stats.heap_pushes++;
                    
                    if ((int)w_furthest.size() >= ef) {
                        w_furthest.pop();
                        stats.heap_pops++;
                    }
                    w_furthest.push({d, neighbor});
                    stats.heap_pushes++;
                }
                t_h1 = std::chrono::high_resolution_clock::now();
                stats.heap_ops_us += std::chrono::duration<double, std::micro>(t_h1 - t_h0).count();
            }
        }
    }
    
    std::vector<Candidate> result;
    while (!w_furthest.empty()) {
        result.push_back(w_furthest.top());
        w_furthest.pop();
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    stats.total_query_us += std::chrono::duration<double, std::micro>(t_end - t_start).count();
    return result;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "     AdaptiveVec Native Search Micro-Profiler (SIFT-100K)        " << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    std::vector<float> base_data;
    size_t n_samples = 0, dim = 0;
    load_fvecs("data/sift_subset_100k.fvecs", base_data, n_samples, dim, 100000);

    std::vector<float> queries;
    size_t n_queries = 0, q_dim = 0;
    load_fvecs("data/sift_query.fvecs", queries, n_queries, q_dim, 1000);

    std::cout << "[+] Building AdaptiveVec Canonical Index (M=16, efC=200) ..." << std::endl;
    PolicyConfig cfg;
    cfg.enable_dynamic_m_efc = false;
    cfg.enable_layer_scaling = false;
    cfg.enable_hubness_regulation = false;
    cfg.enable_ada_ef = false;
    cfg.m_base = 16;
    cfg.ef_c_base = 200;

    AdaptiveHNSWIndex index(dim, SpaceType::L2, false, cfg);
    for (size_t i = 0; i < n_samples; ++i) {
        index.insert(&base_data[i * dim]);
    }
    std::cout << "[+] Built Canonical Graph: " << index.get_total_edges() << " edges." << std::endl;

    // Profile 1,000 queries with efSearch = 64
    ProfileStats total_stats;
    int ef_search = 64;

    std::cout << "[*] Profiling 1,000 queries at efSearch=64 ... " << std::flush;
    for (size_t q = 0; q < n_queries; ++q) {
        tableint curr_ep = index.enter_point;
        // Upper layers
        for (int lc = index.max_level; lc > 0; --lc) {
            auto w = index.search_layer(&queries[q * dim], {curr_ep}, 1, lc, false, false);
            if (!w.empty()) curr_ep = w[0].id;
        }
        // Layer 0 instrumented
        instrumented_search_layer(index, &queries[q * dim], {curr_ep}, ef_search, 0, total_stats);
    }
    std::cout << "Done!" << std::endl;

    double q_count = (double)n_queries;
    double avg_total_us = total_stats.total_query_us / q_count;
    double avg_dist_us = total_stats.dist_calc_us / q_count;
    double avg_visited_us = total_stats.visited_set_us / q_count;
    double avg_heap_us = total_stats.heap_ops_us / q_count;
    double avg_graph_us = total_stats.graph_traversal_us / q_count;
    double other_us = avg_total_us - (avg_dist_us + avg_visited_us + avg_heap_us + avg_graph_us);

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "               MEASURED PROFILING BOTTLENECK REPORT              " << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "Total Average Latency per Query : " << std::fixed << std::setprecision(2) << avg_total_us << " us (" 
              << 1e6 / avg_total_us << " single-thread QPS)" << std::endl;
    std::cout << "Average Distance Evals / Query  : " << (double)total_stats.dist_evals / q_count << std::endl;
    std::cout << "Average Visited Checks / Query  : " << (double)total_stats.visited_checks / q_count << std::endl;
    std::cout << "Average Visited Inserts / Query : " << (double)total_stats.visited_inserts / q_count << std::endl;
    std::cout << "Average Heap Pushes / Query     : " << (double)total_stats.heap_pushes / q_count << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(28) << "Component" 
              << std::setw(15) << "Latency (us)" 
              << std::setw(12) << "Share (%)" 
              << "Primary Mechanism" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    
    auto print_row = [&](const std::string& name, double us, const std::string& desc) {
        double pct = (us / avg_total_us) * 100.0;
        std::cout << std::left << std::setw(28) << name 
                  << std::setw(15) << std::fixed << std::setprecision(2) << us 
                  << std::setw(12) << std::fixed << std::setprecision(1) << pct 
                  << desc << std::endl;
    };

    print_row("1. Visited Set Operations", avg_visited_us, "std::unordered_set heap allocs & hash lookups");
    print_row("2. Distance Evaluations", avg_dist_us, "Single-accumulator AVX2 + scalar std::sqrt");
    print_row("3. Priority Queue Operations", avg_heap_us, "std::priority_queue dynamic pushes/pops");
    print_row("4. Graph Adjacency Access", avg_graph_us, "3-level pointer indirection across std::vector");
    print_row("5. Miscellaneous / Loop", other_us, "Branching, candidate copies, and function calls");
    std::cout << "=================================================================\n" << std::endl;

    return 0;
}
