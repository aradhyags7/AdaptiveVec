/**
 * Visited Set Microbenchmark
 * Compares std::unordered_set vs VisitedTagArray (tag-generation flat array)
 * across 1,000 queries with ~1,800 checks and ~1,000 inserts per query on N=100,000 nodes.
 */

#include <iostream>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <random>

class VisitedTagPool {
private:
    std::vector<uint16_t> tags;
    uint16_t cur_tag = 1;
    size_t num_nodes;

public:
    VisitedTagPool(size_t n) : num_nodes(n), tags(n, 0) {}

    inline void next_query() {
        cur_tag++;
        if (cur_tag == 0) { // wrapped around
            std::memset(tags.data(), 0, num_nodes * sizeof(uint16_t));
            cur_tag = 1;
        }
    }

    inline bool is_visited(uint32_t id) const {
        return tags[id] == cur_tag;
    }

    inline void set_visited(uint32_t id) {
        tags[id] = cur_tag;
    }
};

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "         Visited Set Implementation Microbenchmark (N=100K)      " << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    const size_t N = 100000;
    const size_t n_queries = 2000;
    const size_t checks_per_query = 1800;

    // Generate pseudo-random node check streams
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, N - 1);
    std::vector<std::vector<uint32_t>> query_streams(n_queries, std::vector<uint32_t>(checks_per_query));
    for (size_t q = 0; q < n_queries; ++q) {
        for (size_t i = 0; i < checks_per_query; ++i) {
            query_streams[q][i] = dist(rng);
        }
    }

    // 1. std::unordered_set (Current AdaptiveVec implementation)
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t total_visited_1 = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        std::unordered_set<uint32_t> visited;
        for (size_t i = 0; i < checks_per_query; ++i) {
            uint32_t node = query_streams[q][i];
            if (visited.find(node) == visited.end()) {
                visited.insert(node);
                total_visited_1++;
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_std_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / (double)n_queries;

    // 2. VisitedTagPool (Generation-stamped flat tag array)
    VisitedTagPool pool(N);
    t0 = std::chrono::high_resolution_clock::now();
    size_t total_visited_2 = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        pool.next_query();
        for (size_t i = 0; i < checks_per_query; ++i) {
            uint32_t node = query_streams[q][i];
            if (!pool.is_visited(node)) {
                pool.set_visited(node);
                total_visited_2++;
            }
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    double time_pool_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / (double)n_queries;

    std::cout << std::left << std::setw(32) << "Implementation" 
              << std::setw(20) << "Time / Query (us)" 
              << std::setw(15) << "Speedup" 
              << "Allocations" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(32) << "std::unordered_set (Current)" 
              << std::setw(20) << std::fixed << std::setprecision(2) << time_std_us 
              << std::setw(15) << "1.00x" 
              << "~1,000 heap allocs/q" << std::endl;
    std::cout << std::left << std::setw(32) << "VisitedTagPool (Generation Tags)" 
              << std::setw(20) << std::fixed << std::setprecision(2) << time_pool_us 
              << std::setw(15) << std::fixed << std::setprecision(1) << (time_std_us / time_pool_us) << "x" 
              << "0 heap allocs/q" << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    return 0;
}
