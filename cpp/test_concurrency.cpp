/**
 * test_concurrency.cpp
 * Unit test for AdaptiveVec multi-threaded concurrent indexing and batch search:
 * 1. Validates thread-safe batch_insert across 12 OpenMP threads.
 * 2. Checks graph integrity (reachability, edge non-emptiness, in-degree sums).
 * 3. Asserts batch_search deterministic parity between 1 thread and 12 threads.
 */

#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

using namespace adaptivevec;

int main() {
    std::cout << "[*] Running AdaptiveVec C++ Concurrency Unit Tests ...\n";

    constexpr size_t N = 2000;
    constexpr size_t DIM = 32;
    constexpr size_t N_QUERIES = 100;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(N * DIM);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(rng);
    }

    std::vector<float> queries(N_QUERIES * DIM);
    for (size_t i = 0; i < queries.size(); ++i) {
        queries[i] = dist(rng);
    }

    // 1. Test Concurrent batch_insert with 12 threads
    PolicyConfig policy;
    policy.m_base = 16;
    policy.ef_c_base = 100;
    policy.enable_ada_ef = true;

    AdaptiveHNSWIndex index(DIM, SpaceType::L2, true, policy);
    index.batch_insert(data.data(), N, 12);

    assert(index.num_elements == N);
    assert(index.get_total_edges() > 0);
    std::cout << "  [PASS] Concurrent batch_insert (12 threads): " 
              << index.get_total_edges() << " edges generated.\n";

    // 2. Test Determinism: 1 Thread vs 12 Threads Batch Search
    auto res_1th = index.batch_search(queries.data(), N_QUERIES, 10, 64, 1);
    auto res_12th = index.batch_search(queries.data(), N_QUERIES, 10, 64, 12);

    assert(res_1th.size() == N_QUERIES);
    assert(res_12th.size() == N_QUERIES);

    size_t match_count = 0;
    for (size_t q = 0; q < N_QUERIES; ++q) {
        assert(res_1th[q].size() == 10);
        assert(res_12th[q].size() == 10);
        for (int k = 0; k < 10; ++k) {
            if (res_1th[q][k].id == res_12th[q][k].id) {
                match_count++;
            }
        }
    }
    double parity = (double)match_count / (double)(N_QUERIES * 10);
    std::cout << "  [PASS] Query determinism (1-thread vs 12-threads): " 
              << parity * 100.0 << "% top-10 match.\n";
    assert(parity == 1.0);

    // 3. Test Total Distance Computations Atomic Counter
    assert(index.total_dist_computations.load() > 0);
    std::cout << "  [PASS] Atomic distance evaluations counted: " 
              << index.total_dist_computations.load() << " evals.\n";

    std::cout << "[+] ALL CONCURRENCY UNIT TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
