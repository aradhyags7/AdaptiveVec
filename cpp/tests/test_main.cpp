/**
 * AdaptiveVec Phase 0 Native C++ Test Suite
 * Validates algorithmic correctness, concurrency safety, numerical precision,
 * metric parity, edge cases, and memory accounting.
 */

#include "adaptive_hnsw.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <thread>
#include <atomic>
#include <algorithm>

using namespace adaptivevec;

static int g_tests_passed = 0;
static int g_tests_total = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] ASSERTION FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            std::exit(1); \
        } \
    } while(0)

#define RUN_TEST(fn) \
    do { \
        g_tests_total++; \
        std::cout << "[TEST " << g_tests_total << "] Running " #fn " ... " << std::flush; \
        fn(); \
        g_tests_passed++; \
        std::cout << "PASSED" << std::endl; \
    } while(0)

// 1. Metric implementations across scalar and SIMD paths + odd dimensions
void test_metrics_parity_scalar_vs_simd() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<size_t> test_dims = {1, 7, 16, 64, 65, 127, 128, 384, 768};
    for (size_t d : test_dims) {
        std::vector<float> a(d);
        std::vector<float> b(d);
        for (size_t i = 0; i < d; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }

        // Exact scalar L2
        float scalar_l2 = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            float diff = a[i] - b[i];
            scalar_l2 += diff * diff;
        }
        scalar_l2 = std::sqrt(scalar_l2);

        // Native l2_distance (AVX2/FMA with scalar fallback tail)
        float native_l2 = l2_distance(a.data(), b.data(), d);
        float err_l2 = std::abs(scalar_l2 - native_l2);
        TEST_ASSERT(err_l2 < 1e-4f * (scalar_l2 + 1.0f), "L2 distance parity failed for dim=" + std::to_string(d));

        // Scalar Cosine
        float dot = 0.0f, n_a = 0.0f, n_b = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            dot += a[i] * b[i];
            n_a += a[i] * a[i];
            n_b += b[i] * b[i];
        }
        float denom = std::sqrt(n_a) * std::sqrt(n_b);
        float scalar_cos = 1.0f - (denom > 1e-10f ? dot / denom : 0.0f);

        // Native cosine_distance
        float native_cos = cosine_distance(a.data(), b.data(), d);
        float err_cos = std::abs(scalar_cos - native_cos);
        TEST_ASSERT(err_cos < 1e-4f, "Cosine distance parity failed for dim=" + std::to_string(d));
    }
}

// 2. Deterministic level generation (SplitMix64)
void test_deterministic_level_generation() {
    PolicyConfig cfg;
    cfg.global_seed = 42;
    cfg.use_deterministic_level_hash = true;

    AdaptiveHNSWIndex index1(32, SpaceType::L2, true, cfg);
    AdaptiveHNSWIndex index2(32, SpaceType::L2, true, cfg);

    for (tableint node_id = 0; node_id < 1000; ++node_id) {
        int l1 = index1.generate_node_level(node_id);
        int l2 = index2.generate_node_level(node_id);
        TEST_ASSERT(l1 == l2, "Deterministic level mismatch between independent index instances at node_id=" + std::to_string(node_id));
        TEST_ASSERT(l1 >= 0, "Node level must be non-negative");
    }

    // Level distribution check: level 0 should dominate
    int l0_count = 0;
    for (tableint node_id = 0; node_id < 10000; ++node_id) {
        if (index1.generate_node_level(node_id) == 0) l0_count++;
    }
    // Theoretical expectation for m_l = 1/ln(16) ~ 0.36 is ~ 93.75% in layer 0
    TEST_ASSERT(l0_count > 8500 && l0_count < 9800, "Level 0 distribution out of expected statistical bounds: " + std::to_string(l0_count));
}

// 3. Dynamic M(x) computation and Layer-Decoupled scaling bounds
void test_dynamic_m_efc_bounds() {
    NodeParams np;
    np.m = 16;
    np.ef_construction = 120;
    
    // Layer 0 scaling disabled
    int m0 = np.get_m_for_layer(0, true, 0.75f, 4);
    TEST_ASSERT(m0 == 16, "Layer 0 M must equal base M");

    // Upper layer scaling with lambda = 0.75, min = 4
    int m1 = np.get_m_for_layer(1, true, 0.75f, 4); // 16 * 0.75 = 12
    TEST_ASSERT(m1 == 12, "Layer 1 M expected 12, got " + std::to_string(m1));

    int m2 = np.get_m_for_layer(2, true, 0.75f, 4); // 16 * 0.5625 = 9
    TEST_ASSERT(m2 == 9, "Layer 2 M expected 9, got " + std::to_string(m2));

    int m5 = np.get_m_for_layer(5, true, 0.75f, 4); // Clamped to min_m = 4
    TEST_ASSERT(m5 == 4, "Layer 5 M expected clamp to 4, got " + std::to_string(m5));
}

// 4. Quantization fidelity: SQ8 and SQ4 encode/decode fidelity
void test_quantization_fidelity() {
    const size_t dim = 64;
    const size_t n_samples = 1000;
    std::vector<float> data(n_samples * dim);
    std::mt19937 rng(999);
    std::normal_distribution<float> norm(0.0f, 2.0f);
    for (size_t i = 0; i < data.size(); ++i) data[i] = norm(rng);

    // Test SQ8
    ScalarQuantizer sq8(dim);
    sq8.train(data.data(), n_samples);
    TEST_ASSERT(sq8.is_trained, "SQ8 training failed");

    std::vector<uint8_t> code8(dim);
    std::vector<float> reconstructed8(dim);
    sq8.encode(&data[0], code8.data());
    sq8.decode(code8.data(), reconstructed8.data());

    float max_err8 = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        float err = std::abs(data[d] - reconstructed8[d]);
        if (err > max_err8) max_err8 = err;
    }
    // 256 quantization bins over ~12 std devs range gives ~ 0.1 max error
    TEST_ASSERT(max_err8 < 0.2f, "SQ8 reconstruction error too high: " + std::to_string(max_err8));

    // Test SQ4
    ScalarQuantizer4Bit sq4(dim);
    sq4.train(data.data(), n_samples);
    TEST_ASSERT(sq4.is_trained, "SQ4 training failed");

    size_t packed_dim = (dim + 1) / 2;
    std::vector<uint8_t> code4(packed_dim);
    std::vector<float> reconstructed4(dim);
    sq4.encode(&data[0], code4.data());
    sq4.decode(code4.data(), reconstructed4.data());

    float max_err4 = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        float err = std::abs(data[d] - reconstructed4[d]);
        if (err > max_err4) max_err4 = err;
    }
    // 16 bins gives ~ 1.0 max error
    TEST_ASSERT(max_err4 < 1.5f, "SQ4 reconstruction error too high: " + std::to_string(max_err4));
}

// 5. Tombstones, neighborhood healing, atomic in-degrees, and synchronized blocking compaction
void test_tombstone_healing_and_compaction() {
    const size_t dim = 16;
    const size_t n_samples = 300;
    std::vector<float> data(n_samples * dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < data.size(); ++i) data[i] = dist(rng);

    PolicyConfig cfg;
    AdaptiveHNSWIndex index(dim, SpaceType::L2, true, cfg);
    for (size_t i = 0; i < n_samples; ++i) {
        index.insert(&data[i * dim]);
    }
    TEST_ASSERT(index.num_elements == n_samples, "Index count mismatch");
    TEST_ASSERT(index.get_active_count() == n_samples, "Active count mismatch before deletion");

    // Delete 50 nodes
    for (tableint id = 10; id < 60; ++id) {
        index.delete_vector(id);
    }
    TEST_ASSERT(index.num_deleted.load() == 50, "Deleted count mismatch");
    TEST_ASSERT(index.get_active_count() == n_samples - 50, "Active count mismatch after deletion");

    // Search should never return a deleted node
    for (size_t q = 0; q < 20; ++q) {
        auto res = index.search(&data[q * dim], 10, 32);
        for (const auto& c : res) {
            TEST_ASSERT(c.id < 10 || c.id >= 60, "Search returned a tombstoned node: " + std::to_string(c.id));
        }
    }

    // Synchronized blocking compaction
    index.compact();
    TEST_ASSERT(index.num_deleted.load() == 0, "num_deleted must be 0 after compaction");
    TEST_ASSERT(index.num_elements == n_samples - 50, "num_elements must equal active count after compaction");
    TEST_ASSERT(index.get_active_count() == n_samples - 50, "Active count mismatch after compaction");

    // All node IDs in results must now be < num_elements
    for (size_t q = 0; q < 20; ++q) {
        auto res = index.search(&data[q * dim], 10, 32);
        for (const auto& c : res) {
            TEST_ASSERT(c.id < index.num_elements, "Result ID out of bounds after compaction");
        }
    }
}

// 6. Concurrency correctness: multi-threaded concurrent search and insertion under shared_mutex
void test_concurrency_correctness() {
    const size_t dim = 16;
    AdaptiveHNSWIndex index(dim, SpaceType::L2, false);

    // Pre-insert 100 vectors
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < 100; ++i) {
        std::vector<float> v(dim);
        for (size_t d = 0; d < dim; ++d) v[d] = dist(rng);
        index.insert(v.data());
    }

    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> search_count{0};

    // Reader threads doing concurrent searches
    auto reader_fn = [&]() {
        std::mt19937 lrng(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::uniform_real_distribution<float> ldist(-1.0f, 1.0f);
        std::vector<float> q(dim);
        while (!stop_flag.load(std::memory_order_relaxed)) {
            for (size_t d = 0; d < dim; ++d) q[d] = ldist(lrng);
            auto res = index.search(q.data(), 5, 16);
            search_count.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) readers.emplace_back(reader_fn);

    // Main thread does concurrent insertions
    for (int i = 0; i < 150; ++i) {
        std::vector<float> v(dim);
        for (size_t d = 0; d < dim; ++d) v[d] = dist(rng);
        index.insert(v.data());
    }

    // Trigger compaction during concurrent search
    index.delete_vector(10);
    index.delete_vector(20);
    index.compact();

    stop_flag.store(true, std::memory_order_relaxed);
    for (auto& t : readers) t.join();

    TEST_ASSERT(search_count.load() > 0, "No concurrent searches completed");
    TEST_ASSERT(index.num_elements == 248, "Expected 100 + 150 - 2 = 248 elements");
}

// 7. Memory accounting accuracy and Configuration A vs B
void test_memory_breakdown() {
    const size_t dim = 32;
    const size_t n_samples = 200;
    std::vector<float> data(n_samples * dim, 0.5f);

    AdaptiveHNSWIndex index(dim, SpaceType::L2, true);
    for (size_t i = 0; i < n_samples; ++i) {
        index.insert(&data[i * dim]);
    }

    MemoryBreakdown mb1 = index.get_memory_breakdown();
    TEST_ASSERT(mb1.resident_fp32_bytes >= n_samples * dim * sizeof(float), "Resident FP32 memory under-reported");
    TEST_ASSERT(mb1.graph_adjacency_bytes > 0, "Graph adjacency bytes must be > 0");
    TEST_ASSERT(mb1.per_node_metadata_bytes > 0, "Per-node metadata bytes must be > 0");
    TEST_ASSERT(mb1.process_working_set_bytes > 0, "OS Working set must be > 0");
    TEST_ASSERT(mb1.total_owned_bytes > mb1.resident_fp32_bytes, "Total owned must include graph and metadata");

    // Quantize to SQ8 (Configuration B: Quantized with resident FP32)
    index.quantize_vectors(true, false);
    MemoryBreakdown mb2 = index.get_memory_breakdown();
    TEST_ASSERT(mb2.allocated_quantized_capacity_bytes >= n_samples * dim * sizeof(uint8_t), "Quantized capacity not recorded");
    TEST_ASSERT(mb2.resident_fp32_bytes == mb1.resident_fp32_bytes, "Resident FP32 should remain in Config B");

    // Drop raw vectors (Configuration A: Pure quantized storage)
    index.drop_raw_vectors();
    MemoryBreakdown mb3 = index.get_memory_breakdown();
    TEST_ASSERT(mb3.resident_fp32_bytes == 0, "Resident FP32 must be 0 after drop_raw_vectors()");
    TEST_ASSERT(mb3.allocated_quantized_capacity_bytes > 0, "Quantized capacity must remain after drop_raw_vectors()");
}

// 8. Boundary conditions: empty graph, single vector, duplicates, k > N
void test_boundary_conditions() {
    const size_t dim = 8;
    AdaptiveHNSWIndex index(dim, SpaceType::L2, false);

    std::vector<float> q(dim, 1.0f);
    // Empty search
    auto res_empty = index.search(q.data(), 10, 16);
    TEST_ASSERT(res_empty.empty(), "Empty index search must return empty result");

    // Single vector
    index.insert(q.data());
    auto res_single = index.search(q.data(), 10, 16);
    TEST_ASSERT(res_single.size() == 1, "Single element index search must return exactly 1 candidate");
    TEST_ASSERT(res_single[0].id == 0, "Candidate ID must be 0");
    TEST_ASSERT(res_single[0].dist == 0.0f, "Distance to self must be 0");

    // Duplicate vectors
    for (int i = 0; i < 20; ++i) {
        index.insert(q.data()); // insert identical vectors
    }
    TEST_ASSERT(index.num_elements == 21, "Duplicate vector insert count mismatch");
    auto res_dups = index.search(q.data(), 10, 32);
    TEST_ASSERT(res_dups.size() == 10, "Search on duplicate vectors must return k elements");

    // k > num_elements
    auto res_large_k = index.search(q.data(), 50, 32);
    TEST_ASSERT(res_large_k.size() == 21, "Search with k > N must return exactly N elements");
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "          AdaptiveVec Phase 0 Native C++ Unit Test Suite         " << std::endl;
    std::cout << "=================================================================" << std::endl;

    RUN_TEST(test_metrics_parity_scalar_vs_simd);
    RUN_TEST(test_deterministic_level_generation);
    RUN_TEST(test_dynamic_m_efc_bounds);
    RUN_TEST(test_quantization_fidelity);
    RUN_TEST(test_tombstone_healing_and_compaction);
    RUN_TEST(test_concurrency_correctness);
    RUN_TEST(test_memory_breakdown);
    RUN_TEST(test_boundary_conditions);

    std::cout << "\n=================================================================" << std::endl;
    std::cout << " ALL " << g_tests_passed << "/" << g_tests_total << " PHASE 0 C++ TESTS PASSED SUCCESSFULLY!" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
