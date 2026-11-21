#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <cassert>
#include <cmath>
#include <chrono>
#include <mutex>
#include "adaptive_hnsw.hpp"

using namespace adaptivevec;

int main() {
    std::cout << "=================================================================\n";
    std::cout << " AdaptiveVec: Deterministic Concurrent Mutation Preservation Test\n";
    std::cout << "=================================================================\n\n";

    const size_t dim = 16;
    const size_t initial_count = 1000;
    PolicyConfig cfg;
    cfg.m_base = 16;
    cfg.ef_c_base = 64;
    cfg.keep_raw_vectors = true;
    cfg.enable_sq8 = false;
    cfg.enable_sq4 = false;

    // 1. Initialize Baseline Index with 1,000 vectors
    auto initial_idx = std::make_shared<AdaptiveHNSWIndex>(dim, SpaceType::L2, false, cfg);
    initial_idx->reserve(2000);

    std::unordered_map<uint64_t, std::vector<float>> authoritative_labels;
    std::mutex authoritative_mtx;

    for (size_t i = 0; i < initial_count; ++i) {
        uint64_t label = static_cast<uint64_t>(i);
        std::vector<float> vec(dim);
        for (size_t d = 0; d < dim; ++d) {
            vec[d] = std::sin(static_cast<float>(label * 17 + d * 5) * 0.01f);
        }
        authoritative_labels[label] = vec;
        initial_idx->insert(vec.data());
    }

    std::cout << "[Step 1] Initialized baseline index with " << initial_count << " vectors.\n";

    // 2. Initialize ServingIndexManager
    ServingIndexManager manager(std::move(initial_idx));
    assert(manager.get_active_label_count() == initial_count);

    // Pre-snapshot deletion of 20 labels (10..29) to test compaction on candidate
    for (size_t i = 10; i < 30; ++i) {
        uint64_t l = static_cast<uint64_t>(i);
        manager.mark_deleted(l);
        authoritative_labels.erase(l);
    }
    std::cout << "         Marked deleted 20 labels before snapshot. Active: " 
              << manager.get_active_label_count() << "\n";

    // 3. Create Candidate Snapshot (Simulate start of background maintenance)
    std::cout << "[Step 2] Taking CandidateSnapshot for background compaction/maintenance...\n";
    CandidateSnapshot snap = manager.create_candidate_snapshot();
    assert(snap.index_copy != nullptr);
    assert(snap.label_to_id.size() == authoritative_labels.size());
    uint64_t snapshot_seq = snap.snapshot_seq;
    std::cout << "         Candidate snapshot created at sequence ID: " << snapshot_seq 
              << ", Snapshot labels: " << snap.label_to_id.size() << "\n";

    // 4. Concurrently:
    //    Thread 1: Mutator applies insertions, deletions, and updates to ServingIndexManager
    //    Thread 2 & 3: Readers continuously query ServingIndexManager
    //    Thread 4: Background candidate prepares compaction and replacement
    std::atomic<bool> mutator_done{false};
    std::atomic<bool> stop_readers{false};
    std::atomic<size_t> reader_queries{0};
    std::atomic<size_t> reader_errors{0};

    const size_t num_inserts = 200;  // Labels 1000..1199
    const size_t num_deletes = 100;  // Labels 50..149
    const size_t num_updates = 50;   // Labels 800..849

    std::cout << "[Step 3] Spawning concurrent threads: 1 Mutator, 2 Readers, 1 Background Worker...\n";

    // Reader thread function
    auto reader_fn = [&]() {
        std::vector<float> q(dim, 0.5f);
        while (!stop_readers.load(std::memory_order_relaxed)) {
            auto idx = manager.get_serving_index();
            if (!idx) {
                reader_errors.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            auto res = idx->search(q.data(), 5, 32);
            if (res.empty() && idx->get_active_count() > 0) {
                reader_errors.fetch_add(1, std::memory_order_relaxed);
            }
            reader_queries.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    };

    std::thread reader1(reader_fn);
    std::thread reader2(reader_fn);

    // Mutator thread
    std::thread mutator([&]() {
        // A. Insert new labels (1000..1199)
        for (size_t i = 0; i < num_inserts; ++i) {
            uint64_t label = 1000 + i;
            std::vector<float> vec(dim);
            for (size_t d = 0; d < dim; ++d) {
                vec[d] = 100.0f + std::sin(static_cast<float>(label * 13 + d * 7) * 0.02f);
            }
            {
                std::lock_guard<std::mutex> lk(authoritative_mtx);
                authoritative_labels[label] = vec;
            }
            manager.insert(label, vec.data());
            if (i % 20 == 0) std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // B. Delete existing labels (50..149)
        for (size_t i = 0; i < num_deletes; ++i) {
            uint64_t label = 50 + i;
            {
                std::lock_guard<std::mutex> lk(authoritative_mtx);
                authoritative_labels.erase(label);
            }
            manager.mark_deleted(label);
            if (i % 20 == 0) std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // C. Update existing labels (800..849)
        for (size_t i = 0; i < num_updates; ++i) {
            uint64_t label = 800 + i;
            std::vector<float> vec(dim);
            for (size_t d = 0; d < dim; ++d) {
                vec[d] = 200.0f + std::cos(static_cast<float>(label * 11 + d * 3) * 0.03f);
            }
            {
                std::lock_guard<std::mutex> lk(authoritative_mtx);
                authoritative_labels[label] = vec;
            }
            manager.insert(label, vec.data());
            if (i % 10 == 0) std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        mutator_done.store(true);
    });

    // Background maintenance worker
    ReplacementResult repl_res;
    std::thread background_worker([&]() {
        // Candidate processing: heal and compact candidate index
        std::shared_ptr<AdaptiveHNSWIndex> candidate = snap.index_copy;

        for (size_t i = 10; i < 30; ++i) {
            candidate->heal_neighborhood(static_cast<tableint>(i));
        }
        std::vector<tableint> compaction_map = candidate->compact();

        // Ensure mutator has completed so all concurrent mutations are in the journal
        while (!mutator_done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Validate and atomically deploy candidate with catch-up replay
        std::vector<float> dummy_queries(10 * dim, 0.1f);
        std::vector<std::vector<tableint>> dummy_gt(10);
        for (size_t q = 0; q < 10; ++q) dummy_gt[q] = {0, 1, 2};

        repl_res = manager.evaluate_and_replace(
            candidate,
            dummy_queries.data(),
            10,
            dummy_gt,
            0.0f,    // no strict recall gate for synthetic concurrency test
            500.0,   // generous memory limit
            1000.0f, // latency limit
            5,
            32,
            snapshot_seq,
            snap.label_to_id,
            compaction_map
        );
    });

    mutator.join();
    background_worker.join();

    // Allow readers to execute a few more queries on the newly replaced index
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop_readers.store(true);
    reader1.join();
    reader2.join();

    std::cout << "[Step 4] Concurrency phase finished.\n";
    std::cout << "         Reader queries completed: " << reader_queries.load() << "\n";
    std::cout << "         Reader errors observed:   " << reader_errors.load() << "\n";
    std::cout << "         Replacement status:       " << to_string(repl_res.status) << "\n";
    std::cout << "         Details:                  " << repl_res.details << "\n";
    std::cout << "         Exclusive lock pause:     " << repl_res.exclusive_lock_pause_us << " us\n";
    std::cout << "         Mutations replayed:       " << repl_res.replayed_mutations_count << "\n";

    // 5. Authoritative State Verification
    std::cout << "\n[Step 5] Comparing Serving Index State with Authoritative Mutation Journal...\n";
    assert(repl_res.status == ReplacementStatus::Success);
    assert(reader_errors.load() == 0);
    assert(repl_res.replayed_mutations_count >= (num_inserts + num_deletes + num_updates));

    size_t active_count = manager.get_active_label_count();
    std::cout << "         Authoritative label count: " << authoritative_labels.size() << "\n";
    std::cout << "         Serving index label count: " << active_count << "\n";
    assert(active_count == authoritative_labels.size());

    // Verify all authoritative labels exist with exact vectors
    size_t verified_vectors = 0;
    float max_diff = 0.0f;
    for (const auto& kv : authoritative_labels) {
        uint64_t label = kv.first;
        const auto& expected_vec = kv.second;

        assert(manager.has_label(label));
        std::vector<float> actual_vec(dim, 0.0f);
        bool found = manager.get_vector(label, actual_vec.data());
        assert(found);

        for (size_t d = 0; d < dim; ++d) {
            float diff = std::abs(expected_vec[d] - actual_vec[d]);
            if (diff > max_diff) max_diff = diff;
            assert(diff < 1e-5f);
        }
        verified_vectors++;
    }
    std::cout << "         Verified " << verified_vectors << " vectors against authoritative state (Max Diff = " 
              << max_diff << ")\n";

    // Verify deleted labels do not exist
    size_t verified_deletes = 0;
    for (size_t i = 0; i < num_deletes; ++i) {
        uint64_t label = 50 + i;
        assert(!manager.has_label(label));
        std::vector<float> dummy(dim);
        assert(!manager.get_vector(label, dummy.data()));
        verified_deletes++;
    }
    std::cout << "         Verified " << verified_deletes << " deleted labels are absent from serving index.\n";

    // Verify active count in actual underlying graph
    auto final_serving = manager.get_serving_index();
    assert(final_serving != nullptr);
    assert(final_serving->get_active_count() == authoritative_labels.size());
    std::cout << "         Underlying graph active element count: " << final_serving->get_active_count() << "\n";

    // 6. Reader-Held Pointer Isolation Verification
    std::cout << "\n[Step 6] Verifying Reader Shared-Pointer Isolation across replacement...\n";
    {
        // Reader holds pointer to current index
        auto reader_held_idx = manager.get_serving_index();
        assert(reader_held_idx != nullptr);
        size_t initial_active = reader_held_idx->get_active_count();

        // Create a new candidate and replace again
        std::shared_ptr<AdaptiveHNSWIndex> cand2 = reader_held_idx->clone();
        cand2->insert(std::vector<float>(dim, 999.0f).data());
        
        std::vector<float> q_dummy(5 * dim, 0.1f);
        std::vector<std::vector<tableint>> gt_dummy(5, {0});

        ReplacementResult res2 = manager.evaluate_and_replace(
            cand2, q_dummy.data(), 5, gt_dummy, 0.0f, 500.0, 1000.0f, 5, 32
        );
        assert(res2.status == ReplacementStatus::Success);

        // Reader's held pointer is completely unaffected and still valid
        assert(reader_held_idx->get_active_count() == initial_active);
        std::vector<float> q(dim, 0.5f);
        auto q_res = reader_held_idx->search(q.data(), 5, 32);
        assert(!q_res.empty());

        // New queries observe new index
        auto new_idx = manager.get_serving_index();
        assert(new_idx->get_active_count() == initial_active + 1);
        std::cout << "         Reader shared-pointer isolation verified: Zero corruption or race conditions.\n";
    }

    std::cout << "\n=================================================================\n";
    std::cout << " >>> CONCURRENT MUTATION PRESERVATION: PASSED (ZERO LOST UPDATES)\n";
    std::cout << "=================================================================\n";

    return 0;
}
