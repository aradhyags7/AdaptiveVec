#pragma once

#include "adaptive_hnsw.hpp"
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <iostream>

namespace adaptivevec {

class ExactGroundTruthGenerator {
public:
    /**
     * Compute exact brute-force top-k nearest neighbors for a batch of queries
     * against base_vectors under L2 or Cosine distance.
     * Guaranteed exactness (zero approximations).
     */
    static std::vector<std::vector<tableint>> compute_exact_groundtruth(
        const float* base_data,
        size_t n_base,
        const float* query_data,
        size_t n_queries,
        size_t dim,
        SpaceType space,
        size_t k = 100,
        int num_threads = 0
    ) {
        if (num_threads <= 0) num_threads = omp_get_max_threads();
        std::vector<std::vector<tableint>> ground_truth(n_queries, std::vector<tableint>(k));

        #pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads)
        for (size_t q = 0; q < n_queries; ++q) {
            const float* q_vec = &query_data[q * dim];
            
            // Max-heap of (distance, id) to keep top-k smallest distances
            std::priority_queue<std::pair<dist_t, tableint>> topk_heap;

            for (size_t i = 0; i < n_base; ++i) {
                const float* b_vec = &base_data[i * dim];
                dist_t d = (space == SpaceType::Cosine) 
                           ? cosine_distance(q_vec, b_vec, dim)
                           : l2_sq_distance(q_vec, b_vec, dim);

                if (topk_heap.size() < k) {
                    topk_heap.push({d, static_cast<tableint>(i)});
                } else if (d < topk_heap.top().first) {
                    topk_heap.pop();
                    topk_heap.push({d, static_cast<tableint>(i)});
                }
            }

            // Extract sorted in ascending order
            std::vector<tableint> row(k);
            for (int r = static_cast<int>(k) - 1; r >= 0; --r) {
                row[r] = topk_heap.top().second;
                topk_heap.pop();
            }
            ground_truth[q] = std::move(row);
        }

        return ground_truth;
    }
};

} // namespace adaptivevec
