#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <vector>
#include <chrono>

using namespace adaptivevec;

int main() {
    std::vector<float> data, queries;
    size_t n = 0, d = 0, qn = 0, qd = 0;
    load_fvecs("data/sift_subset_100k.fvecs", data, n, d, 20000);
    load_fvecs("data/sift_query.fvecs", queries, qn, qd, 1000);
    
    PolicyConfig p;
    AdaptiveHNSWIndex idx(d, SpaceType::L2, true, p);
    for (size_t i = 0; i < n; ++i) idx.insert(&data[i * d]);
    std::cout << "Built 20K index!" << std::endl;
    
    for (int th : {1, 2, 4, 8, 12}) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = idx.batch_search(queries.data(), qn, 10, 64, th);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Threads: " << th << " -> " << ms << " ms, QPS: " << (qn / (ms / 1000.0)) << std::endl;
    }
    return 0;
}
