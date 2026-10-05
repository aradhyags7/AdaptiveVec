#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <vector>
#include <chrono>

using namespace adaptivevec;

int main() {
    std::vector<float> data;
    size_t n = 0, d = 0;
    load_fvecs("data/sift_subset_100k.fvecs", data, n, d, 10000);
    std::cout << "Loaded 10K vectors." << std::endl;

    for (int th : {1, 2, 4, 8}) {
        PolicyConfig p;
        AdaptiveHNSWIndex idx(d, SpaceType::L2, true, p);
        auto t0 = std::chrono::high_resolution_clock::now();
        idx.batch_insert(data.data(), n, th);
        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Threads: " << th << " -> " << s << "s, Rate: " << (n / s) << " v/s, Edges: " << idx.get_total_edges() << std::endl;
    }
    return 0;
}
