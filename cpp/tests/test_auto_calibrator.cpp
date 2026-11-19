#include "auto_calibrator.hpp"
#include <iostream>
#include <cassert>

using namespace adaptivevec;

int main() {
    std::cout << "Testing AutoCalibrator compilation and basic functionality..." << std::endl;
    
    // Test bootstrap confidence interval
    std::vector<float> scores = {0.95f, 0.96f, 0.94f, 0.95f, 0.97f, 0.93f, 0.95f, 0.96f};
    auto [ci_low, ci_high] = AutoCalibrator::bootstrap_confidence_interval(scores, 500, 0.05f);
    std::cout << "Bootstrap CI: [" << ci_low << ", " << ci_high << "]" << std::endl;
    assert(ci_low <= ci_high);
    assert(ci_low >= 0.90f && ci_high <= 1.0f);
    
    // Test calibration constraints
    CalibrationConstraints constraints;
    constraints.target_recall = 0.90f;
    constraints.max_memory_mb = 50.0;
    constraints.max_calibration_duration_s = 10.0;
    
    // Synthetic dataset
    size_t n = 200;
    size_t dim = 16;
    std::vector<float> base_vecs(n * dim, 0.5f);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : base_vecs) x = dist(rng);
    
    size_t n_q = 20;
    std::vector<float> query_vecs(n_q * dim);
    for (auto& x : query_vecs) x = dist(rng);
    
    // Compute simple ground truth for 20 queries
    std::vector<std::vector<tableint>> gt(n_q);
    for (size_t q = 0; q < n_q; ++q) {
        std::vector<std::pair<float, tableint>> dlist;
        for (size_t i = 0; i < n; ++i) {
            float d = 0.0f;
            for (size_t d_i = 0; d_i < dim; ++d_i) {
                float diff = query_vecs[q * dim + d_i] - base_vecs[i * dim + d_i];
                d += diff * diff;
            }
            dlist.push_back({d, (tableint)i});
        }
        std::sort(dlist.begin(), dlist.end());
        for (size_t k = 0; k < 10; ++k) gt[q].push_back(dlist[k].second);
    }
    
    std::unique_ptr<AdaptiveHNSWIndex> index = nullptr;
    CalibrationReport report = AutoCalibrator::calibrate(
        base_vecs.data(), n, dim, SpaceType::L2,
        query_vecs.data(), n_q, gt,
        constraints, index, "Synthetic-16D"
    );
    
    std::cout << report.summary() << std::endl;
    assert(!report.evaluated_trials.empty());
    assert(index != nullptr);
    
    // Test search on calibrated index
    auto res = index->search(query_vecs.data(), 5, report.selected_config.ef_search);
    std::cout << "Calibrated index search returned " << res.size() << " candidates. Top candidate ID: " << res[0].id << std::endl;
    assert(!res.empty());
    
    std::cout << "AutoCalibrator basic test PASSED!" << std::endl;
    return 0;
}
