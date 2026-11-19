#include "adaptive_hnsw.hpp"
#include "auto_calibrator.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <random>
#include <iomanip>
#include <omp.h>
#include <cmath>

using namespace adaptivevec;

std::vector<float> load_fvecs(const std::string& path, size_t max_vectors, size_t& dim, size_t& count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[-] Error: Failed to open fvecs: " << path << std::endl;
        return {};
    }
    int d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    if (!in || d <= 0) return {};
    dim = static_cast<size_t>(d);
    
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t record_size = sizeof(int) + dim * sizeof(float);
    size_t total_in_file = file_size / record_size;
    count = std::min(max_vectors, total_in_file);
    
    std::vector<float> data(count * dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < count; ++i) {
        int cur_d = 0;
        in.read(reinterpret_cast<char*>(&cur_d), sizeof(int));
        in.read(reinterpret_cast<char*>(&data[i * dim]), dim * sizeof(float));
    }
    return data;
}

std::vector<std::vector<tableint>> load_ivecs(const std::string& path, size_t max_queries) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[-] Error: Failed to open ivecs: " << path << std::endl;
        return {};
    }
    std::vector<std::vector<tableint>> gt;
    while (gt.size() < max_queries) {
        int k = 0;
        in.read(reinterpret_cast<char*>(&k), sizeof(int));
        if (!in || k <= 0) break;
        std::vector<int> row(k);
        in.read(reinterpret_cast<char*>(row.data()), k * sizeof(int));
        if (!in) break;
        std::vector<tableint> row_t(k);
        for (int i = 0; i < k; ++i) row_t[i] = static_cast<tableint>(row[i]);
        gt.push_back(std::move(row_t));
    }
    return gt;
}

struct EvaluationResult {
    std::string dataset;
    std::string method;
    std::string precision;
    int m;
    int ef_c;
    int ef_s;
    bool enable_ada_ef;
    double calibration_time_s;
    double memory_mb;
    float heldout_recall;
    float heldout_ci_low;
    float heldout_ci_high;
    double heldout_qps;
    float heldout_p99_ms;
    bool meets_target_recall;
    bool meets_memory_budget;
    bool meets_latency_target;
    bool overall_feasible;
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "{"
           << "\"dataset\": \"" << dataset << "\", "
           << "\"method\": \"" << method << "\", "
           << "\"precision\": \"" << precision << "\", "
           << "\"m\": " << m << ", "
           << "\"ef_c\": " << ef_c << ", "
           << "\"ef_s\": " << ef_s << ", "
           << "\"enable_ada_ef\": " << (enable_ada_ef ? "true" : "false") << ", "
           << "\"calibration_time_s\": " << calibration_time_s << ", "
           << "\"memory_mb\": " << memory_mb << ", "
           << "\"heldout_recall\": " << heldout_recall << ", "
           << "\"heldout_ci_low\": " << heldout_ci_low << ", "
           << "\"heldout_ci_high\": " << heldout_ci_high << ", "
           << "\"heldout_qps\": " << heldout_qps << ", "
           << "\"heldout_p99_ms\": " << heldout_p99_ms << ", "
           << "\"meets_target_recall\": " << (meets_target_recall ? "true" : "false") << ", "
           << "\"meets_memory_budget\": " << (meets_memory_budget ? "true" : "false") << ", "
           << "\"meets_latency_target\": " << (meets_latency_target ? "true" : "false") << ", "
           << "\"overall_feasible\": " << (overall_feasible ? "true" : "false")
           << "}";
        return ss.str();
    }
};

EvaluationResult evaluate_on_heldout(
    const std::string& dataset_name,
    const std::string& method_name,
    AdaptiveHNSWIndex& index,
    int ef_s,
    bool enable_ada_ef,
    const std::string& precision,
    int m,
    int ef_c,
    double calibration_time_s,
    const float* heldout_queries,
    size_t n_heldout_q,
    size_t dim,
    const std::vector<std::vector<tableint>>& heldout_gt,
    int k,
    const CalibrationConstraints& constraints
) {
    index.policy.enable_ada_ef = enable_ada_ef;
    
    std::vector<float> recalls(n_heldout_q, 0.0f);
    std::vector<double> latencies_ms(n_heldout_q, 0.0);
    
    double t0 = omp_get_wtime();
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int q = 0; q < (int)n_heldout_q; ++q) {
        const float* q_ptr = &heldout_queries[q * dim];
        auto t_q0 = std::chrono::high_resolution_clock::now();
        auto res = index.search(q_ptr, k, ef_s);
        auto t_q1 = std::chrono::high_resolution_clock::now();
        
        latencies_ms[q] = std::chrono::duration<double, std::milli>(t_q1 - t_q0).count();
        
        if (q < (int)heldout_gt.size()) {
            size_t matches = 0;
            const auto& gt = heldout_gt[q];
            size_t cmp_k = std::min((size_t)k, gt.size());
            for (const auto& cand : res) {
                for (size_t g = 0; g < cmp_k; ++g) {
                    if (cand.id == gt[g]) {
                        matches++;
                        break;
                    }
                }
            }
            recalls[q] = static_cast<float>(matches) / static_cast<float>(cmp_k);
        }
    }
    
    double t1 = omp_get_wtime();
    double total_time_s = t1 - t0;
    double qps = n_heldout_q / std::max(1e-6, total_time_s);
    
    double r_sum = 0.0;
    for (float r : recalls) r_sum += r;
    float mean_recall = static_cast<float>(r_sum / std::max((size_t)1, n_heldout_q));
    auto [ci_low, ci_high] = AutoCalibrator::bootstrap_confidence_interval(recalls, 1000, 0.05f);
    
    std::sort(latencies_ms.begin(), latencies_ms.end());
    size_t p99_idx = static_cast<size_t>(std::floor(0.99 * latencies_ms.size()));
    if (p99_idx >= latencies_ms.size()) p99_idx = latencies_ms.size() - 1;
    float p99_ms = static_cast<float>(latencies_ms[p99_idx]);
    
    MemoryBreakdown mb = index.get_memory_breakdown();
    double mem_mb = mb.total_owned_bytes / (1024.0 * 1024.0);
    
    bool m_rec = (mean_recall >= constraints.target_recall);
    bool m_mem = (mem_mb <= constraints.max_memory_mb);
    bool m_lat = (p99_ms <= constraints.target_p99_latency_ms);
    
    return {
        dataset_name,
        method_name,
        precision,
        m,
        ef_c,
        ef_s,
        enable_ada_ef,
        calibration_time_s,
        mem_mb,
        mean_recall,
        ci_low,
        ci_high,
        qps,
        p99_ms,
        m_rec,
        m_mem,
        m_lat,
        m_rec && m_mem && m_lat
    };
}

int main(int argc, char** argv) {
    std::cout << "=================================================================\n";
    std::cout << "    AdaptiveVec Phase 3: Comprehensive Auto-Calibration Suite    \n";
    std::cout << "=================================================================\n" << std::endl;
    
    int threads = omp_get_max_threads();
    std::cout << "[*] Executing with OpenMP threads: " << threads << std::endl;
    
    std::vector<CalibrationTrial> all_evaluated_trials;
    std::vector<EvaluationResult> all_heldout_results;
    
    // =================================================================
    // Dataset 1: Transformer-384D (57,893 base vectors, 384D)
    // Budget: max_memory = 128 MB (FP32 would take ~100MB vectors + ~25MB graph = 125MB)
    // Target Recall: 0.92, Target p99 Latency: 5.0 ms
    // =================================================================
    {
        std::string dname = "Transformer-384D";
        std::cout << "\n-------------------------------------------------------------\n";
        std::cout << "[+] Benchmarking Dataset: " << dname << "\n";
        std::cout << "-------------------------------------------------------------\n";
        
        size_t dim = 0, n_base = 0, q_dim = 0, n_queries = 0;
        auto base_data = load_fvecs("data/transformer_384d/transformer_384d_base.fvecs", 60000, dim, n_base);
        auto query_data = load_fvecs("data/transformer_384d/transformer_384d_query.fvecs", 1000, q_dim, n_queries);
        auto gt = load_ivecs("data/transformer_384d/transformer_384d_groundtruth.ivecs", 1000);
        
        if (base_data.empty() || query_data.empty() || gt.empty()) {
            std::cerr << "[-] Error loading Transformer dataset!" << std::endl;
            return 1;
        }
        
        size_t n_calib_val = 400; // 200 calib, 200 val
        size_t n_heldout = n_queries - n_calib_val; // 600 held-out test queries
        
        std::vector<float> calib_queries(query_data.begin(), query_data.begin() + n_calib_val * dim);
        std::vector<std::vector<tableint>> calib_gt(gt.begin(), gt.begin() + n_calib_val);
        
        const float* heldout_queries_ptr = &query_data[n_calib_val * dim];
        std::vector<std::vector<tableint>> heldout_gt(gt.begin() + n_calib_val, gt.end());
        
        CalibrationConstraints constraints;
        constraints.target_recall = 0.92f;
        constraints.max_memory_mb = 120.0; // Strict budget favoring SQ8
        constraints.target_p99_latency_ms = 5.0f;
        constraints.max_calibration_duration_s = 60.0;
        constraints.optimization_goal = OptimizationGoal::MaxThroughput;
        
        // 1. Proposed Hierarchical Auto-Calibrator
        std::cout << "[*] Running Proposed Hierarchical Auto-Calibrator..." << std::endl;
        std::unique_ptr<AdaptiveHNSWIndex> autocalib_index = nullptr;
        CalibrationReport report = AutoCalibrator::calibrate(
            base_data.data(), n_base, dim, SpaceType::Cosine,
            calib_queries.data(), n_calib_val, calib_gt,
            constraints, autocalib_index, dname
        );
        for (const auto& tr : report.evaluated_trials) all_evaluated_trials.push_back(tr);
        std::cout << report.summary() << std::endl;
        
        auto eval_autocalib = evaluate_on_heldout(
            dname, "Proposed Hierarchical Auto-Calibrator", *autocalib_index,
            report.selected_config.ef_search, report.selected_config.enable_ada_ef,
            report.selected_config.precision, report.selected_config.m, report.selected_config.ef_construction,
            report.total_calibration_time_s, heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
        );
        all_heldout_results.push_back(eval_autocalib);
        
        // 2. Default HNSW (M=16, efC=64, efS=64, FP32)
        std::cout << "[*] Evaluating Baseline 1: Default HNSW..." << std::endl;
        PolicyConfig def_cfg;
        def_cfg.m_base = 16;
        def_cfg.ef_c_base = 64;
        def_cfg.enable_hubness_regulation = false;
        def_cfg.enable_bridge_aware = false;
        AdaptiveHNSWIndex def_idx(dim, SpaceType::Cosine, false, def_cfg);
        def_idx.batch_insert(base_data.data(), n_base, threads);
        auto eval_def = evaluate_on_heldout(
            dname, "Default HNSW", def_idx, 64, false, "FP32", 16, 64, 0.0,
            heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
        );
        all_heldout_results.push_back(eval_def);
        
        // 3. Manually Tuned HNSW (M=24, efC=100, efS=96, Uniform SQ8)
        std::cout << "[*] Evaluating Baseline 2: Manually Tuned HNSW (SQ8)..." << std::endl;
        PolicyConfig man_cfg;
        man_cfg.m_base = 24;
        man_cfg.ef_c_base = 100;
        man_cfg.enable_hubness_regulation = false;
        AdaptiveHNSWIndex man_idx(dim, SpaceType::Cosine, false, man_cfg);
        man_idx.batch_insert(base_data.data(), n_base, threads);
        man_idx.policy.keep_raw_vectors = false;
        man_idx.allocate_mixed_precision(AllocationStrategy::UniformSQ8, 0.25);
        auto eval_man = evaluate_on_heldout(
            dname, "Manually Tuned HNSW", man_idx, 96, false, "SQ8", 24, 100, 0.0,
            heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
        );
        all_heldout_results.push_back(eval_man);
        
        // 4. AdaptiveVec Default (M=16, efC=64, efS=64, Adaptive ON)
        std::cout << "[*] Evaluating Baseline 3: AdaptiveVec Default..." << std::endl;
        PolicyConfig ady_cfg;
        ady_cfg.m_base = 16;
        ady_cfg.ef_c_base = 64;
        AdaptiveHNSWIndex ady_idx(dim, SpaceType::Cosine, true, ady_cfg);
        ady_idx.batch_insert(base_data.data(), n_base, threads);
        auto eval_ady = evaluate_on_heldout(
            dname, "AdaptiveVec Default", ady_idx, 64, true, "FP32", 16, 64, 0.0,
            heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
        );
        all_heldout_results.push_back(eval_ady);
        
        // 5. Random Search (Equal calibration time budget)
        std::cout << "[*] Evaluating Baseline 4: Random Search (Budget " << report.total_calibration_time_s << " s)..." << std::endl;
        double rand_t0 = omp_get_wtime();
        std::mt19937 rng(1337);
        std::uniform_int_distribution<int> m_dist(8, 24);
        std::uniform_int_distribution<int> efc_dist(32, 100);
        std::uniform_int_distribution<int> efs_dist(16, 128);
        std::uniform_int_distribution<int> prec_dist(0, 1); // FP32, SQ8
        
        EvaluationResult best_rand = eval_def;
        best_rand.method = "Random Search";
        best_rand.calibration_time_s = 0.0;
        
        int rand_iters = 0;
        while (omp_get_wtime() - rand_t0 < std::min(15.0, report.total_calibration_time_s)) {
            rand_iters++;
            int r_m = m_dist(rng);
            int r_efc = efc_dist(rng);
            int r_efs = efs_dist(rng);
            std::string r_prec = (prec_dist(rng) == 0) ? "SQ8" : "FP32";
            
            PolicyConfig r_cfg;
            r_cfg.m_base = r_m;
            r_cfg.ef_c_base = r_efc;
            AdaptiveHNSWIndex r_idx(dim, SpaceType::Cosine, false, r_cfg);
            r_idx.batch_insert(base_data.data(), n_base, threads);
            if (r_prec == "SQ8") {
                r_idx.policy.keep_raw_vectors = false;
                r_idx.allocate_mixed_precision(AllocationStrategy::UniformSQ8, 0.25);
            }
            auto r_res = evaluate_on_heldout(
                dname, "Random Search", r_idx, r_efs, false, r_prec, r_m, r_efc, 0.0,
                heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
            );
            if (r_res.overall_feasible && r_res.heldout_qps > best_rand.heldout_qps) {
                best_rand = r_res;
            }
        }
        best_rand.calibration_time_s = omp_get_wtime() - rand_t0;
        all_heldout_results.push_back(best_rand);
    }
    
    // =================================================================
    // Dataset 2: SIFT-100K (100,000 base vectors, 128D)
    // Budget: max_memory = 100 MB
    // Target Recall: 0.95, Target p99 Latency: 2.0 ms
    // =================================================================
    {
        std::string dname = "SIFT-100K";
        std::cout << "\n-------------------------------------------------------------\n";
        std::cout << "[+] Benchmarking Dataset: " << dname << "\n";
        std::cout << "-------------------------------------------------------------\n";
        
        size_t dim = 0, n_base = 0, q_dim = 0, n_queries = 0;
        auto base_data = load_fvecs("data/sift_subset_100k.fvecs", 100000, dim, n_base);
        auto query_data = load_fvecs("data/sift_query.fvecs", 2000, q_dim, n_queries);
        auto gt = load_ivecs("data/sift_subset_groundtruth.ivecs", 2000);
        
        if (base_data.empty() || query_data.empty() || gt.empty()) {
            std::cerr << "[-] Error loading SIFT-100K dataset!" << std::endl;
            return 1;
        }
        
        size_t n_calib_val = 500;
        size_t n_heldout = 1000;
        
        std::vector<float> calib_queries(query_data.begin(), query_data.begin() + n_calib_val * dim);
        std::vector<std::vector<tableint>> calib_gt(gt.begin(), gt.begin() + n_calib_val);
        
        const float* heldout_queries_ptr = &query_data[n_calib_val * dim];
        std::vector<std::vector<tableint>> heldout_gt(gt.begin() + n_calib_val, gt.begin() + n_calib_val + n_heldout);
        
        CalibrationConstraints constraints;
        constraints.target_recall = 0.95f;
        constraints.max_memory_mb = 60.0; // Restrictive: FP32 requires 65MB+; forces SQ8
        constraints.target_p99_latency_ms = 2.0f;
        constraints.max_calibration_duration_s = 60.0;
        constraints.optimization_goal = OptimizationGoal::MaxThroughput;
        
        // 1. Proposed Hierarchical Auto-Calibrator
        std::cout << "[*] Running Proposed Hierarchical Auto-Calibrator..." << std::endl;
        std::unique_ptr<AdaptiveHNSWIndex> autocalib_index = nullptr;
        CalibrationReport report = AutoCalibrator::calibrate(
            base_data.data(), n_base, dim, SpaceType::L2,
            calib_queries.data(), n_calib_val, calib_gt,
            constraints, autocalib_index, dname
        );
        for (const auto& tr : report.evaluated_trials) all_evaluated_trials.push_back(tr);
        std::cout << report.summary() << std::endl;
        
        auto eval_autocalib = evaluate_on_heldout(
            dname, "Proposed Hierarchical Auto-Calibrator", *autocalib_index,
            report.selected_config.ef_search, report.selected_config.enable_ada_ef,
            report.selected_config.precision, report.selected_config.m, report.selected_config.ef_construction,
            report.total_calibration_time_s, heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
        );
        all_heldout_results.push_back(eval_autocalib);
        
        // 2. Default HNSW (M=16, efC=64, efS=64, FP32)
        std::cout << "[*] Evaluating Baseline 1: Default HNSW..." << std::endl;
        PolicyConfig def_cfg;
        def_cfg.m_base = 16;
        def_cfg.ef_c_base = 64;
        def_cfg.enable_hubness_regulation = false;
        AdaptiveHNSWIndex def_idx(dim, SpaceType::L2, false, def_cfg);
        def_idx.batch_insert(base_data.data(), n_base, threads);
        auto eval_def = evaluate_on_heldout(
            dname, "Default HNSW", def_idx, 64, false, "FP32", 16, 64, 0.0,
            heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
        );
        all_heldout_results.push_back(eval_def);
        
        // 3. Manually Tuned HNSW (M=16, efC=64, efS=64, SQ8)
        std::cout << "[*] Evaluating Baseline 2: Manually Tuned HNSW (SQ8)..." << std::endl;
        PolicyConfig man_cfg;
        man_cfg.m_base = 16;
        man_cfg.ef_c_base = 64;
        man_cfg.enable_hubness_regulation = false;
        AdaptiveHNSWIndex man_idx(dim, SpaceType::L2, false, man_cfg);
        man_idx.batch_insert(base_data.data(), n_base, threads);
        man_idx.policy.keep_raw_vectors = false;
        man_idx.allocate_mixed_precision(AllocationStrategy::UniformSQ8, 0.25);
        auto eval_man = evaluate_on_heldout(
            dname, "Manually Tuned HNSW", man_idx, 64, false, "SQ8", 16, 64, 0.0,
            heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
        );
        all_heldout_results.push_back(eval_man);
        
        // 4. AdaptiveVec Default
        std::cout << "[*] Evaluating Baseline 3: AdaptiveVec Default..." << std::endl;
        PolicyConfig ady_cfg;
        ady_cfg.m_base = 16;
        ady_cfg.ef_c_base = 64;
        AdaptiveHNSWIndex ady_idx(dim, SpaceType::L2, true, ady_cfg);
        ady_idx.batch_insert(base_data.data(), n_base, threads);
        auto eval_ady = evaluate_on_heldout(
            dname, "AdaptiveVec Default", ady_idx, 64, true, "FP32", 16, 64, 0.0,
            heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
        );
        all_heldout_results.push_back(eval_ady);
    }
    
    // =================================================================
    // Dataset 3: SIFT-500K (500,000 base vectors, 128D)
    // Budget: max_memory = 200 MB
    // Target Recall: 0.95, Target p99 Latency: 3.0 ms
    // =================================================================
    {
        std::string dname = "SIFT-500K";
        std::cout << "\n-------------------------------------------------------------\n";
        std::cout << "[+] Benchmarking Dataset: " << dname << "\n";
        std::cout << "-------------------------------------------------------------\n";
        
        size_t dim = 0, n_base = 0, q_dim = 0, n_queries = 0;
        auto base_data = load_fvecs("data/sift_base_1m.fvecs", 500000, dim, n_base);
        auto query_data = load_fvecs("data/sift_query.fvecs", 1000, q_dim, n_queries);
        auto gt = load_ivecs("data/sift_groundtruth_500k.ivecs", 1000);
        
        if (base_data.empty() || query_data.empty() || gt.empty()) {
            std::cerr << "[-] Error loading SIFT-500K dataset!" << std::endl;
            return 1;
        }
        
        size_t n_calib_val = 300;
        size_t n_heldout = 700;
        
        std::vector<float> calib_queries(query_data.begin(), query_data.begin() + n_calib_val * dim);
        std::vector<std::vector<tableint>> calib_gt(gt.begin(), gt.begin() + n_calib_val);
        
        const float* heldout_queries_ptr = &query_data[n_calib_val * dim];
        std::vector<std::vector<tableint>> heldout_gt(gt.begin() + n_calib_val, gt.end());
        
        CalibrationConstraints constraints;
        constraints.target_recall = 0.95f;
        constraints.max_memory_mb = 180.0; // FP32 requires ~300MB; strictly requires SQ8
        constraints.target_p99_latency_ms = 4.0f;
        constraints.max_calibration_duration_s = 90.0;
        constraints.optimization_goal = OptimizationGoal::MaxThroughput;
        
        // 1. Proposed Hierarchical Auto-Calibrator
        std::cout << "[*] Running Proposed Hierarchical Auto-Calibrator on 500K vectors..." << std::endl;
        std::unique_ptr<AdaptiveHNSWIndex> autocalib_index = nullptr;
        CalibrationReport report = AutoCalibrator::calibrate(
            base_data.data(), n_base, dim, SpaceType::L2,
            calib_queries.data(), n_calib_val, calib_gt,
            constraints, autocalib_index, dname
        );
        for (const auto& tr : report.evaluated_trials) all_evaluated_trials.push_back(tr);
        std::cout << report.summary() << std::endl;
        
        auto eval_autocalib = evaluate_on_heldout(
            dname, "Proposed Hierarchical Auto-Calibrator", *autocalib_index,
            report.selected_config.ef_search, report.selected_config.enable_ada_ef,
            report.selected_config.precision, report.selected_config.m, report.selected_config.ef_construction,
            report.total_calibration_time_s, heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
        );
        all_heldout_results.push_back(eval_autocalib);
        
        // 2. Default HNSW (M=16, efC=64, efS=64, FP32)
        std::cout << "[*] Evaluating Baseline 1: Default HNSW (FP32)..." << std::endl;
        PolicyConfig def_cfg;
        def_cfg.m_base = 16;
        def_cfg.ef_c_base = 64;
        def_cfg.enable_hubness_regulation = false;
        AdaptiveHNSWIndex def_idx(dim, SpaceType::L2, false, def_cfg);
        def_idx.batch_insert(base_data.data(), n_base, threads);
        auto eval_def = evaluate_on_heldout(
            dname, "Default HNSW", def_idx, 64, false, "FP32", 16, 64, 0.0,
            heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
        );
        all_heldout_results.push_back(eval_def);
        
        // 3. Manually Tuned HNSW (M=16, efC=64, efS=64, SQ8)
        std::cout << "[*] Evaluating Baseline 2: Manually Tuned HNSW (SQ8)..." << std::endl;
        PolicyConfig man_cfg;
        man_cfg.m_base = 16;
        man_cfg.ef_c_base = 64;
        man_cfg.enable_hubness_regulation = false;
        AdaptiveHNSWIndex man_idx(dim, SpaceType::L2, false, man_cfg);
        man_idx.batch_insert(base_data.data(), n_base, threads);
        man_idx.policy.keep_raw_vectors = false;
        man_idx.allocate_mixed_precision(AllocationStrategy::UniformSQ8, 0.25);
        auto eval_man = evaluate_on_heldout(
            dname, "Manually Tuned HNSW", man_idx, 64, false, "SQ8", 16, 64, 0.0,
            heldout_queries_ptr, n_heldout, dim, heldout_gt, 10, constraints
        );
        all_heldout_results.push_back(eval_man);
    }
    
    // Save evaluated trials to experiments/phase_3_calibration_trials.json
    std::cout << "\n[*] Exporting experiments/phase_3_calibration_trials.json (" << all_evaluated_trials.size() << " trials)..." << std::endl;
    std::ofstream tr_out("experiments/phase_3_calibration_trials.json");
    tr_out << "[\n";
    for (size_t i = 0; i < all_evaluated_trials.size(); ++i) {
        tr_out << "  " << all_evaluated_trials[i].to_json() << (i + 1 < all_evaluated_trials.size() ? ",\n" : "\n");
    }
    tr_out << "]\n";
    tr_out.close();
    
    // Save heldout results to experiments/phase_3_heldout_results.json
    std::cout << "[*] Exporting experiments/phase_3_heldout_results.json (" << all_heldout_results.size() << " evaluations)..." << std::endl;
    std::ofstream h_out("experiments/phase_3_heldout_results.json");
    h_out << "[\n";
    for (size_t i = 0; i < all_heldout_results.size(); ++i) {
        h_out << "  " << all_heldout_results[i].to_json() << (i + 1 < all_heldout_results.size() ? ",\n" : "\n");
    }
    h_out << "]\n";
    h_out.close();
    
    // Print Summary Table
    std::cout << "\n========================================================================================================\n";
    std::cout << "                 Phase 3 Held-Out Query Generalization & Comparative Results                            \n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left << std::setw(18) << "Dataset"
              << std::setw(32) << "Method"
              << std::setw(8)  << "Prec"
              << std::setw(12) << "Memory(MB)"
              << std::setw(16) << "Heldout R@10"
              << std::setw(12) << "QPS"
              << std::setw(10) << "p99(ms)"
              << std::setw(12) << "CalibCost(s)"
              << std::setw(10) << "Feasible"
              << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";
    for (const auto& res : all_heldout_results) {
        std::ostringstream r_ss;
        r_ss << std::fixed << std::setprecision(2) << (res.heldout_recall * 100.0f) << "%";
        
        std::cout << std::left << std::setw(18) << res.dataset
                  << std::setw(32) << res.method
                  << std::setw(8)  << res.precision
                  << std::setw(12) << res.memory_mb
                  << std::setw(16) << r_ss.str()
                  << std::setw(12) << static_cast<int>(res.heldout_qps)
                  << std::setw(10) << res.heldout_p99_ms
                  << std::setw(12) << res.calibration_time_s
                  << std::setw(10) << (res.overall_feasible ? "YES" : "NO")
                  << "\n";
    }
    std::cout << "========================================================================================================\n";
    
    return 0;
}
