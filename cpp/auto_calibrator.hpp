#pragma once

#include "adaptive_hnsw.hpp"
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <numeric>
#include <memory>

namespace adaptivevec {

enum class OptimizationGoal {
    MaxThroughput = 0,
    MinMemory = 1,
    MinLatency = 2,
    MaxRecall = 3
};

inline std::string to_string(OptimizationGoal goal) {
    switch (goal) {
        case OptimizationGoal::MaxThroughput: return "MaxThroughput";
        case OptimizationGoal::MinMemory: return "MinMemory";
        case OptimizationGoal::MinLatency: return "MinLatency";
        case OptimizationGoal::MaxRecall: return "MaxRecall";
        default: return "Unknown";
    }
}

inline OptimizationGoal parse_optimization_goal(const std::string& str) {
    if (str == "MinMemory" || str == "min_memory") return OptimizationGoal::MinMemory;
    if (str == "MinLatency" || str == "min_latency") return OptimizationGoal::MinLatency;
    if (str == "MaxRecall" || str == "max_recall") return OptimizationGoal::MaxRecall;
    return OptimizationGoal::MaxThroughput;
}

enum class FeasibilityStatus {
    VerifiedFeasible = 0,            // Strictly satisfies recall (including 95% bootstrap lower bound), memory, and latency on calibration queries, and passes validation
    ProvisionallyFeasible = 1,       // Satisfies mean recall, memory, and latency, but 95% bootstrap lower bound dips below target or marginal
    ValidationFailed = 2,            // Passed calibration queries, but failed held-out validation queries
    InfeasibleWithinSearchSpace = 3, // One or more hard constraints violated by all evaluated configurations in search space
    InvalidRequirements = 4,         // Target recall, memory budget, or latency constraint is invalid (< 0, > 1.0, etc.)
    EvaluationError = 5              // Evaluation aborted due to null inputs or execution error
};

inline std::string to_string(FeasibilityStatus status) {
    switch (status) {
        case FeasibilityStatus::VerifiedFeasible: return "VERIFIED_FEASIBLE";
        case FeasibilityStatus::ProvisionallyFeasible: return "PROVISIONALLY_FEASIBLE";
        case FeasibilityStatus::ValidationFailed: return "VALIDATION_FAILED";
        case FeasibilityStatus::InfeasibleWithinSearchSpace: return "INFEASIBLE_WITHIN_SEARCH_SPACE";
        case FeasibilityStatus::InvalidRequirements: return "INVALID_REQUIREMENTS";
        case FeasibilityStatus::EvaluationError: return "EVALUATION_ERROR";
        default: return "UNKNOWN";
    }
}

struct NormalizedViolations {
    double v_recall = 0.0;
    double v_memory = 0.0;
    double v_latency = 0.0;
    double v_total = 0.0;
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "{"
           << "\"v_recall\": " << v_recall << ", "
           << "\"v_memory\": " << v_memory << ", "
           << "\"v_latency\": " << v_latency << ", "
           << "\"v_total\": " << v_total
           << "}";
        return ss.str();
    }
};

struct CalibrationConstraints {
    float target_recall = 0.95f;
    double max_memory_mb = 1024.0;
    float target_p99_latency_ms = 10.0f;
    double max_calibration_duration_s = 60.0;
    double max_construction_time_s = 120.0;
    int threads = 0;
    int k = 10;
    OptimizationGoal optimization_goal = OptimizationGoal::MaxThroughput;
    bool sample_construction_if_large = true;
    size_t sample_size_cap = 20000;
    bool require_conservative_recall = false; // If true, requires CI_lower >= target_recall for Feasible status
};

struct CalibrationTrial {
    int trial_id = 0;
    std::string stage = "SearchSweep"; // "Feasibility", "Construction", "SearchSweep"
    int m = 16;
    int ef_construction = 64;
    int ef_search = 64;
    std::string precision = "FP32"; // "FP32", "SQ8", "SQ4"
    bool enable_adaptive = false;
    bool enable_ada_ef = false;
    
    double memory_mb = 0.0;
    float recall_mean = 0.0f;
    float recall_ci_lower = 0.0f;
    float recall_ci_upper = 0.0f;
    double qps = 0.0;
    float p99_latency_ms = 0.0f;
    double trial_duration_s = 0.0;
    
    bool is_feasible = false;
    bool conservative_feasible = false;
    FeasibilityStatus status = FeasibilityStatus::InfeasibleWithinSearchSpace;
    NormalizedViolations violations;
    std::string rejection_reason = "";
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "{"
           << "\"trial_id\": " << trial_id << ", "
           << "\"stage\": \"" << stage << "\", "
           << "\"m\": " << m << ", "
           << "\"ef_construction\": " << ef_construction << ", "
           << "\"ef_search\": " << ef_search << ", "
           << "\"precision\": \"" << precision << "\", "
           << "\"enable_adaptive\": " << (enable_adaptive ? "true" : "false") << ", "
           << "\"enable_ada_ef\": " << (enable_ada_ef ? "true" : "false") << ", "
           << "\"memory_mb\": " << memory_mb << ", "
           << "\"recall_mean\": " << recall_mean << ", "
           << "\"recall_ci_lower\": " << recall_ci_lower << ", "
           << "\"recall_ci_upper\": " << recall_ci_upper << ", "
           << "\"qps\": " << qps << ", "
           << "\"p99_latency_ms\": " << p99_latency_ms << ", "
           << "\"trial_duration_s\": " << trial_duration_s << ", "
           << "\"is_feasible\": " << (is_feasible ? "true" : "false") << ", "
           << "\"conservative_feasible\": " << (conservative_feasible ? "true" : "false") << ", "
           << "\"status\": \"" << to_string(status) << "\", "
           << "\"violations\": " << violations.to_json() << ", "
           << "\"rejection_reason\": \"" << rejection_reason << "\""
           << "}";
        return ss.str();
    }
};

struct CalibrationReport {
    bool is_feasible = false;
    FeasibilityStatus status = FeasibilityStatus::InfeasibleWithinSearchSpace;
    std::vector<std::string> violated_constraints;
    CalibrationTrial selected_config;
    std::vector<CalibrationTrial> evaluated_trials;
    std::vector<CalibrationTrial> pareto_frontier;
    std::vector<CalibrationTrial> fallback_pareto_set; // Set of non-dominated fallbacks when no configuration satisfies all constraints
    double total_calibration_time_s = 0.0;
    double optimizer_overhead_ms = 0.0;
    size_t num_calibration_queries = 0;
    size_t num_validation_queries = 0;
    std::string dataset_name = "Unknown";
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "{\n"
           << "  \"dataset_name\": \"" << dataset_name << "\",\n"
           << "  \"is_feasible\": " << (is_feasible ? "true" : "false") << ",\n"
           << "  \"status\": \"" << to_string(status) << "\",\n"
           << "  \"total_calibration_time_s\": " << total_calibration_time_s << ",\n"
           << "  \"optimizer_overhead_ms\": " << optimizer_overhead_ms << ",\n"
           << "  \"num_calibration_queries\": " << num_calibration_queries << ",\n"
           << "  \"num_validation_queries\": " << num_validation_queries << ",\n"
           << "  \"violated_constraints\": [";
        for (size_t i = 0; i < violated_constraints.size(); ++i) {
            ss << "\"" << violated_constraints[i] << "\"" << (i + 1 < violated_constraints.size() ? ", " : "");
        }
        ss << "],\n";
        ss << "  \"selected_config\": " << selected_config.to_json() << ",\n";
        ss << "  \"pareto_frontier\": [\n";
        for (size_t i = 0; i < pareto_frontier.size(); ++i) {
            ss << "    " << pareto_frontier[i].to_json() << (i + 1 < pareto_frontier.size() ? ",\n" : "\n");
        }
        ss << "  ],\n";
        ss << "  \"fallback_pareto_set\": [\n";
        for (size_t i = 0; i < fallback_pareto_set.size(); ++i) {
            ss << "    " << fallback_pareto_set[i].to_json() << (i + 1 < fallback_pareto_set.size() ? ",\n" : "\n");
        }
        ss << "  ],\n";
        ss << "  \"evaluated_trials\": [\n";
        for (size_t i = 0; i < evaluated_trials.size(); ++i) {
            ss << "    " << evaluated_trials[i].to_json() << (i + 1 < evaluated_trials.size() ? ",\n" : "\n");
        }
        ss << "  ]\n";
        ss << "}\n";
        return ss.str();
    }
    
    std::string summary() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "=== AdaptiveVec Calibration Report ===\n";
        ss << "Dataset: " << dataset_name << "\n";
        ss << "Status: " << to_string(status) << "\n";
        if (status != FeasibilityStatus::VerifiedFeasible && !violated_constraints.empty()) {
            ss << "Violated Constraints:\n";
            for (const auto& vc : violated_constraints) {
                ss << "  - " << vc << "\n";
            }
        }
        ss << "Selected Config:\n";
        ss << "  Precision: " << selected_config.precision << "\n";
        ss << "  M: " << selected_config.m << ", efConstruction: " << selected_config.ef_construction << "\n";
        ss << "  efSearch: " << selected_config.ef_search << ", Ada-ef: " << (selected_config.enable_ada_ef ? "ON" : "OFF") << "\n";
        ss << "  Estimated Recall@10: " << (selected_config.recall_mean * 100.0f) << "% [95% CI: "
           << (selected_config.recall_ci_lower * 100.0f) << "% - " << (selected_config.recall_ci_upper * 100.0f) << "%]\n";
        ss << "  Memory: " << selected_config.memory_mb << " MB\n";
        ss << "  Throughput: " << selected_config.qps << " QPS\n";
        ss << "  p99 Latency: " << selected_config.p99_latency_ms << " ms\n";
        ss << "Calibration Duration: " << total_calibration_time_s << " s (Optimizer Overhead: " << optimizer_overhead_ms << " ms)\n";
        ss << "Total Evaluated Trials: " << evaluated_trials.size() << " (Pareto Frontier: " << pareto_frontier.size();
        if (!fallback_pareto_set.empty()) {
            ss << ", Fallback Pareto Set: " << fallback_pareto_set.size();
        }
        ss << ")\n";
        return ss.str();
    }
};

class AutoCalibrator {
public:
    static std::pair<float, float> bootstrap_confidence_interval(
        const std::vector<float>& scores, 
        size_t n_bootstraps = 1000, 
        float alpha = 0.05f, 
        uint64_t seed = 42
    ) {
        if (scores.empty()) return {0.0f, 0.0f};
        if (scores.size() == 1) return {scores[0], scores[0]};
        
        std::vector<float> sample_means(n_bootstraps);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<size_t> dist(0, scores.size() - 1);
        
        for (size_t b = 0; b < n_bootstraps; ++b) {
            double sum = 0.0;
            for (size_t i = 0; i < scores.size(); ++i) {
                sum += scores[dist(rng)];
            }
            sample_means[b] = static_cast<float>(sum / scores.size());
        }
        
        std::sort(sample_means.begin(), sample_means.end());
        size_t lower_idx = static_cast<size_t>(std::floor((alpha / 2.0f) * n_bootstraps));
        size_t upper_idx = static_cast<size_t>(std::ceil((1.0f - alpha / 2.0f) * n_bootstraps));
        if (upper_idx >= n_bootstraps) upper_idx = n_bootstraps - 1;
        
        return {sample_means[lower_idx], sample_means[upper_idx]};
    }

    static NormalizedViolations compute_normalized_violations(
        float recall_mean, 
        double memory_mb, 
        float p99_latency_ms,
        const CalibrationConstraints& constraints
    ) {
        NormalizedViolations nv;
        if (recall_mean < constraints.target_recall && constraints.target_recall > 0.0f) {
            nv.v_recall = (constraints.target_recall - recall_mean) / constraints.target_recall;
        }
        if (memory_mb > constraints.max_memory_mb && constraints.max_memory_mb > 0.0) {
            nv.v_memory = (memory_mb - constraints.max_memory_mb) / constraints.max_memory_mb;
        }
        if (p99_latency_ms > constraints.target_p99_latency_ms && constraints.target_p99_latency_ms > 0.0f) {
            nv.v_latency = (p99_latency_ms - constraints.target_p99_latency_ms) / constraints.target_p99_latency_ms;
        }
        nv.v_total = nv.v_recall + nv.v_memory + nv.v_latency;
        return nv;
    }

    static void compute_pareto_frontier(const std::vector<CalibrationTrial>& trials, std::vector<CalibrationTrial>& frontier) {
        frontier.clear();
        for (size_t i = 0; i < trials.size(); ++i) {
            bool dominated = false;
            for (size_t j = 0; j < trials.size(); ++j) {
                if (i == j) continue;
                bool j_better_or_equal = (trials[j].recall_mean >= trials[i].recall_mean) &&
                                         (trials[j].qps >= trials[i].qps) &&
                                         (trials[j].memory_mb <= trials[i].memory_mb) &&
                                         (trials[j].p99_latency_ms <= trials[i].p99_latency_ms);
                bool j_strictly_better = (trials[j].recall_mean > trials[i].recall_mean) ||
                                         (trials[j].qps > trials[i].qps) ||
                                         (trials[j].memory_mb < trials[i].memory_mb) ||
                                         (trials[j].p99_latency_ms < trials[i].p99_latency_ms);
                if (j_better_or_equal && j_strictly_better) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) {
                frontier.push_back(trials[i]);
            }
        }
    }

    static void compute_fallback_pareto_set(const std::vector<CalibrationTrial>& trials, std::vector<CalibrationTrial>& fallbacks) {
        fallbacks.clear();
        // A fallback trial dominates another if its normalized violation is lower and QPS higher
        for (size_t i = 0; i < trials.size(); ++i) {
            bool dominated = false;
            for (size_t j = 0; j < trials.size(); ++j) {
                if (i == j) continue;
                bool j_better_or_equal = (trials[j].violations.v_total <= trials[i].violations.v_total) &&
                                         (trials[j].qps >= trials[i].qps);
                bool j_strictly_better = (trials[j].violations.v_total < trials[i].violations.v_total) ||
                                         (trials[j].qps > trials[i].qps);
                if (j_better_or_equal && j_strictly_better) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) {
                fallbacks.push_back(trials[i]);
            }
        }
        std::sort(fallbacks.begin(), fallbacks.end(), [](const CalibrationTrial& a, const CalibrationTrial& b) {
            return a.violations.v_total < b.violations.v_total;
        });
    }

    static CalibrationReport calibrate(
        const float* base_vectors,
        size_t n_vectors,
        size_t dim,
        SpaceType space,
        const float* query_vectors,
        size_t n_queries,
        const std::vector<std::vector<tableint>>& ground_truth,
        const CalibrationConstraints& constraints,
        std::unique_ptr<AdaptiveHNSWIndex>& out_calibrated_index,
        const std::string& dataset_name = "Unknown"
    ) {
        auto t_start = std::chrono::high_resolution_clock::now();
        CalibrationReport report;
        report.dataset_name = dataset_name;

        // Input parameter validation
        if (constraints.target_recall <= 0.0f || constraints.target_recall > 1.0f ||
            constraints.max_memory_mb <= 0.0 || constraints.target_p99_latency_ms <= 0.0f) {
            report.status = FeasibilityStatus::InvalidRequirements;
            report.is_feasible = false;
            report.violated_constraints.push_back("Invalid constraint parameters supplied.");
            return report;
        }
        if (!base_vectors || n_vectors == 0 || !query_vectors || n_queries == 0) {
            report.status = FeasibilityStatus::EvaluationError;
            report.is_feasible = false;
            report.violated_constraints.push_back("Null or empty dataset / query vectors.");
            return report;
        }
        
        size_t n_calib_q = (n_queries > 1) ? (n_queries / 2) : 1;
        size_t n_val_q = (n_queries > 1) ? (n_queries - n_calib_q) : 1;
        report.num_calibration_queries = n_calib_q;
        report.num_validation_queries = n_val_q;
        
        int trial_counter = 1;
        
        // -------------------------------------------------------------
        // Stage 4 Phase A: Resource Feasibility Estimation & Precision Selection
        // -------------------------------------------------------------
        double vec_fp32_mb = (n_vectors * dim * sizeof(float)) / (1024.0 * 1024.0);
        double vec_sq8_mb = (n_vectors * dim * sizeof(uint8_t)) / (1024.0 * 1024.0);
        size_t packed_dim = (dim + 1) / 2;
        double vec_sq4_mb = (n_vectors * packed_dim * sizeof(uint8_t)) / (1024.0 * 1024.0);
        
        // Lower connectivity boundary check (M=8, 24 bytes per node for levels/locks/ptrs)
        double min_graph_mb = (n_vectors * (8 * sizeof(tableint) + 24)) / (1024.0 * 1024.0);
        double min_total_sq4_mb = vec_sq4_mb + min_graph_mb;
        double min_total_sq8_mb = vec_sq8_mb + min_graph_mb;
        double min_total_fp32_mb = vec_fp32_mb + min_graph_mb;
        
        std::vector<std::string> candidate_precisions;
        if (min_total_sq4_mb > constraints.max_memory_mb) {
            report.status = FeasibilityStatus::InfeasibleWithinSearchSpace;
            report.violated_constraints.push_back("Memory budget " + std::to_string(constraints.max_memory_mb) + 
                " MB is below estimated lower bound for evaluated configuration space (" + std::to_string(min_total_sq4_mb) + " MB)");
            candidate_precisions.push_back("SQ4");
        } else if (min_total_sq8_mb > constraints.max_memory_mb) {
            candidate_precisions.push_back("SQ4");
        } else if (min_total_fp32_mb > constraints.max_memory_mb) {
            candidate_precisions.push_back("SQ8");
            candidate_precisions.push_back("SQ4");
        } else {
            candidate_precisions.push_back("SQ8");
            candidate_precisions.push_back("FP32");
        }
        
        // -------------------------------------------------------------
        // Stage 4 Phase B: Construction Configuration via Bounded Candidate Ladder
        // -------------------------------------------------------------
        struct BuildCandidate {
            int m;
            int ef_c;
        };
        std::vector<BuildCandidate> build_candidates = {
            {8, 30},
            {12, 40},
            {16, 64},
            {24, 100}
        };
        
        CalibrationTrial best_feasible_trial;
        std::unique_ptr<AdaptiveHNSWIndex> best_feasible_index = nullptr;
        bool has_feasible = false;

        CalibrationTrial best_provisional_trial;
        std::unique_ptr<AdaptiveHNSWIndex> best_provisional_index = nullptr;
        bool has_provisional = false;

        CalibrationTrial best_fallback_trial;
        std::unique_ptr<AdaptiveHNSWIndex> best_fallback_index = nullptr;
        bool has_fallback = false;
        
        for (const auto& prec_str : candidate_precisions) {
            for (const auto& bc : build_candidates) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_s = std::chrono::duration<double>(now - t_start).count();
                if (elapsed_s >= constraints.max_calibration_duration_s) {
                    break;
                }
                
                double est_graph_mb = (n_vectors * (bc.m * sizeof(tableint) * 2 + 32)) / (1024.0 * 1024.0);
                double est_vec_mb = (prec_str == "FP32") ? vec_fp32_mb : ((prec_str == "SQ8") ? vec_sq8_mb : vec_sq4_mb);
                double est_total_mb = est_vec_mb + est_graph_mb;
                
                if (est_total_mb > constraints.max_memory_mb * 1.15 && !candidate_precisions.empty() && candidate_precisions[0] != prec_str) {
                    CalibrationTrial reject_trial;
                    reject_trial.trial_id = trial_counter++;
                    reject_trial.stage = "Construction";
                    reject_trial.m = bc.m;
                    reject_trial.ef_construction = bc.ef_c;
                    reject_trial.precision = prec_str;
                    reject_trial.memory_mb = est_total_mb;
                    reject_trial.is_feasible = false;
                    reject_trial.status = FeasibilityStatus::InfeasibleWithinSearchSpace;
                    reject_trial.violations = compute_normalized_violations(0.0f, est_total_mb, 0.0f, constraints);
                    reject_trial.rejection_reason = "Estimated memory " + std::to_string(est_total_mb) + " MB exceeds budget in search space";
                    report.evaluated_trials.push_back(reject_trial);
                    continue;
                }
                
                PolicyConfig cfg;
                cfg.m_base = bc.m;
                cfg.ef_c_base = bc.ef_c;
                cfg.enable_hubness_regulation = false;
                cfg.enable_bridge_aware = false;
                
                auto build_start = std::chrono::high_resolution_clock::now();
                auto index = std::make_unique<AdaptiveHNSWIndex>(dim, space, false, cfg);
                
                index->batch_insert(base_vectors, n_vectors, constraints.threads);
                
                if (prec_str == "SQ8") {
                    index->policy.keep_raw_vectors = false;
                    index->allocate_mixed_precision(AllocationStrategy::UniformSQ8, 0.25);
                } else if (prec_str == "SQ4") {
                    index->policy.keep_raw_vectors = false;
                    index->allocate_mixed_precision(AllocationStrategy::UniformSQ4, 0.125);
                }
                
                auto build_end = std::chrono::high_resolution_clock::now();
                double build_time_s = std::chrono::duration<double>(build_end - build_start).count();
                
                MemoryBreakdown mb = index->get_memory_breakdown();
                double measured_memory_mb = mb.total_owned_bytes / (1024.0 * 1024.0);
                
                // ---------------------------------------------------------
                // Stage 4 Phase C: Search Configuration Sweep on Fixed Graph
                // ---------------------------------------------------------
                std::vector<int> ef_sweeps = {16, 32, 64, 96, 128};
                std::vector<bool> ada_ef_sweeps = {false, true};
                
                for (bool ada_val : ada_ef_sweeps) {
                    index->policy.enable_ada_ef = ada_val;
                    
                    for (int ef_s : ef_sweeps) {
                        auto trial_t0 = std::chrono::high_resolution_clock::now();
                        
                        std::vector<float> per_query_recall(n_calib_q, 0.0f);
                        std::vector<double> latencies_ms(n_calib_q, 0.0);
                        
                        double calib_search_t0 = omp_get_wtime();
                        
                        #pragma omp parallel for schedule(dynamic, 16)
                        for (int q = 0; q < (int)n_calib_q; ++q) {
                            const float* q_ptr = &query_vectors[q * dim];
                            
                            auto q_start = std::chrono::high_resolution_clock::now();
                            auto res = index->search(q_ptr, constraints.k, ef_s);
                            auto q_end = std::chrono::high_resolution_clock::now();
                            
                            latencies_ms[q] = std::chrono::duration<double, std::milli>(q_end - q_start).count();
                            
                            if (q < (int)ground_truth.size() && !ground_truth[q].empty()) {
                                size_t match_cnt = 0;
                                const auto& gt = ground_truth[q];
                                size_t compare_k = std::min((size_t)constraints.k, gt.size());
                                for (const auto& cand : res) {
                                    for (size_t g = 0; g < compare_k; ++g) {
                                        if (cand.id == gt[g]) {
                                            match_cnt++;
                                            break;
                                        }
                                    }
                                }
                                per_query_recall[q] = static_cast<float>(match_cnt) / static_cast<float>(compare_k);
                            }
                        }
                        
                        double calib_search_t1 = omp_get_wtime();
                        double total_calib_time_s = calib_search_t1 - calib_search_t0;
                        double measured_qps = n_calib_q / std::max(1e-6, total_calib_time_s);
                        
                        double r_sum = 0.0;
                        for (float r : per_query_recall) r_sum += r;
                        float mean_recall = static_cast<float>(r_sum / std::max((size_t)1, n_calib_q));
                        auto [ci_low, ci_high] = bootstrap_confidence_interval(per_query_recall, 1000, 0.05f);
                        
                        std::sort(latencies_ms.begin(), latencies_ms.end());
                        size_t p99_idx = static_cast<size_t>(std::floor(0.99 * latencies_ms.size()));
                        if (p99_idx >= latencies_ms.size()) p99_idx = latencies_ms.size() - 1;
                        float p99_lat = static_cast<float>(latencies_ms[p99_idx]);
                        
                        auto trial_t1 = std::chrono::high_resolution_clock::now();
                        double trial_sec = std::chrono::duration<double>(trial_t1 - trial_t0).count();
                        
                        NormalizedViolations nv = compute_normalized_violations(mean_recall, measured_memory_mb, p99_lat, constraints);
                        
                        bool satisfies_mean_recall = (mean_recall >= constraints.target_recall);
                        bool satisfies_ci_recall = (ci_low >= constraints.target_recall);
                        bool satisfies_memory = (measured_memory_mb <= constraints.max_memory_mb);
                        bool satisfies_latency = (p99_lat <= constraints.target_p99_latency_ms);
                        
                        FeasibilityStatus trial_status = FeasibilityStatus::InfeasibleWithinSearchSpace;
                        if (satisfies_mean_recall && satisfies_memory && satisfies_latency) {
                            if (satisfies_ci_recall || !constraints.require_conservative_recall) {
                                trial_status = FeasibilityStatus::VerifiedFeasible;
                            } else {
                                trial_status = FeasibilityStatus::ProvisionallyFeasible;
                            }
                        }
                        
                        std::string rej = "";
                        if (!satisfies_mean_recall) rej += "[Recall " + std::to_string(mean_recall) + " < " + std::to_string(constraints.target_recall) + "] ";
                        if (!satisfies_memory) rej += "[Memory " + std::to_string(measured_memory_mb) + " MB > " + std::to_string(constraints.max_memory_mb) + " MB] ";
                        if (!satisfies_latency) rej += "[p99 Latency " + std::to_string(p99_lat) + " ms > " + std::to_string(constraints.target_p99_latency_ms) + " ms]";
                        
                        CalibrationTrial trial;
                        trial.trial_id = trial_counter++;
                        trial.stage = "SearchSweep";
                        trial.m = bc.m;
                        trial.ef_construction = bc.ef_c;
                        trial.ef_search = ef_s;
                        trial.precision = prec_str;
                        trial.enable_adaptive = false;
                        trial.enable_ada_ef = ada_val;
                        trial.memory_mb = measured_memory_mb;
                        trial.recall_mean = mean_recall;
                        trial.recall_ci_lower = ci_low;
                        trial.recall_ci_upper = ci_high;
                        trial.qps = measured_qps;
                        trial.p99_latency_ms = p99_lat;
                        trial.trial_duration_s = trial_sec;
                        trial.is_feasible = (trial_status == FeasibilityStatus::VerifiedFeasible);
                        trial.conservative_feasible = satisfies_ci_recall;
                        trial.status = trial_status;
                        trial.violations = nv;
                        trial.rejection_reason = rej;
                        
                        report.evaluated_trials.push_back(trial);
                        
                        // Rigorous candidate selection logic strictly partitioned by status
                        if (trial_status == FeasibilityStatus::VerifiedFeasible) {
                            bool should_replace = false;
                            if (!has_feasible) {
                                should_replace = true;
                            } else {
                                switch (constraints.optimization_goal) {
                                    case OptimizationGoal::MaxThroughput:
                                        should_replace = (trial.qps > best_feasible_trial.qps);
                                        break;
                                    case OptimizationGoal::MinMemory:
                                        should_replace = (trial.memory_mb < best_feasible_trial.memory_mb) ||
                                                         (std::abs(trial.memory_mb - best_feasible_trial.memory_mb) < 0.1 && trial.qps > best_feasible_trial.qps);
                                        break;
                                    case OptimizationGoal::MinLatency:
                                        should_replace = (trial.p99_latency_ms < best_feasible_trial.p99_latency_ms);
                                        break;
                                    case OptimizationGoal::MaxRecall:
                                        should_replace = (trial.recall_mean > best_feasible_trial.recall_mean);
                                        break;
                                }
                            }
                            if (should_replace) {
                                has_feasible = true;
                                best_feasible_trial = trial;
                                best_feasible_index = index->clone();
                                best_feasible_index->policy.enable_ada_ef = trial.enable_ada_ef;
                            }
                        } else if (trial_status == FeasibilityStatus::ProvisionallyFeasible) {
                            if (!has_provisional || trial.qps > best_provisional_trial.qps) {
                                has_provisional = true;
                                best_provisional_trial = trial;
                                best_provisional_index = index->clone();
                                best_provisional_index->policy.enable_ada_ef = trial.enable_ada_ef;
                            }
                        }
                        
                        // Update fallback strictly among non-feasible candidates (or if none yet)
                        if (!has_fallback || trial.violations.v_total < best_fallback_trial.violations.v_total) {
                            has_fallback = true;
                            best_fallback_trial = trial;
                            best_fallback_index = index->clone();
                            best_fallback_index->policy.enable_ada_ef = trial.enable_ada_ef;
                        }
                    }
                }
            }
        }
        
        // -------------------------------------------------------------
        // Independent Held-Out Validation Query Evaluation
        // -------------------------------------------------------------
        if (has_feasible && best_feasible_index) {
            size_t val_start_idx = (n_queries > 1) ? n_calib_q : 0;
            std::vector<float> val_per_query_recall(n_val_q, 0.0f);
            std::vector<double> val_latencies(n_val_q, 0.0);
            best_feasible_index->policy.enable_ada_ef = best_feasible_trial.enable_ada_ef;
            
            #pragma omp parallel for schedule(dynamic, 16)
            for (int q = 0; q < (int)n_val_q; ++q) {
                size_t q_idx = val_start_idx + q;
                const float* q_ptr = &query_vectors[q_idx * dim];
                auto q_start = std::chrono::high_resolution_clock::now();
                auto res = best_feasible_index->search(q_ptr, constraints.k, best_feasible_trial.ef_search);
                auto q_end = std::chrono::high_resolution_clock::now();
                val_latencies[q] = std::chrono::duration<double, std::milli>(q_end - q_start).count();
                
                if (q_idx < ground_truth.size() && !ground_truth[q_idx].empty()) {
                    size_t match_cnt = 0;
                    const auto& gt = ground_truth[q_idx];
                    size_t comp_k = std::min((size_t)constraints.k, gt.size());
                    for (const auto& cand : res) {
                        for (size_t g = 0; g < comp_k; ++g) {
                            if (cand.id == gt[g]) {
                                match_cnt++;
                                break;
                            }
                        }
                    }
                    val_per_query_recall[q] = static_cast<float>(match_cnt) / static_cast<float>(comp_k);
                }
            }
            
            double val_r_sum = 0.0;
            for (float r : val_per_query_recall) val_r_sum += r;
            float val_mean_recall = static_cast<float>(val_r_sum / std::max((size_t)1, n_val_q));
            
            std::sort(val_latencies.begin(), val_latencies.end());
            size_t p99_idx = static_cast<size_t>(std::floor(0.99 * val_latencies.size()));
            if (p99_idx >= val_latencies.size()) p99_idx = val_latencies.size() - 1;
            float val_p99_lat = static_cast<float>(val_latencies[p99_idx]);
            
            if (val_mean_recall >= constraints.target_recall * 0.98f && val_p99_lat <= constraints.target_p99_latency_ms * 1.10f) {
                report.is_feasible = true;
                report.status = FeasibilityStatus::VerifiedFeasible;
                best_feasible_trial.status = FeasibilityStatus::VerifiedFeasible;
                report.selected_config = best_feasible_trial;
                out_calibrated_index = std::move(best_feasible_index);
            } else {
                report.is_feasible = false;
                report.status = FeasibilityStatus::ValidationFailed;
                best_feasible_trial.status = FeasibilityStatus::ValidationFailed;
                report.selected_config = best_feasible_trial;
                report.violated_constraints.push_back("Held-out validation failed: Validation Recall=" + 
                    std::to_string(val_mean_recall) + " vs target=" + std::to_string(constraints.target_recall) +
                    ", Validation p99=" + std::to_string(val_p99_lat) + " ms vs target=" + std::to_string(constraints.target_p99_latency_ms) + " ms");
                out_calibrated_index = std::move(best_feasible_index);
            }
        } else if (has_provisional && best_provisional_index) {
            report.is_feasible = false;
            report.status = FeasibilityStatus::ProvisionallyFeasible;
            report.selected_config = best_provisional_trial;
            report.violated_constraints.push_back("Selected configuration is provisionally feasible (mean recall met, but bootstrap CI lower bound dips below target).");
            out_calibrated_index = std::move(best_provisional_index);
        } else {
            report.is_feasible = false;
            report.status = FeasibilityStatus::InfeasibleWithinSearchSpace;
            report.selected_config = best_fallback_trial;
            if (best_fallback_trial.recall_mean < constraints.target_recall) {
                report.violated_constraints.push_back("Target recall " + std::to_string(constraints.target_recall) + 
                    " not achieved (best achieved: " + std::to_string(best_fallback_trial.recall_mean) + 
                    ", normalized violation: " + std::to_string(best_fallback_trial.violations.v_recall) + ")");
            }
            if (best_fallback_trial.memory_mb > constraints.max_memory_mb) {
                report.violated_constraints.push_back("Memory budget " + std::to_string(constraints.max_memory_mb) + 
                    " MB exceeded (best achieved: " + std::to_string(best_fallback_trial.memory_mb) + 
                    " MB, normalized violation: " + std::to_string(best_fallback_trial.violations.v_memory) + ")");
            }
            if (best_fallback_trial.p99_latency_ms > constraints.target_p99_latency_ms) {
                report.violated_constraints.push_back("p99 latency target " + std::to_string(constraints.target_p99_latency_ms) + 
                    " ms exceeded (best achieved: " + std::to_string(best_fallback_trial.p99_latency_ms) + 
                    " ms, normalized violation: " + std::to_string(best_fallback_trial.violations.v_latency) + ")");
            }
            if (best_fallback_index) {
                out_calibrated_index = std::move(best_fallback_index);
            }
        }
        
        auto t_end = std::chrono::high_resolution_clock::now();
        report.total_calibration_time_s = std::chrono::duration<double>(t_end - t_start).count();
        
        // Compute Pareto frontier & fallback Pareto set
        compute_pareto_frontier(report.evaluated_trials, report.pareto_frontier);
        compute_fallback_pareto_set(report.evaluated_trials, report.fallback_pareto_set);
        
        return report;
    }

    static CalibrationReport calibrate_search(
        AdaptiveHNSWIndex& existing_index,
        const float* query_vectors,
        size_t n_queries,
        const std::vector<std::vector<tableint>>& ground_truth,
        const CalibrationConstraints& constraints,
        const std::string& dataset_name = "PrebuiltGraph"
    ) {
        auto t_start = std::chrono::high_resolution_clock::now();
        CalibrationReport report;
        report.dataset_name = dataset_name;

        if (constraints.target_recall <= 0.0f || constraints.target_recall > 1.0f ||
            constraints.max_memory_mb <= 0.0 || constraints.target_p99_latency_ms <= 0.0f) {
            report.status = FeasibilityStatus::InvalidRequirements;
            report.is_feasible = false;
            report.violated_constraints.push_back("Invalid constraint parameters supplied.");
            return report;
        }
        if (!query_vectors || n_queries == 0) {
            report.status = FeasibilityStatus::EvaluationError;
            report.is_feasible = false;
            report.violated_constraints.push_back("Null or empty query vectors.");
            return report;
        }

        size_t n_calib_q = (n_queries > 1) ? (n_queries / 2) : 1;
        size_t n_val_q = (n_queries > 1) ? (n_queries - n_calib_q) : 1;
        report.num_calibration_queries = n_calib_q;
        report.num_validation_queries = n_val_q;

        MemoryBreakdown mb = existing_index.get_memory_breakdown();
        double measured_memory_mb = mb.total_owned_bytes / (1024.0 * 1024.0);

        std::vector<int> ef_sweeps = {16, 32, 64, 96, 128};
        std::vector<bool> ada_ef_sweeps = {false, true};

        CalibrationTrial best_feasible_trial;
        bool has_feasible = false;
        CalibrationTrial best_provisional_trial;
        bool has_provisional = false;
        CalibrationTrial best_fallback_trial;
        bool has_fallback = false;

        int trial_counter = 1;
        size_t dim = existing_index.dim;

        for (bool ada_val : ada_ef_sweeps) {
            existing_index.policy.enable_ada_ef = ada_val;
            for (int ef_s : ef_sweeps) {
                auto trial_t0 = std::chrono::high_resolution_clock::now();
                std::vector<float> per_query_recall(n_calib_q, 0.0f);
                std::vector<double> latencies_ms(n_calib_q, 0.0);

                double calib_search_t0 = omp_get_wtime();
                #pragma omp parallel for schedule(dynamic, 16)
                for (int q = 0; q < (int)n_calib_q; ++q) {
                    const float* q_ptr = &query_vectors[q * dim];
                    auto q_start = std::chrono::high_resolution_clock::now();
                    auto res = existing_index.search(q_ptr, constraints.k, ef_s);
                    auto q_end = std::chrono::high_resolution_clock::now();
                    latencies_ms[q] = std::chrono::duration<double, std::milli>(q_end - q_start).count();

                    if (q < (int)ground_truth.size() && !ground_truth[q].empty()) {
                        size_t match_cnt = 0;
                        const auto& gt = ground_truth[q];
                        size_t comp_k = std::min((size_t)constraints.k, gt.size());
                        for (const auto& cand : res) {
                            for (size_t g = 0; g < comp_k; ++g) {
                                if (cand.id == gt[g]) {
                                    match_cnt++;
                                    break;
                                }
                            }
                        }
                        per_query_recall[q] = static_cast<float>(match_cnt) / static_cast<float>(comp_k);
                    }
                }
                double calib_search_t1 = omp_get_wtime();
                double total_calib_time_s = calib_search_t1 - calib_search_t0;
                double measured_qps = n_calib_q / std::max(1e-6, total_calib_time_s);

                double r_sum = 0.0;
                for (float r : per_query_recall) r_sum += r;
                float mean_recall = static_cast<float>(r_sum / std::max((size_t)1, n_calib_q));
                auto [ci_low, ci_high] = bootstrap_confidence_interval(per_query_recall, 1000, 0.05f);

                std::sort(latencies_ms.begin(), latencies_ms.end());
                size_t p99_idx = static_cast<size_t>(std::floor(0.99 * latencies_ms.size()));
                if (p99_idx >= latencies_ms.size()) p99_idx = latencies_ms.size() - 1;
                float p99_lat = static_cast<float>(latencies_ms[p99_idx]);

                auto trial_t1 = std::chrono::high_resolution_clock::now();
                double trial_sec = std::chrono::duration<double>(trial_t1 - trial_t0).count();

                NormalizedViolations nv = compute_normalized_violations(mean_recall, measured_memory_mb, p99_lat, constraints);

                bool satisfies_mean_recall = (mean_recall >= constraints.target_recall);
                bool satisfies_ci_recall = (ci_low >= constraints.target_recall);
                bool satisfies_memory = (measured_memory_mb <= constraints.max_memory_mb);
                bool satisfies_latency = (p99_lat <= constraints.target_p99_latency_ms);

                FeasibilityStatus trial_status = FeasibilityStatus::InfeasibleWithinSearchSpace;
                if (satisfies_mean_recall && satisfies_memory && satisfies_latency) {
                    if (satisfies_ci_recall || !constraints.require_conservative_recall) {
                        trial_status = FeasibilityStatus::VerifiedFeasible;
                    } else {
                        trial_status = FeasibilityStatus::ProvisionallyFeasible;
                    }
                }

                CalibrationTrial trial;
                trial.trial_id = trial_counter++;
                trial.stage = "SearchSweep";
                trial.m = existing_index.policy.m_base;
                trial.ef_construction = existing_index.policy.ef_c_base;
                trial.ef_search = ef_s;
                trial.precision = existing_index.policy.enable_sq4 ? "SQ4" : (existing_index.policy.enable_sq8 ? "SQ8" : "FP32");
                trial.enable_adaptive = existing_index.policy.enable_hubness_regulation;
                trial.enable_ada_ef = ada_val;
                trial.memory_mb = measured_memory_mb;
                trial.recall_mean = mean_recall;
                trial.recall_ci_lower = ci_low;
                trial.recall_ci_upper = ci_high;
                trial.qps = measured_qps;
                trial.p99_latency_ms = p99_lat;
                trial.trial_duration_s = trial_sec;
                trial.is_feasible = (trial_status == FeasibilityStatus::VerifiedFeasible);
                trial.conservative_feasible = satisfies_ci_recall;
                trial.status = trial_status;
                trial.violations = nv;

                report.evaluated_trials.push_back(trial);

                if (trial_status == FeasibilityStatus::VerifiedFeasible) {
                    bool should_replace = false;
                    if (!has_feasible) {
                        should_replace = true;
                    } else {
                        switch (constraints.optimization_goal) {
                            case OptimizationGoal::MaxThroughput:
                                should_replace = (trial.qps > best_feasible_trial.qps);
                                break;
                            case OptimizationGoal::MinMemory:
                                should_replace = (trial.memory_mb < best_feasible_trial.memory_mb) ||
                                                 (std::abs(trial.memory_mb - best_feasible_trial.memory_mb) < 0.1 && trial.qps > best_feasible_trial.qps);
                                break;
                            case OptimizationGoal::MinLatency:
                                should_replace = (trial.p99_latency_ms < best_feasible_trial.p99_latency_ms);
                                break;
                            case OptimizationGoal::MaxRecall:
                                should_replace = (trial.recall_mean > best_feasible_trial.recall_mean);
                                break;
                        }
                    }
                    if (should_replace) {
                        has_feasible = true;
                        best_feasible_trial = trial;
                    }
                } else if (trial_status == FeasibilityStatus::ProvisionallyFeasible) {
                    if (!has_provisional || trial.qps > best_provisional_trial.qps) {
                        has_provisional = true;
                        best_provisional_trial = trial;
                    }
                }

                if (!has_fallback || trial.violations.v_total < best_fallback_trial.violations.v_total) {
                    has_fallback = true;
                    best_fallback_trial = trial;
                }
            }
        }

        // Held-out validation check
        if (has_feasible) {
            size_t val_start_idx = (n_queries > 1) ? n_calib_q : 0;
            std::vector<float> val_per_query_recall(n_val_q, 0.0f);
            std::vector<double> val_latencies(n_val_q, 0.0);
            existing_index.policy.enable_ada_ef = best_feasible_trial.enable_ada_ef;

            #pragma omp parallel for schedule(dynamic, 16)
            for (int q = 0; q < (int)n_val_q; ++q) {
                size_t q_idx = val_start_idx + q;
                const float* q_ptr = &query_vectors[q_idx * dim];
                auto q_start = std::chrono::high_resolution_clock::now();
                auto res = existing_index.search(q_ptr, constraints.k, best_feasible_trial.ef_search);
                auto q_end = std::chrono::high_resolution_clock::now();
                val_latencies[q] = std::chrono::duration<double, std::milli>(q_end - q_start).count();

                if (q_idx < ground_truth.size() && !ground_truth[q_idx].empty()) {
                    size_t match_cnt = 0;
                    const auto& gt = ground_truth[q_idx];
                    size_t comp_k = std::min((size_t)constraints.k, gt.size());
                    for (const auto& cand : res) {
                        for (size_t g = 0; g < comp_k; ++g) {
                            if (cand.id == gt[g]) {
                                match_cnt++;
                                break;
                            }
                        }
                    }
                    val_per_query_recall[q] = static_cast<float>(match_cnt) / static_cast<float>(comp_k);
                }
            }

            double val_r_sum = 0.0;
            for (float r : val_per_query_recall) val_r_sum += r;
            float val_mean_recall = static_cast<float>(val_r_sum / std::max((size_t)1, n_val_q));

            std::sort(val_latencies.begin(), val_latencies.end());
            size_t p99_idx = static_cast<size_t>(std::floor(0.99 * val_latencies.size()));
            if (p99_idx >= val_latencies.size()) p99_idx = val_latencies.size() - 1;
            float val_p99_lat = static_cast<float>(val_latencies[p99_idx]);

            if (val_mean_recall >= constraints.target_recall * 0.98f && val_p99_lat <= constraints.target_p99_latency_ms * 1.10f) {
                report.is_feasible = true;
                report.status = FeasibilityStatus::VerifiedFeasible;
                best_feasible_trial.status = FeasibilityStatus::VerifiedFeasible;
                report.selected_config = best_feasible_trial;
            } else {
                report.is_feasible = false;
                report.status = FeasibilityStatus::ValidationFailed;
                best_feasible_trial.status = FeasibilityStatus::ValidationFailed;
                report.selected_config = best_feasible_trial;
                report.violated_constraints.push_back("Held-out validation failed: Validation Recall=" + 
                    std::to_string(val_mean_recall) + " vs target=" + std::to_string(constraints.target_recall));
            }
        } else if (has_provisional) {
            report.is_feasible = false;
            report.status = FeasibilityStatus::ProvisionallyFeasible;
            report.selected_config = best_provisional_trial;
            report.violated_constraints.push_back("Configuration is provisionally feasible but 95% CI dips below target.");
        } else {
            report.is_feasible = false;
            report.status = FeasibilityStatus::InfeasibleWithinSearchSpace;
            report.selected_config = best_fallback_trial;
            if (best_fallback_trial.recall_mean < constraints.target_recall) {
                report.violated_constraints.push_back("Target recall " + std::to_string(constraints.target_recall) + " not achieved.");
            }
            if (best_fallback_trial.memory_mb > constraints.max_memory_mb) {
                report.violated_constraints.push_back("Memory budget " + std::to_string(constraints.max_memory_mb) + " MB exceeded.");
            }
            if (best_fallback_trial.p99_latency_ms > constraints.target_p99_latency_ms) {
                report.violated_constraints.push_back("p99 latency target " + std::to_string(constraints.target_p99_latency_ms) + " ms exceeded.");
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        report.total_calibration_time_s = std::chrono::duration<double>(t_end - t_start).count();
        compute_pareto_frontier(report.evaluated_trials, report.pareto_frontier);
        compute_fallback_pareto_set(report.evaluated_trials, report.fallback_pareto_set);

        return report;
    }
};

} // namespace adaptivevec
