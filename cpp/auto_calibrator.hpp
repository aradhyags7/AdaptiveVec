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
           << "\"rejection_reason\": \"" << rejection_reason << "\""
           << "}";
        return ss.str();
    }
};

struct CalibrationReport {
    bool is_feasible = false;
    std::vector<std::string> violated_constraints;
    CalibrationTrial selected_config;
    std::vector<CalibrationTrial> evaluated_trials;
    std::vector<CalibrationTrial> pareto_frontier;
    double total_calibration_time_s = 0.0;
    size_t num_calibration_queries = 0;
    size_t num_validation_queries = 0;
    std::string dataset_name = "Unknown";
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "{\n"
           << "  \"dataset_name\": \"" << dataset_name << "\",\n"
           << "  \"is_feasible\": " << (is_feasible ? "true" : "false") << ",\n"
           << "  \"total_calibration_time_s\": " << total_calibration_time_s << ",\n"
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
        ss << "Feasible: " << (is_feasible ? "YES" : "NO") << "\n";
        if (!is_feasible && !violated_constraints.empty()) {
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
        ss << "Calibration Time: " << total_calibration_time_s << " s\n";
        ss << "Total Evaluated Trials: " << evaluated_trials.size() << " (Pareto Frontier: " << pareto_frontier.size() << ")\n";
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

    static void compute_pareto_frontier(std::vector<CalibrationTrial>& trials, std::vector<CalibrationTrial>& frontier) {
        frontier.clear();
        for (size_t i = 0; i < trials.size(); ++i) {
            bool dominated = false;
            for (size_t j = 0; j < trials.size(); ++j) {
                if (i == j) continue;
                // j dominates i if j is >= in recall & QPS, and <= in memory & latency, with at least one strictly better
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
        
        // Split queries: 50% calibration, 50% validation
        size_t n_calib_q = n_queries / 2;
        size_t n_val_q = n_queries - n_calib_q;
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
        
        // Minimum graph overhead estimate (M=8, 24 bytes per node for levels/locks/ptrs)
        double min_graph_mb = (n_vectors * (8 * sizeof(tableint) + 24)) / (1024.0 * 1024.0);
        double min_total_sq4_mb = vec_sq4_mb + min_graph_mb;
        double min_total_sq8_mb = vec_sq8_mb + min_graph_mb;
        double min_total_fp32_mb = vec_fp32_mb + min_graph_mb;
        
        std::vector<std::string> candidate_precisions;
        if (min_total_sq4_mb > constraints.max_memory_mb) {
            report.is_feasible = false;
            report.violated_constraints.push_back("Memory budget " + std::to_string(constraints.max_memory_mb) + 
                " MB is strictly below minimum possible 4-bit index requirement (" + std::to_string(min_total_sq4_mb) + " MB)");
            // Still explore SQ4 to return best-effort configuration
            candidate_precisions.push_back("SQ4");
        } else if (min_total_sq8_mb > constraints.max_memory_mb) {
            candidate_precisions.push_back("SQ4");
        } else if (min_total_fp32_mb > constraints.max_memory_mb) {
            candidate_precisions.push_back("SQ8");
            candidate_precisions.push_back("SQ4");
        } else {
            // Memory allows FP32; test both FP32 and SQ8 (Uniform SQ8 preferred if throughput/memory matches)
            candidate_precisions.push_back("SQ8");
            candidate_precisions.push_back("FP32");
        }
        
        // -------------------------------------------------------------
        // Stage 4 Phase B: Construction Configuration via Bounded Candidate Search
        // -------------------------------------------------------------
        struct BuildCandidate {
            int m;
            int ef_c;
        };
        std::vector<BuildCandidate> build_candidates = {
            {12, 40},
            {16, 64},
            {24, 100}
        };
        
        std::unique_ptr<AdaptiveHNSWIndex> best_index = nullptr;
        CalibrationTrial best_trial;
        bool found_feasible = false;
        
        for (const auto& prec_str : candidate_precisions) {
            for (const auto& bc : build_candidates) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_s = std::chrono::duration<double>(now - t_start).count();
                if (elapsed_s >= constraints.max_calibration_duration_s) {
                    break;
                }
                
                // Estimate memory for this configuration
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
                    reject_trial.rejection_reason = "Estimated memory " + std::to_string(est_total_mb) + " MB exceeds budget";
                    report.evaluated_trials.push_back(reject_trial);
                    continue;
                }
                
                // Construct Index
                PolicyConfig cfg;
                cfg.m_base = bc.m;
                cfg.ef_c_base = bc.ef_c;
                cfg.enable_hubness_regulation = false; // Phase 1/2.1 verified: uniform canonical connectivity provides best baseline
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
                        
                        // Evaluate on Validation Queries
                        std::vector<float> per_query_recall(n_val_q, 0.0f);
                        std::vector<double> latencies_ms(n_val_q, 0.0);
                        
                        double val_search_t0 = omp_get_wtime();
                        
                        #pragma omp parallel for schedule(dynamic, 16)
                        for (int q = 0; q < (int)n_val_q; ++q) {
                            size_t query_idx = n_calib_q + q;
                            const float* q_ptr = &query_vectors[query_idx * dim];
                            
                            auto q_start = std::chrono::high_resolution_clock::now();
                            auto res = index->search(q_ptr, constraints.k, ef_s);
                            auto q_end = std::chrono::high_resolution_clock::now();
                            
                            latencies_ms[q] = std::chrono::duration<double, std::milli>(q_end - q_start).count();
                            
                            // Recall calculation
                            if (query_idx < ground_truth.size() && !ground_truth[query_idx].empty()) {
                                size_t match_cnt = 0;
                                const auto& gt = ground_truth[query_idx];
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
                        
                        double val_search_t1 = omp_get_wtime();
                        double total_val_time_s = val_search_t1 - val_search_t0;
                        double measured_qps = n_val_q / std::max(1e-6, total_val_time_s);
                        
                        // Compute mean recall and 95% bootstrap CI
                        double r_sum = 0.0;
                        for (float r : per_query_recall) r_sum += r;
                        float mean_recall = static_cast<float>(r_sum / std::max((size_t)1, n_val_q));
                        auto [ci_low, ci_high] = bootstrap_confidence_interval(per_query_recall, 1000, 0.05f);
                        
                        // Compute p99 latency
                        std::sort(latencies_ms.begin(), latencies_ms.end());
                        size_t p99_idx = static_cast<size_t>(std::floor(0.99 * latencies_ms.size()));
                        if (p99_idx >= latencies_ms.size()) p99_idx = latencies_ms.size() - 1;
                        float p99_lat = static_cast<float>(latencies_ms[p99_idx]);
                        
                        auto trial_t1 = std::chrono::high_resolution_clock::now();
                        double trial_sec = std::chrono::duration<double>(trial_t1 - trial_t0).count();
                        
                        // Constraint Verification
                        bool satisfies_recall = (mean_recall >= constraints.target_recall);
                        bool satisfies_memory = (measured_memory_mb <= constraints.max_memory_mb);
                        bool satisfies_latency = (p99_lat <= constraints.target_p99_latency_ms);
                        bool trial_feasible = satisfies_recall && satisfies_memory && satisfies_latency;
                        
                        std::string rej = "";
                        if (!satisfies_recall) rej += "[Recall " + std::to_string(mean_recall) + " < " + std::to_string(constraints.target_recall) + "] ";
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
                        trial.is_feasible = trial_feasible;
                        trial.rejection_reason = rej;
                        
                        report.evaluated_trials.push_back(trial);
                        
                        // Track best configuration according to constraints and optimization goal
                        bool should_replace = false;
                        if (trial_feasible) {
                            if (!found_feasible) {
                                should_replace = true;
                                found_feasible = true;
                            } else {
                                switch (constraints.optimization_goal) {
                                    case OptimizationGoal::MaxThroughput:
                                        should_replace = (trial.qps > best_trial.qps);
                                        break;
                                    case OptimizationGoal::MinMemory:
                                        should_replace = (trial.memory_mb < best_trial.memory_mb) ||
                                                         (std::abs(trial.memory_mb - best_trial.memory_mb) < 0.1 && trial.qps > best_trial.qps);
                                        break;
                                    case OptimizationGoal::MinLatency:
                                        should_replace = (trial.p99_latency_ms < best_trial.p99_latency_ms);
                                        break;
                                    case OptimizationGoal::MaxRecall:
                                        should_replace = (trial.recall_mean > best_trial.recall_mean);
                                        break;
                                }
                            }
                        } else if (!found_feasible) {
                            // If nothing is feasible yet, track highest recall configuration within memory budget
                            if (best_trial.trial_id == 0) {
                                should_replace = true;
                            } else if (satisfies_memory && !trial.rejection_reason.empty()) {
                                if (trial.recall_mean > best_trial.recall_mean) {
                                    should_replace = true;
                                }
                            }
                        }
                        
                        if (should_replace) {
                            best_trial = trial;
                            // Keep clone of index or assign best
                            best_index = std::move(index);
                            // Recreate index for subsequent sweeps on this graph
                            index = best_index->clone();
                        }
                    }
                }
            }
        }
        
        auto t_end = std::chrono::high_resolution_clock::now();
        report.total_calibration_time_s = std::chrono::duration<double>(t_end - t_start).count();
        report.is_feasible = found_feasible;
        report.selected_config = best_trial;
        
        if (!found_feasible) {
            if (best_trial.recall_mean < constraints.target_recall) {
                report.violated_constraints.push_back("Target recall " + std::to_string(constraints.target_recall) + 
                    " not achieved (best achieved: " + std::to_string(best_trial.recall_mean) + ")");
            }
            if (best_trial.memory_mb > constraints.max_memory_mb) {
                report.violated_constraints.push_back("Memory budget " + std::to_string(constraints.max_memory_mb) + 
                    " MB exceeded (best achieved: " + std::to_string(best_trial.memory_mb) + " MB)");
            }
            if (best_trial.p99_latency_ms > constraints.target_p99_latency_ms) {
                report.violated_constraints.push_back("p99 latency target " + std::to_string(constraints.target_p99_latency_ms) + 
                    " ms exceeded (best achieved: " + std::to_string(best_trial.p99_latency_ms) + " ms)");
            }
        }
        
        // Compute Pareto frontier
        compute_pareto_frontier(report.evaluated_trials, report.pareto_frontier);
        
        // Return calibrated index
        if (best_index) {
            best_index->policy.enable_ada_ef = best_trial.enable_ada_ef;
            out_calibrated_index = std::move(best_index);
        }
        
        return report;
    }
};

} // namespace adaptivevec
