#pragma once

#include "telemetry.hpp"
#include "adaptive_hnsw.hpp"
#include <vector>
#include <string>
#include <deque>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace adaptivevec {

enum class DriftDetectorType {
    RollingStats = 0,
    EWMA = 1,
    CUSUM = 2
};

inline std::string to_string(DriftDetectorType t) {
    switch (t) {
        case DriftDetectorType::RollingStats: return "RollingStats";
        case DriftDetectorType::EWMA: return "EWMA";
        case DriftDetectorType::CUSUM: return "CUSUM";
        default: return "Unknown";
    }
}

// -------------------------------------------------------------
// Workload Drift Detectors
// -------------------------------------------------------------
class DriftDetector {
public:
    virtual ~DriftDetector() = default;
    virtual void update(float value) = 0;
    virtual bool is_drift_detected() const = 0;
    virtual void reset() = 0;
    virtual std::string name() const = 0;
};

class RollingStatsDriftDetector : public DriftDetector {
public:
    RollingStatsDriftDetector(size_t window_size = 200, float threshold_sigmas = 2.5f)
        : window(window_size), max_window(window_size), k_sigma(threshold_sigmas) {}
        
    void update(float val) override {
        history.push_back(val);
        if (history.size() > max_window) history.pop_front();
        
        if (history.size() >= 30) {
            double sum = 0.0;
            for (float x : history) sum += x;
            double mean = sum / history.size();
            double var = 0.0;
            for (float x : history) var += (x - mean) * (x - mean);
            double std_dev = std::sqrt(var / history.size());
            
            if (baseline_count < 100) {
                baseline_mean = mean;
                baseline_std = std_dev;
                baseline_count++;
            } else {
                if (std::abs(val - baseline_mean) > k_sigma * std::max(0.01, baseline_std)) {
                    drift_flag = true;
                } else {
                    drift_flag = false;
                }
            }
        }
    }
    
    bool is_drift_detected() const override { return drift_flag; }
    void reset() override {
        history.clear();
        baseline_count = 0;
        drift_flag = false;
    }
    std::string name() const override { return "RollingStats"; }
    
private:
    size_t window;
    size_t max_window;
    float k_sigma;
    std::deque<float> history;
    double baseline_mean = 0.0;
    double baseline_std = 1.0;
    size_t baseline_count = 0;
    bool drift_flag = false;
};

class EWMADriftDetector : public DriftDetector {
public:
    EWMADriftDetector(float alpha = 0.05f, float threshold_factor = 2.0f)
        : decay(alpha), factor(threshold_factor) {}
        
    void update(float val) override {
        if (!initialized) {
            ewma_mean = val;
            ewma_var = 1.0f;
            initialized = true;
            return;
        }
        
        float diff = val - ewma_mean;
        ewma_mean += decay * diff;
        ewma_var = (1.0f - decay) * ewma_var + decay * (diff * diff);
        float std_dev = std::sqrt(std::max(1e-4f, ewma_var));
        
        if (warmup < 100) {
            warmup++;
            stable_mean = ewma_mean;
            stable_std = std_dev;
        } else {
            drift_flag = (std::abs(ewma_mean - stable_mean) > factor * stable_std);
        }
    }
    
    bool is_drift_detected() const override { return drift_flag; }
    void reset() override {
        initialized = false;
        warmup = 0;
        drift_flag = false;
    }
    std::string name() const override { return "EWMA"; }
    
private:
    float decay;
    float factor;
    bool initialized = false;
    size_t warmup = 0;
    float ewma_mean = 0.0f;
    float ewma_var = 1.0f;
    float stable_mean = 0.0f;
    float stable_std = 1.0f;
    bool drift_flag = false;
};

class CUSUMDriftDetector : public DriftDetector {
public:
    CUSUMDriftDetector(float delta = 0.5f, float threshold_h = 8.0f)
        : slack(delta), h(threshold_h) {}
        
    void update(float val) override {
        if (warmup < 50) {
            warmup++;
            sum_init += val;
            baseline = sum_init / warmup;
            return;
        }
        
        float diff = val - baseline;
        s_pos = std::max(0.0f, s_pos + diff - slack);
        s_neg = std::max(0.0f, s_neg - diff - slack);
        
        if (s_pos > h || s_neg > h) {
            drift_flag = true;
        } else {
            drift_flag = false;
        }
    }
    
    bool is_drift_detected() const override { return drift_flag; }
    void reset() override {
        warmup = 0;
        sum_init = 0.0f;
        s_pos = 0.0f;
        s_neg = 0.0f;
        drift_flag = false;
    }
    std::string name() const override { return "CUSUM"; }
    
private:
    float slack;
    float h;
    size_t warmup = 0;
    float sum_init = 0.0f;
    float baseline = 0.0f;
    float s_pos = 0.0f;
    float s_neg = 0.0f;
    bool drift_flag = false;
};

// -------------------------------------------------------------
// Reference Query Monitor for Non-Invasive Recall Tracking
// -------------------------------------------------------------
struct ReferenceQuerySample {
    std::vector<float> query;
    std::vector<tableint> ground_truth_topk;
};

class ReferenceRecallTracker {
public:
    void add_sample(const std::vector<float>& q, const std::vector<tableint>& gt) {
        samples.push_back({q, gt});
    }
    
    float evaluate_recall(const AdaptiveHNSWIndex& index, int k, int ef_search) const {
        if (samples.empty()) return 1.0f;
        size_t total_matches = 0;
        size_t total_possible = 0;
        
        for (const auto& smp : samples) {
            auto res = index.search(smp.query.data(), k, ef_search);
            size_t cmp_k = std::min((size_t)k, smp.ground_truth_topk.size());
            total_possible += cmp_k;
            for (const auto& c : res) {
                for (size_t g = 0; g < cmp_k; ++g) {
                    if (c.id == smp.ground_truth_topk[g]) {
                        total_matches++;
                        break;
                    }
                }
            }
        }
        return total_possible > 0 ? static_cast<float>(total_matches) / static_cast<float>(total_possible) : 1.0f;
    }
    
    size_t size() const { return samples.size(); }
    
private:
    std::vector<ReferenceQuerySample> samples;
};

// -------------------------------------------------------------
// Stage 5: Closed-Loop Runtime Workload Controller
// -------------------------------------------------------------
struct ControllerPolicyConfig {
    int current_ef_search = 64;
    bool enable_ada_ef = false;
    int ef_min = 16;
    int ef_max = 192;
    int ef_step = 16;
    
    float target_recall = 0.95f;
    float recall_safety_margin = 0.02f; // Trigger rollback if recall < target - margin
    float hysteresis_deadband = 0.01f;   // Deadband around target recall
    
    size_t cooldown_queries = 150;      // Minimum queries between adaptations
    size_t sample_eval_interval = 50;   // Interval between reference recall checks
};

struct ControllerTraceRecord {
    uint64_t query_idx = 0;
    int active_ef = 64;
    float current_recall = 1.0f;
    float current_p99_ms = 0.0f;
    float current_qps = 0.0f;
    bool drift_detected = false;
    std::string action = "NONE"; // "NONE", "INCREASE_EF", "DECREASE_EF", "ROLLBACK", "DRIFT_ALERT"
};

class WorkloadAwareController {
public:
    WorkloadAwareController(
        const ControllerPolicyConfig& config,
        std::unique_ptr<DriftDetector> detector = std::make_unique<EWMADriftDetector>()
    ) : cfg(config), drift_detector(std::move(detector)) {
        stable_ef_history.push_back(cfg.current_ef_search);
    }
    
    void set_reference_samples(ReferenceRecallTracker tracker) {
        ref_tracker = std::move(tracker);
    }
    
    // Stage 4: Query-Specific Adaptive efSearch
    inline int predict_query_ef(float entry_distance, float baseline_entry_mean) const {
        if (baseline_entry_mean <= 1e-4f) return cfg.current_ef_search;
        float diff_ratio = (entry_distance - baseline_entry_mean) / baseline_entry_mean;
        
        // If query is an outlier far from entry point, scale efSearch safely up by 25-50%
        if (diff_ratio > 0.40f) {
            return std::min(cfg.ef_max, static_cast<int>(cfg.current_ef_search * 1.35f));
        } else if (diff_ratio < -0.30f) {
            // Very close query: safe to reduce search beam slightly
            return std::max(cfg.ef_min, static_cast<int>(cfg.current_ef_search * 0.85f));
        }
        return cfg.current_ef_search;
    }
    
    // Stage 5: Closed-Loop Periodic Adaptation
    ControllerTraceRecord step(
        uint64_t query_id,
        const QueryEvent& event,
        const AdaptiveHNSWIndex& index,
        int k
    ) {
        ControllerTraceRecord rec;
        rec.query_idx = query_id;
        rec.active_ef = cfg.current_ef_search;
        
        // Update drift detector with distance computations as a difficulty proxy
        drift_detector->update(static_cast<float>(event.distance_computations));
        rec.drift_detected = drift_detector->is_drift_detected();
        
        queries_since_last_adaptation++;
        queries_since_last_ref_eval++;
        
        // Periodic reference recall evaluation
        if (queries_since_last_ref_eval >= cfg.sample_eval_interval && ref_tracker.size() > 0) {
            queries_since_last_ref_eval = 0;
            cached_recall = ref_tracker.evaluate_recall(index, k, cfg.current_ef_search);
        }
        rec.current_recall = cached_recall;
        
        // Check for adaptation only if cooldown has passed
        if (queries_since_last_adaptation >= cfg.cooldown_queries) {
            // Safety Check: Rollback if recall dips below safety boundary
            if (cached_recall < cfg.target_recall - cfg.recall_safety_margin) {
                if (stable_ef_history.size() > 1) {
                    stable_ef_history.pop_back();
                    cfg.current_ef_search = stable_ef_history.back();
                    rec.action = "ROLLBACK";
                    num_rollbacks++;
                } else {
                    cfg.current_ef_search = std::min(cfg.ef_max, cfg.current_ef_search + cfg.ef_step * 2);
                    rec.action = "INCREASE_EF";
                }
                queries_since_last_adaptation = 0;
            }
            // Recall below target (outside hysteresis deadband): Increase ef
            else if (cached_recall < cfg.target_recall - cfg.hysteresis_deadband) {
                if (cfg.current_ef_search < cfg.ef_max) {
                    cfg.current_ef_search = std::min(cfg.ef_max, cfg.current_ef_search + cfg.ef_step);
                    stable_ef_history.push_back(cfg.current_ef_search);
                    rec.action = "INCREASE_EF";
                    queries_since_last_adaptation = 0;
                }
            }
            // Recall comfortably exceeds target + deadband: Decrease ef to maximize throughput
            else if (cached_recall > cfg.target_recall + cfg.hysteresis_deadband * 2.0f) {
                if (cfg.current_ef_search > cfg.ef_min) {
                    cfg.current_ef_search = std::max(cfg.ef_min, cfg.current_ef_search - cfg.ef_step);
                    stable_ef_history.push_back(cfg.current_ef_search);
                    rec.action = "DECREASE_EF";
                    queries_since_last_adaptation = 0;
                }
            }
            // Drift Alert Trigger
            else if (rec.drift_detected) {
                rec.action = "DRIFT_ALERT";
            }
        }
        
        rec.active_ef = cfg.current_ef_search;
        trace_history.push_back(rec);
        return rec;
    }
    
    int get_active_ef() const { return cfg.current_ef_search; }
    size_t get_rollback_count() const { return num_rollbacks; }
    const std::vector<ControllerTraceRecord>& get_traces() const { return trace_history; }
    
private:
    ControllerPolicyConfig cfg;
    std::unique_ptr<DriftDetector> drift_detector;
    ReferenceRecallTracker ref_tracker;
    
    size_t queries_since_last_adaptation = 0;
    size_t queries_since_last_ref_eval = 0;
    float cached_recall = 1.0f;
    size_t num_rollbacks = 0;
    
    std::vector<int> stable_ef_history;
    std::vector<ControllerTraceRecord> trace_history;
};

} // namespace adaptivevec
