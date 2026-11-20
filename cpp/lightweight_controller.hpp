#pragma once

#include "adaptive_hnsw.hpp"
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <memory>
#include <iostream>
#include <numeric>

namespace adaptivevec {

// ============================================================================
// Multi-Signal Query Difficulty Features & Estimator
// ============================================================================

struct QueryDifficultySignals {
    dist_t layer0_entry_dist;       // Distance from q to layer-0 entry candidate
    size_t entry_node_degree;       // Out-degree of layer-0 entry candidate
    float topk_margin;              // Normalized margin: (d_k - d_1) / (d_k + 1e-6)
    float estimated_lid;            // Estimated Local Intrinsic Dimensionality (Levina-Bickel MLE)
    float candidate_improvement_rate;// Distance improvement in last iterations
    int visited_count;              // Nodes visited during initial search phase
    dist_t best_dist;               // Best distance found so far
};

class MultiSignalDifficultyEstimator {
public:
    struct Thresholds {
        float min_margin = 0.12f;           // Margin below which neighborhood is ambiguous
        float max_lid = 8.5f;               // LID above which search space is high-dimensional
        dist_t max_l0_dist = 60000.0f;      // L0 distance threshold (dataset calibrated: 60K for SIFT L2, 0.8 for Cosine)
        float min_improvement_rate = 0.05f; // Improvement threshold
    };

    Thresholds thresholds;

    MultiSignalDifficultyEstimator() = default;
    explicit MultiSignalDifficultyEstimator(const Thresholds& th) : thresholds(th) {}

    static Thresholds default_for_sift() {
        Thresholds th;
        th.min_margin = 0.15f;
        th.max_lid = 9.0f;
        th.max_l0_dist = 55000.0f;
        th.min_improvement_rate = 0.02f;
        return th;
    }

    static Thresholds default_for_transformer() {
        Thresholds th;
        th.min_margin = 0.08f;
        th.max_lid = 12.0f;
        th.max_l0_dist = 0.75f;
        th.min_improvement_rate = 0.005f;
        return th;
    }

    /**
     * Compute Levina-Bickel Maximum Likelihood Estimate of Local Intrinsic Dimensionality (LID)
     * from the top-k candidates in the current search beam:
     * LID = [ 1/(k-1) * sum_{i=1}^{k-1} ln(d_k / (d_i + eps)) ]^(-1)
     */
    static float compute_lid(const std::vector<Candidate>& topk_candidates) {
        if (topk_candidates.size() < 3) return 1.0f;
        size_t k = topk_candidates.size();
        float r_k = topk_candidates.back().dist;
        if (r_k <= 1e-6f) return 1.0f;

        float sum_log = 0.0f;
        size_t valid_count = 0;
        for (size_t i = 0; i < k - 1; ++i) {
            float r_i = topk_candidates[i].dist;
            if (r_i > 1e-6f && r_k > r_i) {
                sum_log += std::log(r_k / r_i);
                valid_count++;
            }
        }
        if (valid_count < 2 || sum_log <= 1e-6f) return 1.0f;
        float lid = (float)valid_count / sum_log;
        return std::min(std::max(lid, 1.0f), 50.0f);
    }

    /**
     * Evaluate signals collected during initial search beam ef_init
     * Returns true if incremental expansion is recommended, false if query can terminate.
     */
    bool should_expand(const QueryDifficultySignals& sig) const {
        // Condition 1: Ambiguous local margin with high intrinsic dimensionality
        if (sig.topk_margin < thresholds.min_margin && sig.estimated_lid > thresholds.max_lid) {
            return true;
        }
        // Condition 2: Search was still rapidly improving at boundary of ef_init
        if (sig.candidate_improvement_rate > thresholds.min_improvement_rate) {
            return true;
        }
        // Condition 3: Large initial distance combined with tight margins
        if (sig.layer0_entry_dist > thresholds.max_l0_dist && sig.topk_margin < thresholds.min_margin * 1.5f) {
            return true;
        }
        return false;
    }
};

// ============================================================================
// Resumable Search Engine
// ============================================================================

struct ResumableSearchResult {
    std::vector<Candidate> candidates;
    int ef_used;
    int distance_computations;
    QuerySearchStats stats;
    bool was_expanded;
    QueryDifficultySignals signals;
};

class ResumableSearchEngine {
public:
    /**
     * Executes resumable search:
     * 1. Traverses upper layers (L_max down to 1) to find layer-0 entry candidate.
     * 2. Runs layer-0 beam search with initial budget ef_init.
     * 3. Extracts intermediate multi-signal difficulty features.
     * 4. If estimator recommends expansion, expands ef target to ef_expanded
     *    and resumes search WITHOUT clearing visited tags or recomputing distances.
     */
    static ResumableSearchResult search_resumable(
        const AdaptiveHNSWIndex& index,
        const float* query,
        int k,
        int ef_init,
        int ef_expanded,
        const MultiSignalDifficultyEstimator& estimator,
        bool enable_expansion = true
    ) {
        ResumableSearchResult result;
        result.ef_used = ef_init;
        result.was_expanded = false;
        result.distance_computations = 0;

        if (index.num_elements == 0 || index.get_active_count() == 0) return result;

        tableint curr_ep = index.enter_point;
        if (index.is_deleted(curr_ep)) return result;

        bool use_asym_sq8 = index.policy.enable_sq8 && index.quantizer.is_trained;
        bool use_asym_sq4 = index.policy.enable_sq4 && index.quantizer_sq4.is_trained;
        bool use_asym = use_asym_sq8 || use_asym_sq4 || index.use_mixed_storage;

        QuerySearchStats stats;

        // 1. Upper layers greedy descent
        for (int lc = index.max_level; lc > 0; --lc) {
            auto w = index.search_layer(query, {curr_ep}, 1, lc, false, use_asym_sq8, false, use_asym_sq4, &stats);
            if (!w.empty()) curr_ep = w[0].id;
        }

        dist_t l0_entry_dist = index.get_node_distance_sq(query, curr_ep);
        stats.layer0_dist_evals++;
        stats.visited_nodes++;
        size_t l0_degree = (index.graphs.empty() || curr_ep >= index.graphs[0].size()) ? 0 : index.graphs[0][curr_ep].size();

        // 2. Setup Resumable State for Layer 0
        VisitedTagPool& visited = get_thread_visited_pool();
        visited.reset(index.num_elements);

        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidate_queue;
        std::priority_queue<Candidate, std::vector<Candidate>, std::less<Candidate>> w_furthest;

        visited.mark_visited(curr_ep);
        candidate_queue.push({l0_entry_dist, curr_ep});
        w_furthest.push({l0_entry_dist, curr_ep});
        dist_t best_dist = l0_entry_dist;

        int current_target_ef = ef_init;
        int stagnation_counter = 0;
        dist_t prev_furthest = l0_entry_dist;
        float total_improvement = 0.0f;
        int improvement_steps = 0;

        // Phase 1 Search Loop (up to ef_init)
        while (!candidate_queue.empty()) {
            Candidate curr = candidate_queue.top();
            candidate_queue.pop();
            dist_t furthest_d = w_furthest.top().dist;

            if (curr.dist > furthest_d) break;
            stats.hops++;

            bool improved = false;
            if (curr.id < index.graphs[0].size()) {
                const auto& neighbors = index.graphs[0][curr.id];
                for (size_t n_idx = 0; n_idx < neighbors.size(); ++n_idx) {
                    tableint neighbor = neighbors[n_idx];

                    if (n_idx + 1 < neighbors.size()) {
                        const void* ptr = index.get_node_payload_ptr(neighbors[n_idx + 1]);
                        if (ptr) {
                            #if defined(__GNUC__) || defined(__clang__)
                            __builtin_prefetch(ptr, 0, 3);
                            #elif defined(_MSC_VER)
                            _mm_prefetch((const char*)ptr, _MM_HINT_T0);
                            #endif
                        }
                    }

                    if (!visited.is_visited(neighbor)) {
                        visited.mark_visited(neighbor);
                        dist_t d = index.get_node_distance_sq(query, neighbor);
                        stats.layer0_dist_evals++;
                        stats.visited_nodes++;
                        furthest_d = w_furthest.top().dist;

                        if (d < furthest_d || (int)w_furthest.size() < current_target_ef) {
                            candidate_queue.push({d, neighbor});
                            if ((int)w_furthest.size() >= current_target_ef) {
                                dist_t old_f = w_furthest.top().dist;
                                w_furthest.pop();
                                w_furthest.push({d, neighbor});
                                dist_t new_f = w_furthest.top().dist;
                                if (old_f - new_f > index.policy.ada_ef_epsilon) {
                                    improved = true;
                                    total_improvement += (old_f - new_f);
                                    improvement_steps++;
                                }
                            } else {
                                w_furthest.push({d, neighbor});
                                improved = true;
                            }

                            if (best_dist - d > index.policy.ada_ef_epsilon) {
                                best_dist = d;
                                improved = true;
                            }
                        }
                    }
                }
            }

            if ((int)w_furthest.size() >= current_target_ef) {
                if (improved) {
                    stagnation_counter = 0;
                } else {
                    if (++stagnation_counter >= index.policy.ada_ef_patience) {
                        break;
                    }
                }
            }
        }

        // 3. Extract Intermediate Signals from Current Beam
        std::vector<Candidate> intermediate_topk;
        {
            auto tmp_heap = w_furthest;
            while (!tmp_heap.empty()) {
                intermediate_topk.push_back(tmp_heap.top());
                tmp_heap.pop();
            }
            std::reverse(intermediate_topk.begin(), intermediate_topk.end());
            if ((int)intermediate_topk.size() > k) intermediate_topk.resize(k);
        }

        float topk_margin = 0.0f;
        if (intermediate_topk.size() >= 2) {
            dist_t d_first = intermediate_topk.front().dist;
            dist_t d_last = intermediate_topk.back().dist;
            topk_margin = (d_last - d_first) / (d_last + 1e-6f);
        }
        float est_lid = MultiSignalDifficultyEstimator::compute_lid(intermediate_topk);
        float imp_rate = (improvement_steps > 0 && best_dist > 1e-6f) ? (total_improvement / (best_dist * improvement_steps)) : 0.0f;

        QueryDifficultySignals signals;
        signals.layer0_entry_dist = l0_entry_dist;
        signals.entry_node_degree = l0_degree;
        signals.topk_margin = topk_margin;
        signals.estimated_lid = est_lid;
        signals.candidate_improvement_rate = imp_rate;
        signals.visited_count = stats.visited_nodes;
        signals.best_dist = best_dist;
        result.signals = signals;

        // 4. Decision: Resume and Expand if Needed
        if (enable_expansion && ef_expanded > ef_init && estimator.should_expand(signals)) {
            result.was_expanded = true;
            result.ef_used = ef_expanded;
            current_target_ef = ef_expanded;
            stagnation_counter = 0;

            // Resume exploration: w_furthest can now expand from ef_init up to ef_expanded
            while (!candidate_queue.empty()) {
                Candidate curr = candidate_queue.top();
                candidate_queue.pop();
                dist_t furthest_d = w_furthest.top().dist;

                if (curr.dist > furthest_d) break;

                bool improved = false;
                if (curr.id < index.graphs[0].size()) {
                    const auto& neighbors = index.graphs[0][curr.id];
                    for (size_t n_idx = 0; n_idx < neighbors.size(); ++n_idx) {
                        tableint neighbor = neighbors[n_idx];

                        if (n_idx + 1 < neighbors.size()) {
                            const void* ptr = index.get_node_payload_ptr(neighbors[n_idx + 1]);
                            if (ptr) {
                                #if defined(__GNUC__) || defined(__clang__)
                                __builtin_prefetch(ptr, 0, 3);
                                #elif defined(_MSC_VER)
                                _mm_prefetch((const char*)ptr, _MM_HINT_T0);
                                #endif
                            }
                        }

                        if (!visited.is_visited(neighbor)) {
                            visited.mark_visited(neighbor);
                            dist_t d = index.get_node_distance_sq(query, neighbor);
                            stats.layer0_dist_evals++;
                            stats.visited_nodes++;
                            furthest_d = w_furthest.top().dist;

                            if (d < furthest_d || (int)w_furthest.size() < current_target_ef) {
                                candidate_queue.push({d, neighbor});
                                if ((int)w_furthest.size() >= current_target_ef) {
                                    dist_t old_f = w_furthest.top().dist;
                                    w_furthest.pop();
                                    w_furthest.push({d, neighbor});
                                    dist_t new_f = w_furthest.top().dist;
                                    if (old_f - new_f > index.policy.ada_ef_epsilon) {
                                        improved = true;
                                    }
                                } else {
                                    w_furthest.push({d, neighbor});
                                    improved = true;
                                }

                                if (best_dist - d > index.policy.ada_ef_epsilon) {
                                    best_dist = d;
                                    improved = true;
                                }
                            }
                        }
                    }
                }

                if ((int)w_furthest.size() >= current_target_ef) {
                    if (improved) {
                        stagnation_counter = 0;
                    } else {
                        if (++stagnation_counter >= index.policy.ada_ef_patience) {
                            stats.early_exit_stagnation_count++;
                            break;
                        }
                    }
                }
            }
        }

        result.stats = stats;
        result.distance_computations = stats.total_dist_evals();

        // 5. Final Top-K Extraction & Optional FP32 Re-ranking
        std::vector<Candidate> final_candidates;
        while (!w_furthest.empty()) {
            final_candidates.push_back(w_furthest.top());
            w_furthest.pop();
        }
        std::reverse(final_candidates.begin(), final_candidates.end());

        if (use_asym && index.policy.enable_rerank && !index.data.empty()) {
            std::vector<Candidate> active_w;
            active_w.reserve(final_candidates.size());
            for (auto& cand : final_candidates) {
                if (!index.is_deleted(cand.id)) {
                    cand.dist = index.get_distance(query, index.get_vector(cand.id));
                    stats.rerank_dist_evals++;
                    active_w.push_back(cand);
                }
            }
            std::sort(active_w.begin(), active_w.end());
            if ((int)active_w.size() > k) active_w.resize(k);
            result.candidates = std::move(active_w);
            result.stats = stats;
            result.distance_computations = stats.total_dist_evals();
        } else {
            std::vector<Candidate> active_w;
            active_w.reserve(std::min((size_t)k, final_candidates.size()));
            for (const auto& cand : final_candidates) {
                if (!index.is_deleted(cand.id)) {
                    active_w.push_back(cand);
                    if ((int)active_w.size() >= k) break;
                }
            }
            if (index.space == SpaceType::L2) {
                for (auto& cand : active_w) cand.dist = std::sqrt(cand.dist);
            }
            result.candidates = std::move(active_w);
        }

        return result;
    }
};

// ============================================================================
// Bounded & Incremental Reference Monitor (< 5% CPU Overhead)
// ============================================================================

enum class MonitoringPolicyType {
    PeriodicBatch,       // Run M_ref queries every N_period requests (legacy spike behavior)
    IncrementalSampling, // Run 1 reference query every K_inc requests (smooth, bounded < 5% overhead)
    DriftTriggered       // Run 1 query every K_base requests, accelerate to K_fast when drift detected
};

class BoundedReferenceMonitor {
public:
    struct Sample {
        std::vector<float> query;
        std::vector<tableint> gt_topk;
    };

    struct Stats {
        double cpu_time_ms = 0.0;
        int distance_evals = 0;
        int ref_queries_evaluated = 0;
        int prod_queries_served = 0;
        double cpu_overhead_pct = 0.0;
        float current_rolling_recall = 1.0f;
    };

private:
    std::vector<Sample> reference_pool_;
    size_t next_sample_idx_ = 0;
    size_t k_interval_ = 25; // 1 reference query per 25 production queries = 4.0% overhead
    MonitoringPolicyType policy_type_ = MonitoringPolicyType::IncrementalSampling;
    
    // Rolling window of evaluated reference recalls (capacity = 100)
    std::vector<float> rolling_recalls_;
    size_t rolling_head_ = 0;
    size_t rolling_count_ = 0;
    
    Stats stats_;

public:
    BoundedReferenceMonitor() : rolling_recalls_(100, 1.0f) {}

    void set_policy(MonitoringPolicyType policy, size_t k_interval = 25) {
        policy_type_ = policy;
        k_interval_ = std::max((size_t)10, k_interval);
    }

    void add_reference_sample(std::vector<float> q, std::vector<tableint> gt) {
        reference_pool_.push_back({std::move(q), std::move(gt)});
    }

    void clear() {
        reference_pool_.clear();
        next_sample_idx_ = 0;
        rolling_head_ = 0;
        rolling_count_ = 0;
        stats_ = Stats();
    }

    size_t pool_size() const { return reference_pool_.size(); }
    const Stats& get_stats() const { return stats_; }

    /**
     * Checks if a reference query should be evaluated during production serving.
     * In IncrementalSampling, evaluates exactly 1 sample every k_interval_ queries.
     */
    bool should_evaluate_sample(size_t prod_query_idx, bool drift_detected = false) const {
        if (reference_pool_.empty()) return false;
        if (policy_type_ == MonitoringPolicyType::IncrementalSampling) {
            return (prod_query_idx > 0 && prod_query_idx % k_interval_ == 0);
        } else if (policy_type_ == MonitoringPolicyType::DriftTriggered) {
            size_t effective_interval = drift_detected ? (k_interval_ / 2) : k_interval_;
            return (prod_query_idx > 0 && prod_query_idx % effective_interval == 0);
        } else { // PeriodicBatch (every 500 requests)
            return (prod_query_idx > 0 && prod_query_idx % 500 == 0);
        }
    }

    /**
     * Evaluates next incremental reference query and updates rolling accuracy.
     */
    void evaluate_next_sample(const AdaptiveHNSWIndex& index, int current_ef, int k = 10) {
        if (reference_pool_.empty()) return;

        auto t0 = std::chrono::high_resolution_clock::now();
        size_t idx = next_sample_idx_ % reference_pool_.size();
        next_sample_idx_++;

        const auto& samp = reference_pool_[idx];
        auto res = index.search(samp.query.data(), k, current_ef);

        // Compute recall for this sample
        int hits = 0;
        size_t eval_k = std::min((size_t)k, samp.gt_topk.size());
        for (const auto& cand : res) {
            for (size_t i = 0; i < eval_k; ++i) {
                if (cand.id == samp.gt_topk[i]) {
                    hits++;
                    break;
                }
            }
        }
        float sample_recall = eval_k > 0 ? ((float)hits / eval_k) : 1.0f;

        // Update rolling buffer
        rolling_recalls_[rolling_head_] = sample_recall;
        rolling_head_ = (rolling_head_ + 1) % rolling_recalls_.size();
        if (rolling_count_ < rolling_recalls_.size()) rolling_count_++;

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        stats_.cpu_time_ms += elapsed_ms;
        stats_.ref_queries_evaluated++;
        stats_.distance_evals += current_ef * 5; // approximate distance count
        stats_.current_rolling_recall = get_rolling_recall();
    }

    float get_rolling_recall() const {
        if (rolling_count_ == 0) return 1.0f;
        float sum = 0.0f;
        for (size_t i = 0; i < rolling_count_; ++i) sum += rolling_recalls_[i];
        return sum / (float)rolling_count_;
    }

    void record_prod_query() {
        stats_.prod_queries_served++;
        if (stats_.prod_queries_served > 0) {
            stats_.cpu_overhead_pct = (double)stats_.ref_queries_evaluated / (double)stats_.prod_queries_served * 100.0;
        }
    }
};

// ============================================================================
// Two-Level Lightweight Workload Controller
// ============================================================================

struct LightweightControllerConfig {
    int base_ef_init = 32;
    int base_ef_expanded = 64;
    float target_recall = 0.95f;
    float target_p99_latency_ms = 1.0f;
    int cooldown_queries = 100;
    float hysteresis_deadband = 0.01f; // +/- 1.0% recall deadband
    float safety_rollback_margin = 0.02f; // Trigger rollback if recall < target - 0.02
    size_t monitor_interval = 25; // 1 ref query per 25 requests = 4% CPU overhead
};

class LightweightWorkloadController {
private:
    LightweightControllerConfig config_;
    BoundedReferenceMonitor monitor_;
    MultiSignalDifficultyEstimator estimator_;

    int current_ef_init_;
    int current_ef_expanded_;
    int stable_ef_init_;
    int stable_ef_expanded_;

    int queries_since_last_adaptation_ = 0;
    int total_adaptations_ = 0;
    int total_rollbacks_ = 0;

public:
    explicit LightweightWorkloadController(const LightweightControllerConfig& cfg)
        : config_(cfg),
          current_ef_init_(cfg.base_ef_init),
          current_ef_expanded_(cfg.base_ef_expanded),
          stable_ef_init_(cfg.base_ef_init),
          stable_ef_expanded_(cfg.base_ef_expanded) {
        monitor_.set_policy(MonitoringPolicyType::IncrementalSampling, cfg.monitor_interval);
    }

    BoundedReferenceMonitor& get_monitor() { return monitor_; }
    const BoundedReferenceMonitor& get_monitor() const { return monitor_; }
    MultiSignalDifficultyEstimator& get_estimator() { return estimator_; }

    int get_current_ef_init() const { return current_ef_init_; }
    int get_current_ef_expanded() const { return current_ef_expanded_; }
    int get_total_adaptations() const { return total_adaptations_; }
    int get_total_rollbacks() const { return total_rollbacks_; }

    /**
     * Fast-Path Query Execution:
     * Resumable search + multi-signal difficulty estimation.
     * Zero reference queries evaluated in this critical path.
     */
    ResumableSearchResult execute_query(
        const AdaptiveHNSWIndex& index,
        const float* query,
        int k,
        bool enable_resumable = true
    ) {
        return ResumableSearchEngine::search_resumable(
            index, query, k,
            current_ef_init_, current_ef_expanded_,
            estimator_, enable_resumable
        );
    }

    /**
     * Slower Workload-Level Adaptation Check:
     * Called periodically (e.g. after each batch or every query).
     * Bounded to < 5% CPU via incremental reference sampling.
     */
    void step_workload(const AdaptiveHNSWIndex& index, size_t query_idx, double last_latency_ms) {
        monitor_.record_prod_query();

        // Check if an incremental reference sample should be evaluated
        if (monitor_.should_evaluate_sample(query_idx)) {
            monitor_.evaluate_next_sample(index, current_ef_init_, 10);
        }

        queries_since_last_adaptation_++;
        if (queries_since_last_adaptation_ < config_.cooldown_queries) return;

        float rolling_r = monitor_.get_rolling_recall();

        // 1. Safety Rollback Check: If rolling recall drops below safety threshold
        if (rolling_r < config_.target_recall - config_.safety_rollback_margin) {
            current_ef_init_ = std::max(current_ef_init_, stable_ef_init_);
            current_ef_expanded_ = std::max(current_ef_expanded_, stable_ef_expanded_);
            queries_since_last_adaptation_ = -config_.cooldown_queries; // double cooldown on rollback
            total_rollbacks_++;
            return;
        }

        // 2. Upward Adaptation: If recall is below target - deadband
        if (rolling_r < config_.target_recall - config_.hysteresis_deadband) {
            int new_init = std::min(current_ef_init_ + 16, 128);
            int new_exp = std::min(current_ef_expanded_ + 16, 192);
            if (new_init != current_ef_init_ || new_exp != current_ef_expanded_) {
                current_ef_init_ = new_init;
                current_ef_expanded_ = new_exp;
                total_adaptations_++;
                queries_since_last_adaptation_ = 0;
            }
            return;
        }

        // 3. Downward Adaptation: If recall is comfortably above target + 2*deadband and latency allows
        if (rolling_r > config_.target_recall + 2.0f * config_.hysteresis_deadband) {
            int new_init = std::max(current_ef_init_ - 8, 16);
            int new_exp = std::max(current_ef_expanded_ - 16, 32);
            if (new_init != current_ef_init_ || new_exp != current_ef_expanded_) {
                current_ef_init_ = new_init;
                current_ef_expanded_ = new_exp;
                total_adaptations_++;
                queries_since_last_adaptation_ = 0;
            }
            return;
        }
    }
};

} // namespace adaptivevec
