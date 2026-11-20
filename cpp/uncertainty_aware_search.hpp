#pragma once

#include "adaptive_hnsw.hpp"
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <iostream>

namespace adaptivevec {

enum class TerminationReason {
    FrontierExhausted,      // Candidates heap empty or distance exceeded furthest
    FixedPatienceEarlyExit, // Stagnation exceeded fixed patience
    UncertaintyAwareEarlyExit, // Dynamic patience p(q, t) exhausted
    HighMarginConvergence,  // Extremely high margin (>0.40) & low LID indicates decisive convergence
    BudgetExhausted         // Hit max_ef limit
};

inline const char* to_string(TerminationReason reason) {
    switch (reason) {
        case TerminationReason::FrontierExhausted: return "FrontierExhausted";
        case TerminationReason::FixedPatienceEarlyExit: return "FixedPatienceEarlyExit";
        case TerminationReason::UncertaintyAwareEarlyExit: return "UncertaintyAwareEarlyExit";
        case TerminationReason::HighMarginConvergence: return "HighMarginConvergence";
        case TerminationReason::BudgetExhausted: return "BudgetExhausted";
        default: return "Unknown";
    }
}

// ============================================================================
// Search Checkpoint State (for Stage 4 data collection and utility estimation)
// ============================================================================

struct SearchStateCheckpoint {
    // Live Traversal Features (available during online search)
    int step = 0;
    int current_ef = 0;
    int candidate_queue_size = 0;
    dist_t best_dist = 0.0f;
    dist_t furthest_dist = 0.0f;
    float topk_margin = 0.0f;
    float improvement_velocity = 0.0f;
    int stagnation_count = 0;
    uint32_t visited_nodes = 0;
    uint32_t measured_dist_evals = 0;
    dist_t layer0_entry_dist = 0.0f;
    float estimated_lid = 0.0f;
    float uncertainty_score = 0.0f;
    float convergence_score = 0.0f;
    int dynamic_patience = 0;

    // Offline Diagnostic Ground-Truth Labels (available only offline)
    float current_recall_at_10 = 0.0f;
    float max_possible_recall = 0.0f;
    float marginal_recall_gain = 0.0f;
    bool will_gain_recall = false;
    bool is_censored = false; // Cannot reach 100% even at max budget
};

// ============================================================================
// Dynamic Patience Configuration
// ============================================================================

struct DynamicPatienceConfig {
    int p_base = 6;
    int p_min = 2;
    int p_max = 16;
    float alpha = 2.5f; // Weight for uncertainty (increases patience)
    float beta = 2.0f;  // Weight for convergence (decreases patience)
    float high_margin_thresh = 0.40f; // Decisive convergence threshold
    float low_lid_thresh = 6.0f;
    float epsilon = 1e-4f;
};

// ============================================================================
// Uncertainty-Aware Search Engine
// ============================================================================

struct UncertaintySearchResult {
    std::vector<Candidate> candidates;
    QuerySearchStats stats;
    TerminationReason termination_reason = TerminationReason::FrontierExhausted;
    float final_uncertainty = 0.0f;
    float final_convergence = 0.0f;
    int effective_patience_used = 0;
    std::vector<SearchStateCheckpoint> checkpoints;
};

class UncertaintyAwareSearchEngine {
public:
    /**
     * Compute Levina-Bickel Maximum Likelihood Estimate of Local Intrinsic Dimensionality (LID)
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
     * Executes uncertainty-aware beam search on layer 0 with dynamic patience p(q, t)
     * Optional checkpoint recording for Stage 4 dataset export.
     */
    static UncertaintySearchResult search_layer0_uncertainty_aware(
        const AdaptiveHNSWIndex& index,
        const float* query,
        int k,
        int ef,
        const DynamicPatienceConfig& cfg,
        bool collect_checkpoints = false,
        const std::vector<tableint>* ground_truth = nullptr
    ) {
        UncertaintySearchResult result;
        QuerySearchStats stats;

        if (index.num_elements == 0 || index.get_active_count() == 0) return result;

        tableint curr_ep = index.enter_point;
        if (index.is_deleted(curr_ep)) return result;

        bool use_asym_sq8 = index.policy.enable_sq8 && index.quantizer.is_trained;
        bool use_asym_sq4 = index.policy.enable_sq4 && index.quantizer_sq4.is_trained;
        bool use_asym = use_asym_sq8 || use_asym_sq4 || index.use_mixed_storage;

        // 1. Upper layers greedy descent
        for (int lc = index.max_level; lc > 0; --lc) {
            auto w = index.search_layer(query, {curr_ep}, 1, lc, false, use_asym_sq8, false, use_asym_sq4, &stats);
            if (!w.empty()) curr_ep = w[0].id;
        }

        dist_t l0_entry_dist = index.get_node_distance_sq(query, curr_ep);
        stats.layer0_dist_evals++;
        stats.visited_nodes++;

        // 2. Layer 0 setup
        VisitedTagPool& visited = get_thread_visited_pool();
        visited.reset(index.num_elements);

        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidate_queue;
        std::priority_queue<Candidate, std::vector<Candidate>, std::less<Candidate>> w_furthest;

        visited.mark_visited(curr_ep);
        candidate_queue.push({l0_entry_dist, curr_ep});
        w_furthest.push({l0_entry_dist, curr_ep});
        dist_t best_dist = l0_entry_dist;

        int stagnation_counter = 0;
        float total_improvement = 0.0f;
        int improvement_steps = 0;
        int current_patience = cfg.p_base;
        TerminationReason term_reason = TerminationReason::FrontierExhausted;

        int step_count = 0;

        // Helper lambda to extract current top-k candidates
        auto get_current_topk = [&](int target_k) {
            auto tmp_heap = w_furthest;
            std::vector<Candidate> current_w;
            while (!tmp_heap.empty()) {
                current_w.push_back(tmp_heap.top());
                tmp_heap.pop();
            }
            std::reverse(current_w.begin(), current_w.end());
            if ((int)current_w.size() > target_k) current_w.resize(target_k);
            return current_w;
        };

        // Main Layer-0 Traversal Loop
        while (!candidate_queue.empty()) {
            Candidate curr = candidate_queue.top();
            candidate_queue.pop();
            dist_t furthest_d = w_furthest.top().dist;

            if (curr.dist > furthest_d) {
                term_reason = TerminationReason::FrontierExhausted;
                break;
            }
            stats.hops++;
            step_count++;

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

                        if (d < furthest_d || (int)w_furthest.size() < ef) {
                            candidate_queue.push({d, neighbor});
                            if ((int)w_furthest.size() >= ef) {
                                dist_t old_f = w_furthest.top().dist;
                                w_furthest.pop();
                                w_furthest.push({d, neighbor});
                                dist_t new_f = w_furthest.top().dist;
                                if (old_f - new_f > cfg.epsilon) {
                                    improved = true;
                                    total_improvement += (old_f - new_f);
                                    improvement_steps++;
                                }
                            } else {
                                w_furthest.push({d, neighbor});
                                improved = true;
                            }

                            if (best_dist - d > cfg.epsilon) {
                                best_dist = d;
                                improved = true;
                            }
                        }
                    }
                }
            }

            // Stagnation and dynamic patience computation
            if ((int)w_furthest.size() >= ef) {
                if (improved) {
                    stagnation_counter = 0;
                } else {
                    stagnation_counter++;
                }

                // Intermediate Multi-Signal Extraction
                auto current_topk = get_current_topk(k);
                float topk_margin = 0.0f;
                if (current_topk.size() >= 2) {
                    dist_t d1 = current_topk.front().dist;
                    dist_t dk = current_topk.back().dist;
                    topk_margin = (dk - d1) / (dk + 1e-6f);
                }
                float est_lid = compute_lid(current_topk);
                float imp_velocity = (improvement_steps > 0 && best_dist > 1e-6f)
                                     ? (total_improvement / (best_dist * improvement_steps))
                                     : 0.0f;

                // 1. Decisive Convergence Fast-Exit
                if (topk_margin > cfg.high_margin_thresh && est_lid < cfg.low_lid_thresh && stagnation_counter >= cfg.p_min) {
                    term_reason = TerminationReason::HighMarginConvergence;
                    stats.early_exit_stagnation_count++;
                    break;
                }

                // 2. Dynamic Patience Calculation:
                // p(q, t) = clip(p_base + alpha * uncertainty - beta * convergence, p_min, p_max)
                float uncertainty = std::max(0.0f, std::min(3.0f, (est_lid / 10.0f) + (1.0f - topk_margin)));
                float convergence = std::max(0.0f, std::min(3.0f, ((float)stagnation_counter / cfg.p_base) + (1.0f - imp_velocity * 10.0f)));
                
                int dyn_p = static_cast<int>(std::round(cfg.p_base + cfg.alpha * uncertainty - cfg.beta * convergence));
                current_patience = std::max(cfg.p_min, std::min(cfg.p_max, dyn_p));

                result.final_uncertainty = uncertainty;
                result.final_convergence = convergence;
                result.effective_patience_used = current_patience;

                // Stage 4 Checkpoint Data Collection
                if (collect_checkpoints && step_count % 10 == 0) {
                    SearchStateCheckpoint cp;
                    cp.step = step_count;
                    cp.current_ef = ef;
                    cp.candidate_queue_size = static_cast<int>(candidate_queue.size());
                    cp.best_dist = best_dist;
                    cp.furthest_dist = w_furthest.top().dist;
                    cp.topk_margin = topk_margin;
                    cp.improvement_velocity = imp_velocity;
                    cp.stagnation_count = stagnation_counter;
                    cp.visited_nodes = stats.visited_nodes;
                    cp.measured_dist_evals = stats.total_dist_evals();
                    cp.layer0_entry_dist = l0_entry_dist;
                    cp.estimated_lid = est_lid;
                    cp.uncertainty_score = uncertainty;
                    cp.convergence_score = convergence;
                    cp.dynamic_patience = current_patience;

                    if (ground_truth && !ground_truth->empty()) {
                        int hits = 0;
                        size_t eval_k = std::min((size_t)k, ground_truth->size());
                        for (const auto& c : current_topk) {
                            for (size_t g_i = 0; g_i < eval_k; ++g_i) {
                                if (c.id == (*ground_truth)[g_i]) { hits++; break; }
                            }
                        }
                        cp.current_recall_at_10 = (float)hits / eval_k;
                    }
                    result.checkpoints.push_back(cp);
                }

                // Check termination condition against dynamic patience
                if (stagnation_counter >= current_patience) {
                    term_reason = TerminationReason::UncertaintyAwareEarlyExit;
                    stats.early_exit_stagnation_count++;
                    break;
                }
            }
        }

        result.stats = stats;
        result.termination_reason = term_reason;

        // 3. Extract final candidates
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

} // namespace adaptivevec
