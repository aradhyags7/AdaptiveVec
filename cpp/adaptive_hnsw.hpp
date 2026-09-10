#pragma once

/**
 * AdaptiveVec: High-Performance C++ Core Engine
 * 
 * Implements:
 * 1. Stock HNSW Baseline (Algorithms 1-5 from Malkov & Yashunin, 2020)
 * 2. AdaptiveVec Dynamic Multi-Layer Proximity Graph with online LID & Density Estimation
 * 3. Cache-friendly neighbor heuristic and distance metrics (L2 and Cosine)
 */

#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <memory>
#include <string>

namespace adaptivevec {

using tableint = uint32_t;
using dist_t = float;

enum class SpaceType {
    L2,
    Cosine
};

enum class PolicyType {
    Continuous,
    Quantile,
    LidOnly,
    DensityOnly
};

struct PolicyConfig {
    PolicyType type = PolicyType::Continuous;
    int m_base = 16;
    int m_min = 8;
    int m_max = 24;
    int ef_c_base = 120;
    int ef_c_min = 40;
    int ef_c_max = 220;
    float alpha_lid = 0.5f;
    float beta_density = 0.5f;
    float sensitivity = 0.4f;
    
    float density_mean = 1.0f;
    float density_std = 1.0f;
    float lid_mean = 4.0f;
    float lid_std = 2.0f;
};

struct NodeParams {
    int m;
    int m_max;
    int m_max0;
    int ef_construction;
    float score;
    float lid;
    float density;
};

struct Candidate {
    dist_t dist;
    tableint id;
    
    bool operator>(const Candidate& other) const {
        return dist > other.dist;
    }
    bool operator<(const Candidate& other) const {
        return dist < other.dist;
    }
};

// Distance Functions
inline dist_t l2_distance(const float* a, const float* b, size_t dim) {
    dist_t sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dist_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline dist_t cosine_distance(const float* a, const float* b, size_t dim) {
    dist_t dot = 0.0f;
    dist_t norm_a = 0.0f;
    dist_t norm_b = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-9f || norm_b < 1e-9f) return 1.0f;
    dist_t cos_sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    return std::max(0.0f, 1.0f - cos_sim);
}

// Online Signal Estimators
inline float estimate_mle_lid(const std::vector<dist_t>& dists, int k = 15) {
    if (dists.size() < 3) return 2.0f;
    std::vector<dist_t> sorted_d = dists;
    std::sort(sorted_d.begin(), sorted_d.end());
    
    int n = std::min((int)sorted_d.size(), k);
    dist_t r_k = sorted_d[n - 1];
    if (r_k <= 1e-7f) return 1.0f;
    
    float log_sum = 0.0f;
    for (int i = 0; i < n - 1; ++i) {
        float ratio = std::max(1e-7f, std::min(1.0f - 1e-7f, sorted_d[i] / r_k));
        log_sum += std::log(ratio);
    }
    
    if (std::abs(log_sum) < 1e-7f) return 1.0f;
    float lid = -((float)(n - 1) / log_sum);
    return std::max(1.0f, std::min(lid, 1000.0f));
}

inline float estimate_local_density(const std::vector<dist_t>& dists) {
    if (dists.empty()) return 1.0f;
    float sum = 0.0f;
    for (dist_t d : dists) sum += d;
    return sum / (float)dists.size();
}

inline NodeParams evaluate_policy(float density, float lid, const PolicyConfig& cfg) {
    float norm_lid = (lid - cfg.lid_mean) / (cfg.lid_std + 1e-6f);
    float norm_density = (density - cfg.density_mean) / (cfg.density_std + 1e-6f);
    
    float score = 0.0f;
    if (cfg.type == PolicyType::LidOnly) {
        score = std::max(-2.5f, std::min(norm_lid, 2.5f));
    } else if (cfg.type == PolicyType::DensityOnly) {
        score = std::max(-2.5f, std::min(norm_density, 2.5f));
    } else {
        score = std::max(-2.5f, std::min(cfg.alpha_lid * norm_lid + cfg.beta_density * norm_density, 2.5f));
    }
    
    int m = cfg.m_base;
    int ef_c = cfg.ef_c_base;
    
    if (cfg.type == PolicyType::Quantile) {
        if (score < -0.4f) {
            m = cfg.m_min;
            ef_c = cfg.ef_c_min;
        } else if (score > 0.4f) {
            m = cfg.m_max;
            ef_c = cfg.ef_c_max;
        }
    } else {
        float factor = 1.0f + cfg.sensitivity * score;
        m = std::max(cfg.m_min, std::min((int)std::round(cfg.m_base * factor), cfg.m_max));
        ef_c = std::max(cfg.ef_c_min, std::min((int)std::round(cfg.ef_c_base * factor), cfg.ef_c_max));
    }
    
    return {m, m, 2 * m, ef_c, score, lid, density};
}

/**
 * Adaptive HNSW Graph Index
 */
class AdaptiveHNSWIndex {
public:
    size_t dim;
    SpaceType space;
    PolicyConfig policy;
    bool is_adaptive;
    
    // Flat vector storage: data[i * dim ... (i+1) * dim - 1]
    std::vector<float> data;
    size_t num_elements = 0;
    
    // Multi-layer adjacency: graphs[layer][node_id] = std::vector<tableint>
    std::vector<std::vector<std::vector<tableint>>> graphs;
    std::vector<int> node_levels;
    std::vector<NodeParams> node_params;
    
    tableint enter_point = 0;
    int max_level = -1;
    double m_l;
    
    mutable uint64_t total_dist_computations = 0;
    
    AdaptiveHNSWIndex(size_t dimension, SpaceType sp = SpaceType::L2, bool adaptive = true, PolicyConfig cfg = PolicyConfig())
        : dim(dimension), space(sp), is_adaptive(adaptive), policy(cfg) {
        m_l = 1.0 / std::log(policy.m_base > 1 ? policy.m_base : 2);
    }
    
    inline dist_t get_distance(const float* a, const float* b) const {
        total_dist_computations++;
        return space == SpaceType::Cosine ? cosine_distance(a, b, dim) : l2_distance(a, b, dim);
    }
    
    inline const float* get_vector(tableint id) const {
        return &data[id * dim];
    }
    
    void calibrate(const float* sample_data, size_t sample_count, int k = 15) {
        if (sample_count < 10) return;
        std::vector<float> sample_densities;
        std::vector<float> sample_lids;
        
        for (size_t i = 0; i < sample_count; ++i) {
            const float* cur_v = &sample_data[i * dim];
            std::vector<dist_t> dists;
            for (size_t j = 0; j < sample_count; ++j) {
                if (i == j) continue;
                dists.push_back(get_distance(cur_v, &sample_data[j * dim]));
            }
            std::sort(dists.begin(), dists.end());
            if (dists.size() >= (size_t)k) {
                dists.resize(k);
                sample_densities.push_back(estimate_local_density(dists));
                sample_lids.push_back(estimate_mle_lid(dists, k));
            }
        }
        
        if (!sample_densities.empty()) {
            float sum_d = std::accumulate(sample_densities.begin(), sample_densities.end(), 0.0f);
            policy.density_mean = sum_d / sample_densities.size();
            float sq_sum_d = 0.0f;
            for (float d : sample_densities) sq_sum_d += (d - policy.density_mean) * (d - policy.density_mean);
            policy.density_std = std::sqrt(sq_sum_d / sample_densities.size());
            if (policy.density_std < 1e-4f) policy.density_std = 1.0f;
            
            float sum_l = std::accumulate(sample_lids.begin(), sample_lids.end(), 0.0f);
            policy.lid_mean = sum_l / sample_lids.size();
            float sq_sum_l = 0.0f;
            for (float l : sample_lids) sq_sum_l += (l - policy.lid_mean) * (l - policy.lid_mean);
            policy.lid_std = std::sqrt(sq_sum_l / sample_lids.size());
            if (policy.lid_std < 1e-4f) policy.lid_std = 1.0f;
        }
    }

    int generate_random_level() {
        static thread_local std::mt19937 gen(1337);
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        double r = dis(gen);
        while (r == 0.0) r = dis(gen);
        return (int)(-std::log(r) * m_l);
    }
    
    // Algorithm 2: SEARCH-LAYER
    std::vector<Candidate> search_layer(const float* query, const std::vector<tableint>& enter_points, int ef, int lc) const {
        std::unordered_set<tableint> visited(enter_points.begin(), enter_points.end());
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
        std::priority_queue<Candidate, std::vector<Candidate>, std::less<Candidate>> w_furthest;
        
        for (tableint ep : enter_points) {
            dist_t d = get_distance(query, get_vector(ep));
            candidates.push({d, ep});
            w_furthest.push({d, ep});
        }
        
        while (!candidates.empty()) {
            Candidate curr = candidates.top();
            candidates.pop();
            dist_t furthest_d = w_furthest.top().dist;
            
            if (curr.dist > furthest_d) break;
            
            if (lc < (int)graphs.size() && curr.id < graphs[lc].size()) {
                for (tableint neighbor : graphs[lc][curr.id]) {
                    if (visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        dist_t d = get_distance(query, get_vector(neighbor));
                        furthest_d = w_furthest.top().dist;
                        
                        if (d < furthest_d || (int)w_furthest.size() < ef) {
                            candidates.push({d, neighbor});
                            w_furthest.push({d, neighbor});
                            
                            if ((int)w_furthest.size() > ef) {
                                w_furthest.pop();
                            }
                        }
                    }
                }
            }
        }
        
        std::vector<Candidate> result;
        while (!w_furthest.empty()) {
            result.push_back(w_furthest.top());
            w_furthest.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }
    
    // Algorithm 4: SELECT-NEIGHBORS-HEURISTIC
    std::vector<tableint> select_neighbors_heuristic(const float* query, const std::vector<Candidate>& candidates, int m_limit) const {
        std::vector<Candidate> sorted_c = candidates;
        std::sort(sorted_c.begin(), sorted_c.end());
        
        std::vector<tableint> result;
        std::vector<Candidate> discarded;
        
        for (const auto& c : sorted_c) {
            if ((int)result.size() >= m_limit) break;
            
            const float* c_vec = get_vector(c.id);
            bool is_diverse = true;
            for (tableint r_id : result) {
                dist_t dist_c_r = get_distance(c_vec, get_vector(r_id));
                if (dist_c_r < c.dist) {
                    is_diverse = false;
                    break;
                }
            }
            
            if (is_diverse) {
                result.push_back(c.id);
            } else {
                discarded.push_back(c);
            }
        }
        
        for (const auto& d : discarded) {
            if ((int)result.size() >= m_limit) break;
            result.push_back(d.id);
        }
        
        return result;
    }
    
    tableint insert(const float* vec) {
        tableint q_idx = (tableint)num_elements++;
        data.insert(data.end(), vec, vec + dim);
        
        if (q_idx == 0) {
            enter_point = 0;
            max_level = 0;
            node_levels.push_back(0);
            node_params.push_back({policy.m_base, policy.m_base, 2 * policy.m_base, policy.ef_c_base, 0, policy.lid_mean, policy.density_mean});
            graphs.resize(1);
            graphs[0].resize(1);
            return 0;
        }
        
        int q_level = generate_random_level();
        node_levels.push_back(q_level);
        
        if (q_level >= (int)graphs.size()) {
            graphs.resize(q_level + 1);
        }
        for (int l = 0; l <= q_level; ++l) {
            if (q_idx >= graphs[l].size()) graphs[l].resize(q_idx + 1);
        }
        
        tableint curr_ep = enter_point;
        int top_level = max_level;
        
        // Phase 1: Top-down greedy 1-NN traversal (ef=1)
        for (int lc = top_level; lc > q_level; --lc) {
            auto w = search_layer(vec, {curr_ep}, 1, lc);
            curr_ep = w[0].id;
        }
        
        // Online Signal Estimation & Policy Assignment
        NodeParams params;
        if (is_adaptive) {
            int probe_lc = std::min(top_level, q_level);
            auto probe_w = search_layer(vec, {curr_ep}, std::min(25, policy.ef_c_base / 4), probe_lc);
            std::vector<dist_t> probe_dists;
            for (const auto& c : probe_w) probe_dists.push_back(c.dist);
            
            float density = estimate_local_density(probe_dists);
            float lid = estimate_mle_lid(probe_dists);
            params = evaluate_policy(density, lid, policy);
        } else {
            params = {policy.m_base, policy.m_base, 2 * policy.m_base, policy.ef_c_base, 0, 0, 0};
        }
        node_params.push_back(params);
        
        // Phase 2: Insert into levels min(top_level, q_level) down to 0
        for (int lc = std::min(top_level, q_level); lc >= 0; --lc) {
            auto w = search_layer(vec, {curr_ep}, params.ef_construction, lc);
            auto neighbors = select_neighbors_heuristic(vec, w, params.m);
            
            graphs[lc][q_idx] = neighbors;
            
            for (tableint neighbor : neighbors) {
                graphs[lc][neighbor].push_back(q_idx);
                
                int n_limit = (lc == 0) ? node_params[neighbor].m_max0 : node_params[neighbor].m_max;
                if ((int)graphs[lc][neighbor].size() > n_limit) {
                    std::vector<Candidate> n_candidates;
                    const float* n_vec = get_vector(neighbor);
                    for (tableint cand : graphs[lc][neighbor]) {
                        n_candidates.push_back({get_distance(n_vec, get_vector(cand)), cand});
                    }
                    graphs[lc][neighbor] = select_neighbors_heuristic(n_vec, n_candidates, n_limit);
                }
            }
            
            if (!w.empty()) curr_ep = w[0].id;
        }
        
        if (q_level > max_level) {
            max_level = q_level;
            enter_point = q_idx;
        }
        
        return q_idx;
    }
    
    std::vector<Candidate> search(const float* query, int k, int ef) const {
        if (num_elements == 0) return {};
        
        tableint curr_ep = enter_point;
        for (int lc = max_level; lc > 0; --lc) {
            auto w = search_layer(query, {curr_ep}, 1, lc);
            curr_ep = w[0].id;
        }
        
        auto w = search_layer(query, {curr_ep}, ef, 0);
        if ((int)w.size() > k) {
            w.resize(k);
        }
        return w;
    }
    
    size_t get_total_edges() const {
        size_t total = 0;
        for (const auto& layer_graph : graphs) {
            for (const auto& adj : layer_graph) {
                total += adj.size();
            }
        }
        return total;
    }
};

} // namespace adaptivevec
