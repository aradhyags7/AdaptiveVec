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
#include <cstdint>
#include <cstddef>
#include <cstring>

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
    
    // Ablation toggles
    bool enable_dynamic_m_efc = true;
    bool enable_layer_scaling = true;
    bool enable_hubness_regulation = true;
    bool enable_ada_ef = true;
    bool enable_sq8 = false;
    
    // Mechanics parameters
    float lambda_layer = 0.75f;
    int m_min_layer = 4;
    float hubness_mu = 0.15f;
    int ada_ef_patience = 6;
    float ada_ef_epsilon = 1e-4f;
    int sq8_rerank_factor = 2; // K = 2 * k
    
    // Online tracking state
    bool use_streaming_welford = true;
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

    int get_m_for_layer(int layer, bool enable_scaling = true, float lambda = 0.75f, int min_m = 4) const {
        if (layer == 0 || !enable_scaling) return m;
        float factor = std::pow(lambda, (float)layer);
        return std::max(min_m, (int)std::floor((float)m * factor));
    }

    int get_m_max_for_layer(int layer, bool enable_scaling = true, float lambda = 0.75f, int min_m = 4) const {
        if (layer == 0) return m_max0;
        if (!enable_scaling) return m_max;
        float factor = std::pow(lambda, (float)layer);
        return std::max(min_m, (int)std::floor((float)m_max * factor));
    }
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

// Distance Functions with AVX2 FMA SIMD acceleration
#if defined(__AVX2__)
#include <immintrin.h>

inline float _mm256_reduce_add_ps(__m256 x) {
    __m128 vlow  = _mm256_castps256_ps128(x);
    __m128 vhigh = _mm256_extractf128_ps(x, 1);
    __m128 v128  = _mm_add_ps(vlow, vhigh);
    __m128 v64   = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
    __m128 v32   = _mm_add_ss(v64, _mm_shuffle_ps(v64, v64, 0x55));
    return _mm_cvtss_f32(v32);
}

inline dist_t l2_distance(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
#if defined(__FMA__)
        sum256 = _mm256_fmadd_ps(diff, diff, sum256);
#else
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff, diff));
#endif
    }
    float total = _mm256_reduce_add_ps(sum256);
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return std::sqrt(total);
}

inline dist_t cosine_distance(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    __m256 dot256 = _mm256_setzero_ps();
    __m256 norm_a256 = _mm256_setzero_ps();
    __m256 norm_b256 = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        dot256 = _mm256_fmadd_ps(va, vb, dot256);
        norm_a256 = _mm256_fmadd_ps(va, va, norm_a256);
        norm_b256 = _mm256_fmadd_ps(vb, vb, norm_b256);
#else
        dot256 = _mm256_add_ps(dot256, _mm256_mul_ps(va, vb));
        norm_a256 = _mm256_add_ps(norm_a256, _mm256_mul_ps(va, va));
        norm_b256 = _mm256_add_ps(norm_b256, _mm256_mul_ps(vb, vb));
#endif
    }
    float dot = _mm256_reduce_add_ps(dot256);
    float norm_a = _mm256_reduce_add_ps(norm_a256);
    float norm_b = _mm256_reduce_add_ps(norm_b256);
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-9f || norm_b < 1e-9f) return 1.0f;
    dist_t cos_sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    return std::max(0.0f, 1.0f - cos_sim);
}
#else
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
#endif

struct StreamingStatsTracker {
    uint64_t count = 0;
    float mean = 0.0f;
    float m2 = 0.0f;
    bool use_ema = false;
    float ema_alpha = 0.05f;

    void update(float x) {
        count++;
        if (use_ema && count > 10) {
            float delta = x - mean;
            mean += ema_alpha * delta;
            m2 = (1.0f - ema_alpha) * (m2 + ema_alpha * delta * delta);
        } else {
            float delta = x - mean;
            mean += delta / (float)count;
            float delta2 = x - mean;
            m2 += delta * delta2;
        }
    }

    float get_mean() const {
        return mean;
    }

    float get_std() const {
        if (count < 2) return 1.0f;
        float var = m2 / (float)(count - 1);
        return std::sqrt(std::max(1e-6f, var));
    }
};

// Scalar Quantizer (SQ8) with fast asymmetric distance
struct ScalarQuantizer {
    size_t dim = 0;
    std::vector<float> min_vals;
    std::vector<float> scales;
    bool is_trained = false;

    ScalarQuantizer() = default;
    ScalarQuantizer(size_t d) : dim(d), min_vals(d, 0.0f), scales(d, 1.0f) {}

    void train(const float* vectors, size_t n_vectors) {
        if (n_vectors == 0 || dim == 0) return;
        min_vals.assign(dim, 1e9f);
        std::vector<float> max_vals(dim, -1e9f);
        scales.resize(dim);

        for (size_t i = 0; i < n_vectors; ++i) {
            const float* v = vectors + i * dim;
            for (size_t d = 0; d < dim; ++d) {
                if (v[d] < min_vals[d]) min_vals[d] = v[d];
                if (v[d] > max_vals[d]) max_vals[d] = v[d];
            }
        }

        for (size_t d = 0; d < dim; ++d) {
            float range = max_vals[d] - min_vals[d];
            scales[d] = (range > 1e-7f) ? (range / 255.0f) : 1.0f;
        }
        is_trained = true;
    }

    void encode(const float* vec, uint8_t* code) const {
        for (size_t d = 0; d < dim; ++d) {
            float scaled = (vec[d] - min_vals[d]) / scales[d];
            int c = (int)std::round(scaled);
            code[d] = (uint8_t)std::max(0, std::min(255, c));
        }
    }

    void decode(const uint8_t* code, float* vec) const {
        for (size_t d = 0; d < dim; ++d) {
            vec[d] = min_vals[d] + (float)code[d] * scales[d];
        }
    }

    dist_t asymmetric_l2_distance(const float* q, const uint8_t* code) const {
        float sum = 0.0f;
        size_t d = 0;
#if defined(__AVX2__)
        __m256 sum256 = _mm256_setzero_ps();
        for (; d + 8 <= dim; d += 8) {
            __m256 vq = _mm256_loadu_ps(q + d);
            __m256 vmin = _mm256_loadu_ps(&min_vals[d]);
            __m256 vscale = _mm256_loadu_ps(&scales[d]);

            __m128i b = _mm_loadl_epi64((const __m128i*)(code + d));
            __m256i epi32 = _mm256_cvtepu8_epi32(b);
            __m256 vc = _mm256_cvtepi32_ps(epi32);

#if defined(__FMA__)
            __m256 vapprox = _mm256_fmadd_ps(vc, vscale, vmin);
            __m256 diff = _mm256_sub_ps(vq, vapprox);
            sum256 = _mm256_fmadd_ps(diff, diff, sum256);
#else
            __m256 vapprox = _mm256_add_ps(vmin, _mm256_mul_ps(vc, vscale));
            __m256 diff = _mm256_sub_ps(vq, vapprox);
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff, diff));
#endif
        }
        sum = _mm256_reduce_add_ps(sum256);
#endif
        for (; d < dim; ++d) {
            float approx = min_vals[d] + (float)code[d] * scales[d];
            float diff = q[d] - approx;
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    dist_t asymmetric_cosine_distance(const float* q, const uint8_t* code) const {
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float approx = min_vals[d] + (float)code[d] * scales[d];
            dot += q[d] * approx;
            norm_a += q[d] * q[d];
            norm_b += approx * approx;
        }
        if (norm_a < 1e-9f || norm_b < 1e-9f) return 1.0f;
        dist_t cos_sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        return std::max(0.0f, 1.0f - cos_sim);
    }
};

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
    if (!cfg.enable_dynamic_m_efc) {
        return {cfg.m_base, cfg.m_base, 2 * cfg.m_base, cfg.ef_c_base, 0.0f, lid, density};
    }

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
    std::vector<uint8_t> quantized_data;
    ScalarQuantizer quantizer;
    size_t num_elements = 0;
    
    // Multi-layer adjacency: graphs[layer][node_id] = std::vector<tableint>
    std::vector<std::vector<std::vector<tableint>>> graphs;
    std::vector<int> node_levels;
    std::vector<NodeParams> node_params;
    
    // Hubness regulation: layer-0 in-degree tracking
    std::vector<uint32_t> in_degrees;
    uint64_t total_in_degrees = 0;
    
    // Online Streaming Welford Trackers
    StreamingStatsTracker density_tracker;
    StreamingStatsTracker lid_tracker;
    
    tableint enter_point = 0;
    int max_level = -1;
    double m_l;
    
    mutable uint64_t total_dist_computations = 0;
    
    AdaptiveHNSWIndex(size_t dimension, SpaceType sp = SpaceType::L2, bool adaptive = true, PolicyConfig cfg = PolicyConfig())
        : dim(dimension), space(sp), policy(cfg), is_adaptive(adaptive), quantizer(dimension) {
        m_l = 1.0 / std::log(policy.m_base > 1 ? policy.m_base : 2);
    }
    
    inline dist_t get_distance(const float* a, const float* b) const {
        total_dist_computations++;
        return space == SpaceType::Cosine ? cosine_distance(a, b, dim) : l2_distance(a, b, dim);
    }

    inline dist_t get_sq8_distance(const float* query, tableint id) const {
        total_dist_computations++;
        const uint8_t* code = &quantized_data[id * dim];
        return space == SpaceType::Cosine 
               ? quantizer.asymmetric_cosine_distance(query, code) 
               : quantizer.asymmetric_l2_distance(query, code);
    }
    
    inline const float* get_vector(tableint id) const {
        return &data[id * dim];
    }
    
    void train_quantizer(const float* sample_data, size_t sample_count) {
        quantizer.train(sample_data, sample_count);
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
    
    // Algorithm 2: SEARCH-LAYER with hardware cache prefetching and optional stagnation early exit
    std::vector<Candidate> search_layer(
        const float* query,
        const std::vector<tableint>& enter_points,
        int ef,
        int lc,
        bool early_exit = true,
        bool use_asym_sq8 = false
    ) const {
        std::unordered_set<tableint> visited(enter_points.begin(), enter_points.end());
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
        std::priority_queue<Candidate, std::vector<Candidate>, std::less<Candidate>> w_furthest;
        
        dist_t best_dist = 1e9f;
        int stagnation_counter = 0;
        
        for (tableint ep : enter_points) {
            dist_t d = (use_asym_sq8 && quantizer.is_trained) 
                       ? get_sq8_distance(query, ep) 
                       : get_distance(query, get_vector(ep));
            candidates.push({d, ep});
            w_furthest.push({d, ep});
            if (d < best_dist) best_dist = d;
        }
        
        while (!candidates.empty()) {
            Candidate curr = candidates.top();
            candidates.pop();
            dist_t furthest_d = w_furthest.top().dist;
            
            if (curr.dist > furthest_d) break;
            
            bool improved = false;
            
            if (lc < (int)graphs.size() && curr.id < graphs[lc].size()) {
                const auto& neighbors = graphs[lc][curr.id];
                for (size_t n_idx = 0; n_idx < neighbors.size(); ++n_idx) {
                    tableint neighbor = neighbors[n_idx];
                    
                    // Hardware Cache Prefetching for next candidate vector
                    if (n_idx + 1 < neighbors.size()) {
                        tableint next_neighbor = neighbors[n_idx + 1];
                        #if defined(__GNUC__) || defined(__clang__)
                        __builtin_prefetch(get_vector(next_neighbor), 0, 3);
                        #elif defined(_MSC_VER)
                        _mm_prefetch((const char*)get_vector(next_neighbor), _MM_HINT_T0);
                        #endif
                    }
                    
                    if (visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        dist_t d = (use_asym_sq8 && quantizer.is_trained) 
                                   ? get_sq8_distance(query, neighbor) 
                                   : get_distance(query, get_vector(neighbor));
                        furthest_d = w_furthest.top().dist;
                        
                        if (d < furthest_d || (int)w_furthest.size() < ef) {
                            candidates.push({d, neighbor});
                            
                            if ((int)w_furthest.size() >= ef) {
                                dist_t old_f = w_furthest.top().dist;
                                w_furthest.pop();
                                w_furthest.push({d, neighbor});
                                dist_t new_f = w_furthest.top().dist;
                                if (old_f - new_f > policy.ada_ef_epsilon) {
                                    improved = true;
                                }
                            } else {
                                w_furthest.push({d, neighbor});
                                improved = true;
                            }
                            
                            if (best_dist - d > policy.ada_ef_epsilon) {
                                best_dist = d;
                                improved = true;
                            }
                        }
                    }
                }
            }
            
            // Ada-ef Stagnation early exit for layer 0 search:
            // Activates only after beam W has accumulated at least ef candidates,
            // tracking whether the candidate expansion successfully improved W.
            if (early_exit && lc == 0 && (int)w_furthest.size() >= ef) {
                if (improved) {
                    stagnation_counter = 0;
                } else {
                    if (++stagnation_counter >= policy.ada_ef_patience) {
                        break;
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
    
    // Algorithm 4: SELECT-NEIGHBORS-HEURISTIC with Hubness-Aware In-Degree Regulation
    std::vector<tableint> select_neighbors_heuristic(
        const float* /* query */,
        const std::vector<Candidate>& candidates,
        int m_limit,
        int lc
    ) const {
        std::vector<Candidate> sorted_c = candidates;
        
        // Candidates sorted by true metric distance to base query
        std::sort(sorted_c.begin(), sorted_c.end());
        
        float mean_deg = 1.0f;
        if (policy.enable_hubness_regulation && lc == 0 && !in_degrees.empty() && num_elements > 0) {
            mean_deg = (float)total_in_degrees / (float)std::max((size_t)1, num_elements);
            mean_deg = std::max(1.0f, mean_deg);
        }
        
        std::vector<tableint> result;
        std::vector<Candidate> discarded;
        
        for (const auto& c : sorted_c) {
            if ((int)result.size() >= m_limit) break;
            
            const float* c_vec = get_vector(c.id);
            bool is_diverse = true;
            for (tableint r_id : result) {
                dist_t dist_c_r = get_distance(c_vec, get_vector(r_id));
                
                // Hubness Regulation: Penalize high-degree hubs from shadowing diverse neighbors
                if (policy.enable_hubness_regulation && lc == 0 && !in_degrees.empty()) {
                    float deg_r = (r_id < in_degrees.size()) ? (float)in_degrees[r_id] : 0.0f;
                    dist_c_r *= (1.0f + policy.hubness_mu * (deg_r / mean_deg));
                }
                
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
        
        if (policy.enable_sq8) {
            if (!quantizer.is_trained) {
                // Initialize default training bounds if not yet trained
                quantizer.train(vec, 1);
            }
            size_t old_size = quantized_data.size();
            quantized_data.resize(old_size + dim);
            quantizer.encode(vec, &quantized_data[old_size]);
        }
        
        in_degrees.push_back(0);
        
        if (q_idx == 0) {
            enter_point = 0;
            max_level = 0;
            node_levels.push_back(0);
            node_params.push_back({policy.m_base, policy.m_base, 2 * policy.m_base, policy.ef_c_base, 0.0f, policy.lid_mean, policy.density_mean});
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
            auto w = search_layer(vec, {curr_ep}, 1, lc, false, false);
            if (!w.empty()) curr_ep = w[0].id;
        }
        
        // Online Signal Estimation & Streaming Tracking
        NodeParams params;
        if (is_adaptive) {
            int probe_lc = std::min(top_level, q_level);
            int probe_budget = std::min(25, std::max(10, policy.ef_c_base / 4));
            auto probe_w = search_layer(vec, {curr_ep}, probe_budget, probe_lc, false, false);
            std::vector<dist_t> probe_dists;
            for (const auto& c : probe_w) probe_dists.push_back(c.dist);
            
            float density = estimate_local_density(probe_dists);
            float lid = estimate_mle_lid(probe_dists);
            
            if (policy.use_streaming_welford) {
                density_tracker.update(density);
                lid_tracker.update(lid);
                
                if (density_tracker.count >= 10) {
                    policy.density_mean = density_tracker.get_mean();
                    policy.density_std = density_tracker.get_std();
                    policy.lid_mean = lid_tracker.get_mean();
                    policy.lid_std = lid_tracker.get_std();
                }
            }
            
            params = evaluate_policy(density, lid, policy);
        } else {
            params = {policy.m_base, policy.m_base, 2 * policy.m_base, policy.ef_c_base, 0.0f, 0.0f, 0.0f};
        }
        node_params.push_back(params);
        
        // Phase 2: Insert into levels min(top_level, q_level) down to 0
        for (int lc = std::min(top_level, q_level); lc >= 0; --lc) {
            auto w = search_layer(vec, {curr_ep}, params.ef_construction, lc, false, false);
            
            int m_curr = params.get_m_for_layer(lc, policy.enable_layer_scaling, policy.lambda_layer, policy.m_min_layer);
            auto neighbors = select_neighbors_heuristic(vec, w, m_curr, lc);
            
            graphs[lc][q_idx] = neighbors;
            
            for (tableint neighbor : neighbors) {
                graphs[lc][neighbor].push_back(q_idx);
                if (lc == 0) {
                    in_degrees[neighbor]++;
                    in_degrees[q_idx]++;
                    total_in_degrees += 2;
                }
                
                int n_limit = node_params[neighbor].get_m_max_for_layer(lc, policy.enable_layer_scaling, policy.lambda_layer, policy.m_min_layer);
                if ((int)graphs[lc][neighbor].size() > n_limit) {
                    std::vector<Candidate> n_candidates;
                    const float* n_vec = get_vector(neighbor);
                    for (tableint cand : graphs[lc][neighbor]) {
                        n_candidates.push_back({get_distance(n_vec, get_vector(cand)), cand});
                    }
                    auto pruned = select_neighbors_heuristic(n_vec, n_candidates, n_limit, lc);
                    
                    if (lc == 0) {
                        std::unordered_set<tableint> kept(pruned.begin(), pruned.end());
                        for (tableint old_cand : graphs[lc][neighbor]) {
                            if (kept.find(old_cand) == kept.end()) {
                                if (in_degrees[old_cand] > 0) in_degrees[old_cand]--;
                                if (in_degrees[neighbor] > 0) in_degrees[neighbor]--;
                                if (total_in_degrees >= 2) total_in_degrees -= 2;
                            }
                        }
                    }
                    graphs[lc][neighbor] = pruned;
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
        bool use_asym = policy.enable_sq8 && quantizer.is_trained;
        bool use_ada = policy.enable_ada_ef;
        
        for (int lc = max_level; lc > 0; --lc) {
            auto w = search_layer(query, {curr_ep}, 1, lc, false, use_asym);
            if (!w.empty()) curr_ep = w[0].id;
        }
        
        int search_beam = use_asym ? std::max(policy.sq8_rerank_factor * k, ef) : ef;
        auto w = search_layer(query, {curr_ep}, search_beam, 0, use_ada, use_asym);
        
        if (use_asym) {
            // Two-stage re-ranking with exact float32 distance
            for (auto& cand : w) {
                cand.dist = get_distance(query, get_vector(cand.id));
            }
            std::sort(w.begin(), w.end());
        }
        
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

    size_t get_layer0_edges() const {
        size_t total = 0;
        if (!graphs.empty()) {
            for (const auto& adj : graphs[0]) {
                total += adj.size();
            }
        }
        return total;
    }

    size_t get_upper_layer_edges() const {
        size_t total = 0;
        for (size_t l = 1; l < graphs.size(); ++l) {
            for (const auto& adj : graphs[l]) {
                total += adj.size();
            }
        }
        return total;
    }

    size_t get_vector_memory_bytes() const {
        if (policy.enable_sq8 && quantizer.is_trained) {
            return num_elements * dim * sizeof(uint8_t) + 2 * dim * sizeof(float);
        }
        return num_elements * dim * sizeof(float);
    }

    size_t get_edge_memory_bytes() const {
        return get_total_edges() * sizeof(tableint);
    }

    size_t get_total_memory_bytes() const {
        size_t total = get_vector_memory_bytes() + get_edge_memory_bytes();
        total += num_elements * sizeof(int); // node_levels
        total += in_degrees.size() * sizeof(uint32_t); // in_degrees
        return total;
    }
};

} // namespace adaptivevec
