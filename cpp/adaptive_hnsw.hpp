// AdaptiveVec HNSW Engine with fine-grained mutex striping and AVX2 vectorization
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
#include <atomic>
#include <mutex>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif

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
    bool enable_sq4 = false;
    
    // Frontier 1: Query-Time Dynamic efSearch
    bool enable_dynamic_ef_search = false;
    float dynamic_ef_factor = 0.40f;
    int dynamic_ef_min = 16;
    int dynamic_ef_max = 128;
    
    // Mechanics parameters
    float lambda_layer = 0.75f;
    int m_min_layer = 4;
    float hubness_mu = 0.15f;
    int ada_ef_patience = 6;
    float ada_ef_epsilon = 1e-4f;
    int sq8_rerank_factor = 2; // K = 2 * k
    int sq4_rerank_factor = 3; // K = 3 * k (Frontier 3 Sub-Byte)
    
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

// Frontier 3: 4-Bit Sub-Byte Scalar Quantizer (SQ4) with AVX2 SIMD nibble unpacking
struct ScalarQuantizer4Bit {
    size_t dim = 0;
    size_t packed_dim = 0;
    std::vector<float> min_vals;
    std::vector<float> scales;
    bool is_trained = false;

    ScalarQuantizer4Bit() = default;
    ScalarQuantizer4Bit(size_t d) 
        : dim(d), packed_dim((d + 1) / 2), min_vals(d, 0.0f), scales(d, 1.0f) {}

    void train(const float* vectors, size_t n_vectors) {
        if (n_vectors == 0 || dim == 0) return;
        packed_dim = (dim + 1) / 2;
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
            scales[d] = (range > 1e-7f) ? (range / 15.0f) : 1.0f;
        }
        is_trained = true;
    }

    void encode(const float* vec, uint8_t* code) const {
        for (size_t i = 0; i < packed_dim; ++i) {
            size_t d0 = i * 2;
            size_t d1 = d0 + 1;

            float scaled0 = (vec[d0] - min_vals[d0]) / scales[d0];
            int c0 = (int)std::round(scaled0);
            uint8_t nibble0 = (uint8_t)std::max(0, std::min(15, c0));

            uint8_t nibble1 = 0;
            if (d1 < dim) {
                float scaled1 = (vec[d1] - min_vals[d1]) / scales[d1];
                int c1 = (int)std::round(scaled1);
                nibble1 = (uint8_t)std::max(0, std::min(15, c1));
            }

            code[i] = (uint8_t)((nibble1 << 4) | (nibble0 & 0x0F));
        }
    }

    void decode(const uint8_t* code, float* vec) const {
        for (size_t i = 0; i < packed_dim; ++i) {
            size_t d0 = i * 2;
            size_t d1 = d0 + 1;
            uint8_t byte = code[i];
            uint8_t nibble0 = byte & 0x0F;
            uint8_t nibble1 = (byte >> 4) & 0x0F;

            vec[d0] = min_vals[d0] + (float)nibble0 * scales[d0];
            if (d1 < dim) {
                vec[d1] = min_vals[d1] + (float)nibble1 * scales[d1];
            }
        }
    }

    dist_t asymmetric_l2_distance(const float* q, const uint8_t* code) const {
        float sum = 0.0f;
        size_t d = 0;
#if defined(__AVX2__)
        __m256 sum256 = _mm256_setzero_ps();
        const __m128i mask_low = _mm_set1_epi8(0x0F);

        // Process 16 dimensions per 8 packed bytes
        for (; d + 16 <= dim; d += 16) {
            size_t byte_idx = d / 2;
            __m128i raw = _mm_loadl_epi64((const __m128i*)(code + byte_idx));
            __m128i low_nibbles = _mm_and_si128(raw, mask_low);
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_low);
            __m128i unpacked = _mm_unpacklo_epi8(low_nibbles, high_nibbles);

            // First 8 dimensions
            __m256i epi32_lo = _mm256_cvtepu8_epi32(unpacked);
            __m256 vc_lo = _mm256_cvtepi32_ps(epi32_lo);
            __m256 vq_lo = _mm256_loadu_ps(q + d);
            __m256 vmin_lo = _mm256_loadu_ps(&min_vals[d]);
            __m256 vscale_lo = _mm256_loadu_ps(&scales[d]);
#if defined(__FMA__)
            __m256 approx_lo = _mm256_fmadd_ps(vc_lo, vscale_lo, vmin_lo);
            __m256 diff_lo = _mm256_sub_ps(vq_lo, approx_lo);
            sum256 = _mm256_fmadd_ps(diff_lo, diff_lo, sum256);
#else
            __m256 approx_lo = _mm256_add_ps(vmin_lo, _mm256_mul_ps(vc_lo, vscale_lo));
            __m256 diff_lo = _mm256_sub_ps(vq_lo, approx_lo);
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff_lo, diff_lo));
#endif

            // Second 8 dimensions
            __m128i unpacked_hi = _mm_srli_si128(unpacked, 8);
            __m256i epi32_hi = _mm256_cvtepu8_epi32(unpacked_hi);
            __m256 vc_hi = _mm256_cvtepi32_ps(epi32_hi);
            __m256 vq_hi = _mm256_loadu_ps(q + d + 8);
            __m256 vmin_hi = _mm256_loadu_ps(&min_vals[d + 8]);
            __m256 vscale_hi = _mm256_loadu_ps(&scales[d + 8]);
#if defined(__FMA__)
            __m256 approx_hi = _mm256_fmadd_ps(vc_hi, vscale_hi, vmin_hi);
            __m256 diff_hi = _mm256_sub_ps(vq_hi, approx_hi);
            sum256 = _mm256_fmadd_ps(diff_hi, diff_hi, sum256);
#else
            __m256 approx_hi = _mm256_add_ps(vmin_hi, _mm256_mul_ps(vc_hi, vscale_hi));
            __m256 diff_hi = _mm256_sub_ps(vq_hi, approx_hi);
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff_hi, diff_hi));
#endif
        }
        sum = _mm256_reduce_add_ps(sum256);
#endif
        // Remainder scalar loop
        for (; d < dim; ++d) {
            size_t byte_idx = d / 2;
            uint8_t byte = code[byte_idx];
            uint8_t nibble = (d % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            float approx = min_vals[d] + (float)nibble * scales[d];
            float diff = q[d] - approx;
            sum += diff * diff;
        }
        return std::sqrt(sum);
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
    std::vector<uint8_t> quantized_data_sq4;
    ScalarQuantizer quantizer;
    ScalarQuantizer4Bit quantizer_sq4;
    size_t num_elements = 0;
    
    // Multi-layer adjacency: graphs[layer][node_id] = std::vector<tableint>
    std::vector<std::vector<std::vector<tableint>>> graphs;
    std::vector<int> node_levels;
    std::vector<NodeParams> node_params;
    
    // Hubness regulation: layer-0 in-degree tracking
    std::vector<uint32_t> in_degrees;
    uint64_t total_in_degrees = 0;
    
    // Frontier 2: Dynamic Streaming Node Deletions & Tombstone Tracking
    std::vector<uint8_t> deleted;
    std::atomic<size_t> num_deleted{0};
    
    // Online Streaming Welford Trackers
    StreamingStatsTracker density_tracker;
    StreamingStatsTracker lid_tracker;
    
    tableint enter_point = 0;
    int max_level = -1;
    double m_l;
    
    static constexpr size_t NUM_NODE_LOCKS = 4096;
    mutable std::vector<std::mutex> node_locks;
    mutable std::atomic<uint64_t> total_dist_computations{0};
    mutable std::mutex entry_point_mutex;
    mutable std::mutex tracker_mutex;
    
    AdaptiveHNSWIndex(size_t dimension, SpaceType sp = SpaceType::L2, bool adaptive = true, PolicyConfig cfg = PolicyConfig())
        : dim(dimension), space(sp), policy(cfg), is_adaptive(adaptive), quantizer(dimension), quantizer_sq4(dimension), node_locks(NUM_NODE_LOCKS) {
        m_l = 1.0 / std::log(policy.m_base > 1 ? policy.m_base : 2);
    }

    inline bool is_deleted(tableint id) const {
        return id < deleted.size() && deleted[id] != 0;
    }

    inline size_t get_active_count() const {
        return num_elements > num_deleted.load() ? (num_elements - num_deleted.load()) : 0;
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

    inline dist_t get_sq4_distance(const float* query, tableint id) const {
        total_dist_computations++;
        size_t packed_dim = (dim + 1) / 2;
        const uint8_t* code = &quantized_data_sq4[id * packed_dim];
        return quantizer_sq4.asymmetric_l2_distance(query, code);
    }
    
    inline const float* get_vector(tableint id) const {
        return &data[id * dim];
    }
    
    void train_quantizer(const float* sample_data, size_t sample_count) {
        if (policy.enable_sq4) {
            quantizer_sq4.train(sample_data, sample_count);
        }
        if (policy.enable_sq8) {
            quantizer.train(sample_data, sample_count);
        }
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
        bool use_asym_sq8 = false,
        bool concurrent_search = false,
        bool use_asym_sq4 = false
    ) const {
        std::unordered_set<tableint> visited(enter_points.begin(), enter_points.end());
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
        std::priority_queue<Candidate, std::vector<Candidate>, std::less<Candidate>> w_furthest;
        
        dist_t best_dist = 1e9f;
        int stagnation_counter = 0;
        
        for (tableint ep : enter_points) {
            dist_t d = (use_asym_sq4 && quantizer_sq4.is_trained)
                       ? get_sq4_distance(query, ep)
                       : (use_asym_sq8 && quantizer.is_trained) 
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
                std::vector<tableint> local_neighbors;
                if (concurrent_search) {
                    std::lock_guard<std::mutex> lk(node_locks[curr.id % NUM_NODE_LOCKS]);
                    local_neighbors = graphs[lc][curr.id];
                }
                const auto& neighbors = concurrent_search ? local_neighbors : graphs[lc][curr.id];
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
                        dist_t d = (use_asym_sq4 && quantizer_sq4.is_trained)
                                   ? get_sq4_distance(query, neighbor)
                                   : (use_asym_sq8 && quantizer.is_trained) 
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

        if (policy.enable_sq4) {
            if (!quantizer_sq4.is_trained) {
                quantizer_sq4.train(vec, 1);
            }
            size_t packed_dim = (dim + 1) / 2;
            size_t old_size = quantized_data_sq4.size();
            quantized_data_sq4.resize(old_size + packed_dim);
            quantizer_sq4.encode(vec, &quantized_data_sq4[old_size]);
        }
        
        in_degrees.push_back(0);
        deleted.push_back(0);
        
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
    
    /**
     * Multi-Threaded Concurrent Batch Insertion
     * Parallelizes vector insertion across worker threads using striped node locking.
     */
    void batch_insert(const float* vectors, size_t n_vectors, int num_threads = 0) {
        if (n_vectors == 0) return;
        
        if (num_threads <= 0) {
            #if defined(_OPENMP)
            num_threads = omp_get_max_threads();
            #else
            num_threads = (int)std::max(1u, std::thread::hardware_concurrency());
            #endif
        }

        // If single thread requested, simply run sequential insert
        if (num_threads == 1) {
            for (size_t i = 0; i < n_vectors; ++i) {
                insert(&vectors[i * dim]);
            }
            return;
        }

        // Seed Phase: Insert the first min(n_vectors, 256) sequentially to establish graph root hierarchy
        size_t seed_count = std::min((size_t)256, n_vectors);
        for (size_t i = 0; i < seed_count; ++i) {
            insert(&vectors[i * dim]);
        }
        if (seed_count >= n_vectors) return;

        // Reserve and copy vector memory up-front
        size_t start_idx = num_elements;
        num_elements = n_vectors;
        data.resize(n_vectors * dim);
        std::memcpy(&data[start_idx * dim], &vectors[start_idx * dim], (n_vectors - start_idx) * dim * sizeof(float));

        if (policy.enable_sq8) {
            quantized_data.resize(n_vectors * dim);
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (size_t i = start_idx; i < n_vectors; ++i) {
                quantizer.encode(&data[i * dim], &quantized_data[i * dim]);
            }
        }

        if (policy.enable_sq4) {
            size_t packed_dim = (dim + 1) / 2;
            quantized_data_sq4.resize(n_vectors * packed_dim);
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (size_t i = start_idx; i < n_vectors; ++i) {
                quantizer_sq4.encode(&data[i * dim], &quantized_data_sq4[i * packed_dim]);
            }
        }

        node_levels.resize(n_vectors);
        node_params.resize(n_vectors);
        for (size_t i = start_idx; i < n_vectors; ++i) {
            node_params[i] = NodeParams{policy.m_base, policy.m_base, 2 * policy.m_base, policy.ef_c_base, 0.0f, 0.0f, 0.0f};
        }
        in_degrees.resize(n_vectors, 0);
        deleted.resize(n_vectors, 0);

        // Pre-generate levels and ensure graphs layers exist
        int expected_max_level = max_level;
        for (size_t i = start_idx; i < n_vectors; ++i) {
            int lvl = generate_random_level();
            node_levels[i] = lvl;
            if (lvl > expected_max_level) expected_max_level = lvl;
        }
        if (expected_max_level >= (int)graphs.size()) {
            graphs.resize(expected_max_level + 1);
        }
        for (size_t l = 0; l < graphs.size(); ++l) {
            if (graphs[l].size() < n_vectors) {
                graphs[l].resize(n_vectors);
            }
            for (size_t i = 0; i < n_vectors; ++i) {
                graphs[l][i].reserve(64);
            }
        }

        // Parallel insertion phase for remaining vectors
        #if defined(_OPENMP)
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 64)
        #endif
        for (size_t q_idx = start_idx; q_idx < n_vectors; ++q_idx) {
            const float* vec = &data[q_idx * dim];
            int q_level = node_levels[q_idx];
            
            tableint curr_ep;
            int top_level;
            {
                std::lock_guard<std::mutex> lk(entry_point_mutex);
                curr_ep = enter_point;
                top_level = max_level;
            }

            // Phase 1: Top-down greedy 1-NN traversal (ef=1)
            for (int lc = top_level; lc > q_level; --lc) {
                auto w = search_layer(vec, {curr_ep}, 1, lc, false, false, true);
                if (!w.empty()) curr_ep = w[0].id;
            }

            // Online Signal Estimation
            NodeParams params;
            if (is_adaptive) {
                int probe_lc = std::min(top_level, q_level);
                int probe_budget = std::min(25, std::max(10, policy.ef_c_base / 4));
                auto probe_w = search_layer(vec, {curr_ep}, probe_budget, probe_lc, false, false, true);
                std::vector<dist_t> probe_dists;
                for (const auto& c : probe_w) probe_dists.push_back(c.dist);

                float density = estimate_local_density(probe_dists);
                float lid = estimate_mle_lid(probe_dists);

                if (policy.use_streaming_welford) {
                    std::lock_guard<std::mutex> lk(tracker_mutex);
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
            node_params[q_idx] = params;

            // Phase 2: Insert into levels min(top_level, q_level) down to 0
            for (int lc = std::min(top_level, q_level); lc >= 0; --lc) {
                auto w = search_layer(vec, {curr_ep}, params.ef_construction, lc, false, false, true);
                int m_curr = params.get_m_for_layer(lc, policy.enable_layer_scaling, policy.lambda_layer, policy.m_min_layer);
                auto neighbors = select_neighbors_heuristic(vec, w, m_curr, lc);

                {
                    std::lock_guard<std::mutex> lk(node_locks[q_idx % NUM_NODE_LOCKS]);
                    graphs[lc][q_idx] = neighbors;
                }

                for (tableint neighbor : neighbors) {
                    std::lock_guard<std::mutex> lk(node_locks[neighbor % NUM_NODE_LOCKS]);
                    graphs[lc][neighbor].push_back((tableint)q_idx);
                    if (lc == 0) {
                        __atomic_fetch_add(&in_degrees[neighbor], 1, __ATOMIC_RELAXED);
                        __atomic_fetch_add(&in_degrees[q_idx], 1, __ATOMIC_RELAXED);
                    }

                    int n_limit = (node_params[neighbor].m > 0)
                        ? node_params[neighbor].get_m_max_for_layer(lc, policy.enable_layer_scaling, policy.lambda_layer, policy.m_min_layer)
                        : policy.m_base * 2;
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
                                    if (in_degrees[old_cand] > 0) {
                                        __atomic_fetch_sub(&in_degrees[old_cand], 1, __ATOMIC_RELAXED);
                                    }
                                }
                            }
                        }
                        graphs[lc][neighbor] = pruned;
                    }
                }

                if (!w.empty()) curr_ep = w[0].id;
            }

            if (q_level > max_level) {
                std::lock_guard<std::mutex> lk(entry_point_mutex);
                if (q_level > max_level) {
                    max_level = q_level;
                    enter_point = (tableint)q_idx;
                }
            }
        }

        // Recompute exact in-degrees in parallel directly from layer 0 graph
        std::fill(in_degrees.begin(), in_degrees.end(), 0);
        #if defined(_OPENMP)
        #pragma omp parallel
        {
            std::vector<uint32_t> local_in(n_vectors, 0);
            #pragma omp for schedule(static)
            for (size_t u = 0; u < n_vectors; ++u) {
                if (!graphs[0].empty() && u < graphs[0].size()) {
                    for (tableint v : graphs[0][u]) {
                        if (v < n_vectors) local_in[v]++;
                    }
                }
            }
            #pragma omp critical
            {
                for (size_t i = 0; i < n_vectors; ++i) {
                    in_degrees[i] += local_in[i];
                }
            }
        }
        #else
        for (size_t u = 0; u < n_vectors; ++u) {
            for (tableint v : graphs[0][u]) {
                if (v < n_vectors) in_degrees[v]++;
            }
        }
        #endif

        uint64_t total = 0;
        for (size_t i = 0; i < n_vectors; ++i) total += in_degrees[i];
        total_in_degrees = total;
    }

    /**
     * Frontier 2: Dynamic Streaming Node Deletions & Localized Graph Healing
     */
    void migrate_entry_point() {
        for (int lc = max_level; lc >= 0; --lc) {
            if (lc >= (int)graphs.size()) continue;
            tableint best_candidate = 0;
            bool found = false;
            size_t max_deg = 0;

            for (tableint u = 0; u < graphs[lc].size(); ++u) {
                if (!is_deleted(u) && !graphs[lc][u].empty()) {
                    if (!found || graphs[lc][u].size() > max_deg) {
                        best_candidate = u;
                        max_deg = graphs[lc][u].size();
                        found = true;
                    }
                }
            }
            if (found) {
                enter_point = best_candidate;
                max_level = lc;
                return;
            }
        }

        for (tableint u = 0; u < num_elements; ++u) {
            if (!is_deleted(u)) {
                enter_point = u;
                max_level = 0;
                return;
            }
        }

        max_level = -1;
        enter_point = 0;
    }

    void heal_neighborhood(tableint v) {
        if (v >= num_elements) return;
        int v_level = (v < node_levels.size()) ? node_levels[v] : 0;

        for (int lc = v_level; lc >= 0; --lc) {
            if (lc >= (int)graphs.size() || v >= graphs[lc].size()) continue;

            // Collect surviving neighbors of v
            std::vector<tableint> v_surviving_neighbors;
            {
                std::lock_guard<std::mutex> lk_v(node_locks[v % NUM_NODE_LOCKS]);
                for (tableint nb : graphs[lc][v]) {
                    if (!is_deleted(nb)) {
                        v_surviving_neighbors.push_back(nb);
                    }
                }
            }

            // Sever (u -> v) from each neighbor u and heal with candidates from v's surviving neighbors
            for (tableint u : v_surviving_neighbors) {
                if (is_deleted(u)) continue;

                std::vector<tableint> added_reps;
                {
                    std::lock_guard<std::mutex> lk_u(node_locks[u % NUM_NODE_LOCKS]);
                    auto& u_adj = graphs[lc][u];
                    auto it = std::find(u_adj.begin(), u_adj.end(), v);
                    if (it != u_adj.end()) {
                        u_adj.erase(it);
                        if (lc == 0) {
                            if (in_degrees[v] > 0) in_degrees[v]--;
                            if (in_degrees[u] > 0) in_degrees[u]--;
                        }

                        // Candidate replacements for u: other surviving neighbors of v
                        std::vector<Candidate> repair_cands;
                        const float* u_vec = get_vector(u);
                        for (tableint cand : v_surviving_neighbors) {
                            if (cand != u && !is_deleted(cand) && std::find(u_adj.begin(), u_adj.end(), cand) == u_adj.end()) {
                                repair_cands.push_back({get_distance(u_vec, get_vector(cand)), cand});
                            }
                        }

                        if (!repair_cands.empty()) {
                            std::sort(repair_cands.begin(), repair_cands.end());
                            int u_limit = (u < node_params.size())
                                ? node_params[u].get_m_max_for_layer(lc, policy.enable_layer_scaling, policy.lambda_layer, policy.m_min_layer)
                                : policy.m_base * 2;
                            
                            // Reconnect u with up to 3 best candidates up to u_limit
                            int max_to_add = std::min(3, (int)repair_cands.size());
                            for (int ci = 0; ci < max_to_add; ++ci) {
                                if ((int)u_adj.size() >= u_limit) break;
                                tableint rep_id = repair_cands[ci].id;
                                u_adj.push_back(rep_id);
                                added_reps.push_back(rep_id);
                                if (lc == 0) {
                                    in_degrees[rep_id]++;
                                    in_degrees[u]++;
                                    total_in_degrees += 2;
                                }
                            }
                        }
                    }
                }

                // Add reciprocal edges for added replacements
                for (tableint best_rep : added_reps) {
                    if (best_rep != u && !is_deleted(best_rep)) {
                        std::lock_guard<std::mutex> lk_rep(node_locks[best_rep % NUM_NODE_LOCKS]);
                        auto& rep_adj = graphs[lc][best_rep];
                        if (std::find(rep_adj.begin(), rep_adj.end(), u) == rep_adj.end()) {
                            int rep_limit = (best_rep < node_params.size())
                                ? node_params[best_rep].get_m_max_for_layer(lc, policy.enable_layer_scaling, policy.lambda_layer, policy.m_min_layer)
                                : policy.m_base * 2;
                            if ((int)rep_adj.size() < rep_limit) {
                                rep_adj.push_back(u);
                                if (lc == 0) {
                                    in_degrees[u]++;
                                    in_degrees[best_rep]++;
                                    total_in_degrees += 2;
                                }
                            }
                        }
                    }
                }
            }

            // Mutual neighbor bridging between surviving neighbors of v
            if (v_surviving_neighbors.size() >= 2) {
                for (size_t i = 0; i < v_surviving_neighbors.size(); ++i) {
                    tableint w1 = v_surviving_neighbors[i];
                    if (is_deleted(w1)) continue;

                    for (size_t j = i + 1; j < v_surviving_neighbors.size(); ++j) {
                        tableint w2 = v_surviving_neighbors[j];
                        if (is_deleted(w2)) continue;

                        int w1_limit = (w1 < node_params.size())
                            ? node_params[w1].get_m_max_for_layer(lc, policy.enable_layer_scaling, policy.lambda_layer, policy.m_min_layer)
                            : policy.m_base * 2;
                        int w2_limit = (w2 < node_params.size())
                            ? node_params[w2].get_m_max_for_layer(lc, policy.enable_layer_scaling, policy.lambda_layer, policy.m_min_layer)
                            : policy.m_base * 2;

                        // Check w1 -> w2
                        {
                            std::lock_guard<std::mutex> lk1(node_locks[w1 % NUM_NODE_LOCKS]);
                            if ((int)graphs[lc][w1].size() < w1_limit) {
                                if (std::find(graphs[lc][w1].begin(), graphs[lc][w1].end(), w2) == graphs[lc][w1].end()) {
                                    graphs[lc][w1].push_back(w2);
                                    if (lc == 0) { in_degrees[w2]++; in_degrees[w1]++; total_in_degrees += 2; }
                                }
                            }
                        }
                        // Check w2 -> w1
                        {
                            std::lock_guard<std::mutex> lk2(node_locks[w2 % NUM_NODE_LOCKS]);
                            if ((int)graphs[lc][w2].size() < w2_limit) {
                                if (std::find(graphs[lc][w2].begin(), graphs[lc][w2].end(), w1) == graphs[lc][w2].end()) {
                                    graphs[lc][w2].push_back(w1);
                                    if (lc == 0) { in_degrees[w1]++; in_degrees[w2]++; total_in_degrees += 2; }
                                }
                            }
                        }
                    }
                }
            }

            // Maintain outgoing links from v to only surviving neighbors (soft transit bridge)
            {
                std::lock_guard<std::mutex> lk_v(node_locks[v % NUM_NODE_LOCKS]);
                graphs[lc][v] = v_surviving_neighbors;
            }
        }
    }

    bool mark_deleted(tableint id, bool heal = true) {
        if (id >= num_elements || is_deleted(id)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lk(node_locks[id % NUM_NODE_LOCKS]);
            deleted[id] = 1;
            num_deleted.fetch_add(1, std::memory_order_relaxed);
        }

        if (heal) {
            heal_neighborhood(id);
        }

        if (id == enter_point) {
            std::lock_guard<std::mutex> lk(entry_point_mutex);
            if (id == enter_point) {
                migrate_entry_point();
            }
        }

        return true;
    }

    size_t batch_delete(const std::vector<tableint>& ids, bool heal = true) {
        size_t count = 0;
        for (tableint id : ids) {
            if (mark_deleted(id, heal)) {
                count++;
            }
        }
        return count;
    }

    void compact() {
        if (num_deleted.load() == 0) return;

        std::vector<tableint> id_map(num_elements, (tableint)-1);
        tableint new_count = 0;
        for (tableint i = 0; i < num_elements; ++i) {
            if (!is_deleted(i)) {
                id_map[i] = new_count++;
            }
        }

        if (new_count == 0) {
            data.clear();
            quantized_data.clear();
            quantized_data_sq4.clear();
            graphs.clear();
            node_levels.clear();
            node_params.clear();
            in_degrees.clear();
            deleted.clear();
            num_elements = 0;
            num_deleted = 0;
            max_level = -1;
            enter_point = 0;
            return;
        }

        // Compact vector data
        std::vector<float> new_data(new_count * dim);
        for (tableint i = 0; i < num_elements; ++i) {
            if (id_map[i] != (tableint)-1) {
                std::memcpy(&new_data[id_map[i] * dim], &data[i * dim], dim * sizeof(float));
            }
        }
        data = std::move(new_data);

        // Compact quantized data
        if (policy.enable_sq8 && !quantized_data.empty()) {
            std::vector<uint8_t> new_qdata(new_count * dim);
            for (tableint i = 0; i < num_elements; ++i) {
                if (id_map[i] != (tableint)-1) {
                    std::memcpy(&new_qdata[id_map[i] * dim], &quantized_data[i * dim], dim * sizeof(uint8_t));
                }
            }
            quantized_data = std::move(new_qdata);
        }

        if (policy.enable_sq4 && !quantized_data_sq4.empty()) {
            size_t packed_dim = (dim + 1) / 2;
            std::vector<uint8_t> new_qdata4(new_count * packed_dim);
            for (tableint i = 0; i < num_elements; ++i) {
                if (id_map[i] != (tableint)-1) {
                    std::memcpy(&new_qdata4[id_map[i] * packed_dim], &quantized_data_sq4[i * packed_dim], packed_dim * sizeof(uint8_t));
                }
            }
            quantized_data_sq4 = std::move(new_qdata4);
        }

        // Compact node levels and node params
        std::vector<int> new_levels(new_count);
        std::vector<NodeParams> new_params(new_count);
        for (tableint i = 0; i < num_elements; ++i) {
            if (id_map[i] != (tableint)-1) {
                new_levels[id_map[i]] = node_levels[i];
                new_params[id_map[i]] = node_params[i];
            }
        }
        node_levels = std::move(new_levels);
        node_params = std::move(new_params);

        // Compact graphs and re-map node IDs
        std::vector<std::vector<std::vector<tableint>>> new_graphs(graphs.size());
        for (size_t l = 0; l < graphs.size(); ++l) {
            new_graphs[l].resize(new_count);
            for (tableint i = 0; i < graphs[l].size() && i < num_elements; ++i) {
                if (id_map[i] != (tableint)-1) {
                    tableint new_u = id_map[i];
                    for (tableint v : graphs[l][i]) {
                        if (v < num_elements && id_map[v] != (tableint)-1) {
                            new_graphs[l][new_u].push_back(id_map[v]);
                        }
                    }
                }
            }
        }
        graphs = std::move(new_graphs);

        num_elements = new_count;
        num_deleted = 0;
        deleted.assign(new_count, 0);

        in_degrees.assign(new_count, 0);
        uint64_t total = 0;
        if (!graphs.empty()) {
            for (tableint u = 0; u < new_count; ++u) {
                for (tableint v : graphs[0][u]) {
                    if (v < new_count) in_degrees[v]++;
                }
            }
            for (tableint u = 0; u < new_count; ++u) total += in_degrees[u];
        }
        total_in_degrees = total;

        if (enter_point < id_map.size() && id_map[enter_point] != (tableint)-1) {
            enter_point = id_map[enter_point];
        } else {
            migrate_entry_point();
        }
    }
    
    std::vector<Candidate> search(const float* query, int k, int ef) const {
        if (num_elements == 0 || get_active_count() == 0) return {};
        
        tableint curr_ep = enter_point;
        if (is_deleted(curr_ep)) {
            const_cast<AdaptiveHNSWIndex*>(this)->migrate_entry_point();
            curr_ep = enter_point;
            if (is_deleted(curr_ep)) return {};
        }

        bool use_asym_sq8 = policy.enable_sq8 && quantizer.is_trained;
        bool use_asym_sq4 = policy.enable_sq4 && quantizer_sq4.is_trained;
        bool use_asym = use_asym_sq8 || use_asym_sq4;
        bool use_ada = policy.enable_ada_ef;
        
        for (int lc = max_level; lc > 0; --lc) {
            auto w = search_layer(query, {curr_ep}, 1, lc, false, use_asym_sq8, false, use_asym_sq4);
            if (!w.empty()) curr_ep = w[0].id;
        }
        
        int target_ef = ef;
        if (policy.enable_dynamic_ef_search && curr_ep < node_params.size()) {
            float node_score = node_params[curr_ep].score;
            dist_t d_entry = (use_asym_sq4 && quantizer_sq4.is_trained)
                             ? get_sq4_distance(query, curr_ep)
                             : ((use_asym_sq8 && quantizer.is_trained) 
                                ? get_sq8_distance(query, curr_ep) 
                                : get_distance(query, get_vector(curr_ep)));
            float norm_d = (d_entry - policy.density_mean) / (policy.density_std + 1e-6f);
            float query_hardness = 0.7f * node_score + 0.3f * std::max(-2.0f, std::min(norm_d, 2.0f));
            float factor = 1.0f + policy.dynamic_ef_factor * query_hardness;
            target_ef = std::max(policy.dynamic_ef_min, 
                        std::min((int)std::round((float)ef * factor), policy.dynamic_ef_max));
        }

        // Expand candidate beam if tombstones are present
        int effective_ef = target_ef;
        if (num_deleted.load() > 0 && num_elements > 0) {
            float tombstone_ratio = (float)num_deleted.load() / (float)num_elements;
            effective_ef = target_ef + (int)((float)target_ef * tombstone_ratio);
            effective_ef = std::min(effective_ef, target_ef * 2);
        }
        
        int rerank_factor = use_asym_sq4 ? policy.sq4_rerank_factor : (use_asym_sq8 ? policy.sq8_rerank_factor : 1);
        int search_beam = use_asym ? std::max(rerank_factor * k, effective_ef) : effective_ef;
        auto w = search_layer(query, {curr_ep}, search_beam, 0, use_ada, use_asym_sq8, false, use_asym_sq4);
        
        if (use_asym) {
            // Two-stage re-ranking with exact float32 distance, excluding deleted nodes
            std::vector<Candidate> active_w;
            active_w.reserve(w.size());
            for (auto& cand : w) {
                if (!is_deleted(cand.id)) {
                    cand.dist = get_distance(query, get_vector(cand.id));
                    active_w.push_back(cand);
                }
            }
            std::sort(active_w.begin(), active_w.end());
            if ((int)active_w.size() > k) active_w.resize(k);
            return active_w;
        }
        
        // Filter out deleted nodes
        std::vector<Candidate> final_results;
        final_results.reserve(k);
        for (const auto& cand : w) {
            if (!is_deleted(cand.id)) {
                final_results.push_back(cand);
                if ((int)final_results.size() >= k) break;
            }
        }
        return final_results;
    }
    
    /**
     * Multi-Threaded Batch Search
     * Executes queries in parallel across worker threads with dynamic load balancing.
     */
    std::vector<std::vector<Candidate>> batch_search(
        const float* queries,
        size_t n_queries,
        int k,
        int ef,
        int num_threads = 0
    ) const {
        if (num_threads <= 0) {
            #if defined(_OPENMP)
            num_threads = omp_get_max_threads();
            #else
            num_threads = (int)std::max(1u, std::thread::hardware_concurrency());
            #endif
        }
        
        std::vector<std::vector<Candidate>> results(n_queries);
        
        #if defined(_OPENMP)
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 32)
        for (size_t q = 0; q < n_queries; ++q) {
            results[q] = search(&queries[q * dim], k, ef);
        }
        #else
        std::vector<std::thread> workers;
        size_t chunk_size = (n_queries + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, n_queries);
            if (start >= end) continue;
            workers.emplace_back([this, queries, &results, k, ef, start, end]() {
                for (size_t q = start; q < end; ++q) {
                    results[q] = this->search(&queries[q * dim], k, ef);
                }
            });
        }
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
        #endif
        
        return results;
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
        if (policy.enable_sq4 && quantizer_sq4.is_trained) {
            size_t packed_dim = (dim + 1) / 2;
            return num_elements * packed_dim * sizeof(uint8_t) + 2 * dim * sizeof(float);
        }
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
