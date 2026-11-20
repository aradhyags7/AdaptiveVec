#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <cassert>
#include <iomanip>

namespace adaptivevec {

struct GeometrySignal {
    float lid = 0.0f;
    float density = 0.0f;
    float sparsity = 0.0f;
    float standardized_lid = 0.0f;
    float standardized_sparsity = 0.0f;
    float combined_score = 0.0f; // S_i
};

class BudgetAllocator {
public:
    // Helper to standardize signals
    static void standardize_signals(std::vector<GeometrySignal>& signals, float alpha_lid = 0.5f, float beta_spars = 0.5f) {
        size_t n = signals.size();
        if (n == 0) return;

        double sum_lid = 0.0, sum_sq_lid = 0.0;
        double sum_spars = 0.0, sum_sq_spars = 0.0;

        for (size_t i = 0; i < n; ++i) {
            sum_lid += signals[i].lid;
            sum_sq_lid += (double)signals[i].lid * signals[i].lid;
            sum_spars += signals[i].sparsity;
            sum_sq_spars += (double)signals[i].sparsity * signals[i].sparsity;
        }

        double mean_lid = sum_lid / (double)n;
        double var_lid = std::max(1e-8, (sum_sq_lid / (double)n) - (mean_lid * mean_lid));
        double std_lid = std::sqrt(var_lid);

        double mean_spars = sum_spars / (double)n;
        double var_spars = std::max(1e-8, (sum_sq_spars / (double)n) - (mean_spars * mean_spars));
        double std_spars = std::sqrt(var_spars);

        for (size_t i = 0; i < n; ++i) {
            float s_lid = (signals[i].lid - (float)mean_lid) / (float)std_lid;
            float s_spars = (signals[i].sparsity - (float)mean_spars) / (float)std_spars;

            // Clip to [-3, +3] to prevent extreme outlier distortion
            s_lid = std::max(-3.0f, std::min(3.0f, s_lid));
            s_spars = std::max(-3.0f, std::min(3.0f, s_spars));

            signals[i].standardized_lid = s_lid;
            signals[i].standardized_sparsity = s_spars;
            signals[i].combined_score = alpha_lid * s_lid + beta_spars * s_spars;
        }
    }

    // Approach A: Uniform Allocation
    static std::vector<int> allocate_uniform(size_t n, int m_base = 16) {
        return std::vector<int>(n, m_base);
    }

    // Approach B: Original AdaptiveVec (Unconstrained Clipped)
    static std::vector<int> allocate_original_adaptive(
        const std::vector<GeometrySignal>& signals,
        int m_base = 16, int m_min = 8, int m_max = 24, float beta = 1.0f
    ) {
        size_t n = signals.size();
        std::vector<int> alloc(n);
        for (size_t i = 0; i < n; ++i) {
            float delta = (float)m_base * (signals[i].combined_score / 2.5f) * beta;
            int val = (int)std::round((float)m_base + delta);
            alloc[i] = std::max(m_min, std::min(m_max, val));
        }
        return alloc;
    }

    // Hare-Niemeyer / Largest-Remainder Integer Discretization
    static std::vector<int> largest_remainder_discretization(
        const std::vector<double>& continuous_m,
        int total_budget,
        int m_min,
        int m_max
    ) {
        size_t n = continuous_m.size();
        std::vector<int> m(n);
        int current_sum = 0;

        for (size_t i = 0; i < n; ++i) {
            int floor_val = (int)std::floor(continuous_m[i]);
            floor_val = std::max(m_min, std::min(m_max, floor_val));
            m[i] = floor_val;
            current_sum += floor_val;
        }

        int deficit = total_budget - current_sum;

        if (deficit > 0) {
            // Sort by descending fractional remainder
            std::vector<std::pair<double, size_t>> remainders;
            remainders.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (m[i] < m_max) {
                    double rem = continuous_m[i] - std::floor(continuous_m[i]);
                    remainders.emplace_back(rem, i);
                }
            }
            std::sort(remainders.begin(), remainders.end(), [](const auto& a, const auto& b) {
                if (std::abs(a.first - b.first) > 1e-9) return a.first > b.first;
                return a.second < b.second; // Deterministic tie-breaking on index
            });

            for (size_t k = 0; k < (size_t)deficit && k < remainders.size(); ++k) {
                m[remainders[k].second]++;
            }
        } else if (deficit < 0) {
            int surplus = -deficit;
            // Sort by ascending fractional remainder
            std::vector<std::pair<double, size_t>> remainders;
            remainders.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (m[i] > m_min) {
                    double rem = continuous_m[i] - std::floor(continuous_m[i]);
                    remainders.emplace_back(rem, i);
                }
            }
            std::sort(remainders.begin(), remainders.end(), [](const auto& a, const auto& b) {
                if (std::abs(a.first - b.first) > 1e-9) return a.first < b.first;
                return a.second < b.second;
            });

            for (size_t k = 0; k < (size_t)surplus && k < remainders.size(); ++k) {
                m[remainders[k].second]--;
            }
        }

        // Verify budget invariant
        long long check_sum = 0;
        for (size_t i = 0; i < n; ++i) {
            assert(m[i] >= m_min && m[i] <= m_max);
            check_sum += m[i];
        }
        assert(check_sum == total_budget);

        return m;
    }

    // Approach C: Normalized Linear Allocation
    static std::vector<int> allocate_normalized_linear(
        const std::vector<GeometrySignal>& signals,
        int m_base = 16, int m_min = 8, int m_max = 24, float beta = 1.0f
    ) {
        size_t n = signals.size();
        int total_budget = (int)n * m_base;
        int min_budget = (int)n * m_min;
        int residual_budget = total_budget - min_budget;

        std::vector<double> weights(n);
        double sum_w = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double w = std::max(1e-4, 1.0 + (double)beta * (signals[i].combined_score / 2.5));
            weights[i] = w;
            sum_w += w;
        }

        std::vector<double> continuous_m(n);
        for (size_t i = 0; i < n; ++i) {
            continuous_m[i] = (double)m_min + (double)residual_budget * (weights[i] / sum_w);
        }

        return largest_remainder_discretization(continuous_m, total_budget, m_min, m_max);
    }

    // Approach D: Normalized Exponential Allocation
    static std::vector<int> allocate_normalized_exponential(
        const std::vector<GeometrySignal>& signals,
        int m_base = 16, int m_min = 8, int m_max = 24, float beta = 0.5f
    ) {
        size_t n = signals.size();
        int total_budget = (int)n * m_base;
        int min_budget = (int)n * m_min;
        int residual_budget = total_budget - min_budget;

        std::vector<double> weights(n);
        double sum_w = 0.0;
        for (size_t i = 0; i < n; ++i) {
            // Clip exponent argument to [-2.0, 2.0] to prevent overflow
            double arg = std::max(-2.0, std::min(2.0, (double)beta * signals[i].combined_score));
            double w = std::exp(arg);
            weights[i] = w;
            sum_w += w;
        }

        std::vector<double> continuous_m(n);
        for (size_t i = 0; i < n; ++i) {
            continuous_m[i] = (double)m_min + (double)residual_budget * (weights[i] / sum_w);
        }

        return largest_remainder_discretization(continuous_m, total_budget, m_min, m_max);
    }

    // Direct interface to apply normalized exponential budget allocation to an AdaptiveHNSWIndex
    static void apply_normalized_exponential(
        AdaptiveHNSWIndex& index,
        const std::vector<GeometrySignal>& signals,
        int m_base = 16, int m_min = 8, int m_max = 24, float beta = 0.5f
    ) {
        std::vector<int> alloc = allocate_normalized_exponential(signals, m_base, m_min, m_max, beta);
        std::vector<float> lids(signals.size());
        std::vector<float> densities(signals.size());
        for (size_t i = 0; i < signals.size(); ++i) {
            lids[i] = signals[i].lid;
            densities[i] = signals[i].density;
        }
        index.assign_preassigned_degrees(alloc, &lids, &densities);
    }

    // Approach E: Geometry Quantile Allocation (5 Quintiles)
    static std::vector<int> allocate_quantile(
        const std::vector<GeometrySignal>& signals,
        int m_base = 16
    ) {
        size_t n = signals.size();
        std::vector<std::pair<float, size_t>> score_indices(n);
        for (size_t i = 0; i < n; ++i) {
            score_indices[i] = {signals[i].combined_score, i};
        }
        std::sort(score_indices.begin(), score_indices.end(), [](const auto& a, const auto& b) {
            if (std::abs(a.first - b.first) > 1e-9) return a.first < b.first;
            return a.second < b.second;
        });

        // 5 Quintiles: 10, 13, 16, 19, 22. Sum = 80 / 5 = 16 = m_base!
        const int quintile_degrees[5] = {10, 13, 16, 19, 22};
        std::vector<int> alloc(n);

        size_t chunk_size = n / 5;
        for (size_t q = 0; q < 5; ++q) {
            size_t start = q * chunk_size;
            size_t end = (q == 4) ? n : (start + chunk_size);
            for (size_t idx = start; idx < end; ++idx) {
                alloc[score_indices[idx].second] = quintile_degrees[q];
            }
        }

        // Exact budget check
        long long check_sum = 0;
        for (int m : alloc) check_sum += m;
        int total_budget = (int)n * m_base;
        if (check_sum != total_budget) {
            int diff = total_budget - (int)check_sum;
            alloc[score_indices[n / 2].second] += diff;
        }

        return alloc;
    }

    // Control 1: Randomized Budget Allocation
    static std::vector<int> allocate_randomized(
        const std::vector<int>& base_allocation,
        uint32_t seed = 42
    ) {
        std::vector<int> shuffled = base_allocation;
        std::mt19937 rng(seed);
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        return shuffled;
    }

    // Control 2: Reversed Geometry Allocation (Inverted)
    static std::vector<int> allocate_reversed(
        const std::vector<int>& base_allocation,
        const std::vector<GeometrySignal>& signals
    ) {
        size_t n = signals.size();
        std::vector<std::pair<float, size_t>> score_indices(n);
        for (size_t i = 0; i < n; ++i) {
            score_indices[i] = {signals[i].combined_score, i};
        }
        // Sort ascending by score
        std::sort(score_indices.begin(), score_indices.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Sort degree allocations descending
        std::vector<int> sorted_degrees = base_allocation;
        std::sort(sorted_degrees.begin(), sorted_degrees.end(), std::greater<int>());

        // Assign highest degrees to lowest scores
        std::vector<int> reversed_alloc(n);
        for (size_t i = 0; i < n; ++i) {
            reversed_alloc[score_indices[i].second] = sorted_degrees[i];
        }
        return reversed_alloc;
    }
};

} // namespace adaptivevec
