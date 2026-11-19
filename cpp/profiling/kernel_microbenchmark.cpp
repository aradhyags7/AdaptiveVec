/**
 * AdaptiveVec Phase 0.5 Distance Kernel Microbenchmark
 * Compares:
 * 1. Scalar L2 (with sqrt)
 * 2. Current AVX2 (single accumulator + sqrt)
 * 3. AVX2 Squared L2 (single accumulator, no sqrt)
 * 4. AVX2 4-Way Unrolled Squared L2 (4 independent accumulators, no sqrt)
 * 5. Specialized 128-D Kernel (fully unrolled 4x32 floats, no sqrt)
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <immintrin.h>

// 1. Scalar L2
float scalar_l2(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

// 2. Current AVX2 (Single accumulator + sqrt)
inline float _mm256_reduce_add_ps(__m256 x) {
    __m128 vlow = _mm256_castps256_ps128(x);
    __m128 vhigh = _mm256_extractf128_ps(x, 1);
    __m128 v128 = _mm_add_ps(vlow, vhigh);
    __m128 v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
    __m128 v32 = _mm_add_ss(v64, _mm_shuffle_ps(v64, v64, 0x55));
    return _mm_cvtss_f32(v32);
}

float current_avx2_l2(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum256 = _mm256_fmadd_ps(diff, diff, sum256);
    }
    float total = _mm256_reduce_add_ps(sum256);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return std::sqrt(total);
}

// 3. AVX2 Squared L2 (Single accumulator, no sqrt)
float avx2_sq_l2(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum256 = _mm256_fmadd_ps(diff, diff, sum256);
    }
    float total = _mm256_reduce_add_ps(sum256);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

// 4. AVX2 4-Way Unrolled Squared L2 (4 independent accumulators to break latency dependency)
float avx2_unrolled_sq_l2(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (; i + 32 <= dim; i += 32) {
        __m256 diff0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 diff1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        __m256 diff2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 diff3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));

        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }

    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    for (; i + 8 <= dim; i += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        sum0 = _mm256_fmadd_ps(diff, diff, sum0);
    }

    float total = _mm256_reduce_add_ps(sum0);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

// 5. Specialized 128-D Kernel (Exactly 4x32 unrolled steps, no loop branches)
inline float avx2_sq_l2_128d(const float* a, const float* b) {
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
    __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
    __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));

    __m256 s0 = _mm256_mul_ps(d0, d0);
    __m256 s1 = _mm256_mul_ps(d1, d1);
    __m256 s2 = _mm256_mul_ps(d2, d2);
    __m256 s3 = _mm256_mul_ps(d3, d3);

    __m256 d4 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
    __m256 d5 = _mm256_sub_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40));
    __m256 d6 = _mm256_sub_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48));
    __m256 d7 = _mm256_sub_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56));

    s0 = _mm256_fmadd_ps(d4, d4, s0);
    s1 = _mm256_fmadd_ps(d5, d5, s1);
    s2 = _mm256_fmadd_ps(d6, d6, s2);
    s3 = _mm256_fmadd_ps(d7, d7, s3);

    __m256 d8 = _mm256_sub_ps(_mm256_loadu_ps(a + 64), _mm256_loadu_ps(b + 64));
    __m256 d9 = _mm256_sub_ps(_mm256_loadu_ps(a + 72), _mm256_loadu_ps(b + 72));
    __m256 d10 = _mm256_sub_ps(_mm256_loadu_ps(a + 80), _mm256_loadu_ps(b + 80));
    __m256 d11 = _mm256_sub_ps(_mm256_loadu_ps(a + 88), _mm256_loadu_ps(b + 88));

    s0 = _mm256_fmadd_ps(d8, d8, s0);
    s1 = _mm256_fmadd_ps(d9, d9, s1);
    s2 = _mm256_fmadd_ps(d10, d10, s2);
    s3 = _mm256_fmadd_ps(d11, d11, s3);

    __m256 d12 = _mm256_sub_ps(_mm256_loadu_ps(a + 96), _mm256_loadu_ps(b + 96));
    __m256 d13 = _mm256_sub_ps(_mm256_loadu_ps(a + 104), _mm256_loadu_ps(b + 104));
    __m256 d14 = _mm256_sub_ps(_mm256_loadu_ps(a + 112), _mm256_loadu_ps(b + 112));
    __m256 d15 = _mm256_sub_ps(_mm256_loadu_ps(a + 120), _mm256_loadu_ps(b + 120));

    s0 = _mm256_fmadd_ps(d12, d12, s0);
    s1 = _mm256_fmadd_ps(d13, d13, s1);
    s2 = _mm256_fmadd_ps(d14, d14, s2);
    s3 = _mm256_fmadd_ps(d15, d15, s3);

    __m256 sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    return _mm256_reduce_add_ps(sum);
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "        AdaptiveVec Distance Kernel Microbenchmark (128-D)       " << std::endl;
    std::cout << "=================================================================\n" << std::endl;

    const size_t dim = 128;
    const size_t n_iters = 2000000; // 2 million evaluations
    std::vector<float> a(dim, 1.234f);
    std::vector<float> b(dim, 5.678f);

    auto bench = [&](const std::string& name, auto fn) {
        // Warmup
        volatile float sink = 0.0f;
        for (int i = 0; i < 10000; ++i) sink += fn(a.data(), b.data());

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n_iters; ++i) {
            sink += fn(a.data(), b.data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double ns_per_eval = total_ns / (double)n_iters;
        double mega_evals = (double)n_iters / (total_ns / 1e9) / 1e6;

        std::cout << std::left << std::setw(38) << name 
                  << std::setw(15) << std::fixed << std::setprecision(2) << ns_per_eval << " ns/eval"
                  << std::setw(15) << std::fixed << std::setprecision(1) << mega_evals << " M evals/s"
                  << std::endl;
    };

    std::cout << std::left << std::setw(38) << "Kernel Implementation" 
              << std::setw(23) << "Latency" 
              << "Throughput" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;

    bench("1. Scalar L2 (with sqrt)", [](const float* x, const float* y) { return scalar_l2(x, y, 128); });
    bench("2. Current AVX2 (Single Acc + sqrt)", [](const float* x, const float* y) { return current_avx2_l2(x, y, 128); });
    bench("3. AVX2 Squared L2 (Single Acc, no sqrt)", [](const float* x, const float* y) { return avx2_sq_l2(x, y, 128); });
    bench("4. AVX2 4-Way Unrolled (4 Accs, no sqrt)", [](const float* x, const float* y) { return avx2_unrolled_sq_l2(x, y, 128); });
    bench("5. Specialized 128-D Kernel (Full Unroll)", [](const float* x, const float* y) { return avx2_sq_l2_128d(x, y); });

    std::cout << "=================================================================\n" << std::endl;
    return 0;
}
