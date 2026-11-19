#include "lightweight_controller.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <set>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace adaptivevec;

std::vector<float> load_fvecs(const std::string& path, size_t max_vectors, size_t& dim, size_t& count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    int d = 0;
    in.read((char*)&d, sizeof(int));
    dim = d;
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t record_size = sizeof(int) + dim * sizeof(float);
    count = std::min(file_size / record_size, max_vectors);
    std::vector<float> data(count * dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < count; ++i) {
        int cur_d = 0;
        in.read((char*)&cur_d, sizeof(int));
        in.read((char*)&data[i * dim], dim * sizeof(float));
    }
    return data;
}

std::vector<std::vector<tableint>> load_ivecs(const std::string& path, size_t max_queries) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    std::vector<std::vector<tableint>> gt;
    size_t q = 0;
    while (in.peek() != EOF && q < max_queries) {
        int k = 0;
        in.read((char*)&k, sizeof(int));
        if (k <= 0 || k > 10000) break;
        std::vector<int> row(k);
        in.read((char*)row.data(), k * sizeof(int));
        std::vector<tableint> row_t(k);
        for (int i = 0; i < k; ++i) row_t[i] = static_cast<tableint>(row[i]);
        gt.push_back(std::move(row_t));
        q++;
    }
    return gt;
}

struct MatchedRecallRecord {
    std::string dataset;
    std::string method;
    double recall_at_10;
    double throughput_qps;
    double p50_latency_ms;
    double p95_latency_ms;
    double p99_latency_ms;
    double mean_dist_comps;
    double memory_mb;
    int constraint_violations;
    int adaptations;
    int rollbacks;
    double monitoring_overhead_pct;

    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "{"
           << "\"dataset\": \"" << dataset << "\", "
           << "\"method\": \"" << method << "\", "
           << "\"recall_at_10\": " << recall_at_10 << ", "
           << "\"throughput_qps\": " << throughput_qps << ", "
           << "\"p50_latency_ms\": " << p50_latency_ms << ", "
           << "\"p95_latency_ms\": " << p95_latency_ms << ", "
           << "\"p99_latency_ms\": " << p99_latency_ms << ", "
           << "\"mean_dist_comps\": " << mean_dist_comps << ", "
           << "\"memory_mb\": " << memory_mb << ", "
           << "\"constraint_violations\": " << constraint_violations << ", "
           << "\"adaptations\": " << adaptations << ", "
           << "\"rollbacks\": " << rollbacks << ", "
           << "\"monitoring_overhead_pct\": " << monitoring_overhead_pct
           << "}";
        return ss.str();
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  AdaptiveVec Phase 4.2: Matched-Recall Comparative Suite       \n";
    std::cout << "=================================================================\n" << std::endl;

    int threads = omp_get_max_threads();
    std::cout << "[*] Executing with OpenMP concurrency: " << threads << " worker threads" << std::endl;

    std::vector<MatchedRecallRecord> all_records;

    struct DatasetConfig {
        std::string name;
        std::string base_path;
        std::string query_path;
        std::string gt_path;
        size_t max_base;
        size_t max_queries;
        SpaceType space;
        float target_recall;
        float target_p99_ms;
    };

    std::vector<DatasetConfig> dataset_configs = {
        {"SIFT-100K", "data/sift_subset_100k.fvecs", "data/sift_query.fvecs", "data/sift_subset_groundtruth.ivecs", 100000, 1000, SpaceType::L2, 0.95f, 0.50f},
        {"SIFT-500K", "data/sift_base_1m.fvecs", "data/sift_query.fvecs", "data/sift_groundtruth_500k.ivecs", 500000, 1000, SpaceType::L2, 0.95f, 0.80f},
        {"Transformer-384D", "data/transformer_384d/transformer_384d_base.fvecs", "data/transformer_384d/transformer_384d_query.fvecs", "data/transformer_384d/transformer_384d_groundtruth.ivecs", 60000, 1000, SpaceType::Cosine, 0.90f, 1.50f}
    };

    for (const auto& dcfg : dataset_configs) {
        std::cout << "\n-------------------------------------------------------------\n";
        std::cout << "[+] Benchmarking Dataset: " << dcfg.name << "\n";
        std::cout << "-------------------------------------------------------------\n";

        size_t dim = 0, n_base = 0, q_dim = 0, n_queries = 0;
        auto base_data = load_fvecs(dcfg.base_path, dcfg.max_base, dim, n_base);
        auto query_data = load_fvecs(dcfg.query_path, dcfg.max_queries, q_dim, n_queries);
        auto gt = load_ivecs(dcfg.gt_path, dcfg.max_queries);

        if (base_data.empty() || query_data.empty() || gt.empty()) {
            std::cerr << "[-] Error loading " << dcfg.name << " (Skipping)" << std::endl;
            continue;
        }

        std::cout << "  [*] Loaded " << n_base << " base vectors, " << n_queries << " test queries (dim=" << dim << ")\n";

        // Build Base SQ8 Index
        std::cout << "  [*] Constructing Base SQ8 Index..." << std::endl;
        PolicyConfig p_cfg;
        p_cfg.m_base = 16;
        p_cfg.ef_c_base = 64;
        p_cfg.enable_hubness_regulation = false;
        AdaptiveHNSWIndex base_idx(dim, dcfg.space, false, p_cfg);
        base_idx.batch_insert(base_data.data(), n_base, threads);
        base_idx.policy.keep_raw_vectors = false;
        base_idx.allocate_mixed_precision(AllocationStrategy::UniformSQ8, 0.25);

        double index_ram_mb = base_idx.get_total_memory_bytes() / (1024.0 * 1024.0);
        std::cout << "  [*] Index Constructed. Resident RAM: " << std::fixed << std::setprecision(2) << index_ram_mb << " MB\n";

        // Setup offline reference samples (100 samples)
        std::vector<std::pair<std::vector<float>, std::vector<tableint>>> ref_samples;
        size_t ref_pool_start = std::min((size_t)800, n_queries - 100);
        for (size_t r = ref_pool_start; r < ref_pool_start + 100; ++r) {
            std::vector<float> r_q(&query_data[r * dim], &query_data[(r + 1) * dim]);
            std::vector<tableint> r_gt(gt[r].begin(), gt[r].begin() + 10);
            ref_samples.push_back({r_q, r_gt});
        }

        size_t test_n = std::min((size_t)800, n_queries);

        // Pre-compute Oracle search budgets for diagnostic comparison (Method 8)
        std::cout << "  [*] Calibrating Oracle baseline for diagnostic comparison..." << std::endl;
        std::vector<int> oracle_budgets(test_n, 128);
        for (size_t q = 0; q < test_n; ++q) {
            const float* q_ptr = &query_data[q * dim];
            std::set<tableint> gt_set(gt[q].begin(), gt[q].begin() + 10);
            for (int cand_ef : {16, 24, 32, 48, 64, 96, 128}) {
                auto res = base_idx.search(q_ptr, 10, cand_ef);
                int hits = 0;
                for (const auto& c : res) if (gt_set.count(c.id)) hits++;
                if (hits >= 9) { // 90% or 100% matched
                    oracle_budgets[q] = cand_ef;
                    break;
                }
            }
        }

        // Methods to evaluate
        struct MethodEvalSpec {
            std::string name;
            int ef_fixed;
            bool enable_ada_ef;
            bool is_p41_controller;
            bool is_proposed_lightweight;
            bool is_query_level_resumable;
            bool is_oracle;
        };

        std::vector<MethodEvalSpec> methods = {
            {"Fixed efSearch=32", 32, false, false, false, false, false},
            {"Fixed efSearch=48", 48, false, false, false, false, false},
            {"Manually Tuned Static (ef=64)", 64, false, false, false, false, false},
            {"Phase 3 Calibrated (Ada-ef ON)", 64, true, false, false, false, false},
            {"Existing Ada-ef", 48, true, false, false, false, false},
            {"Phase 4.1 Runtime Controller", 64, true, true, false, false, false},
            {"Query-Level Resumable (No Workload)", 32, false, false, false, true, false},
            {"Proposed Lightweight Controller", 32, true, false, true, false, false},
            {"Oracle Search-Budget Policy", 0, false, false, false, false, true}
        };

        for (const auto& m : methods) {
            std::unique_ptr<AdaptiveHNSWIndex> test_idx = base_idx.clone();
            test_idx->policy.enable_ada_ef = m.enable_ada_ef;

            MultiSignalDifficultyEstimator estimator = (dcfg.space == SpaceType::Cosine) 
                ? MultiSignalDifficultyEstimator(MultiSignalDifficultyEstimator::default_for_transformer())
                : MultiSignalDifficultyEstimator(MultiSignalDifficultyEstimator::default_for_sift());

            LightweightControllerConfig l_cfg;
            l_cfg.base_ef_init = 32;
            l_cfg.base_ef_expanded = 64;
            l_cfg.target_recall = dcfg.target_recall;
            l_cfg.monitor_interval = 25; // 4% bounded monitoring
            LightweightWorkloadController lw_controller(l_cfg);
            lw_controller.get_estimator() = estimator;
            for (const auto& s : ref_samples) {
                lw_controller.get_monitor().add_reference_sample(s.first, s.second);
            }

            // Phase 4.1 heavy monitor setup
            BoundedReferenceMonitor p41_mon;
            p41_mon.set_policy(MonitoringPolicyType::PeriodicBatch, 50);
            for (const auto& s : ref_samples) p41_mon.add_reference_sample(s.first, s.second);

            std::vector<double> latencies_ms(test_n);
            std::vector<double> recalls(test_n);
            std::vector<int> dist_comps(test_n);
            int violations = 0;
            int adaptations = 0;
            int rollbacks = 0;
            double monitoring_overhead_pct = 0.0;

            auto t_bench_0 = std::chrono::high_resolution_clock::now();

            for (size_t q = 0; q < test_n; ++q) {
                const float* q_ptr = &query_data[q * dim];
                std::set<tableint> gt_set(gt[q].begin(), gt[q].begin() + 10);

                auto t_q0 = std::chrono::high_resolution_clock::now();

                std::vector<Candidate> res;
                int d_comps = 0;

                if (m.is_oracle) {
                    int o_ef = oracle_budgets[q];
                    res = test_idx->search(q_ptr, 10, o_ef);
                    d_comps = o_ef * 6;
                } else if (m.is_query_level_resumable) {
                    auto r = ResumableSearchEngine::search_resumable(*test_idx, q_ptr, 10, 32, 64, estimator, true);
                    res = std::move(r.candidates);
                    d_comps = r.distance_computations;
                } else if (m.is_proposed_lightweight) {
                    auto r = lw_controller.execute_query(*test_idx, q_ptr, 10, true);
                    res = std::move(r.candidates);
                    d_comps = r.distance_computations;
                } else {
                    res = test_idx->search(q_ptr, 10, m.ef_fixed);
                    d_comps = m.ef_fixed * 6;
                }

                auto t_q1 = std::chrono::high_resolution_clock::now();
                double q_lat_ms = std::chrono::duration<double, std::milli>(t_q1 - t_q0).count();

                // Slow path controller step (includes reference monitoring work)
                if (m.is_proposed_lightweight) {
                    lw_controller.step_workload(*test_idx, q, q_lat_ms);
                } else if (m.is_p41_controller) {
                    p41_mon.record_prod_query();
                    if (q > 0 && q % 50 == 0) {
                        for (size_t r = 0; r < 100; ++r) {
                            p41_mon.evaluate_next_sample(*test_idx, m.ef_fixed, 10);
                        }
                    }
                }

                latencies_ms[q] = q_lat_ms;
                dist_comps[q] = d_comps;

                // Recall calculation
                int hits = 0;
                for (const auto& c : res) if (gt_set.count(c.id)) hits++;
                double rec = (double)hits / 10.0;
                recalls[q] = rec;

                if (rec < dcfg.target_recall || q_lat_ms > dcfg.target_p99_ms) {
                    violations++;
                }
            }

            auto t_bench_1 = std::chrono::high_resolution_clock::now();
            double total_wall_ms = std::chrono::duration<double, std::milli>(t_bench_1 - t_bench_0).count();
            double qps = (double)test_n / (total_wall_ms / 1000.0);

            // Compute recall and latency percentiles
            double mean_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / test_n;
            double mean_dc = std::accumulate(dist_comps.begin(), dist_comps.end(), 0.0) / test_n;

            std::sort(latencies_ms.begin(), latencies_ms.end());
            double p50 = latencies_ms[(size_t)(test_n * 0.50)];
            double p95 = latencies_ms[(size_t)(test_n * 0.95)];
            double p99 = latencies_ms[(size_t)(test_n * 0.99)];

            if (m.is_proposed_lightweight) {
                adaptations = lw_controller.get_total_adaptations();
                rollbacks = lw_controller.get_total_rollbacks();
                monitoring_overhead_pct = lw_controller.get_monitor().get_stats().cpu_overhead_pct;
            } else if (m.is_p41_controller) {
                monitoring_overhead_pct = p41_mon.get_stats().cpu_overhead_pct;
            }

            MatchedRecallRecord rec;
            rec.dataset = dcfg.name;
            rec.method = m.name;
            rec.recall_at_10 = mean_recall;
            rec.throughput_qps = qps;
            rec.p50_latency_ms = p50;
            rec.p95_latency_ms = p95;
            rec.p99_latency_ms = p99;
            rec.mean_dist_comps = mean_dc;
            rec.memory_mb = index_ram_mb;
            rec.constraint_violations = violations;
            rec.adaptations = adaptations;
            rec.rollbacks = rollbacks;
            rec.monitoring_overhead_pct = monitoring_overhead_pct;

            all_records.push_back(rec);

            std::cout << "  " << std::left << std::setw(36) << m.name
                      << " | R@10: " << std::fixed << std::setprecision(2) << (mean_recall * 100.0) << "%"
                      << " | QPS: " << std::setw(7) << (int)qps
                      << " | p99: " << std::setprecision(3) << p99 << " ms"
                      << " | Violations: " << std::setw(4) << violations << "\n";
        }
    }

    // Export JSON
    std::cout << "\n[*] Exporting experiments/phase_4_2_matched_recall_results.json..." << std::endl;
    {
        std::ofstream out("experiments/phase_4_2_matched_recall_results.json");
        out << "[\n";
        for (size_t i = 0; i < all_records.size(); ++i) {
            out << "  " << all_records[i].to_json() << (i + 1 < all_records.size() ? ",\n" : "\n");
        }
        out << "]\n";
    }
    std::cout << "[*] Phase 4.2 Matched-Recall Benchmark Suite Completed Successfully!\n";

    return 0;
}
