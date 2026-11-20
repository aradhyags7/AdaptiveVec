#include "uncertainty_aware_search.hpp"
#include "groundtruth_generator.hpp"
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

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  AdaptiveVec Phase 4.3: Uncertainty-Aware Search Engine Suite   \n";
    std::cout << "=================================================================\n" << std::endl;

    int threads = omp_get_max_threads();
    std::cout << "[*] OpenMP Execution Concurrency: " << threads << " worker threads" << std::endl;

    // Load SIFT-100K Dataset for baseline ablation and search-state collection
    size_t dim = 0, n_base = 0, q_dim = 0, n_queries = 0;
    auto base_data = load_fvecs("data/sift_subset_100k.fvecs", 100000, dim, n_base);
    auto query_data = load_fvecs("data/sift_query.fvecs", 1000, q_dim, n_queries);
    auto gt = load_ivecs("data/sift_subset_groundtruth.ivecs", 1000);

    std::cout << "[*] Loaded SIFT-100K: " << n_base << " base, " << n_queries << " test queries\n";

    PolicyConfig p_cfg;
    p_cfg.m_base = 16;
    p_cfg.ef_c_base = 64;
    p_cfg.enable_hubness_regulation = false;
    AdaptiveHNSWIndex sift_idx(dim, SpaceType::L2, false, p_cfg);
    sift_idx.batch_insert(base_data.data(), n_base, threads);
    sift_idx.policy.keep_raw_vectors = false;
    sift_idx.allocate_mixed_precision(AllocationStrategy::UniformSQ8, 0.25);

    // ========================================================================
    // Stage 3 — Patience & efSearch Grid Ablation
    // ========================================================================
    std::cout << "\n-------------------------------------------------------------\n";
    std::cout << "[+] Stage 3: Evaluating efSearch and Patience Combinations\n";
    std::cout << "-------------------------------------------------------------\n";

    struct PatienceAblationRecord {
        int ef_search;
        int patience;
        bool patience_disabled;
        double recall_at_10;
        double actual_mean_dist_evals;
        double throughput_qps;
        double p99_latency_ms;
        double early_exit_pct;
    };
    std::vector<PatienceAblationRecord> ablation_records;

    std::vector<int> ef_sweep = {16, 24, 32, 48, 64, 96, 128, 192, 256};
    std::vector<int> patience_sweep = {2, 4, 6, 8, 12, 16, 24, 0}; // 0 = disabled

    size_t test_n = std::min((size_t)500, n_queries);

    for (int ef : ef_sweep) {
        for (int p : patience_sweep) {
            bool disabled = (p == 0);
            sift_idx.policy.enable_ada_ef = !disabled;
            sift_idx.policy.ada_ef_patience = disabled ? 999999 : p;

            std::vector<double> latencies_ms(test_n);
            double total_rec = 0.0;
            double total_dist = 0.0;
            int early_exits = 0;

            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t q = 0; q < test_n; ++q) {
                const float* q_ptr = &query_data[q * dim];
                std::set<tableint> gt_set(gt[q].begin(), gt[q].begin() + 10);

                auto t_q0 = std::chrono::high_resolution_clock::now();
                QuerySearchStats stats;
                auto res = sift_idx.search(q_ptr, 10, ef, stats);
                auto t_q1 = std::chrono::high_resolution_clock::now();

                latencies_ms[q] = std::chrono::duration<double, std::milli>(t_q1 - t_q0).count();
                total_dist += stats.total_dist_evals();
                if (stats.early_exit_stagnation_count > 0) early_exits++;

                int hits = 0;
                for (const auto& c : res) if (gt_set.count(c.id)) hits++;
                total_rec += (double)hits / 10.0;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::sort(latencies_ms.begin(), latencies_ms.end());
            double p99 = latencies_ms[static_cast<size_t>(0.99 * test_n)];
            double mean_rec = total_rec / test_n;
            double mean_dist = total_dist / test_n;
            double qps = (double)test_n / (wall_ms / 1000.0);
            double ee_pct = (double)early_exits / test_n * 100.0;

            ablation_records.push_back({ef, p, disabled, mean_rec, mean_dist, qps, p99, ee_pct});

            if (p == 6 || disabled) {
                std::cout << "  ef=" << std::setw(3) << ef 
                          << " | p=" << (disabled ? "none" : std::to_string(p))
                          << " | R@10: " << std::fixed << std::setprecision(2) << (mean_rec * 100.0) << "%"
                          << " | Mean Dist: " << std::setw(6) << std::setprecision(1) << mean_dist
                          << " | QPS: " << std::setw(7) << (int)qps
                          << " | p99: " << std::setprecision(3) << p99 << " ms"
                          << " | EE%: " << std::setprecision(1) << ee_pct << "%\n";
            }
        }
    }

    // Export Stage 3 JSON
    std::cout << "[*] Exporting experiments/phase_4_3_patience_ablation.json...\n";
    {
        std::ofstream out("experiments/phase_4_3_patience_ablation.json");
        out << "[\n";
        for (size_t i = 0; i < ablation_records.size(); ++i) {
            const auto& r = ablation_records[i];
            out << "  {"
                << "\"ef_search\": " << r.ef_search << ", "
                << "\"patience\": " << r.patience << ", "
                << "\"patience_disabled\": " << (r.patience_disabled ? "true" : "false") << ", "
                << "\"recall_at_10\": " << r.recall_at_10 << ", "
                << "\"actual_mean_dist_evals\": " << r.actual_mean_dist_evals << ", "
                << "\"throughput_qps\": " << r.throughput_qps << ", "
                << "\"p99_latency_ms\": " << r.p99_latency_ms << ", "
                << "\"early_exit_pct\": " << r.early_exit_pct
                << "}" << (i + 1 < ablation_records.size() ? ",\n" : "\n");
        }
        out << "]\n";
    }

    // ========================================================================
    // Stage 4 — Search-State Checkpoint Data Collection
    // ========================================================================
    std::cout << "\n-------------------------------------------------------------\n";
    std::cout << "[+] Stage 4: Collecting Live Search-State Checkpoints\n";
    std::cout << "-------------------------------------------------------------\n";

    DynamicPatienceConfig dyn_cfg;
    std::vector<SearchStateCheckpoint> dataset_checkpoints;

    for (size_t q = 0; q < 200; ++q) {
        const float* q_ptr = &query_data[q * dim];
        std::vector<tableint> q_gt(gt[q].begin(), gt[q].begin() + 10);

        // Precompute max possible recall at ef=192
        QuerySearchStats max_stats;
        auto max_res = sift_idx.search(q_ptr, 10, 192, max_stats);
        int max_hits = 0;
        std::set<tableint> gt_set(q_gt.begin(), q_gt.end());
        for (const auto& c : max_res) if (gt_set.count(c.id)) max_hits++;
        float max_r = (float)max_hits / 10.0f;

        auto u_res = UncertaintyAwareSearchEngine::search_layer0_uncertainty_aware(
            sift_idx, q_ptr, 10, 64, dyn_cfg, true, &q_gt
        );

        for (auto& cp : u_res.checkpoints) {
            cp.max_possible_recall = max_r;
            cp.marginal_recall_gain = std::max(0.0f, max_r - cp.current_recall_at_10);
            cp.will_gain_recall = (cp.marginal_recall_gain > 0.01f);
            cp.is_censored = (max_r < 0.90f);
            dataset_checkpoints.push_back(cp);
        }
    }

    std::cout << "  [*] Collected " << dataset_checkpoints.size() << " structured search-state checkpoints\n";
    std::cout << "[*] Exporting experiments/phase_4_3_search_state_dataset.json...\n";
    {
        std::ofstream out("experiments/phase_4_3_search_state_dataset.json");
        out << "[\n";
        for (size_t i = 0; i < std::min((size_t)1000, dataset_checkpoints.size()); ++i) {
            const auto& cp = dataset_checkpoints[i];
            out << "  {"
                << "\"step\": " << cp.step << ", "
                << "\"current_ef\": " << cp.current_ef << ", "
                << "\"best_dist\": " << cp.best_dist << ", "
                << "\"furthest_dist\": " << cp.furthest_dist << ", "
                << "\"topk_margin\": " << cp.topk_margin << ", "
                << "\"improvement_velocity\": " << cp.improvement_velocity << ", "
                << "\"stagnation_count\": " << cp.stagnation_count << ", "
                << "\"visited_nodes\": " << cp.visited_nodes << ", "
                << "\"measured_dist_evals\": " << cp.measured_dist_evals << ", "
                << "\"estimated_lid\": " << cp.estimated_lid << ", "
                << "\"uncertainty_score\": " << cp.uncertainty_score << ", "
                << "\"convergence_score\": " << cp.convergence_score << ", "
                << "\"dynamic_patience\": " << cp.dynamic_patience << ", "
                << "\"current_recall_at_10\": " << cp.current_recall_at_10 << ", "
                << "\"max_possible_recall\": " << cp.max_possible_recall << ", "
                << "\"marginal_recall_gain\": " << cp.marginal_recall_gain << ", "
                << "\"will_gain_recall\": " << (cp.will_gain_recall ? "true" : "false") << ", "
                << "\"is_censored\": " << (cp.is_censored ? "true" : "false")
                << "}" << (i + 1 < std::min((size_t)1000, dataset_checkpoints.size()) ? ",\n" : "\n");
        }
        out << "]\n";
    }

    // ========================================================================
    // Stage 6 & 7 — Matched-Recall Comparative Evaluation Across Datasets
    // ========================================================================
    std::cout << "\n-------------------------------------------------------------\n";
    std::cout << "[+] Stage 7: Matched-Recall Independent Comparative Suite\n";
    std::cout << "-------------------------------------------------------------\n";

    struct ComparisonRecord {
        std::string dataset;
        std::string method;
        double recall_at_1;
        double recall_at_10;
        double throughput_qps;
        double p50_latency_ms;
        double p95_latency_ms;
        double p99_latency_ms;
        double actual_mean_dist_evals;
        double upper_layer_dist_evals;
        double layer0_dist_evals;
        double predictor_overhead_us;
        double memory_mb;
        double false_early_exit_pct;
        std::string primary_termination_reason;

        std::string to_json() const {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4);
            ss << "{"
               << "\"dataset\": \"" << dataset << "\", "
               << "\"method\": \"" << method << "\", "
               << "\"recall_at_1\": " << recall_at_1 << ", "
               << "\"recall_at_10\": " << recall_at_10 << ", "
               << "\"throughput_qps\": " << throughput_qps << ", "
               << "\"p50_latency_ms\": " << p50_latency_ms << ", "
               << "\"p95_latency_ms\": " << p95_latency_ms << ", "
               << "\"p99_latency_ms\": " << p99_latency_ms << ", "
               << "\"actual_mean_dist_evals\": " << actual_mean_dist_evals << ", "
               << "\"upper_layer_dist_evals\": " << upper_layer_dist_evals << ", "
               << "\"layer0_dist_evals\": " << layer0_dist_evals << ", "
               << "\"predictor_overhead_us\": " << predictor_overhead_us << ", "
               << "\"memory_mb\": " << memory_mb << ", "
               << "\"false_early_exit_pct\": " << false_early_exit_pct << ", "
               << "\"primary_termination_reason\": \"" << primary_termination_reason << "\""
               << "}";
            return ss.str();
        }
    };
    std::vector<ComparisonRecord> final_comparisons;

    struct DatasetSpec {
        std::string name;
        std::string base_path;
        std::string query_path;
        std::string gt_path;
        size_t max_base;
        size_t max_queries;
        SpaceType space;
        int tuned_ef;
        int tuned_patience;
    };

    std::vector<DatasetSpec> specs = {
        {"SIFT-100K", "data/sift_subset_100k.fvecs", "data/sift_query.fvecs", "data/sift_subset_groundtruth.ivecs", 100000, 1000, SpaceType::L2, 64, 8},
        {"Transformer-384D", "data/transformer_384d/transformer_384d_base.fvecs", "data/transformer_384d/transformer_384d_query.fvecs", "data/transformer_384d/transformer_384d_groundtruth.ivecs", 57893, 1000, SpaceType::Cosine, 64, 8}
    };

    for (const auto& spec : specs) {
        std::cout << "\n-------------------------------------------------------------\n";
        std::cout << "[+] Benchmarking Matched-Recall Suite on " << spec.name << "\n";
        std::cout << "-------------------------------------------------------------\n";

        size_t d_dim = 0, d_nbase = 0, d_qdim = 0, d_nq = 0;
        auto d_base = load_fvecs(spec.base_path, spec.max_base, d_dim, d_nbase);
        auto d_queries = load_fvecs(spec.query_path, spec.max_queries, d_qdim, d_nq);
        auto d_gt = load_ivecs(spec.gt_path, spec.max_queries);

        PolicyConfig idx_cfg;
        idx_cfg.m_base = 16;
        idx_cfg.ef_c_base = 64;
        AdaptiveHNSWIndex test_index(d_dim, spec.space, false, idx_cfg);
        test_index.batch_insert(d_base.data(), d_nbase, threads);
        test_index.policy.keep_raw_vectors = false;
        test_index.allocate_mixed_precision(AllocationStrategy::UniformSQ8, 0.25);
        double ram_mb = test_index.get_total_memory_bytes() / (1024.0 * 1024.0);

        size_t eval_queries = std::min((size_t)800, d_nq);

        // Precompute ground truth max recall for false-termination accounting
        std::vector<float> oracle_recalls(eval_queries);
        for (size_t q = 0; q < eval_queries; ++q) {
            QuerySearchStats max_s;
            auto r = test_index.search(&d_queries[q * d_dim], 10, 192, max_s);
            int hits = 0;
            std::set<tableint> gt_set(d_gt[q].begin(), d_gt[q].begin() + 10);
            for (const auto& c : r) if (gt_set.count(c.id)) hits++;
            oracle_recalls[q] = (float)hits / 10.0f;
        }

        // Methods to compare at matched recall:
        struct EvalMethod {
            std::string name;
            int ef;
            int patience;
            bool enable_ada;
            bool is_uncertainty_aware;
            bool is_resumable;
        };

        std::vector<EvalMethod> methods = {
            {"Canonical HNSW (Fixed Budget)", spec.tuned_ef, 0, false, false, false},
            {"Fixed Stagnation (p=4)", spec.tuned_ef, 4, true, false, false},
            {"Fixed Stagnation (p=8 - Tuned)", spec.tuned_ef, 8, true, false, false},
            {"Fixed Stagnation (p=16)", spec.tuned_ef, 16, true, false, false},
            {"Resumable Search (ef=32->64)", 32, 8, true, false, true},
            {"Proposed Uncertainty-Aware Search", spec.tuned_ef, 6, true, true, false}
        };

        for (const auto& m : methods) {
            test_index.policy.enable_ada_ef = m.enable_ada;
            test_index.policy.ada_ef_patience = (m.patience == 0) ? 999999 : m.patience;

            DynamicPatienceConfig p_cfg;
            p_cfg.p_base = m.patience > 0 ? m.patience : 6;
            p_cfg.p_min = 2;
            p_cfg.p_max = 16;

            MultiSignalDifficultyEstimator estimator = (spec.space == SpaceType::Cosine)
                ? MultiSignalDifficultyEstimator(MultiSignalDifficultyEstimator::default_for_transformer())
                : MultiSignalDifficultyEstimator(MultiSignalDifficultyEstimator::default_for_sift());

            std::vector<double> latencies_ms(eval_queries);
            double total_r1 = 0, total_r10 = 0;
            double total_dist = 0, total_upper = 0, total_l0 = 0;
            int false_early_exits = 0;
            double predictor_us = 0.0;
            std::string last_reason = "FrontierExhausted";

            auto t_bench_0 = std::chrono::high_resolution_clock::now();

            for (size_t q = 0; q < eval_queries; ++q) {
                const float* q_ptr = &d_queries[q * d_dim];
                std::set<tableint> gt_set(d_gt[q].begin(), d_gt[q].begin() + 10);
                tableint gt1 = d_gt[q][0];

                auto t_q0 = std::chrono::high_resolution_clock::now();
                std::vector<Candidate> res;
                QuerySearchStats stats;

                if (m.is_uncertainty_aware) {
                    auto t_p0 = std::chrono::high_resolution_clock::now();
                    auto u_res = UncertaintyAwareSearchEngine::search_layer0_uncertainty_aware(
                        test_index, q_ptr, 10, m.ef, p_cfg, false, nullptr
                    );
                    auto t_p1 = std::chrono::high_resolution_clock::now();
                    predictor_us += std::chrono::duration<double, std::micro>(t_p1 - t_p0).count();
                    res = std::move(u_res.candidates);
                    stats = u_res.stats;
                    last_reason = to_string(u_res.termination_reason);
                } else if (m.is_resumable) {
                    auto r = ResumableSearchEngine::search_resumable(test_index, q_ptr, 10, 32, 64, estimator, true);
                    res = std::move(r.candidates);
                    stats = r.stats;
                    last_reason = r.was_expanded ? "ResumedExpanded" : "ResumedConverged";
                } else {
                    res = test_index.search(q_ptr, 10, m.ef, stats);
                    last_reason = (stats.early_exit_stagnation_count > 0) ? "FixedPatienceEarlyExit" : "FrontierExhausted";
                }

                auto t_q1 = std::chrono::high_resolution_clock::now();
                latencies_ms[q] = std::chrono::duration<double, std::milli>(t_q1 - t_q0).count();

                total_dist += stats.total_dist_evals();
                total_upper += stats.upper_layer_dist_evals;
                total_l0 += stats.layer0_dist_evals;

                int hits = 0;
                for (const auto& c : res) if (gt_set.count(c.id)) hits++;
                double rec10 = (double)hits / 10.0;
                double rec1 = (!res.empty() && res[0].id == gt1) ? 1.0 : 0.0;

                total_r10 += rec10;
                total_r1 += rec1;

                if (rec10 < oracle_recalls[q] - 0.10f && stats.early_exit_stagnation_count > 0) {
                    false_early_exits++;
                }
            }

            auto t_bench_1 = std::chrono::high_resolution_clock::now();
            double wall_ms = std::chrono::duration<double, std::milli>(t_bench_1 - t_bench_0).count();
            double qps = (double)eval_queries / (wall_ms / 1000.0);

            std::sort(latencies_ms.begin(), latencies_ms.end());
            double p50 = latencies_ms[static_cast<size_t>(0.50 * eval_queries)];
            double p95 = latencies_ms[static_cast<size_t>(0.95 * eval_queries)];
            double p99 = latencies_ms[static_cast<size_t>(0.99 * eval_queries)];

            ComparisonRecord rec;
            rec.dataset = spec.name;
            rec.method = m.name;
            rec.recall_at_1 = total_r1 / eval_queries;
            rec.recall_at_10 = total_r10 / eval_queries;
            rec.throughput_qps = qps;
            rec.p50_latency_ms = p50;
            rec.p95_latency_ms = p95;
            rec.p99_latency_ms = p99;
            rec.actual_mean_dist_evals = total_dist / eval_queries;
            rec.upper_layer_dist_evals = total_upper / eval_queries;
            rec.layer0_dist_evals = total_l0 / eval_queries;
            rec.predictor_overhead_us = predictor_us / eval_queries;
            rec.memory_mb = ram_mb;
            rec.false_early_exit_pct = (double)false_early_exits / eval_queries * 100.0;
            rec.primary_termination_reason = last_reason;

            final_comparisons.push_back(rec);

            std::cout << "  " << std::left << std::setw(36) << m.name
                      << " | R@10: " << std::fixed << std::setprecision(2) << (rec.recall_at_10 * 100.0) << "%"
                      << " | Mean Dist: " << std::setw(6) << std::setprecision(1) << rec.actual_mean_dist_evals
                      << " | QPS: " << std::setw(7) << (int)qps
                      << " | p99: " << std::setprecision(3) << p99 << " ms"
                      << " | False EE%: " << std::setprecision(1) << rec.false_early_exit_pct << "%\n";
        }
    }

    // Export Stage 7 JSON
    std::cout << "\n[*] Exporting experiments/phase_4_3_matched_recall_results.json...\n";
    {
        std::ofstream out("experiments/phase_4_3_matched_recall_results.json");
        out << "[\n";
        for (size_t i = 0; i < final_comparisons.size(); ++i) {
            out << "  " << final_comparisons[i].to_json() << (i + 1 < final_comparisons.size() ? ",\n" : "\n");
        }
        out << "]\n";
    }

    std::cout << "[*] Phase 4.3 Uncertainty-Aware Search Engine Suite Finished Successfully!\n";
    return 0;
}
