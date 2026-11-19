#include "lightweight_controller.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <set>
#include <numeric>
#include <cmath>

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

double compute_pearson(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = x.size();
    if (n < 2) return 0.0;
    double mx = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double my = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double num = 0.0, d1 = 0.0, d2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mx;
        double dy = y[i] - my;
        num += dx * dy;
        d1 += dx * dx;
        d2 += dy * dy;
    }
    if (d1 <= 1e-12 || d2 <= 1e-12) return 0.0;
    return num / (std::sqrt(d1) * std::sqrt(d2));
}

double compute_spearman(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = x.size();
    if (n < 2) return 0.0;
    auto get_ranks = [](const std::vector<double>& v) {
        size_t m = v.size();
        std::vector<size_t> idx(m);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&v](size_t a, size_t b) { return v[a] < v[b]; });
        std::vector<double> ranks(m);
        for (size_t r = 0; r < m; ++r) ranks[idx[r]] = (double)r;
        return ranks;
    };
    return compute_pearson(get_ranks(x), get_ranks(y));
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  AdaptiveVec Phase 4.2: Stages 2 & 3 Investigation Benchmark    \n";
    std::cout << "=================================================================\n" << std::endl;

    size_t dim = 0, n_base = 0, q_dim = 0, n_queries = 0;
    auto base_data = load_fvecs("data/sift_subset_100k.fvecs", 100000, dim, n_base);
    auto query_data = load_fvecs("data/sift_query.fvecs", 2000, q_dim, n_queries);
    auto gt = load_ivecs("data/sift_subset_groundtruth.ivecs", 2000);

    std::cout << "[*] Loaded SIFT-100K: " << n_base << " base, " << n_queries << " queries, dim: " << dim << std::endl;

    PolicyConfig cfg;
    cfg.m_base = 16;
    cfg.ef_c_base = 64;
    cfg.enable_hubness_regulation = false;
    AdaptiveHNSWIndex index(dim, SpaceType::L2, false, cfg);
    index.batch_insert(base_data.data(), n_base, 12);

    // ------------------------------------------------------------------------
    // Part 1: Stage 2 — Reference Monitoring Cost Profiling
    // ------------------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------\n";
    std::cout << "[+] Stage 2: Profiling Reference Monitoring Architectures\n";
    std::cout << "-------------------------------------------------------------\n";

    // Setup 100 reference samples
    BoundedReferenceMonitor mon_legacy;
    mon_legacy.set_policy(MonitoringPolicyType::PeriodicBatch, 50); // Legacy: 100 ref queries every 50
    BoundedReferenceMonitor mon_periodic;
    mon_periodic.set_policy(MonitoringPolicyType::PeriodicBatch, 500); // 100 ref queries every 500
    BoundedReferenceMonitor mon_incremental_25;
    mon_incremental_25.set_policy(MonitoringPolicyType::IncrementalSampling, 25); // 1 ref query every 25 (4%)
    BoundedReferenceMonitor mon_incremental_50;
    mon_incremental_50.set_policy(MonitoringPolicyType::IncrementalSampling, 50); // 1 ref query every 50 (2%)

    for (size_t i = 1000; i < 1100 && i < gt.size(); ++i) {
        std::vector<float> q(&query_data[i * dim], &query_data[(i + 1) * dim]);
        std::vector<tableint> g(gt[i].begin(), gt[i].begin() + 10);
        mon_legacy.add_reference_sample(q, g);
        mon_periodic.add_reference_sample(q, g);
        mon_incremental_25.add_reference_sample(q, g);
        mon_incremental_50.add_reference_sample(q, g);
    }

    struct MonitoringProfile {
        std::string policy_name;
        double wall_clock_ms;
        double throughput_qps;
        int prod_queries;
        int ref_queries;
        double cpu_time_ms;
        double overhead_pct;
    };
    std::vector<MonitoringProfile> profiles;

    // Helper to evaluate monitoring policy under 1,000 queries
    auto profile_policy = [&](const std::string& name, BoundedReferenceMonitor& mon, bool is_legacy_batch) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int ref_count = 0;
        for (size_t q = 0; q < 1000; ++q) {
            index.search(&query_data[q * dim], 10, 64);
            mon.record_prod_query();

            if (is_legacy_batch) {
                if (q > 0 && q % 50 == 0) {
                    for (size_t r = 0; r < 100; ++r) {
                        mon.evaluate_next_sample(index, 64, 10);
                        ref_count++;
                    }
                }
            } else {
                if (mon.should_evaluate_sample(q)) {
                    mon.evaluate_next_sample(index, 64, 10);
                    ref_count++;
                }
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double qps = 1000.0 / (wall_ms / 1000.0);
        double overhead = ((double)ref_count / 1000.0) * 100.0;

        profiles.push_back({name, wall_ms, qps, 1000, ref_count, mon.get_stats().cpu_time_ms, overhead});

        std::cout << "  " << std::left << std::setw(38) << name
                  << " | QPS: " << std::setw(7) << (int)qps
                  << " | Ref Queries: " << std::setw(5) << ref_count
                  << " | CPU Overhead: " << std::fixed << std::setprecision(2) << overhead << "%\n";
    };

    // Baseline (Zero monitoring)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t q = 0; q < 1000; ++q) index.search(&query_data[q * dim], 10, 64);
        auto t1 = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double qps = 1000.0 / (wall_ms / 1000.0);
        profiles.push_back({"Unmonitored Baseline (0% Target)", wall_ms, qps, 1000, 0, 0.0, 0.0});
        std::cout << "  " << std::left << std::setw(38) << "Unmonitored Baseline (0% Target)"
                  << " | QPS: " << std::setw(7) << (int)qps
                  << " | Ref Queries: " << std::setw(5) << 0
                  << " | CPU Overhead: 0.00%\n";
    }

    profile_policy("Phase 4.1 Fixed Batch (100q / 50req)", mon_legacy, true);
    profile_policy("Low-Frequency Periodic (100q / 500req)", mon_periodic, false);
    profile_policy("Proposed Incremental (1q / 25req)", mon_incremental_25, false);
    profile_policy("Proposed Incremental (1q / 50req)", mon_incremental_50, false);

    // Export Stage 2 JSON
    {
        std::ofstream out("experiments/phase_4_2_monitoring_cost.json");
        out << "[\n";
        for (size_t i = 0; i < profiles.size(); ++i) {
            const auto& p = profiles[i];
            out << "  {"
                << "\"policy\": \"" << p.policy_name << "\", "
                << "\"wall_clock_ms\": " << p.wall_clock_ms << ", "
                << "\"throughput_qps\": " << p.throughput_qps << ", "
                << "\"prod_queries\": " << p.prod_queries << ", "
                << "\"ref_queries_evaluated\": " << p.ref_queries << ", "
                << "\"cpu_time_ms\": " << p.cpu_time_ms << ", "
                << "\"cpu_overhead_pct\": " << p.overhead_pct
                << "}" << (i + 1 < profiles.size() ? ",\n" : "\n");
        }
        out << "]\n";
    }
    std::cout << "[*] Exported experiments/phase_4_2_monitoring_cost.json\n";

    // ------------------------------------------------------------------------
    // Part 2: Stage 3 — Multi-Signal Query Difficulty Evaluation
    // ------------------------------------------------------------------------
    std::cout << "\n-------------------------------------------------------------\n";
    std::cout << "[+] Stage 3: Multi-Signal Difficulty Feature Analysis\n";
    std::cout << "-------------------------------------------------------------\n";

    size_t eval_n = 1000;
    std::vector<double> v_l0_dist, v_margin, v_lid, v_imp_rate, v_degree, v_min_ef;

    MultiSignalDifficultyEstimator estimator(MultiSignalDifficultyEstimator::default_for_sift());
    int total_expanded = 0;
    double feature_calc_time_us = 0.0;

    for (size_t q = 0; q < eval_n; ++q) {
        const float* q_ptr = &query_data[q * dim];

        // Find min_ef for 100% Recall@10
        int min_ef = 128;
        std::set<tableint> gt_set(gt[q].begin(), gt[q].begin() + 10);
        for (int ef : {16, 24, 32, 48, 64, 96, 128}) {
            auto res = index.search(q_ptr, 10, ef);
            int hits = 0;
            for (const auto& c : res) if (gt_set.count(c.id)) hits++;
            if (hits == 10) { min_ef = ef; break; }
        }

        // Run Resumable search with signal extraction
        auto t_f0 = std::chrono::high_resolution_clock::now();
        auto res = ResumableSearchEngine::search_resumable(index, q_ptr, 10, 32, 64, estimator, true);
        auto t_f1 = std::chrono::high_resolution_clock::now();
        feature_calc_time_us += std::chrono::duration<double, std::micro>(t_f1 - t_f0).count();

        if (res.was_expanded) total_expanded++;

        v_l0_dist.push_back(res.signals.layer0_entry_dist);
        v_margin.push_back(res.signals.topk_margin);
        v_lid.push_back(res.signals.estimated_lid);
        v_imp_rate.push_back(res.signals.candidate_improvement_rate);
        v_degree.push_back((double)res.signals.entry_node_degree);
        v_min_ef.push_back(min_ef);
    }

    double rho_l0 = compute_spearman(v_l0_dist, v_min_ef);
    double rho_margin = compute_spearman(v_margin, v_min_ef);
    double rho_lid = compute_spearman(v_lid, v_min_ef);
    double rho_imp = compute_spearman(v_imp_rate, v_min_ef);
    double rho_deg = compute_spearman(v_degree, v_min_ef);

    std::cout << "  Layer-0 Distance Spearman Rho:  " << std::fixed << std::setprecision(4) << rho_l0 << "\n";
    std::cout << "  Top-10 Margin Spearman Rho:     " << rho_margin << " (stronger separation = lower required ef)\n";
    std::cout << "  Estimated LID Spearman Rho:     " << rho_lid << " (higher dimensionality = higher required ef)\n";
    std::cout << "  Improvement Rate Spearman Rho:  " << rho_imp << "\n";
    std::cout << "  Entry Node Degree Spearman Rho: " << rho_deg << "\n";
    std::cout << "  Expansion Rate (ef=32 -> 64):   " << (double)total_expanded / eval_n * 100.0 << "%\n";
    std::cout << "  Avg Search+Inference Time:      " << feature_calc_time_us / eval_n << " us/query\n";

    // Export Stage 3 JSON
    {
        std::ofstream out("experiments/phase_4_2_predictor_results.json");
        out << "{\n"
            << "  \"dataset\": \"SIFT-100K\",\n"
            << "  \"evaluated_queries\": " << eval_n << ",\n"
            << "  \"feature_spearman_rho\": {\n"
            << "    \"layer0_entry_dist\": " << rho_l0 << ",\n"
            << "    \"top10_margin\": " << rho_margin << ",\n"
            << "    \"estimated_lid\": " << rho_lid << ",\n"
            << "    \"candidate_improvement_rate\": " << rho_imp << ",\n"
            << "    \"entry_node_degree\": " << rho_deg << "\n"
            << "  },\n"
            << "  \"expansion_rate_pct\": " << ((double)total_expanded / eval_n * 100.0) << ",\n"
            << "  \"avg_inference_latency_us\": " << (feature_calc_time_us / eval_n) << "\n"
            << "}\n";
    }
    std::cout << "[*] Exported experiments/phase_4_2_predictor_results.json\n";

    return 0;
}
