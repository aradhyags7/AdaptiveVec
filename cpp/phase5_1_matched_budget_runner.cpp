#include "adaptive_hnsw.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <numeric>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace adaptivevec;

struct ParetoPoint {
    int ef_search;
    double recall_at_1;
    double recall_at_10;
    double recall_at_100;
    double qps;
    double p50_latency_ms;
    double p95_latency_ms;
    double p99_latency_ms;
    double mean_dist_evals;
};

struct SeriesResult {
    std::string model_id;
    std::string model_name;
    std::string dataset;
    size_t total_edges;
    double memory_mb;
    std::vector<ParetoPoint> points;
};

SeriesResult evaluate_pareto(
    const std::string& model_id,
    const std::string& model_name,
    const std::string& dataset_name,
    const AdaptiveHNSWIndex& index,
    const std::vector<float>& queries,
    size_t nq,
    size_t dim,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    const std::vector<int>& ef_list
) {
    SeriesResult sr;
    sr.model_id = model_id;
    sr.model_name = model_name;
    sr.dataset = dataset_name;
    sr.total_edges = index.get_total_edges();
    sr.memory_mb = (double)index.get_total_memory_bytes() / (1024.0 * 1024.0);

    for (int ef : ef_list) {
        std::vector<double> r1(nq, 0.0), r10(nq, 0.0), r100(nq, 0.0);
        std::vector<double> latencies_ms(nq, 0.0);
        uint64_t total_dist_evals = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_dist_evals)
        for (size_t q = 0; q < nq; ++q) {
            auto q_start = std::chrono::high_resolution_clock::now();
            QuerySearchStats stats;
            auto results = index.search(&queries[q * dim], 100, ef, stats);
            auto q_end = std::chrono::high_resolution_clock::now();
            latencies_ms[q] = std::chrono::duration<double, std::milli>(q_end - q_start).count();
            total_dist_evals += stats.total_dist_evals();

            std::unordered_set<uint32_t> gt1, gt10, gt100;
            if (q < ground_truth.size()) {
                for (size_t k = 0; k < 1 && k < ground_truth[q].size(); ++k) gt1.insert(ground_truth[q][k]);
                for (size_t k = 0; k < 10 && k < ground_truth[q].size(); ++k) gt10.insert(ground_truth[q][k]);
                for (size_t k = 0; k < 100 && k < ground_truth[q].size(); ++k) gt100.insert(ground_truth[q][k]);
            }
            size_t m1 = 0, m10 = 0, m100 = 0;
            for (size_t k = 0; k < results.size(); ++k) {
                tableint cid = results[k].id;
                if (k < 1 && gt1.find(cid) != gt1.end()) m1++;
                if (k < 10 && gt10.find(cid) != gt10.end()) m10++;
                if (k < 100 && gt100.find(cid) != gt100.end()) m100++;
            }
            r1[q] = (double)m1 / 1.0;
            r10[q] = (double)m10 / 10.0;
            r100[q] = (double)m100 / 100.0;
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_time_s = std::chrono::duration<double>(t1 - t0).count();
        double qps = (double)nq / total_time_s;
        std::sort(latencies_ms.begin(), latencies_ms.end());

        ParetoPoint pt;
        pt.ef_search = ef;
        pt.recall_at_1 = std::accumulate(r1.begin(), r1.end(), 0.0) / (double)nq;
        pt.recall_at_10 = std::accumulate(r10.begin(), r10.end(), 0.0) / (double)nq;
        pt.recall_at_100 = std::accumulate(r100.begin(), r100.end(), 0.0) / (double)nq;
        pt.qps = qps;
        pt.mean_dist_evals = (double)total_dist_evals / (double)nq;
        pt.p50_latency_ms = latencies_ms[nq / 2];
        pt.p95_latency_ms = latencies_ms[(size_t)(nq * 0.95)];
        pt.p99_latency_ms = latencies_ms[(size_t)(nq * 0.99)];

        sr.points.push_back(pt);

        std::cout << "  [" << model_id << "] efSearch=" << ef << " -> Recall@10: " << std::fixed << std::setprecision(2)
                  << pt.recall_at_10 * 100.0 << "%, QPS: " << std::setprecision(1) << pt.qps
                  << ", MeanDist: " << pt.mean_dist_evals << std::endl;
    }
    return sr;
}

int main() {
    std::cout << "=============================================================" << std::endl;
    std::cout << "  AdaptiveVec Phase 5.1: Matched-Budget Pareto Runner         " << std::endl;
    std::cout << "=============================================================" << std::endl;

    // Load SIFT-100K
    std::string sift_base_path = "data/sift_subset_100k.fvecs";
    std::string sift_query_path = "data/sift_query.fvecs";
    std::string sift_gt_path = "data/sift_subset_groundtruth.ivecs";

    size_t sift_n = 0, sift_dim = 0;
    std::vector<float> sift_data;
    load_fvecs(sift_base_path, sift_data, sift_n, sift_dim);
    size_t sift_nq = 0, sift_qdim = 0;
    std::vector<float> sift_queries;
    load_fvecs(sift_query_path, sift_queries, sift_nq, sift_qdim);
    std::vector<std::vector<uint32_t>> sift_gt;
    load_ivecs(sift_gt_path, sift_gt);

    // Load Transformer-384D
    std::string tf_base_path = "data/transformer_384d/transformer_384d_base.fvecs";
    std::string tf_query_path = "data/transformer_384d/transformer_384d_query.fvecs";
    std::string tf_gt_path = "data/transformer_384d/transformer_384d_groundtruth.ivecs";

    size_t tf_n = 0, tf_dim = 0;
    std::vector<float> tf_data;
    load_fvecs(tf_base_path, tf_data, tf_n, tf_dim);
    size_t tf_nq = 0, tf_qdim = 0;
    std::vector<float> tf_queries;
    load_fvecs(tf_query_path, tf_queries, tf_nq, tf_qdim);
    std::vector<std::vector<uint32_t>> tf_gt;
    load_ivecs(tf_gt_path, tf_gt);

    std::vector<int> ef_search_list = {16, 32, 64, 128, 256};
    std::vector<SeriesResult> all_series;

    // 1. Uniform Canonical HNSW on SIFT-100K
    PolicyConfig cfg_uniform;
    cfg_uniform.m_base = 16;
    cfg_uniform.ef_c_base = 120;
    cfg_uniform.enable_layer_scaling = false;
    cfg_uniform.enable_hubness_regulation = false;

    std::cout << "\nBuilding Uniform Canonical HNSW on SIFT-100K..." << std::endl;
    AdaptiveHNSWIndex idx_uniform_sift(sift_dim, SpaceType::L2, false, cfg_uniform);
    for (size_t i = 0; i < sift_n; ++i) idx_uniform_sift.insert(&sift_data[i * sift_dim]);
    all_series.push_back(evaluate_pareto("Uniform-HNSW-SIFT", "Uniform Canonical HNSW (M=16)", "SIFT-100K",
                                         idx_uniform_sift, sift_queries, sift_nq, sift_dim, sift_gt, ef_search_list));

    // 2. Architecture B on SIFT-100K
    PolicyConfig cfg_arch_b;
    cfg_arch_b.m_base = 16;
    cfg_arch_b.ef_c_base = 120;
    cfg_arch_b.m_min = 8;
    cfg_arch_b.m_max = 24;
    cfg_arch_b.ef_c_min = 40;
    cfg_arch_b.ef_c_max = 220;
    cfg_arch_b.alpha_lid = 0.5f;
    cfg_arch_b.beta_density = 0.5f;
    cfg_arch_b.sensitivity = 1.0f;
    cfg_arch_b.enable_layer_scaling = true;
    cfg_arch_b.lambda_layer = 0.75f;
    cfg_arch_b.m_min_layer = 4;
    cfg_arch_b.enable_hubness_regulation = true;
    cfg_arch_b.hubness_mu = 0.15f;
    cfg_arch_b.use_streaming_welford = true;

    std::cout << "\nBuilding Architecture B on SIFT-100K..." << std::endl;
    AdaptiveHNSWIndex idx_arch_b_sift(sift_dim, SpaceType::L2, true, cfg_arch_b);
    for (size_t i = 0; i < sift_n; ++i) idx_arch_b_sift.insert(&sift_data[i * sift_dim]);
    all_series.push_back(evaluate_pareto("Arch-B-SIFT", "Architecture B (Adaptive M/efC)", "SIFT-100K",
                                         idx_arch_b_sift, sift_queries, sift_nq, sift_dim, sift_gt, ef_search_list));

    // 3. Phase 5.1 Geometry-Conditioned Pruning (Full Joint) on SIFT-100K
    PolicyConfig cfg_gcp = cfg_arch_b;
    cfg_gcp.enable_geometry_pruning = true;
    cfg_gcp.pruning_alpha_base = 1.0f;
    cfg_gcp.pruning_lid_weight = 0.12f;
    cfg_gcp.pruning_density_weight = 0.10f;

    std::cout << "\nBuilding Phase 5.1 GCP (Joint Allocation & Pruning) on SIFT-100K..." << std::endl;
    AdaptiveHNSWIndex idx_gcp_sift(sift_dim, SpaceType::L2, true, cfg_gcp);
    for (size_t i = 0; i < sift_n; ++i) idx_gcp_sift.insert(&sift_data[i * sift_dim]);
    all_series.push_back(evaluate_pareto("Phase-5.1-GCP-SIFT", "AdaptiveVec 5.1 GCP (Joint)", "SIFT-100K",
                                         idx_gcp_sift, sift_queries, sift_nq, sift_dim, sift_gt, ef_search_list));

    // 4. Transformer-384D Comparison
    if (tf_n > 0) {
        std::cout << "\nBuilding Uniform Canonical HNSW on Transformer-384D..." << std::endl;
        AdaptiveHNSWIndex idx_uniform_tf(tf_dim, SpaceType::Cosine, false, cfg_uniform);
        for (size_t i = 0; i < tf_n; ++i) idx_uniform_tf.insert(&tf_data[i * tf_dim]);
        all_series.push_back(evaluate_pareto("Uniform-HNSW-TF", "Uniform Canonical HNSW (M=16)", "Transformer-384D",
                                             idx_uniform_tf, tf_queries, tf_nq, tf_dim, tf_gt, ef_search_list));

        std::cout << "\nBuilding Architecture B on Transformer-384D..." << std::endl;
        AdaptiveHNSWIndex idx_arch_b_tf(tf_dim, SpaceType::Cosine, true, cfg_arch_b);
        for (size_t i = 0; i < tf_n; ++i) idx_arch_b_tf.insert(&tf_data[i * tf_dim]);
        all_series.push_back(evaluate_pareto("Arch-B-TF", "Architecture B (Adaptive M/efC)", "Transformer-384D",
                                             idx_arch_b_tf, tf_queries, tf_nq, tf_dim, tf_gt, ef_search_list));

        std::cout << "\nBuilding Phase 5.1 GCP on Transformer-384D..." << std::endl;
        AdaptiveHNSWIndex idx_gcp_tf(tf_dim, SpaceType::Cosine, true, cfg_gcp);
        for (size_t i = 0; i < tf_n; ++i) idx_gcp_tf.insert(&tf_data[i * tf_dim]);
        all_series.push_back(evaluate_pareto("Phase-5.1-GCP-TF", "AdaptiveVec 5.1 GCP (Joint)", "Transformer-384D",
                                             idx_gcp_tf, tf_queries, tf_nq, tf_dim, tf_gt, ef_search_list));
    }

    // Export Matched Budget Results JSON
    std::ofstream out_json("experiments/phase_5_1_matched_budget_results.json");
    out_json << "{\n";
    out_json << "  \"timestamp\": \"2026-11-20T13:00:00Z\",\n";
    out_json << "  \"pareto_series\": [\n";
    for (size_t s = 0; s < all_series.size(); ++s) {
        const auto& sr = all_series[s];
        out_json << "    {\n";
        out_json << "      \"model_id\": \"" << sr.model_id << "\",\n";
        out_json << "      \"model_name\": \"" << sr.model_name << "\",\n";
        out_json << "      \"dataset\": \"" << sr.dataset << "\",\n";
        out_json << "      \"total_edges\": " << sr.total_edges << ",\n";
        out_json << "      \"memory_mb\": " << sr.memory_mb << ",\n";
        out_json << "      \"points\": [\n";
        for (size_t p = 0; p < sr.points.size(); ++p) {
            const auto& pt = sr.points[p];
            out_json << "        {\n";
            out_json << "          \"ef_search\": " << pt.ef_search << ",\n";
            out_json << "          \"recall_at_1\": " << pt.recall_at_1 << ",\n";
            out_json << "          \"recall_at_10\": " << pt.recall_at_10 << ",\n";
            out_json << "          \"recall_at_100\": " << pt.recall_at_100 << ",\n";
            out_json << "          \"qps\": " << pt.qps << ",\n";
            out_json << "          \"mean_dist_evals\": " << pt.mean_dist_evals << ",\n";
            out_json << "          \"p50_latency_ms\": " << pt.p50_latency_ms << ",\n";
            out_json << "          \"p95_latency_ms\": " << pt.p95_latency_ms << ",\n";
            out_json << "          \"p99_latency_ms\": " << pt.p99_latency_ms << "\n";
            out_json << "        }" << (p + 1 < sr.points.size() ? "," : "") << "\n";
        }
        out_json << "      ]\n";
        out_json << "    }" << (s + 1 < all_series.size() ? "," : "") << "\n";
    }
    out_json << "  ]\n";
    out_json << "}\n";
    out_json.close();
    std::cout << "\nSaved Matched Budget Pareto Results to experiments/phase_5_1_matched_budget_results.json" << std::endl;

    return 0;
}
