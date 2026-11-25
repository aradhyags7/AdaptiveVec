/**
 * AdaptiveVec Phase 0 pybind11 C++ Extensions
 * Exposes AdaptiveHNSWIndex and PolicyConfig to Python with NumPy buffer interoperability.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "adaptive_hnsw.hpp"
#include "auto_calibrator.hpp"
#include "budget_allocator.hpp"

namespace py = pybind11;
using namespace adaptivevec;

PYBIND11_MODULE(_adaptivevec_cpp, m) {
    m.doc() = "AdaptiveVec Phase 0 C++ Native Engine with pybind11";

    // Enums
    py::enum_<SpaceType>(m, "SpaceType")
        .value("L2", SpaceType::L2)
        .value("Cosine", SpaceType::Cosine)
        .export_values();

    py::enum_<PolicyType>(m, "PolicyType")
        .value("Continuous", PolicyType::Continuous)
        .value("Quantile", PolicyType::Quantile)
        .value("LidOnly", PolicyType::LidOnly)
        .value("DensityOnly", PolicyType::DensityOnly)
        .export_values();

    py::enum_<VectorPrecision>(m, "VectorPrecision")
        .value("FP32", VectorPrecision::FP32)
        .value("SQ8", VectorPrecision::SQ8)
        .value("SQ4", VectorPrecision::SQ4)
        .export_values();

    py::enum_<AllocationStrategy>(m, "AllocationStrategy")
        .value("UniformFP32", AllocationStrategy::UniformFP32)
        .value("UniformSQ8", AllocationStrategy::UniformSQ8)
        .value("UniformSQ4", AllocationStrategy::UniformSQ4)
        .value("Random", AllocationStrategy::Random)
        .value("ReconstructionError", AllocationStrategy::ReconstructionError)
        .value("Geometry", AllocationStrategy::Geometry)
        .value("RoutingImportance", AllocationStrategy::RoutingImportance)
        .value("CombinedSensitivity", AllocationStrategy::CombinedSensitivity)
        .export_values();

    py::class_<PrecisionDistribution>(m, "PrecisionDistribution")
        .def(py::init<>())
        .def_readonly("n_fp32", &PrecisionDistribution::n_fp32)
        .def_readonly("n_sq8", &PrecisionDistribution::n_sq8)
        .def_readonly("n_sq4", &PrecisionDistribution::n_sq4)
        .def_readonly("vector_storage_mb", &PrecisionDistribution::vector_storage_mb)
        .def_readonly("savings_ratio", &PrecisionDistribution::savings_ratio);

    // PolicyConfig struct
    py::class_<PolicyConfig>(m, "PolicyConfig")
        .def(py::init<>())
        .def_readwrite("type", &PolicyConfig::type)
        .def_readwrite("m_base", &PolicyConfig::m_base)
        .def_readwrite("m_min", &PolicyConfig::m_min)
        .def_readwrite("m_max", &PolicyConfig::m_max)
        .def_readwrite("ef_c_base", &PolicyConfig::ef_c_base)
        .def_readwrite("ef_c_min", &PolicyConfig::ef_c_min)
        .def_readwrite("ef_c_max", &PolicyConfig::ef_c_max)
        .def_readwrite("alpha_lid", &PolicyConfig::alpha_lid)
        .def_readwrite("beta_density", &PolicyConfig::beta_density)
        .def_readwrite("sensitivity", &PolicyConfig::sensitivity)
        .def_readwrite("enable_dynamic_m_efc", &PolicyConfig::enable_dynamic_m_efc)
        .def_readwrite("enable_layer_scaling", &PolicyConfig::enable_layer_scaling)
        .def_readwrite("lambda_layer", &PolicyConfig::lambda_layer)
        .def_readwrite("m_min_layer", &PolicyConfig::m_min_layer)
        .def_readwrite("enable_hubness_regulation", &PolicyConfig::enable_hubness_regulation)
        .def_readwrite("hubness_mu", &PolicyConfig::hubness_mu)
        .def_readwrite("enable_ada_ef", &PolicyConfig::enable_ada_ef)
        .def_readwrite("ada_ef_patience", &PolicyConfig::ada_ef_patience)
        .def_readwrite("ada_ef_epsilon", &PolicyConfig::ada_ef_epsilon)
        .def_readwrite("enable_dynamic_ef_search", &PolicyConfig::enable_dynamic_ef_search)
        .def_readwrite("enable_sq8", &PolicyConfig::enable_sq8)
        .def_readwrite("enable_sq4", &PolicyConfig::enable_sq4)
        .def_readwrite("sq8_rerank_factor", &PolicyConfig::sq8_rerank_factor)
        .def_readwrite("sq4_rerank_factor", &PolicyConfig::sq4_rerank_factor)
        .def_readwrite("keep_raw_vectors", &PolicyConfig::keep_raw_vectors)
        .def_readwrite("enable_rerank", &PolicyConfig::enable_rerank)
        .def_readwrite("global_seed", &PolicyConfig::global_seed)
        .def_readwrite("use_deterministic_level_hash", &PolicyConfig::use_deterministic_level_hash)
        .def_readwrite("enable_bridge_aware", &PolicyConfig::enable_bridge_aware)
        .def_readwrite("bridge_alpha", &PolicyConfig::bridge_alpha)
        .def_readwrite("bridge_estimator", &PolicyConfig::bridge_estimator)
        .def_readwrite("enable_mixed_precision", &PolicyConfig::enable_mixed_precision)
        .def_readwrite("mixed_precision_strategy", &PolicyConfig::mixed_precision_strategy)
        .def_readwrite("mixed_precision_budget_ratio", &PolicyConfig::mixed_precision_budget_ratio);

    // MemoryBreakdown struct
    py::class_<MemoryBreakdown>(m, "MemoryBreakdown")
        .def(py::init<>())
        .def_readonly("logical_compressed_bytes", &MemoryBreakdown::logical_compressed_bytes)
        .def_readonly("allocated_quantized_capacity_bytes", &MemoryBreakdown::allocated_quantized_capacity_bytes)
        .def_readonly("resident_fp32_bytes", &MemoryBreakdown::resident_fp32_bytes)
        .def_readonly("graph_adjacency_bytes", &MemoryBreakdown::graph_adjacency_bytes)
        .def_readonly("per_node_metadata_bytes", &MemoryBreakdown::per_node_metadata_bytes)
        .def_readonly("quantization_params_bytes", &MemoryBreakdown::quantization_params_bytes)
        .def_readonly("synchronization_bytes", &MemoryBreakdown::synchronization_bytes)
        .def_readonly("allocator_overhead_bytes", &MemoryBreakdown::allocator_overhead_bytes)
        .def_readonly("total_owned_bytes", &MemoryBreakdown::total_owned_bytes)
        .def_readonly("process_working_set_bytes", &MemoryBreakdown::process_working_set_bytes)
        .def_readonly("peak_process_working_set_bytes", &MemoryBreakdown::peak_process_working_set_bytes)
        .def_readonly("process_private_bytes", &MemoryBreakdown::process_private_bytes);

    // Candidate struct
    py::class_<Candidate>(m, "Candidate")
        .def_readonly("id", &Candidate::id)
        .def_readonly("dist", &Candidate::dist)
        .def("__repr__", [](const Candidate& c) {
            return "<Candidate id=" + std::to_string(c.id) + " dist=" + std::to_string(c.dist) + ">";
        });

    // AdaptiveHNSWIndex class (shared_ptr holder supports ServingIndexManager sharing)
    py::class_<AdaptiveHNSWIndex, std::shared_ptr<AdaptiveHNSWIndex>>(m, "AdaptiveHNSWIndex")
        .def(py::init<size_t, SpaceType, bool, PolicyConfig>(),
             py::arg("dimension"),
             py::arg("space") = SpaceType::L2,
             py::arg("adaptive") = true,
             py::arg("policy") = PolicyConfig())
        
        .def("insert", [](AdaptiveHNSWIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> vec) {
            py::buffer_info info = vec.request();
            if (info.ndim != 1 || (size_t)info.shape[0] != self.dim) {
                throw std::runtime_error("Vector shape must be 1D with length matching index dimension (" + std::to_string(self.dim) + ")");
            }
            return self.insert(static_cast<const float*>(info.ptr));
        }, py::arg("vec"), "Insert a single vector into the index")

        .def("batch_insert", [](AdaptiveHNSWIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> vectors, int num_threads) {
            py::buffer_info info = vectors.request();
            if (info.ndim != 2 || (size_t)info.shape[1] != self.dim) {
                throw std::runtime_error("Vectors must be a 2D float32 array of shape (N, dim)");
            }
            size_t n_vectors = info.shape[0];
            self.batch_insert(static_cast<const float*>(info.ptr), n_vectors, num_threads);
        }, py::arg("vectors"), py::arg("num_threads") = 0, "Insert multiple vectors in parallel")

        .def("search", [](AdaptiveHNSWIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> query, int k, int ef) {
            py::buffer_info info = query.request();
            if (info.ndim != 1 || (size_t)info.shape[0] != self.dim) {
                throw std::runtime_error("Query shape must be 1D with length matching index dimension (" + std::to_string(self.dim) + ")");
            }
            return self.search(static_cast<const float*>(info.ptr), k, ef);
        }, py::arg("query"), py::arg("k") = 10, py::arg("ef") = 64, "Search for top-k nearest neighbors")

        .def("batch_search", [](AdaptiveHNSWIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> queries, int k, int ef, int num_threads) {
            py::buffer_info info = queries.request();
            if (info.ndim != 2 || (size_t)info.shape[1] != self.dim) {
                throw std::runtime_error("Queries must be a 2D float32 array of shape (Q, dim)");
            }
            size_t n_queries = info.shape[0];
            return self.batch_search(static_cast<const float*>(info.ptr), n_queries, k, ef, num_threads);
        }, py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 64, py::arg("num_threads") = 0, "Batch search for top-k nearest neighbors across threads")

        .def("delete_vector", &AdaptiveHNSWIndex::delete_vector, py::arg("node_id"), py::arg("heal") = true, "Tombstone delete a vector and heal graph")
        .def("compact", &AdaptiveHNSWIndex::compact, "Synchronized blocking compaction: remove tombstones and rebuild dense IDs")
        .def("train_quantizer", [](AdaptiveHNSWIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> sample_data) {
            py::buffer_info info = sample_data.request();
            if (info.ndim != 2 || (size_t)info.shape[1] != self.dim) {
                throw std::runtime_error("Sample data must be a 2D float32 array of shape (N, dim)");
            }
            self.train_quantizer(static_cast<const float*>(info.ptr), info.shape[0]);
        }, py::arg("sample_data"), "Train quantizer on representative sample")
        
        .def("quantize_vectors", &AdaptiveHNSWIndex::quantize_vectors, py::arg("sq8") = true, py::arg("sq4") = false, "Quantize resident vectors")
        .def("drop_raw_vectors", &AdaptiveHNSWIndex::drop_raw_vectors, "Drop FP32 vectors for pure quantized retrieval (Configuration A)")
        .def("get_memory_breakdown", &AdaptiveHNSWIndex::get_memory_breakdown, "Return comprehensive MemoryBreakdown")
        .def("get_graph_fingerprint", &AdaptiveHNSWIndex::get_graph_fingerprint, "Return 64-bit canonical graph topology fingerprint")
        .def("get_total_edges", &AdaptiveHNSWIndex::get_total_edges, "Return total directed edges")
        .def("get_layer0_edges", &AdaptiveHNSWIndex::get_layer0_edges, "Return layer 0 edges")
        .def("get_upper_layer_edges", &AdaptiveHNSWIndex::get_upper_layer_edges, "Return upper layer edges")
        .def("get_active_count", &AdaptiveHNSWIndex::get_active_count, "Return count of active non-deleted vectors")
        .def("get_bridge_score", &AdaptiveHNSWIndex::get_bridge_score, py::arg("node_id"), "Return bridge score B(v) for node")
        .def("get_bridge_scores", &AdaptiveHNSWIndex::get_bridge_scores, "Return all bridge scores")
        .def("get_average_bridge_score", &AdaptiveHNSWIndex::get_average_bridge_score, "Return average bridge score across active nodes")
        .def("get_bridge_preservation_ratio", &AdaptiveHNSWIndex::get_bridge_preservation_ratio, py::arg("threshold") = 0.5f, "Return connectivity preservation ratio for bridge nodes")
        .def("update_bridge_scores", &AdaptiveHNSWIndex::update_bridge_scores, "Recompute bridge scores for all nodes")
        .def("allocate_mixed_precision", &AdaptiveHNSWIndex::allocate_mixed_precision,
             py::arg("strategy") = AllocationStrategy::CombinedSensitivity,
             py::arg("budget_ratio") = 0.5,
             "Allocate resource-budgeted mixed precisions (SQ4/SQ8/FP32)")
        .def("get_precision_distribution", &AdaptiveHNSWIndex::get_precision_distribution, "Return precision distribution and memory stats")
        .def("get_vector_precision", &AdaptiveHNSWIndex::get_vector_precision, py::arg("node_id"), "Return precision of node")
        .def("is_mixed_precision_active", &AdaptiveHNSWIndex::is_mixed_precision_active, "Return whether mixed-precision storage is active")
        .def_readonly("num_elements", &AdaptiveHNSWIndex::num_elements)
        .def_readonly("dim", &AdaptiveHNSWIndex::dim)
        .def_readwrite("policy", &AdaptiveHNSWIndex::policy);

    // Geometry Budget Allocator (Milestone I)
    py::class_<GeometrySignal>(m, "GeometrySignal")
        .def(py::init<>())
        .def_readwrite("lid", &GeometrySignal::lid)
        .def_readwrite("density", &GeometrySignal::density)
        .def_readwrite("sparsity", &GeometrySignal::sparsity)
        .def_readonly("standardized_lid", &GeometrySignal::standardized_lid)
        .def_readonly("standardized_sparsity", &GeometrySignal::standardized_sparsity)
        .def_readonly("combined_score", &GeometrySignal::combined_score);

    py::class_<BudgetAllocator>(m, "BudgetAllocator")
        .def_static("standardize_signals", &BudgetAllocator::standardize_signals,
                    py::arg("signals"), py::arg("alpha_lid") = 0.5f, py::arg("beta_spars") = 0.5f)
        .def_static("allocate_uniform", &BudgetAllocator::allocate_uniform,
                    py::arg("n"), py::arg("m_base") = 16)
        .def_static("allocate_original_adaptive", &BudgetAllocator::allocate_original_adaptive,
                    py::arg("signals"), py::arg("m_base") = 16, py::arg("m_min") = 8, py::arg("m_max") = 24, py::arg("beta") = 1.0f)
        .def_static("largest_remainder_discretization", &BudgetAllocator::largest_remainder_discretization,
                    py::arg("continuous_m"), py::arg("total_budget"), py::arg("m_min"), py::arg("m_max"))
        .def_static("allocate_normalized_linear", &BudgetAllocator::allocate_normalized_linear,
                    py::arg("signals"), py::arg("m_base") = 16, py::arg("m_min") = 8, py::arg("m_max") = 24, py::arg("beta") = 1.0f)
        .def_static("allocate_normalized_exponential", &BudgetAllocator::allocate_normalized_exponential,
                    py::arg("signals"), py::arg("m_base") = 16, py::arg("m_min") = 8, py::arg("m_max") = 24, py::arg("beta") = 0.5f)
        .def_static("apply_normalized_exponential", &BudgetAllocator::apply_normalized_exponential,
                    py::arg("index"), py::arg("signals"), py::arg("m_base") = 16, py::arg("m_min") = 8, py::arg("m_max") = 24, py::arg("beta") = 0.5f)
        .def_static("allocate_quantile", &BudgetAllocator::allocate_quantile,
                    py::arg("signals"), py::arg("m_base") = 16)
        .def_static("allocate_randomized", &BudgetAllocator::allocate_randomized,
                    py::arg("base_allocation"), py::arg("seed") = 42)
        .def_static("allocate_reversed", &BudgetAllocator::allocate_reversed,
                    py::arg("base_allocation"), py::arg("signals"));

    // Controlled Replacement & ServingIndexManager (Milestone III)
    py::enum_<ReplacementStatus>(m, "ReplacementStatus")
        .value("Success", ReplacementStatus::Success)
        .value("RejectedSlaViolation", ReplacementStatus::RejectedSlaViolation)
        .value("RejectedPeakMemoryViolation", ReplacementStatus::RejectedPeakMemoryViolation)
        .value("RejectedGraphInvariantViolation", ReplacementStatus::RejectedGraphInvariantViolation)
        .export_values();

    py::class_<CandidateSnapshot>(m, "CandidateSnapshot")
        .def_readonly("index_copy", &CandidateSnapshot::index_copy)
        .def_readonly("label_to_id", &CandidateSnapshot::label_to_id)
        .def_readonly("snapshot_seq", &CandidateSnapshot::snapshot_seq);

    py::class_<ReplacementResult>(m, "ReplacementResult")
        .def(py::init<>())
        .def_readonly("status", &ReplacementResult::status)
        .def_readonly("details", &ReplacementResult::details)
        .def_readonly("peak_memory_mb", &ReplacementResult::peak_memory_mb)
        .def_readonly("steady_state_memory_mb", &ReplacementResult::steady_state_memory_mb)
        .def_readonly("manager_overhead_mb", &ReplacementResult::manager_overhead_mb)
        .def_readonly("candidate_val_recall", &ReplacementResult::candidate_val_recall)
        .def_readonly("candidate_val_ci_lower", &ReplacementResult::candidate_val_ci_lower)
        .def_readonly("candidate_val_ci_upper", &ReplacementResult::candidate_val_ci_upper)
        .def_readonly("candidate_p99_latency_ms", &ReplacementResult::candidate_p99_latency_ms)
        .def_readonly("baseline_val_recall", &ReplacementResult::baseline_val_recall)
        .def_readonly("compaction_recall_loss", &ReplacementResult::compaction_recall_loss)
        .def_readonly("exclusive_lock_pause_us", &ReplacementResult::exclusive_lock_pause_us)
        .def_readonly("replayed_mutations_count", &ReplacementResult::replayed_mutations_count);

    py::class_<ServingIndexManager>(m, "ServingIndexManager")
        .def(py::init<std::shared_ptr<AdaptiveHNSWIndex>>(), py::arg("initial_index"))
        .def(py::init<std::shared_ptr<AdaptiveHNSWIndex>, const std::vector<uint64_t>&>(),
             py::arg("initial_index"), py::arg("labels"))
        .def("get_serving_index", &ServingIndexManager::get_serving_index)
        .def("insert", [](ServingIndexManager& self, uint64_t label, py::array_t<float, py::array::c_style | py::array::forcecast> vec) {
            py::buffer_info info = vec.request();
            if (info.ndim != 1) throw std::runtime_error("Vector must be a 1D float32 array");
            auto idx = self.get_serving_index();
            if (idx && static_cast<size_t>(info.shape[0]) != idx->dim) {
                throw std::runtime_error("Vector dimension mismatch");
            }
            return self.insert(label, static_cast<const float*>(info.ptr));
        }, py::arg("label"), py::arg("vec"))
        .def("mark_deleted", &ServingIndexManager::mark_deleted, py::arg("label"))
        .def("has_label", &ServingIndexManager::has_label, py::arg("label"))
        .def("get_vector", [](const ServingIndexManager& self, uint64_t label) -> py::object {
            auto idx = self.get_serving_index();
            if (!idx) return py::none();
            std::vector<float> buf(idx->dim);
            if (!self.get_vector(label, buf.data())) return py::none();
            return py::array_t<float>(buf.size(), buf.data());
        }, py::arg("label"))
        .def("get_manager_overhead_bytes", &ServingIndexManager::get_manager_overhead_bytes)
        .def("get_active_label_count", &ServingIndexManager::get_active_label_count)
        .def("get_active_labels", &ServingIndexManager::get_active_labels)
        .def("create_candidate_snapshot", &ServingIndexManager::create_candidate_snapshot)
        .def("release_candidate_snapshot", &ServingIndexManager::release_candidate_snapshot, py::arg("snapshot_seq"))
        .def("evaluate_and_replace", [](
            ServingIndexManager& self,
            std::shared_ptr<AdaptiveHNSWIndex> candidate,
            py::array_t<float, py::array::c_style | py::array::forcecast> val_queries,
            py::array_t<int, py::array::c_style | py::array::forcecast> val_gt,
            float target_recall,
            double max_memory_mb,
            float target_p99_latency_ms,
            int k,
            int ef_search,
            uint64_t snapshot_seq,
            std::unordered_map<uint64_t, tableint> candidate_label_to_id,
            std::vector<tableint> compaction_id_map,
            bool enforce_conservative_ci,
            double max_compaction_loss,
            double baseline_recall
        ) {
            py::buffer_info q_info = val_queries.request();
            py::buffer_info gt_info = val_gt.request();
            if (q_info.ndim != 2 || gt_info.ndim != 2) {
                throw std::runtime_error("val_queries and val_gt must be 2D arrays");
            }
            size_t n_q = q_info.shape[0];
            size_t gt_k = gt_info.shape[1];
            std::vector<std::vector<tableint>> gt(n_q, std::vector<tableint>(gt_k));
            const int* gt_ptr = static_cast<const int*>(gt_info.ptr);
            for (size_t i = 0; i < n_q; ++i) {
                for (size_t j = 0; j < gt_k; ++j) {
                    gt[i][j] = static_cast<tableint>(gt_ptr[i * gt_k + j]);
                }
            }
            return self.evaluate_and_replace(
                std::move(candidate),
                static_cast<const float*>(q_info.ptr),
                n_q,
                gt,
                target_recall,
                max_memory_mb,
                target_p99_latency_ms,
                k,
                ef_search,
                snapshot_seq,
                std::move(candidate_label_to_id),
                compaction_id_map,
                enforce_conservative_ci,
                max_compaction_loss,
                baseline_recall
            );
        }, py::arg("candidate"), py::arg("val_queries"), py::arg("val_gt"),
           py::arg("target_recall") = 0.0f, py::arg("max_memory_mb") = 0.0,
           py::arg("target_p99_latency_ms") = 0.0f, py::arg("k") = 10, py::arg("ef_search") = 64,
           py::arg("snapshot_seq") = 0,
           py::arg("candidate_label_to_id") = std::unordered_map<uint64_t, tableint>{},
           py::arg("compaction_id_map") = std::vector<tableint>{},
           py::arg("enforce_conservative_ci") = true,
           py::arg("max_compaction_loss") = 0.0020,
           py::arg("baseline_recall") = -1.0);

    // Auto-Calibration Enums & Structures
    py::enum_<FeasibilityStatus>(m, "FeasibilityStatus")
        .value("VerifiedFeasible", FeasibilityStatus::VerifiedFeasible)
        .value("ProvisionallyFeasible", FeasibilityStatus::ProvisionallyFeasible)
        .value("ValidationFailed", FeasibilityStatus::ValidationFailed)
        .value("InfeasibleWithinSearchSpace", FeasibilityStatus::InfeasibleWithinSearchSpace)
        .value("InvalidRequirements", FeasibilityStatus::InvalidRequirements)
        .value("EvaluationError", FeasibilityStatus::EvaluationError)
        .export_values();

    py::enum_<OptimizationGoal>(m, "OptimizationGoal")
        .value("MaxThroughput", OptimizationGoal::MaxThroughput)
        .value("MinMemory", OptimizationGoal::MinMemory)
        .value("MinLatency", OptimizationGoal::MinLatency)
        .value("MaxRecall", OptimizationGoal::MaxRecall)
        .export_values();

    py::class_<CalibrationConstraints>(m, "CalibrationConstraints")
        .def(py::init<>())
        .def_readwrite("target_recall", &CalibrationConstraints::target_recall)
        .def_readwrite("max_memory_mb", &CalibrationConstraints::max_memory_mb)
        .def_readwrite("target_p99_latency_ms", &CalibrationConstraints::target_p99_latency_ms)
        .def_readwrite("max_calibration_duration_s", &CalibrationConstraints::max_calibration_duration_s)
        .def_readwrite("max_construction_time_s", &CalibrationConstraints::max_construction_time_s)
        .def_readwrite("threads", &CalibrationConstraints::threads)
        .def_readwrite("k", &CalibrationConstraints::k)
        .def_readwrite("optimization_goal", &CalibrationConstraints::optimization_goal)
        .def_readwrite("sample_construction_if_large", &CalibrationConstraints::sample_construction_if_large)
        .def_readwrite("sample_size_cap", &CalibrationConstraints::sample_size_cap)
        .def_readwrite("require_conservative_recall", &CalibrationConstraints::require_conservative_recall)
        .def_readwrite("enable_adaptive_geometry", &CalibrationConstraints::enable_adaptive_geometry);

    py::class_<CalibrationTrial>(m, "CalibrationTrial")
        .def(py::init<>())
        .def_readonly("trial_id", &CalibrationTrial::trial_id)
        .def_readonly("stage", &CalibrationTrial::stage)
        .def_readonly("m", &CalibrationTrial::m)
        .def_readonly("ef_construction", &CalibrationTrial::ef_construction)
        .def_readonly("ef_search", &CalibrationTrial::ef_search)
        .def_readonly("precision", &CalibrationTrial::precision)
        .def_readonly("enable_adaptive", &CalibrationTrial::enable_adaptive)
        .def_readonly("enable_ada_ef", &CalibrationTrial::enable_ada_ef)
        .def_readonly("memory_mb", &CalibrationTrial::memory_mb)
        .def_readonly("recall_mean", &CalibrationTrial::recall_mean)
        .def_readonly("recall_ci_lower", &CalibrationTrial::recall_ci_lower)
        .def_readonly("recall_ci_upper", &CalibrationTrial::recall_ci_upper)
        .def_readonly("qps", &CalibrationTrial::qps)
        .def_readonly("p99_latency_ms", &CalibrationTrial::p99_latency_ms)
        .def_readonly("trial_duration_s", &CalibrationTrial::trial_duration_s)
        .def_readonly("is_feasible", &CalibrationTrial::is_feasible)
        .def_readonly("conservative_feasible", &CalibrationTrial::conservative_feasible)
        .def_readonly("status", &CalibrationTrial::status)
        .def_readonly("rejection_reason", &CalibrationTrial::rejection_reason)
        .def("to_json", &CalibrationTrial::to_json);

    py::class_<CalibrationReport>(m, "CalibrationReport")
        .def(py::init<>())
        .def_readonly("is_feasible", &CalibrationReport::is_feasible)
        .def_readonly("status", &CalibrationReport::status)
        .def_readonly("violated_constraints", &CalibrationReport::violated_constraints)
        .def_readonly("selected_config", &CalibrationReport::selected_config)
        .def_readonly("evaluated_trials", &CalibrationReport::evaluated_trials)
        .def_readonly("pareto_frontier", &CalibrationReport::pareto_frontier)
        .def_readonly("fallback_pareto_set", &CalibrationReport::fallback_pareto_set)
        .def_readonly("total_calibration_time_s", &CalibrationReport::total_calibration_time_s)
        .def_readonly("optimizer_overhead_ms", &CalibrationReport::optimizer_overhead_ms)
        .def_readonly("num_calibration_queries", &CalibrationReport::num_calibration_queries)
        .def_readonly("num_validation_queries", &CalibrationReport::num_validation_queries)
        .def_readonly("dataset_name", &CalibrationReport::dataset_name)
        .def("to_json", &CalibrationReport::to_json)
        .def("summary", &CalibrationReport::summary)
        .def("__repr__", &CalibrationReport::summary);

    m.def("calibrate_index", [](
        py::array_t<float, py::array::c_style | py::array::forcecast> base_vectors,
        SpaceType space,
        py::array_t<float, py::array::c_style | py::array::forcecast> query_vectors,
        py::array_t<int, py::array::c_style | py::array::forcecast> ground_truth,
        const CalibrationConstraints& constraints,
        const std::string& dataset_name
    ) {
        py::buffer_info b_info = base_vectors.request();
        py::buffer_info q_info = query_vectors.request();
        py::buffer_info gt_info = ground_truth.request();
        
        if (b_info.ndim != 2) throw std::runtime_error("base_vectors must be 2D array");
        if (q_info.ndim != 2) throw std::runtime_error("query_vectors must be 2D array");
        if (gt_info.ndim != 2) throw std::runtime_error("ground_truth must be 2D array");
        
        size_t n_vectors = b_info.shape[0];
        size_t dim = b_info.shape[1];
        size_t n_queries = q_info.shape[0];
        size_t gt_k = gt_info.shape[1];
        
        std::vector<std::vector<tableint>> gt(n_queries, std::vector<tableint>(gt_k));
        const int* gt_ptr = static_cast<const int*>(gt_info.ptr);
        for (size_t i = 0; i < n_queries; ++i) {
            for (size_t j = 0; j < gt_k; ++j) {
                gt[i][j] = static_cast<tableint>(gt_ptr[i * gt_k + j]);
            }
        }
        
        std::unique_ptr<AdaptiveHNSWIndex> index = nullptr;
        const float* b_ptr = static_cast<const float*>(b_info.ptr);
        const float* q_ptr = static_cast<const float*>(q_info.ptr);
        CalibrationReport report;
        {
            py::gil_scoped_release release;
            report = AutoCalibrator::calibrate(
                b_ptr, n_vectors, dim, space,
                q_ptr, n_queries, gt,
                constraints, index, dataset_name
            );
        }
        std::shared_ptr<AdaptiveHNSWIndex> shared_index(std::move(index));
        return std::make_pair(std::move(report), std::move(shared_index));
    }, py::arg("base_vectors"), py::arg("space"), py::arg("query_vectors"), py::arg("ground_truth"),
       py::arg("constraints"), py::arg("dataset_name") = "Unknown",
       "Run hierarchical auto-calibration and return (CalibrationReport, AdaptiveHNSWIndex)");

    m.def("calibrate_search", [](
        AdaptiveHNSWIndex& existing_index,
        py::array_t<float, py::array::c_style | py::array::forcecast> query_vectors,
        py::array_t<int, py::array::c_style | py::array::forcecast> ground_truth,
        const CalibrationConstraints& constraints,
        const std::string& dataset_name
    ) {
        py::buffer_info q_info = query_vectors.request();
        py::buffer_info gt_info = ground_truth.request();
        if (q_info.ndim != 2) throw std::runtime_error("query_vectors must be 2D array");
        if (gt_info.ndim != 2) throw std::runtime_error("ground_truth must be 2D array");

        size_t n_queries = q_info.shape[0];
        size_t gt_k = gt_info.shape[1];
        std::vector<std::vector<tableint>> gt(n_queries, std::vector<tableint>(gt_k));
        const int* gt_ptr = static_cast<const int*>(gt_info.ptr);
        for (size_t i = 0; i < n_queries; ++i) {
            for (size_t j = 0; j < gt_k; ++j) {
                gt[i][j] = static_cast<tableint>(gt_ptr[i * gt_k + j]);
            }
        }
        const float* q_ptr = static_cast<const float*>(q_info.ptr);
        py::gil_scoped_release release;
        return AutoCalibrator::calibrate_search(
            existing_index, q_ptr, n_queries, gt, constraints, dataset_name
        );
    }, py::arg("existing_index"), py::arg("query_vectors"), py::arg("ground_truth"),
       py::arg("constraints"), py::arg("dataset_name") = "PrebuiltGraph",
       "Recalibrate search parameters on an existing AdaptiveHNSWIndex");
}
