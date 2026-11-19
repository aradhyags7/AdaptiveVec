/**
 * AdaptiveVec Phase 0 pybind11 C++ Extensions
 * Exposes AdaptiveHNSWIndex and PolicyConfig to Python with NumPy buffer interoperability.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "adaptive_hnsw.hpp"

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
        .def_readwrite("bridge_estimator", &PolicyConfig::bridge_estimator);

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

    // AdaptiveHNSWIndex class
    py::class_<AdaptiveHNSWIndex>(m, "AdaptiveHNSWIndex")
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
        .def_readonly("num_elements", &AdaptiveHNSWIndex::num_elements)
        .def_readonly("dim", &AdaptiveHNSWIndex::dim)
        .def_readwrite("policy", &AdaptiveHNSWIndex::policy);
}
