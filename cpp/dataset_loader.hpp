#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>

namespace adaptivevec {

struct VectorDataset {
    std::string name;
    size_t n_samples = 0;
    size_t dim = 0;
    size_t n_queries = 0;
    std::vector<float> data;
    std::vector<float> queries;
    std::vector<std::vector<uint32_t>> ground_truth;
};

// Fast fvecs loader: reads vectors of floats with 4-byte dimension header
inline bool load_fvecs(const std::string& filepath, std::vector<float>& out_data, size_t& out_n, size_t& out_dim, size_t max_vectors = 0) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    int32_t d = 0;
    file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    if (file.gcount() != sizeof(int32_t) || d <= 0) return false;

    out_dim = static_cast<size_t>(d);

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t vec_size_bytes = sizeof(int32_t) + out_dim * sizeof(float);
    size_t total_vectors = file_size / vec_size_bytes;

    out_n = (max_vectors > 0 && max_vectors < total_vectors) ? max_vectors : total_vectors;
    out_data.resize(out_n * out_dim);

    file.seekg(0, std::ios::beg);
    for (size_t i = 0; i < out_n; ++i) {
        int32_t cur_d = 0;
        file.read(reinterpret_cast<char*>(&cur_d), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&out_data[i * out_dim]), out_dim * sizeof(float));
    }

    return true;
}

// Fast ivecs loader: reads ground truth neighbor IDs with 4-byte count header
inline bool load_ivecs(const std::string& filepath, std::vector<std::vector<uint32_t>>& out_gt, size_t max_queries = 0) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    int32_t k = 0;
    file.read(reinterpret_cast<char*>(&k), sizeof(int32_t));
    if (file.gcount() != sizeof(int32_t) || k <= 0) return false;

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t vec_size_bytes = sizeof(int32_t) + k * sizeof(int32_t);
    size_t total_queries = file_size / vec_size_bytes;

    size_t n_q = (max_queries > 0 && max_queries < total_queries) ? max_queries : total_queries;
    out_gt.resize(n_q, std::vector<uint32_t>(k));

    file.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n_q; ++i) {
        int32_t cur_k = 0;
        file.read(reinterpret_cast<char*>(&cur_k), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(out_gt[i].data()), k * sizeof(uint32_t));
    }

    return true;
}

// Writes fvecs file for binary caching
inline bool save_fvecs(const std::string& filepath, const float* data, size_t n, size_t dim) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    int32_t d = static_cast<int32_t>(dim);
    for (size_t i = 0; i < n; ++i) {
        file.write(reinterpret_cast<const char*>(&d), sizeof(int32_t));
        file.write(reinterpret_cast<const char*>(data + i * dim), dim * sizeof(float));
    }
    return true;
}

// Writes ivecs file for binary caching
inline bool save_ivecs(const std::string& filepath, const std::vector<std::vector<uint32_t>>& gt) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    for (size_t i = 0; i < gt.size(); ++i) {
        int32_t k = static_cast<int32_t>(gt[i].size());
        file.write(reinterpret_cast<const char*>(&k), sizeof(int32_t));
        file.write(reinterpret_cast<const char*>(gt[i].data()), k * sizeof(uint32_t));
    }
    return true;
}

} // namespace adaptivevec
