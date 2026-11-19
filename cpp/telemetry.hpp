#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <atomic>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <numeric>

namespace adaptivevec {

struct QueryEvent {
    uint64_t timestamp_us = 0;
    float latency_us = 0.0f;
    uint32_t distance_computations = 0;
    uint32_t visited_nodes = 0;
    uint32_t queue_size_peak = 0;
    float entry_point_distance = 0.0f;
    float top1_distance = 0.0f;
    float topk_distance = 0.0f;
    float distance_margin = 0.0f; // (topk - top1) / max(1e-6, top1)
    float query_norm = 0.0f;
    int ef_search_used = 64;
    bool early_exit_triggered = false;
};

struct TelemetrySummary {
    size_t total_queries = 0;
    float mean_latency_us = 0.0f;
    float p50_latency_us = 0.0f;
    float p95_latency_us = 0.0f;
    float p99_latency_us = 0.0f;
    float mean_distance_computations = 0.0f;
    float mean_visited_nodes = 0.0f;
    float mean_distance_margin = 0.0f;
    float mean_entry_distance = 0.0f;
    float mean_query_norm = 0.0f;
    float early_exit_rate = 0.0f;
    double current_qps = 0.0;
    double memory_mb = 0.0;
    float estimated_reference_recall = 1.0f;
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "{"
           << "\"total_queries\": " << total_queries << ", "
           << "\"mean_latency_us\": " << mean_latency_us << ", "
           << "\"p50_latency_us\": " << p50_latency_us << ", "
           << "\"p95_latency_us\": " << p95_latency_us << ", "
           << "\"p99_latency_us\": " << p99_latency_us << ", "
           << "\"mean_dist_comps\": " << mean_distance_computations << ", "
           << "\"mean_visited\": " << mean_visited_nodes << ", "
           << "\"mean_margin\": " << mean_distance_margin << ", "
           << "\"mean_entry_dist\": " << mean_entry_distance << ", "
           << "\"early_exit_rate\": " << early_exit_rate << ", "
           << "\"current_qps\": " << current_qps << ", "
           << "\"memory_mb\": " << memory_mb << ", "
           << "\"estimated_reference_recall\": " << estimated_reference_recall
           << "}";
        return ss.str();
    }
};

class AdaptiveTelemetry {
public:
    explicit AdaptiveTelemetry(size_t ring_buffer_size = 2048, bool enabled = true)
        : buffer_capacity(ring_buffer_size), is_enabled(enabled) {
        ring_buffer.resize(buffer_capacity);
    }
    
    inline bool enabled() const { return is_enabled.load(std::memory_order_relaxed); }
    inline void set_enabled(bool en) { is_enabled.store(en, std::memory_order_relaxed); }

    inline void record(const QueryEvent& event) {
        if (!is_enabled.load(std::memory_order_relaxed)) return;
        
        uint64_t idx = global_event_counter.fetch_add(1, std::memory_order_relaxed);
        size_t slot = idx % buffer_capacity;
        
        // Fast lockless slot overwrite using bounded circular buffer
        ring_buffer[slot] = event;
    }

    TelemetrySummary get_summary(double current_mem_mb = 0.0, float ref_recall = 1.0f) const {
        TelemetrySummary s;
        s.memory_mb = current_mem_mb;
        s.estimated_reference_recall = ref_recall;
        
        uint64_t total = global_event_counter.load(std::memory_order_relaxed);
        s.total_queries = total;
        if (total == 0) return s;
        
        size_t count = std::min((size_t)total, buffer_capacity);
        std::vector<float> latencies;
        latencies.reserve(count);
        
        double sum_lat = 0.0;
        double sum_dist = 0.0;
        double sum_visited = 0.0;
        double sum_margin = 0.0;
        double sum_entry = 0.0;
        double sum_norm = 0.0;
        size_t early_exits = 0;
        
        for (size_t i = 0; i < count; ++i) {
            const auto& ev = ring_buffer[i];
            latencies.push_back(ev.latency_us);
            sum_lat += ev.latency_us;
            sum_dist += ev.distance_computations;
            sum_visited += ev.visited_nodes;
            sum_margin += ev.distance_margin;
            sum_entry += ev.entry_point_distance;
            sum_norm += ev.query_norm;
            if (ev.early_exit_triggered) early_exits++;
        }
        
        s.mean_latency_us = static_cast<float>(sum_lat / count);
        s.mean_distance_computations = static_cast<float>(sum_dist / count);
        s.mean_visited_nodes = static_cast<float>(sum_visited / count);
        s.mean_distance_margin = static_cast<float>(sum_margin / count);
        s.mean_entry_distance = static_cast<float>(sum_entry / count);
        s.mean_query_norm = static_cast<float>(sum_norm / count);
        s.early_exit_rate = static_cast<float>(early_exits) / static_cast<float>(count);
        
        if (s.mean_latency_us > 0.0f) {
            s.current_qps = 1000000.0 / s.mean_latency_us;
        }
        
        std::sort(latencies.begin(), latencies.end());
        size_t p50_idx = static_cast<size_t>(0.50 * latencies.size());
        size_t p95_idx = static_cast<size_t>(0.95 * latencies.size());
        size_t p99_idx = static_cast<size_t>(0.99 * latencies.size());
        if (p95_idx >= latencies.size()) p95_idx = latencies.size() - 1;
        if (p99_idx >= latencies.size()) p99_idx = latencies.size() - 1;
        
        s.p50_latency_us = latencies[p50_idx];
        s.p95_latency_us = latencies[p95_idx];
        s.p99_latency_us = latencies[p99_idx];
        
        return s;
    }
    
    void reset() {
        global_event_counter.store(0, std::memory_order_relaxed);
    }

private:
    size_t buffer_capacity;
    std::atomic<bool> is_enabled{true};
    std::atomic<uint64_t> global_event_counter{0};
    std::vector<QueryEvent> ring_buffer;
};

} // namespace adaptivevec
