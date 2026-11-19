#!/usr/bin/env python3
"""
Isolated SIFT-500K Benchmark for Faiss IndexHNSWFlat.
Measures initial, loaded, peak, and steady-state memory separately.
"""

import os
import sys
import time
import json
import numpy as np
import faiss

# Windows memory query
def get_mem():
    import ctypes
    from ctypes import wintypes
    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]
    psapi = ctypes.windll.psapi
    kernel32 = ctypes.windll.kernel32
    hProcess = kernel32.GetCurrentProcess()
    pmc = PROCESS_MEMORY_COUNTERS_EX()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    if psapi.GetProcessMemoryInfo(hProcess, ctypes.byref(pmc), pmc.cb):
        return {
            "working_set_mb": pmc.WorkingSetSize / (1024.0 * 1024.0),
            "peak_working_set_mb": pmc.PeakWorkingSetSize / (1024.0 * 1024.0),
            "private_mb": pmc.PrivateUsage / (1024.0 * 1024.0)
        }
    return {"working_set_mb": 0.0, "peak_working_set_mb": 0.0, "private_mb": 0.0}

def read_fvecs(filename, max_vectors=None):
    with open(filename, "rb") as f:
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0)
        record_size = 4 + dim * 4
        file_size = os.path.getsize(filename)
        total_vectors = file_size // record_size
        count = total_vectors if max_vectors is None else min(total_vectors, max_vectors)
        raw = f.read(count * record_size)
        raw_arr = np.frombuffer(raw, dtype=np.float32).reshape(count, dim + 1)
        return raw_arr[:, 1:].copy()

def read_ivecs(filename, max_records=None):
    with open(filename, "rb") as f:
        k = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0)
        record_size = 4 + k * 4
        file_size = os.path.getsize(filename)
        total_records = file_size // record_size
        count = total_records if max_records is None else min(total_records, max_records)
        raw = f.read(count * record_size)
        raw_arr = np.frombuffer(raw, dtype=np.int32).reshape(count, k + 1)
        return raw_arr[:, 1:].astype(np.uint32).copy()

def compute_recall(results_indices, ground_truth, k=10):
    total_matches = 0
    n_queries = len(results_indices)
    for q in range(n_queries):
        gt_set = set(ground_truth[q][:k])
        matches = len(gt_set.intersection(set(results_indices[q][:k])))
        total_matches += matches
    return total_matches / (n_queries * k)

def main():
    m_init = get_mem()
    print("=================================================================")
    print("  AdaptiveVec Phase 1: Isolated SIFT-500K Benchmark [FAISS]      ")
    print("=================================================================\n")
    print(f"[*] Process Startup -> WorkingSet: {m_init['working_set_mb']:.1f} MB, Private: {m_init['private_mb']:.1f} MB")

    base = read_fvecs("data/sift_base_1m.fvecs", 500000)
    queries = read_fvecs("data/sift_query.fvecs", 1000)
    gt = read_ivecs("data/sift_groundtruth_500k.ivecs", 1000)

    m_loaded = get_mem()
    dim = base.shape[1]
    n_samples = len(base)
    n_queries = len(queries)
    print(f"[+] Loaded Dataset: N={n_samples:,}, D={dim}")
    print(f"[*] Memory After Load -> WorkingSet: {m_loaded['working_set_mb']:.1f} MB, Private: {m_loaded['private_mb']:.1f} MB")

    index = faiss.IndexHNSWFlat(dim, 16)
    index.hnsw.efConstruction = 200

    print("[*] Building Faiss IndexHNSWFlat (M=16, efC=200) ...")
    t0 = time.time()
    index.add(base)
    build_time = time.time() - t0
    print(f"[+] Build Time: {build_time:.2f} s")

    # Free base buffer
    del base
    import gc
    gc.collect()

    m_post_build = get_mem()
    print(f"[*] Memory After Build (caller buffer freed) -> WorkingSet: {m_post_build['working_set_mb']:.1f} MB, PeakWorkingSet: {m_post_build['peak_working_set_mb']:.1f} MB, Private: {m_post_build['private_mb']:.1f} MB\n")

    ef_search_vals = [16, 32, 64, 128, 256]
    evaluations = []

    for ef in ef_search_vals:
        index.hnsw.efSearch = ef
        # Warmup
        index.search(queries[:100], 10)

        t0 = time.time()
        dists, idxs = index.search(queries, 10)
        q_time = time.time() - t0
        qps = n_queries / q_time
        rec = compute_recall(idxs, gt, 10)

        # Latencies
        lats = []
        for q in range(n_queries):
            q_v = queries[q:q+1]
            l0 = time.perf_counter()
            index.search(q_v, 10)
            l1 = time.perf_counter()
            lats.append((l1 - l0) * 1e6)
        lats.sort()
        p50 = lats[int(n_queries * 0.50)]
        p95 = lats[int(n_queries * 0.95)]
        p99 = lats[int(n_queries * 0.99)]
        mean_lat = sum(lats) / len(lats)

        print(f"  ef={ef:3d} -> Recall@10: {rec*100:6.2f}%, MT-QPS: {qps:8.1f}, Latency (μs): mean={mean_lat:.1f}, p50={p50:.1f}, p95={p95:.1f}, p99={p99:.1f}")
        evaluations.append({
            "ef_search": ef,
            "recall_at_10": float(rec),
            "qps": float(qps),
            "latency_mean_us": float(mean_lat),
            "latency_p50_us": float(p50),
            "latency_p95_us": float(p95),
            "latency_p99_us": float(p99)
        })

    report = {
        "engine": "faiss_hnsw",
        "n_samples": n_samples,
        "dim": dim,
        "build_time_sec": build_time,
        "initial_working_set_mb": m_init["working_set_mb"],
        "initial_private_mb": m_init["private_mb"],
        "dataset_loaded_working_set_mb": m_loaded["working_set_mb"],
        "peak_construction_working_set_mb": m_post_build["peak_working_set_mb"],
        "steady_state_working_set_mb": m_post_build["working_set_mb"],
        "steady_state_private_mb": m_post_build["private_mb"],
        "evaluations": evaluations
    }

    out_path = "experiments/sift500k_isolated_faiss.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Saved isolated FAISS benchmark report to: {out_path}")

if __name__ == "__main__":
    main()
