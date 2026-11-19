"""
benchmarks/run_external_baselines.py
Phase 0 Independent Baseline Benchmarks:
Evaluates:
1. Official hnswlib (via experiments/hnswlib_baseline_results.json)
2. Faiss IndexHNSWFlat
3. AdaptiveVec Canonical Baseline HNSW
4. AdaptiveVec Full Continuous Adaptive Graph
Under matched datasets (SIFT-100K), identical queries (10,000), and identical recall targets.
"""

import os
import sys
import json
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import faiss
import adaptivevec._adaptivevec_cpp as ac
DATA_DIR = os.path.join(REPO_ROOT, "data")
EXP_DIR = os.path.join(REPO_ROOT, "experiments")
os.makedirs(EXP_DIR, exist_ok=True)


def read_fvecs(filename, max_vectors=None):
    with open(filename, "rb") as f:
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0)
        file_size = os.path.getsize(filename)
        record_size = 4 + dim * 4
        total_vectors = file_size // record_size
        count = total_vectors if max_vectors is None else min(total_vectors, max_vectors)
        raw = f.read(count * record_size)
        raw_arr = np.frombuffer(raw, dtype=np.float32).reshape(count, dim + 1)
        return raw_arr[:, 1:].copy()


def read_ivecs(filename, max_records=None):
    with open(filename, "rb") as f:
        k = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0)
        file_size = os.path.getsize(filename)
        record_size = 4 + k * 4
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
    print("=================================================================")
    print("    AdaptiveVec Phase 0: Independent Baselines Benchmark        ")
    print("=================================================================\n")

    sift_100k_file = os.path.join(DATA_DIR, "sift_subset_100k.fvecs")
    query_file = os.path.join(DATA_DIR, "sift_query.fvecs")
    gt_file = os.path.join(DATA_DIR, "sift_subset_groundtruth.ivecs")

    base_100k = read_fvecs(sift_100k_file)
    queries = read_fvecs(query_file)
    gt = read_ivecs(gt_file)

    dim = base_100k.shape[1]
    n_base = len(base_100k)
    n_queries = len(queries)
    print(f"[+] Loaded SIFT-100K: N={n_base:,}, D={dim}, Q={n_queries:,}")

    ef_search_vals = [16, 32, 64, 128, 256]
    results = {}

    # 1. Faiss IndexHNSWFlat
    print("\n[*] Evaluating Faiss IndexHNSWFlat (M=16, efC=200) ...")
    faiss_index = faiss.IndexHNSWFlat(dim, 16)
    faiss_index.hnsw.efConstruction = 200
    
    t0 = time.time()
    faiss_index.add(base_100k)
    faiss_build_time = time.time() - t0
    print(f"[+] Faiss Build Time: {faiss_build_time:.2f} s")

    faiss_evals = []
    for ef in ef_search_vals:
        faiss_index.hnsw.efSearch = ef
        t0 = time.time()
        dists, idxs = faiss_index.search(queries, 10)
        q_time = time.time() - t0
        qps = n_queries / q_time
        rec = compute_recall(idxs, gt, 10)
        faiss_evals.append({"ef_search": ef, "recall_at_10": rec, "qps": qps})
        print(f"    ef={ef:3d} -> Recall@10: {rec:.4f}, QPS: {qps:8.1f}")
    
    results["faiss_hnsw"] = {
        "build_time_sec": faiss_build_time,
        "evaluations": faiss_evals
    }

    # 2. AdaptiveVec Canonical Baseline HNSW (Fixed M=16, efC=200)
    print("\n[*] Evaluating AdaptiveVec Canonical Baseline HNSW (M=16, efC=200) ...")
    cfg_base = ac.PolicyConfig()
    cfg_base.enable_dynamic_m_efc = False
    cfg_base.enable_layer_scaling = false if hasattr(ac, 'false') else False
    cfg_base.enable_hubness_regulation = False
    cfg_base.enable_ada_ef = False
    cfg_base.m_base = 16
    cfg_base.ef_c_base = 200

    idx_base = ac.AdaptiveHNSWIndex(dim, ac.SpaceType.L2, False, cfg_base)
    t0 = time.time()
    idx_base.batch_insert(base_100k)
    base_build_time = time.time() - t0
    mb_base = idx_base.get_memory_breakdown()
    print(f"[+] AdaptiveVec Baseline Build Time: {base_build_time:.2f} s (RAM: {mb_base.total_owned_bytes / (1024*1024):.1f} MB, Edges: {idx_base.get_total_edges():,})")

    base_evals = []
    for ef in ef_search_vals:
        t0 = time.time()
        cands = idx_base.batch_search(queries, k=10, ef=ef)
        q_time = time.time() - t0
        qps = n_queries / q_time
        idxs = [[c.id for c in row] for row in cands]
        rec = compute_recall(idxs, gt, 10)
        base_evals.append({"ef_search": ef, "recall_at_10": rec, "qps": qps})
        print(f"    ef={ef:3d} -> Recall@10: {rec:.4f}, QPS: {qps:8.1f}")

    results["adaptivevec_baseline"] = {
        "build_time_sec": base_build_time,
        "ram_mb": mb_base.total_owned_bytes / (1024*1024),
        "total_edges": idx_base.get_total_edges(),
        "evaluations": base_evals
    }

    # 3. AdaptiveVec Full Continuous Adaptive Graph
    print("\n[*] Evaluating AdaptiveVec Full Continuous Architecture ...")
    cfg_adapt = ac.PolicyConfig()
    cfg_adapt.enable_dynamic_m_efc = True
    cfg_adapt.enable_layer_scaling = True
    cfg_adapt.enable_hubness_regulation = True
    cfg_adapt.enable_ada_ef = True
    cfg_adapt.hubness_mu = 0.15
    cfg_adapt.m_base = 16
    cfg_adapt.ef_c_base = 120

    idx_adapt = ac.AdaptiveHNSWIndex(dim, ac.SpaceType.L2, True, cfg_adapt)
    t0 = time.time()
    idx_adapt.batch_insert(base_100k)
    adapt_build_time = time.time() - t0
    mb_adapt = idx_adapt.get_memory_breakdown()
    print(f"[+] AdaptiveVec Full Build Time: {adapt_build_time:.2f} s (RAM: {mb_adapt.total_owned_bytes / (1024*1024):.1f} MB, Edges: {idx_adapt.get_total_edges():,})")

    adapt_evals = []
    for ef in ef_search_vals:
        t0 = time.time()
        cands = idx_adapt.batch_search(queries, k=10, ef=ef)
        q_time = time.time() - t0
        qps = n_queries / q_time
        idxs = [[c.id for c in row] for row in cands]
        rec = compute_recall(idxs, gt, 10)
        adapt_evals.append({"ef_search": ef, "recall_at_10": rec, "qps": qps})
        print(f"    ef={ef:3d} -> Recall@10: {rec:.4f}, QPS: {qps:8.1f}")

    results["adaptivevec_full"] = {
        "build_time_sec": adapt_build_time,
        "ram_mb": mb_adapt.total_owned_bytes / (1024*1024),
        "total_edges": idx_adapt.get_total_edges(),
        "evaluations": adapt_evals
    }

    # 4. Load hnswlib results
    hnswlib_json = os.path.join(EXP_DIR, "hnswlib_baseline_results.json")
    if os.path.exists(hnswlib_json):
        with open(hnswlib_json, "r") as f:
            results["hnswlib"] = json.load(f)

    # Save comprehensive comparison JSON
    out_comparison = os.path.join(EXP_DIR, "external_baselines_comparison.json")
    with open(out_comparison, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved complete baseline comparison to: {out_comparison}")

    # Print Comparative Pareto Table
    print("\n==================================================================================================")
    print("               UNIFIED PARETO COMPARISON ON SIFT-100K (10,000 QUERIES, TOP-10)                   ")
    print("==================================================================================================")
    print(f"{'Engine':<28} {'Build (s)':<12} {'RAM (MB)':<12} {'Edges':<12} {'efSearch':<10} {'Recall@10':<12} {'QPS':<12}")
    print("--------------------------------------------------------------------------------------------------")
    
    # Print sample comparison for ef=64
    for engine_name, eng_data in [
        ("Official hnswlib", results.get("hnswlib", {})),
        ("Faiss IndexHNSWFlat", results.get("faiss_hnsw", {})),
        ("AdaptiveVec Baseline", results.get("adaptivevec_baseline", {})),
        ("AdaptiveVec Continuous", results.get("adaptivevec_full", {}))
    ]:
        b_time = eng_data.get("build_time_sec", 0.0)
        ram = eng_data.get("ram_mb", eng_data.get("working_set_mb", 0.0))
        edges = eng_data.get("total_edges", "N/A")
        edges_str = f"{edges:,}" if isinstance(edges, int) else str(edges)
        
        evals = eng_data.get("evaluations", [])
        for ev in evals:
            if ev["ef_search"] == 64:
                print(f"{engine_name:<28} {b_time:<12.2f} {ram:<12.1f} {edges_str:<12} {ev['ef_search']:<10} {ev['recall_at_10']:<12.4f} {ev['qps']:<12.1f}")
    print("==================================================================================================\n")


if __name__ == "__main__":
    main()
