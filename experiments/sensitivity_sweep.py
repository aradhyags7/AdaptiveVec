"""
Policy Sensitivity Factor (γ) Sweep on SIFT-100K.

Tests γ ∈ {0.2, 0.4, 0.6, 0.8, 1.0} to characterize how aggressively
the adaptive policy modulates M(x) and efC(x) in response to difficulty scores.
"""

import sys
import os
import json
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptivevec.adaptive_hnsw import AdaptiveHNSW
from adaptivevec.policy import AdaptivePolicyConfig
from adaptivevec.datasets import load_or_generate_dataset, compute_ground_truth


def run_sensitivity_sweep():
    print("=" * 70)
    print("POLICY SENSITIVITY (gamma) SWEEP - SIFT-100K")
    print("=" * 70)

    gamma_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    k = 10
    ef_search = 50

    # Load SIFT-100K
    print("\nLoading SIFT-100K dataset...")
    data, queries, metadata = load_or_generate_dataset(
        dataset_name="sift", n_samples=100000, dim=128, n_queries=10000
    )

    n_samples = len(data)
    dim = data.shape[1]
    n_queries = len(queries)
    print(f"Dataset: {metadata.get('name', 'SIFT-100K')}, N={n_samples}, D={dim}, Q={n_queries}")

    # Ground truth (load precomputed if available to save 20 mins)
    gt_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sift_subset_groundtruth.ivecs")
    if os.path.exists(gt_file):
        print(f"Loading precomputed ground truth from {gt_file}...")
        from adaptivevec.datasets import read_ivecs
        gt_idx = read_ivecs(gt_file)
    else:
        print("Computing brute-force ground truth...")
        gt_idx, _ = compute_ground_truth(data, queries, k=k, space="l2")

    results = []

    for gamma in gamma_values:
        print(f"\n{'-' * 50}")
        print(f"  Testing gamma = {gamma}")
        print(f"{'-' * 50}")

        config = AdaptivePolicyConfig(
            policy_type="continuous",
            m_base=16, m_min=8, m_max=24,
            sensitivity=gamma,
            alpha_lid=0.5,
            beta_density=0.5,
            lambda_layer=0.75,
            m_min_layer=4,
        )

        index = AdaptiveHNSW(
            dim=dim,
            policy_config=config,
            space="l2",
            heuristic=True,
            early_exit=False,  # Steps 1-2 only (dynamic M + efC)
            hubness_regulation=False,
            quantize=False,
        )

        sample = data[:min(500, len(data))]
        index.calibrate(sample)

        t0 = time.perf_counter()
        for vec in data:
            index.insert(vec)
        build_time = time.perf_counter() - t0

        stats = index.get_stats()

        # M distribution
        assigned_ms = [p.m for p in index.node_params.values()]
        ef_cs = [p.ef_construction for p in index.node_params.values()]

        # Query
        total_found = 0
        latencies = []
        init_comps = index.total_dist_computations

        for qi in range(n_queries):
            t0q = time.perf_counter()
            res = index.search(queries[qi], k=k, ef=ef_search, early_exit=False)
            t1q = time.perf_counter()
            latencies.append((t1q - t0q) * 1000.0)

            found_nodes = set(node for _, node in res[:k])
            gt_nodes = set(gt_idx[qi, :k])
            total_found += len(found_nodes.intersection(gt_nodes))

        recall = total_found / (n_queries * k)
        total_query_s = sum(latencies) / 1000.0
        qps = n_queries / total_query_s if total_query_s > 0 else 0
        dist_per_q = (index.total_dist_computations - init_comps) / n_queries

        result = {
            "gamma": gamma,
            "recall_at_10": round(recall, 4),
            "qps": round(qps, 1),
            "build_time_sec": round(build_time, 2),
            "total_edges": stats["total_edges"],
            "memory_mb": round(stats["total_memory_mb"], 2),
            "dist_evals_per_query": round(dist_per_q, 1),
            "m_distribution": {
                "min": int(np.min(assigned_ms)),
                "max": int(np.max(assigned_ms)),
                "mean": round(float(np.mean(assigned_ms)), 2),
                "std": round(float(np.std(assigned_ms)), 2),
                "median": float(np.median(assigned_ms)),
            },
            "efc_distribution": {
                "min": int(np.min(ef_cs)),
                "max": int(np.max(ef_cs)),
                "mean": round(float(np.mean(ef_cs)), 2),
                "std": round(float(np.std(ef_cs)), 2),
            },
        }
        results.append(result)

        print(f"  Recall@10  = {result['recall_at_10']}")
        print(f"  QPS        = {result['qps']}")
        print(f"  Edges      = {result['total_edges']}")
        print(f"  M dist     = {result['m_distribution']}")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensitivity_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "sensitivity_gamma_sweep",
            "dataset": "SIFT-100K",
            "n_samples": n_samples,
            "dim": dim,
            "n_queries": n_queries,
            "k": k,
            "gamma_values_tested": gamma_values,
            "results": results
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 70}")

    # Summary table
    print(f"\n{'gamma':>6} | {'Recall@10':>10} | {'QPS':>8} | {'Edges':>10} | {'M mean':>7} | {'M std':>6}")
    print("-" * 60)
    for r in results:
        m = r['m_distribution']
        print(f"{r['gamma']:>6.2f} | {r['recall_at_10']:>10.4f} | {r['qps']:>8.1f} | {r['total_edges']:>10} | {m['mean']:>7.2f} | {m['std']:>6.2f}")

    return results


if __name__ == "__main__":
    run_sensitivity_sweep()
