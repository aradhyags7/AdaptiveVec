"""
Stagnation Early-Exit Patience (p) Sweep on SIFT-100K.

Tests p ∈ {3, 4, 5, 6, 8, 10} to map the recall-throughput tradeoff curve
of the distance stagnation early-exit mechanism.
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


def run_patience_sweep():
    print("=" * 70)
    print("STAGNATION PATIENCE (p) SWEEP - SIFT-100K")
    print("=" * 70)

    patience_values = [3, 4, 5, 6, 8, 10]
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

    # Build index ONCE with Steps 1-4 (no early exit), then test different patience values at query time
    print("\nBuilding AdaptiveVec index (Steps 1-4: Dynamic M + Layer Scaling + Hubness)...")
    config = AdaptivePolicyConfig(
        policy_type="continuous",
        m_base=16, m_min=8, m_max=24,
        sensitivity=0.4,
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
        early_exit=True,  # Enable early exit
        hubness_regulation=True,
        hubness_penalty_weight=0.15,
        quantize=False,
    )

    sample = data[:min(500, len(data))]
    index.calibrate(sample)

    t0 = time.perf_counter()
    for vec in data:
        index.insert(vec)
    build_time = time.perf_counter() - t0
    print(f"Build complete in {build_time:.1f}s")

    stats = index.get_stats()
    print(f"Total edges: {stats['total_edges']}, Memory: {stats['total_memory_mb']:.2f} MB")

    # Also test with early_exit=False as control
    test_configs = [("no_exit", False, 0, 0)] + [(f"p={p}", True, p, 1e-4) for p in patience_values]
    results = []

    for name, use_exit, patience, epsilon in test_configs:
        print(f"\n{'-' * 50}")
        print(f"  Testing: {name}")
        print(f"{'-' * 50}")

        # Update index parameters for this sweep point
        index.stagnation_patience = patience
        index.stagnation_epsilon = epsilon

        total_found = 0
        latencies = []
        init_comps = index.total_dist_computations

        for qi in range(n_queries):
            t0 = time.perf_counter()
            res = index.search(queries[qi], k=k, ef=ef_search, early_exit=use_exit)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            found_nodes = set(node for _, node in res[:k])
            gt_nodes = set(gt_idx[qi, :k])
            total_found += len(found_nodes.intersection(gt_nodes))

        recall = total_found / (n_queries * k)
        total_query_s = sum(latencies) / 1000.0
        qps = n_queries / total_query_s if total_query_s > 0 else 0
        dist_per_q = (index.total_dist_computations - init_comps) / n_queries

        result = {
            "config": name,
            "early_exit": use_exit,
            "patience": patience,
            "epsilon": epsilon,
            "recall_at_10": round(recall, 4),
            "qps": round(qps, 1),
            "dist_evals_per_query": round(dist_per_q, 1),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 3),
        }
        results.append(result)

        print(f"  Recall@10     = {result['recall_at_10']}")
        print(f"  QPS           = {result['qps']}")
        print(f"  Dist evals/q  = {result['dist_evals_per_query']}")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patience_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "stagnation_patience_sweep",
            "dataset": "SIFT-100K",
            "n_samples": n_samples,
            "dim": dim,
            "n_queries": n_queries,
            "k": k,
            "ef_search": ef_search,
            "build_time_sec": round(build_time, 2),
            "index_edges": stats["total_edges"],
            "index_memory_mb": round(stats["total_memory_mb"], 2),
            "patience_values_tested": patience_values,
            "results": results
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 70}")

    # Summary table
    print(f"\n{'Config':>10} | {'Recall@10':>10} | {'QPS':>8} | {'Dist/q':>8} | {'p50ms':>7} | {'p95ms':>7}")
    print("-" * 65)
    for r in results:
        print(f"{r['config']:>10} | {r['recall_at_10']:>10.4f} | {r['qps']:>8.1f} | {r['dist_evals_per_query']:>8.1f} | {r['latency_p50_ms']:>7.3f} | {r['latency_p95_ms']:>7.3f}")

    return results


if __name__ == "__main__":
    run_patience_sweep()
