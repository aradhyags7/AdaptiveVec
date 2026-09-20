"""
Hubness Penalty Weight (μ) Sweep on Synthetic-Multi-Cluster.

Tests μ ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.30} to characterize:
1. Whether the 0.9145→0.7821 recall drop is μ-sensitivity
2. Graph connectivity effects (reachability from entry point)
3. Degree distribution changes
"""

import sys
import os
import json
import time
import numpy as np
from collections import deque

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptivevec.adaptive_hnsw import AdaptiveHNSW
from adaptivevec.policy import AdaptivePolicyConfig
from adaptivevec.datasets import compute_ground_truth

def generate_synthetic_multi_cluster(n_samples=50000, dim=64, n_queries=1000, seed=42):
    """Reproduces the exact Synthetic-Multi-Cluster dataset from benchmark_results.json."""
    np.random.seed(seed)
    n_clusters = 8
    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples - samples_per_cluster * n_clusters

    vectors = []
    cluster_labels = []
    for c in range(n_clusters):
        n_c = samples_per_cluster + (1 if c < remainder else 0)
        # Vary spread and center per cluster
        center = np.random.uniform(-50, 50, dim)
        spread = np.random.uniform(0.5, 5.0)
        cluster_data = np.random.normal(center, spread, size=(n_c, dim)).astype(np.float32)
        vectors.append(cluster_data)
        cluster_labels.extend([c] * n_c)

    data = np.vstack(vectors)
    # Shuffle
    perm = np.random.permutation(len(data))
    data = data[perm]
    cluster_labels = [cluster_labels[i] for i in perm]

    # Generate queries near cluster centers
    queries = []
    for _ in range(n_queries):
        c = np.random.randint(0, n_clusters)
        center = np.random.uniform(-50, 50, dim)
        q = np.random.normal(center, 2.0, dim).astype(np.float32)
        queries.append(q)
    queries = np.array(queries)

    return data, queries, cluster_labels


def measure_reachability(index, entry_point, layer=0):
    """BFS from entry point to measure fraction of reachable nodes."""
    if entry_point is None or layer >= len(index.graphs):
        return 0.0
    graph = index.graphs[layer]
    total_nodes = len(graph)
    if total_nodes == 0:
        return 0.0

    visited = set()
    queue = deque([entry_point])
    visited.add(entry_point)

    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) / total_nodes


def run_hubness_sweep():
    print("=" * 70)
    print("HUBNESS PENALTY WEIGHT (mu) SWEEP - Synthetic-Multi-Cluster")
    print("=" * 70)

    mu_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    n_samples = 50000
    dim = 64
    n_queries = 1000
    k = 10

    print(f"\nGenerating dataset: N={n_samples}, D={dim}, Q={n_queries}")
    data, queries, cluster_labels = generate_synthetic_multi_cluster(n_samples, dim, n_queries)

    # Compute ground truth
    print("Computing brute-force ground truth...")
    gt_idx, gt_dist = compute_ground_truth(data, queries, k=k, space="l2")

    results = []

    for mu in mu_values:
        print(f"\n{'-' * 50}")
        print(f"  Testing mu = {mu}")
        print(f"{'-' * 50}")

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
            early_exit=False,  # Steps 1-4 only, no early exit
            hubness_regulation=(mu > 0),
            hubness_penalty_weight=mu,
            quantize=False,
        )

        # Calibrate
        sample = data[:min(500, len(data))]
        index.calibrate(sample)

        # Build
        t0 = time.perf_counter()
        for vec in data:
            index.insert(vec)
        build_time = time.perf_counter() - t0

        stats = index.get_stats()

        # Measure reachability
        reachability = measure_reachability(index, index.enter_point, layer=0)

        # Query
        total_found = 0
        latencies = []
        init_comps = index.total_dist_computations

        for qi in range(n_queries):
            t0 = time.perf_counter()
            res = index.search(queries[qi], k=k, ef=50, early_exit=False)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            found_nodes = set(node for _, node in res[:k])
            gt_nodes = set(gt_idx[qi, :k])
            total_found += len(found_nodes.intersection(gt_nodes))

        recall = total_found / (n_queries * k)
        total_query_s = sum(latencies) / 1000.0
        qps = n_queries / total_query_s if total_query_s > 0 else 0
        dist_per_q = (index.total_dist_computations - init_comps) / n_queries

        # Degree distribution
        l0_degrees = [len(n) for n in index.graphs[0].values()]

        result = {
            "mu": mu,
            "recall_at_10": round(recall, 4),
            "qps": round(qps, 1),
            "build_time_sec": round(build_time, 2),
            "total_edges": stats["total_edges"],
            "memory_mb": round(stats["total_memory_mb"], 2),
            "reachability_pct": round(reachability * 100, 2),
            "dist_evals_per_query": round(dist_per_q, 1),
            "degree_min": int(np.min(l0_degrees)) if l0_degrees else 0,
            "degree_max": int(np.max(l0_degrees)) if l0_degrees else 0,
            "degree_median": float(np.median(l0_degrees)) if l0_degrees else 0,
            "degree_mean": round(float(np.mean(l0_degrees)), 2) if l0_degrees else 0,
            "degree_std": round(float(np.std(l0_degrees)), 2) if l0_degrees else 0,
        }
        results.append(result)

        print(f"  Recall@10 = {result['recall_at_10']}")
        print(f"  QPS       = {result['qps']}")
        print(f"  Edges     = {result['total_edges']}")
        print(f"  Reachable = {result['reachability_pct']}%")
        print(f"  Degree    = {result['degree_median']} median, [{result['degree_min']}, {result['degree_max']}] range")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hubness_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "hubness_penalty_weight_sweep",
            "dataset": "Synthetic-Multi-Cluster",
            "n_samples": n_samples,
            "dim": dim,
            "n_queries": n_queries,
            "k": k,
            "mu_values_tested": mu_values,
            "results": results
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 70}")

    # Summary table
    print(f"\n{'mu':>6} | {'Recall@10':>10} | {'QPS':>8} | {'Reachable%':>10} | {'Edges':>10} | {'Deg Median':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['mu']:>6.2f} | {r['recall_at_10']:>10.4f} | {r['qps']:>8.1f} | {r['reachability_pct']:>9.2f}% | {r['total_edges']:>10} | {r['degree_median']:>10.1f}")

    return results


if __name__ == "__main__":
    run_hubness_sweep()
