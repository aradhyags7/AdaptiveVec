"""
Synthetic-Multi-Cluster Dataset Characterization.

Demonstrates the heterogeneity (cluster distributions, LID/density variations)
that AdaptiveVec is designed to exploit, rather than presenting the dataset
as a black-box benchmark.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptivevec.signals import compute_mle_lid, compute_local_density


def generate_synthetic_multi_cluster(n_samples=50000, dim=64, seed=42):
    """Reproduces the Synthetic-Multi-Cluster dataset with cluster labels."""
    np.random.seed(seed)
    n_clusters = 8
    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples - samples_per_cluster * n_clusters

    vectors = []
    cluster_labels = []
    cluster_params = []

    for c in range(n_clusters):
        n_c = samples_per_cluster + (1 if c < remainder else 0)
        center = np.random.uniform(-50, 50, dim)
        spread = np.random.uniform(0.5, 5.0)
        cluster_data = np.random.normal(center, spread, size=(n_c, dim)).astype(np.float32)
        vectors.append(cluster_data)
        cluster_labels.extend([c] * n_c)
        cluster_params.append({
            "cluster_id": c,
            "center_norm": float(np.linalg.norm(center)),
            "spread": float(spread),
            "n_samples": n_c,
        })

    data = np.vstack(vectors)
    # Keep track of original cluster assignments before shuffle
    return data, cluster_labels, cluster_params


def run_synthetic_analysis():
    print("=" * 70)
    print("SYNTHETIC-MULTI-CLUSTER DATASET CHARACTERIZATION")
    print("=" * 70)

    n_samples = 50000
    dim = 64
    k_neighbors = 15

    print(f"\nGenerating dataset: N={n_samples}, D={dim}")
    data, cluster_labels, cluster_params = generate_synthetic_multi_cluster(n_samples, dim)

    n_clusters = len(cluster_params)
    print(f"Clusters: {n_clusters}")

    # Per-cluster analysis
    cluster_results = []

    for c in range(n_clusters):
        print(f"\n{'-' * 50}")
        print(f"  Cluster {c}")
        print(f"{'-' * 50}")

        # Get cluster indices
        indices = [i for i, l in enumerate(cluster_labels) if l == c]
        cluster_data = data[indices]
        n_c = len(cluster_data)

        # Sample for LID/density estimation
        sample_size = min(200, n_c)
        sample_indices = np.random.choice(n_c, size=sample_size, replace=False)

        lids = []
        densities = []

        for si in sample_indices:
            point = cluster_data[si]
            # Compute distances to other cluster members
            diffs = cluster_data - point
            dists = np.linalg.norm(diffs, axis=1)
            dists = dists[dists > 1e-9]  # Remove self
            if len(dists) >= k_neighbors:
                sorted_dists = np.sort(dists)[:k_neighbors]
                lid = compute_mle_lid(sorted_dists)
                density = compute_local_density(sorted_dists)
                lids.append(lid)
                densities.append(density)

        lids = np.array(lids)
        densities = np.array(densities)

        # Inter-cluster distance to nearest other cluster center
        cluster_center = np.mean(cluster_data, axis=0)
        inter_dists = []
        for c2 in range(n_clusters):
            if c2 != c:
                c2_indices = [i for i, l in enumerate(cluster_labels) if l == c2]
                c2_center = np.mean(data[c2_indices], axis=0)
                inter_dists.append(float(np.linalg.norm(cluster_center - c2_center)))

        result = {
            "cluster_id": c,
            "n_samples": n_c,
            "spread": cluster_params[c]["spread"],
            "lid": {
                "mean": round(float(np.mean(lids)), 4),
                "std": round(float(np.std(lids)), 4),
                "min": round(float(np.min(lids)), 4),
                "max": round(float(np.max(lids)), 4),
                "q25": round(float(np.percentile(lids, 25)), 4),
                "q50": round(float(np.percentile(lids, 50)), 4),
                "q75": round(float(np.percentile(lids, 75)), 4),
            },
            "density": {
                "mean": round(float(np.mean(densities)), 4),
                "std": round(float(np.std(densities)), 4),
                "min": round(float(np.min(densities)), 4),
                "max": round(float(np.max(densities)), 4),
            },
            "inter_cluster_dist_min": round(float(np.min(inter_dists)), 2),
            "inter_cluster_dist_mean": round(float(np.mean(inter_dists)), 2),
        }
        cluster_results.append(result)

        print(f"  N = {n_c}, Spread = {cluster_params[c]['spread']:.2f}")
        print(f"  LID:     mean={result['lid']['mean']:.2f}, std={result['lid']['std']:.2f}, range=[{result['lid']['min']:.2f}, {result['lid']['max']:.2f}]")
        print(f"  Density: mean={result['density']['mean']:.4f}, std={result['density']['std']:.4f}")
        print(f"  Inter-cluster dist: min={result['inter_cluster_dist_min']:.2f}, mean={result['inter_cluster_dist_mean']:.2f}")

    # Global stats
    all_lids = []
    all_densities = []
    for r in cluster_results:
        all_lids.append(r["lid"]["mean"])
        all_densities.append(r["density"]["mean"])

    global_stats = {
        "lid_range_across_clusters": [round(float(np.min(all_lids)), 4), round(float(np.max(all_lids)), 4)],
        "density_range_across_clusters": [round(float(np.min(all_densities)), 4), round(float(np.max(all_densities)), 4)],
        "lid_heterogeneity_ratio": round(float(np.max(all_lids) / np.min(all_lids)), 2) if np.min(all_lids) > 0 else 0,
        "density_heterogeneity_ratio": round(float(np.max(all_densities) / np.min(all_densities)), 2) if np.min(all_densities) > 0 else 0,
    }

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "synthetic_dataset_characterization",
            "dataset": "Synthetic-Multi-Cluster",
            "n_samples": n_samples,
            "dim": dim,
            "n_clusters": n_clusters,
            "cluster_params": cluster_params,
            "per_cluster_analysis": cluster_results,
            "global_heterogeneity": global_stats,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print("GLOBAL HETEROGENEITY SUMMARY")
    print(f"{'=' * 70}")
    print(f"  LID range across clusters:     {global_stats['lid_range_across_clusters']}")
    print(f"  Density range across clusters:  {global_stats['density_range_across_clusters']}")
    print(f"  LID heterogeneity ratio:        {global_stats['lid_heterogeneity_ratio']}x")
    print(f"  Density heterogeneity ratio:    {global_stats['density_heterogeneity_ratio']}x")
    print(f"\nResults saved to: {out_path}")

    return cluster_results, global_stats


if __name__ == "__main__":
    run_synthetic_analysis()
