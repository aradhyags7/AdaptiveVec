"""
LID / Difficulty Score Validation on SIFT-100K.

Validates that the adaptive difficulty score S(x) correlates with actual
search difficulty by:
1. Computing S(x) for a sample of 1000 vectors
2. Measuring actual search difficulty (distance evaluations to find true 10-NN)
3. Computing Spearman rank correlation

This is critical evidence that the signal estimation is meaningful rather
than arbitrary.
"""

import sys
import os
import json
import time
import numpy as np
from scipy import stats as scipy_stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptivevec.adaptive_hnsw import AdaptiveHNSW
from adaptivevec.policy import AdaptivePolicyConfig
from adaptivevec.signals import compute_mle_lid, compute_local_density
from adaptivevec.datasets import load_or_generate_dataset, compute_ground_truth


def run_lid_validation():
    print("=" * 70)
    print("LID / DIFFICULTY SCORE VALIDATION - SIFT-100K")
    print("=" * 70)

    k = 10
    ef_search = 50
    sample_size = 1000

    # Load SIFT-100K
    print("\nLoading SIFT-100K dataset...")
    data, queries, metadata = load_or_generate_dataset(
        dataset_name="sift", n_samples=100000, dim=128, n_queries=10000
    )

    n_samples = len(data)
    dim = data.shape[1]
    n_queries = len(queries)
    print(f"Dataset: {metadata.get('name', 'SIFT-100K')}, N={n_samples}, D={dim}, Q={n_queries}")

    # Build a standard AdaptiveVec index (Steps 1-4)
    print("\nBuilding AdaptiveVec index...")
    config = AdaptivePolicyConfig(
        policy_type="continuous",
        m_base=16, m_min=8, m_max=24,
        sensitivity=0.4,
        alpha_lid=0.5,
        beta_density=0.5,
    )

    index = AdaptiveHNSW(
        dim=dim,
        policy_config=config,
        space="l2",
        heuristic=True,
        early_exit=False,
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

    # Ground truth for queries (load precomputed if available)
    gt_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sift_subset_groundtruth.ivecs")
    if os.path.exists(gt_file):
        print(f"Loading precomputed ground truth from {gt_file}...")
        from adaptivevec.datasets import read_ivecs
        gt_idx = read_ivecs(gt_file)
    else:
        print("Computing brute-force ground truth...")
        gt_idx, _ = compute_ground_truth(data, queries, k=100, space="l2")

    # Sample queries for validation
    np.random.seed(42)
    sample_indices = np.random.choice(n_queries, size=min(sample_size, n_queries), replace=False)

    print(f"\nValidating S(x) correlation on {len(sample_indices)} sampled queries...")

    difficulty_scores = []
    search_difficulties = []
    lids = []
    densities = []
    recall_per_query = []

    for qi in sample_indices:
        q = queries[qi]

        # 1. Compute difficulty score S(x) for this query using its neighborhood
        # Use brute-force distances to compute LID and density
        diffs = data - q
        all_dists = np.linalg.norm(diffs, axis=1)
        sorted_dists = np.sort(all_dists)[:25]

        lid = compute_mle_lid(sorted_dists)
        density = compute_local_density(sorted_dists[:15])
        score = index.policy.compute_difficulty_score(density, lid)

        difficulty_scores.append(score)
        lids.append(lid)
        densities.append(density)

        # 2. Measure actual search difficulty: distance evaluations needed
        init_comps = index.total_dist_computations
        res = index.search(q, k=k, ef=ef_search, early_exit=False)
        actual_comps = index.total_dist_computations - init_comps
        search_difficulties.append(actual_comps)

        # 3. Recall for this query
        found_nodes = set(node for _, node in res[:k])
        gt_nodes = set(gt_idx[qi, :k])
        q_recall = len(found_nodes.intersection(gt_nodes)) / k
        recall_per_query.append(q_recall)

    # Compute correlations
    difficulty_scores = np.array(difficulty_scores)
    search_difficulties = np.array(search_difficulties)
    lids = np.array(lids)
    densities = np.array(densities)
    recall_per_query = np.array(recall_per_query)

    # Spearman correlations
    spearman_score_difficulty, p_score_difficulty = scipy_stats.spearmanr(difficulty_scores, search_difficulties)
    spearman_lid_difficulty, p_lid_difficulty = scipy_stats.spearmanr(lids, search_difficulties)
    spearman_density_difficulty, p_density_difficulty = scipy_stats.spearmanr(densities, search_difficulties)
    spearman_score_recall, p_score_recall = scipy_stats.spearmanr(difficulty_scores, recall_per_query)

    print(f"\n{'-' * 50}")
    print("  CORRELATION RESULTS")
    print(f"{'-' * 50}")
    print(f"  S(x) ↔ Dist Evals:  ρ = {spearman_score_difficulty:.4f}  (p = {p_score_difficulty:.2e})")
    print(f"  LID  ↔ Dist Evals:  ρ = {spearman_lid_difficulty:.4f}  (p = {p_lid_difficulty:.2e})")
    print(f"  Dens ↔ Dist Evals:  ρ = {spearman_density_difficulty:.4f}  (p = {p_density_difficulty:.2e})")
    print(f"  S(x) ↔ Recall:      ρ = {spearman_score_recall:.4f}  (p = {p_score_recall:.2e})")

    # Distribution stats
    print(f"\n  S(x) distribution:  mean={np.mean(difficulty_scores):.4f}, std={np.std(difficulty_scores):.4f}, range=[{np.min(difficulty_scores):.4f}, {np.max(difficulty_scores):.4f}]")
    print(f"  LID distribution:   mean={np.mean(lids):.4f}, std={np.std(lids):.4f}, range=[{np.min(lids):.4f}, {np.max(lids):.4f}]")
    print(f"  Dens distribution:  mean={np.mean(densities):.4f}, std={np.std(densities):.4f}")

    # Build per-query validation data (sample for inclusion in paper)
    validation_samples = []
    for i in range(min(20, len(sample_indices))):
        validation_samples.append({
            "query_idx": int(sample_indices[i]),
            "difficulty_score": round(float(difficulty_scores[i]), 4),
            "lid": round(float(lids[i]), 4),
            "density": round(float(densities[i]), 4),
            "dist_evals": int(search_difficulties[i]),
            "recall_at_10": round(float(recall_per_query[i]), 4),
        })

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lid_validation_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "lid_difficulty_score_validation",
            "dataset": "SIFT-100K",
            "sample_size": len(sample_indices),
            "correlations": {
                "score_vs_dist_evals": {
                    "spearman_rho": round(spearman_score_difficulty, 4),
                    "p_value": float(p_score_difficulty),
                    "interpretation": "positive = higher S(x) → more distance evaluations (harder search)"
                },
                "lid_vs_dist_evals": {
                    "spearman_rho": round(spearman_lid_difficulty, 4),
                    "p_value": float(p_lid_difficulty),
                },
                "density_vs_dist_evals": {
                    "spearman_rho": round(spearman_density_difficulty, 4),
                    "p_value": float(p_density_difficulty),
                },
                "score_vs_recall": {
                    "spearman_rho": round(spearman_score_recall, 4),
                    "p_value": float(p_score_recall),
                    "interpretation": "negative = higher S(x) → lower recall (harder search)"
                },
            },
            "distributions": {
                "difficulty_score": {
                    "mean": round(float(np.mean(difficulty_scores)), 4),
                    "std": round(float(np.std(difficulty_scores)), 4),
                    "min": round(float(np.min(difficulty_scores)), 4),
                    "max": round(float(np.max(difficulty_scores)), 4),
                    "q25": round(float(np.percentile(difficulty_scores, 25)), 4),
                    "q50": round(float(np.percentile(difficulty_scores, 50)), 4),
                    "q75": round(float(np.percentile(difficulty_scores, 75)), 4),
                },
                "lid": {
                    "mean": round(float(np.mean(lids)), 4),
                    "std": round(float(np.std(lids)), 4),
                    "min": round(float(np.min(lids)), 4),
                    "max": round(float(np.max(lids)), 4),
                },
                "density": {
                    "mean": round(float(np.mean(densities)), 4),
                    "std": round(float(np.std(densities)), 4),
                },
                "search_difficulty_dist_evals": {
                    "mean": round(float(np.mean(search_difficulties)), 1),
                    "std": round(float(np.std(search_difficulties)), 1),
                    "min": int(np.min(search_difficulties)),
                    "max": int(np.max(search_difficulties)),
                },
            },
            "sample_data": validation_samples,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 70}")

    return {
        "spearman_score_difficulty": spearman_score_difficulty,
        "spearman_lid_difficulty": spearman_lid_difficulty,
        "spearman_score_recall": spearman_score_recall,
    }


if __name__ == "__main__":
    run_lid_validation()
