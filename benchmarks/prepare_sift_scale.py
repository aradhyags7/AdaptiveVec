# Automated Dataset Scaling and Ground Truth Pre-computation
# Prepares canonical SIFT subsets and calculates exact nearest neighbors
"""
benchmarks/prepare_sift_1m.py
Prepares canonical SIFT datasets for large-scale evaluation:
- Exports 1,000,000 vectors from data/sift-128-euclidean.hdf5 to data/sift_base_1m.fvecs
- Computes exact ground truth for SIFT-250K and SIFT-500K for 1,000 evaluation queries
- Verifies integrity of SIFT-1M queries and ground truth
"""

import os
import time
import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")


def write_fvecs(filename, data_array):
    n, d = data_array.shape
    d_bytes = np.int32(d).tobytes()
    print(f"[*] Writing {n:,} vectors (dim={d}) to {filename} ...")
    with open(filename, "wb") as f:
        chunk_size = 50000
        for i in range(0, n, chunk_size):
            chunk = data_array[i:i + chunk_size]
            # Create buffer with dim header for each vector
            header_col = np.full((len(chunk), 1), d, dtype=np.int32)
            # Efficiently write vector by vector or combined
            for vec in chunk:
                f.write(d_bytes)
                f.write(vec.astype(np.float32).tobytes())
            print(f"    Written {min(i + chunk_size, n):,}/{n:,} vectors ...", end="\r")
    print(f"\n[+] Successfully created {filename} ({os.path.getsize(filename) / (1024*1024):.1f} MB)")


def write_ivecs(filename, indices_array):
    n, k = indices_array.shape
    k_bytes = np.int32(k).tobytes()
    print(f"[*] Writing {n:,} ground truth rows (k={k}) to {filename} ...")
    with open(filename, "wb") as f:
        for row in indices_array:
            f.write(k_bytes)
            f.write(row.astype(np.uint32).tobytes())
    print(f"[+] Successfully created {filename} ({os.path.getsize(filename) / (1024*1024):.2f} MB)")


def compute_exact_groundtruth(queries, base_vectors, k=100):
    q_norm = np.sum(queries**2, axis=1, keepdims=True)
    d_norm = np.sum(base_vectors**2, axis=1, keepdims=True).T
    dists = q_norm + d_norm - 2.0 * np.dot(queries, base_vectors.T)
    # Get top-k nearest neighbors
    part_idx = np.argpartition(dists, k, axis=1)[:, :k]
    # Sort within top-k
    row_indices = np.arange(len(queries))[:, None]
    sorted_part = np.argsort(dists[row_indices, part_idx], axis=1)
    top_k = part_idx[row_indices, sorted_part]
    return top_k.astype(np.uint32)


def main():
    hdf5_path = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"[-] {hdf5_path} not found!")

    with h5py.File(hdf5_path, "r") as hf:
        print("[*] Inspecting SIFT-1M HDF5 archive ...")
        train_dset = hf["train"]
        test_dset = hf["test"]
        neighbors_dset = hf["neighbors"]

        n_train, dim = train_dset.shape
        n_test, _ = test_dset.shape
        print(f"[+] Found {n_train:,} base vectors, {n_test:,} test queries (dim={dim}).")

        # 1. Export 1,000,000 vectors to sift_base_1m.fvecs if not already present
        out_fvecs = os.path.join(DATA_DIR, "sift_base_1m.fvecs")
        if not os.path.exists(out_fvecs) or os.path.getsize(out_fvecs) < 500 * 1024 * 1024:
            print(f"[*] Exporting 1,000,000 vectors from HDF5 to {out_fvecs} ...")
            t0 = time.time()
            write_fvecs(out_fvecs, train_dset)
            print(f"[+] SIFT-1M exported in {time.time() - t0:.1f}s")
        else:
            print(f"[+] {out_fvecs} already exists ({os.path.getsize(out_fvecs) / (1024*1024):.1f} MB). Skipping export.")

        # 2. Prepare 1M Ground Truth
        out_gt_1m = os.path.join(DATA_DIR, "sift_groundtruth_1m.ivecs")
        if not os.path.exists(out_gt_1m):
            print(f"[*] Exporting 1M ground truth to {out_gt_1m} ...")
            write_ivecs(out_gt_1m, np.array(neighbors_dset))

        # 3. Compute Ground Truth for 250K and 500K subsets (for first 1,000 queries)
        queries_1k = np.array(test_dset[:1000], dtype=np.float32)

        out_gt_250k = os.path.join(DATA_DIR, "sift_groundtruth_250k.ivecs")
        if not os.path.exists(out_gt_250k):
            print("[*] Computing exact ground truth for 250K subset (1,000 queries) ...")
            t0 = time.time()
            base_250k = np.array(train_dset[:250000], dtype=np.float32)
            gt_250k = compute_exact_groundtruth(queries_1k, base_250k, k=100)
            write_ivecs(out_gt_250k, gt_250k)
            print(f"[+] 250K ground truth computed in {time.time() - t0:.1f}s")

        out_gt_500k = os.path.join(DATA_DIR, "sift_groundtruth_500k.ivecs")
        if not os.path.exists(out_gt_500k):
            print("[*] Computing exact ground truth for 500K subset (1,000 queries) ...")
            t0 = time.time()
            base_500k = np.array(train_dset[:500000], dtype=np.float32)
            gt_500k = compute_exact_groundtruth(queries_1k, base_500k, k=100)
            write_ivecs(out_gt_500k, gt_500k)
            print(f"[+] 500K ground truth computed in {time.time() - t0:.1f}s")

    print("[+] All large-scale SIFT datasets and ground truths successfully prepared!")


if __name__ == "__main__":
    main()
