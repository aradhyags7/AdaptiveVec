import os
import numpy as np
import pyarrow.parquet as pq

def export_fvecs(filename, array):
    """Writes a 2D float32 numpy array to .fvecs format."""
    n, d = array.shape
    array = array.astype(np.float32)
    header = np.array([d], dtype=np.int32)
    with open(filename, 'wb') as f:
        for i in range(n):
            header.tofile(f)
            array[i].tofile(f)
    print(f"Exported {filename}: shape ({n}, {d}), size {os.path.getsize(filename)/(1024*1024):.2f} MB")

def export_ivecs(filename, array):
    """Writes a 2D int32/uint32 numpy array to .ivecs format."""
    n, k = array.shape
    array = array.astype(np.int32)
    header = np.array([k], dtype=np.int32)
    with open(filename, 'wb') as f:
        for i in range(n):
            header.tofile(f)
            array[i].tofile(f)
    print(f"Exported {filename}: shape ({n}, {k}), size {os.path.getsize(filename)/(1024*1024):.2f} MB")

def main():
    data_dir = "data/transformer_384d"
    os.makedirs(data_dir, exist_ok=True)
    
    passages = [
        ("scifact", os.path.join(data_dir, "scifact_passage.parquet")),
        ("nfcorpus", os.path.join(data_dir, "nfcorpus_passage.parquet")),
        ("arguana", os.path.join(data_dir, "arguana_passage.parquet")),
        ("webmasters", os.path.join(data_dir, "webmasters_passage.parquet")),
        ("android", os.path.join(data_dir, "android_passage.parquet")),
    ]
    
    all_vectors = []
    domain_labels = []
    
    for domain_name, p_path in passages:
        print(f"Loading {domain_name} from {p_path}...")
        t = pq.read_table(p_path, columns=['embedding'])
        emb_list = t['embedding'].to_pylist()
        arr = np.array(emb_list, dtype=np.float32)
        all_vectors.append(arr)
        domain_labels.extend([domain_name] * len(arr))
        print(f"  Loaded {len(arr)} vectors from {domain_name}")
        
    base_matrix = np.vstack(all_vectors)
    n_base, d = base_matrix.shape
    print(f"\nConsolidated Base Corpus: {n_base} vectors, {d} dimensions.")
    
    # Load queries
    query_files = [
        ("scifact", os.path.join(data_dir, "scifact_query.parquet")),
        ("nfcorpus", os.path.join(data_dir, "nfcorpus_query.parquet")),
        ("arguana", os.path.join(data_dir, "arguana_query.parquet")),
    ]
    
    all_queries = []
    for q_domain, q_path in query_files:
        t = pq.read_table(q_path, columns=['embedding'])
        emb_list = t['embedding'].to_pylist()
        arr = np.array(emb_list, dtype=np.float32)
        all_queries.append(arr)
        print(f"Loaded {len(arr)} queries from {q_domain}")
        
    query_matrix_all = np.vstack(all_queries)
    # Stratified sample of 1,000 queries
    np.random.seed(42)
    sample_indices = np.random.choice(len(query_matrix_all), size=1000, replace=False)
    query_matrix = query_matrix_all[sample_indices]
    n_query, q_d = query_matrix.shape
    print(f"Sampled Query Matrix: {n_query} queries, {q_d} dimensions.")
    
    # Export fvecs
    base_fvecs_path = os.path.join(data_dir, "transformer_384d_base.fvecs")
    query_fvecs_path = os.path.join(data_dir, "transformer_384d_query.fvecs")
    gt_ivecs_path = os.path.join(data_dir, "transformer_384d_groundtruth.ivecs")
    
    export_fvecs(base_fvecs_path, base_matrix)
    export_fvecs(query_fvecs_path, query_matrix)
    
    # Brute-force ground truth computation (Top 100 nearest neighbors)
    print("\nComputing exact brute-force ground truth (Top 100 L2 distance)...")
    k = 100
    ground_truth = np.zeros((n_query, k), dtype=np.int32)
    
    # Blocked matrix computation to prevent memory spikes
    # L2 distance: ||q - x||^2 = ||q||^2 + ||x||^2 - 2 * q @ x.T
    base_sq_norms = np.sum(base_matrix ** 2, axis=1) # (N,)
    
    batch_size = 100
    for i in range(0, n_query, batch_size):
        q_batch = query_matrix[i:i+batch_size] # (B, D)
        q_sq_norms = np.sum(q_batch ** 2, axis=1, keepdims=True) # (B, 1)
        # distances: (B, N)
        dots = np.dot(q_batch, base_matrix.T) # (B, N)
        dists = q_sq_norms + base_sq_norms.reshape(1, -1) - 2.0 * dots
        # argpartition or argsort
        top_k_indices = np.argpartition(dists, k, axis=1)[:, :k]
        for b in range(len(q_batch)):
            sorted_k = top_k_indices[b][np.argsort(dists[b, top_k_indices[b]])]
            ground_truth[i + b] = sorted_k
        print(f"  Processed {min(i + batch_size, n_query)}/{n_query} queries...")
        
    export_ivecs(gt_ivecs_path, ground_truth)
    print("\n[SUCCESS] Transformer 384D benchmark dataset prepared successfully!")

if __name__ == "__main__":
    main()
