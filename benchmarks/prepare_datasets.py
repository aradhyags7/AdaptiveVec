import os
import sys
import struct
import urllib.request
import numpy as np
import h5py

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def write_fvecs(filename, data):
    print(f'[*] Writing {filename} ({data.shape[0]} vectors, dim {data.shape[1]})...')
    with open(filename, 'wb') as f:
        dim = np.int32(data.shape[1])
        dim_bytes = dim.tobytes()
        for i in range(data.shape[0]):
            f.write(dim_bytes)
            f.write(data[i].astype(np.float32).tobytes())

def write_ivecs(filename, data):
    print(f'[*] Writing {filename} ({data.shape[0]} queries, k {data.shape[1]})...')
    with open(filename, 'wb') as f:
        k = np.int32(data.shape[1])
        k_bytes = k.tobytes()
        for i in range(data.shape[0]):
            f.write(k_bytes)
            f.write(data[i].astype(np.uint32).tobytes())

def download_file(url, target_path):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f'[+] Found cached {target_path} ({os.path.getsize(target_path)/(1024*1024):.1f} MB)')
        return
    print(f'[*] Downloading {url} -> {target_path}...')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(target_path, 'wb') as out_f:
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024 * 2
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = (downloaded / total) * 100.0
                print(f'\r    {downloaded/(1024*1024):.1f}/{total/(1024*1024):.1f} MB ({percent:.1f}%)', end='', flush=True)
        print()
    print(f'[+] Downloaded {target_path}')

import hashlib
import tarfile
import zipfile

TEXMEX_SIFT_URL = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
TEXMEX_SIFT_MD5 = 'b23d1b3b2ee8469d819b61ca900ef0ed'
STANFORD_GLOVE_URL = 'https://nlp.stanford.edu/data/glove.6B.zip'

def verify_md5(filepath, expected_md5):
    """Verify file integrity using MD5 checksum."""
    print(f'[*] Verifying MD5 for {os.path.basename(filepath)}...')
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024 * 4):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual.lower() != expected_md5.lower():
        raise ValueError(
            f'MD5 mismatch for {filepath}!\n'
            f'  Expected: {expected_md5}\n'
            f'  Actual:   {actual}\n'
            f'The downloaded archive is corrupted or invalid. Aborting.'
        )
    print(f'[+] MD5 verified: {actual}')
    return True

def prepare_canonical_sift_irisa():
    """Downloads directly from canonical Texmex server (IRISA INRIA) with MD5 check."""
    tar_path = os.path.join(DATA_DIR, 'sift.tar.gz')
    download_file(TEXMEX_SIFT_URL, tar_path)
    verify_md5(tar_path, TEXMEX_SIFT_MD5)
    
    print('[*] Extracting canonical Texmex archive (sift.tar.gz)...')
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.fvecs') or member.name.endswith('.ivecs'):
                tar.extract(member, path=DATA_DIR)
                base_name = os.path.basename(member.name)
                extracted_file = os.path.join(DATA_DIR, member.name)
                dest_file = os.path.join(DATA_DIR, base_name)
                if extracted_file != dest_file and os.path.exists(extracted_file):
                    os.replace(extracted_file, dest_file)
    print('[+] Canonical Texmex SIFT-1M extraction complete from IRISA source.')

def prepare_canonical_glove_stanford():
    """Downloads directly from canonical Stanford NLP server (nlp.stanford.edu)."""
    zip_path = os.path.join(DATA_DIR, 'glove.6B.zip')
    download_file(STANFORD_GLOVE_URL, zip_path)
    
    print('[*] Extracting Stanford GloVe-100 (glove.6B.100d.txt)...')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extract('glove.6B.100d.txt', path=DATA_DIR)
    print('[+] Canonical Stanford GloVe extraction complete from Stanford NLP source.')

def prepare_sift_subset_from_canonical(num_subset=100000):
    """Generates a verifiable N-sample subset directly from canonical Texmex sift_base.fvecs."""
    canonical_base = os.path.join(DATA_DIR, 'sift_base.fvecs')
    canonical_query = os.path.join(DATA_DIR, 'sift_query.fvecs')
    
    if not os.path.exists(canonical_base) or not os.path.exists(canonical_query):
        print('[*] Canonical SIFT files missing, downloading from Texmex IRISA...')
        prepare_canonical_sift_irisa()
    
    subset_fvecs = os.path.join(DATA_DIR, f'sift_subset_{num_subset//1000}k.fvecs')
    query_fvecs = os.path.join(DATA_DIR, 'sift_query.fvecs')
    subset_gt = os.path.join(DATA_DIR, f'sift_subset_groundtruth.ivecs')
    
    if os.path.exists(subset_fvecs) and os.path.exists(subset_gt):
        print(f'[+] Subset {subset_fvecs} and groundtruth already exist.')
        return
    
    # Read first num_subset vectors from canonical base
    print(f'[*] Extracting first {num_subset} vectors from canonical Texmex {canonical_base}...')
    with open(canonical_base, 'rb') as f:
        dim = struct.unpack('<i', f.read(4))[0]
        f.seek(0)
        record_size = 4 + dim * 4
        raw_data = f.read(record_size * num_subset)
    
    with open(subset_fvecs, 'wb') as f:
        f.write(raw_data)
    print(f'[+] Wrote {subset_fvecs}')
    
    # Compute ground truth against this exact canonical subset
    print(f'[*] Reading {num_subset} subset vectors and 10,000 query vectors for exact ground truth...')
    from adaptivevec.datasets import read_fvecs
    base_vectors = read_fvecs(subset_fvecs)
    query_vectors = read_fvecs(query_fvecs)[:10000]
    
    print('[*] Computing exact ground truth for subset...')
    q_norms = np.sum(query_vectors ** 2, axis=1, keepdims=True)
    v_norms = np.sum(base_vectors ** 2, axis=1, keepdims=True).T
    
    batch_size = 1000
    gt = np.zeros((10000, 100), dtype=np.uint32)
    for b in range(0, 10000, batch_size):
        end_b = min(10000, b + batch_size)
        q_sub = query_vectors[b:end_b]
        dists = q_norms[b:end_b] + v_norms - 2.0 * np.dot(q_sub, base_vectors.T)
        top100 = np.argpartition(dists, 100, axis=1)[:, :100]
        for i in range(end_b - b):
            sorted_top = top100[i, np.argsort(dists[i, top100[i]])]
            gt[b + i] = sorted_top
            
    write_ivecs(subset_gt, gt)
    print(f'[+] Computed and saved exact ground truth to {subset_gt}')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--canonical-glove':
        prepare_canonical_glove_stanford()
    elif len(sys.argv) > 1 and sys.argv[1] == '--canonical-sift-full':
        prepare_canonical_sift_irisa()
    else:
        # Default: prepare canonical Texmex SIFT and verify
        prepare_canonical_sift_irisa()
        prepare_sift_subset_from_canonical(100000)
