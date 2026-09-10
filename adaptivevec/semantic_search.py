"""
Semantic Document Search Engine powered by AdaptiveVec and Stock HNSW.
Demonstrates practical RAG / Semantic Retrieval on real text corpora.
"""

import math
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from .hnsw_base import StockHNSW
from .adaptive_hnsw import AdaptiveHNSW
from .policy import AdaptivePolicyConfig

# Curated high-quality corpus of technical and scientific documents
SAMPLE_DOCUMENTS = [
    {
        "id": 1,
        "title": "Hierarchical Navigable Small World Graphs (HNSW)",
        "category": "Vector Search & Indexing",
        "content": "HNSW creates a multi-layer graph where upper layers have long-range express links and lower layers have dense local links, achieving logarithmic search complexity for approximate nearest neighbors."
    },
    {
        "id": 2,
        "title": "Local Intrinsic Dimensionality (LID) in High-Dimensional Data",
        "category": "Machine Learning & Geometry",
        "content": "LID measures the local rate of space expansion as distance increases. Points on low-dimensional manifolds embedded in high-dimensional spaces exhibit small LID values regardless of ambient dimension."
    },
    {
        "id": 3,
        "title": "Product Quantization for Billion-Scale Vector Search",
        "category": "Vector Search & Indexing",
        "content": "Product Quantization decomposes vector spaces into Cartesian products of low-dimensional subspaces and quantizes each subspace separately, compressing vectors into compact byte codes."
    },
    {
        "id": 4,
        "title": "Retrieval-Augmented Generation (RAG) Architecture",
        "category": "Large Language Models",
        "content": "RAG enhances generative language models by retrieving relevant external context from vector databases before generating responses, reducing hallucinations and providing up-to-date domain knowledge."
    },
    {
        "id": 5,
        "title": "Skip Lists: Probabilistic Alternative to Balanced Trees",
        "category": "Data Structures",
        "content": "Skip lists are linked lists with layered forward pointers chosen by coin flips, providing expected O(log N) search, insertion, and deletion without complex tree rebalancing."
    },
    {
        "id": 6,
        "title": "Locality-Sensitive Hashing (LSH) for Cosine Similarity",
        "category": "Vector Search & Indexing",
        "content": "LSH hashes high-dimensional points into buckets such that similar items have a much higher collision probability than dissimilar ones, enabling sub-linear nearest neighbor retrieval."
    },
    {
        "id": 7,
        "title": "Graph Neural Networks for Molecule Property Prediction",
        "category": "Deep Learning",
        "content": "GNNs operate directly on molecular graph structures using message passing between atomic nodes and chemical bonds to predict bioactivity, toxicity, and chemical reactivity."
    },
    {
        "id": 8,
        "title": "Attention Is All You Need: The Transformer Model",
        "category": "Deep Learning",
        "content": "The Transformer replaces recurrent neural networks with multi-head self-attention mechanisms, enabling massive parallelization during training and capturing long-range contextual token dependencies."
    },
    {
        "id": 9,
        "title": "Delaunay Triangulation and Relative Neighborhood Graphs",
        "category": "Computational Geometry",
        "content": "Relative Neighborhood Graphs are minimal subgraphs of Delaunay triangulations where two points share an edge if no third point is closer to both, preserving directional neighborhood diversity."
    },
    {
        "id": 10,
        "title": "Vector Databases in Modern Production AI Systems",
        "category": "Systems & Databases",
        "content": "Vector databases like Milvus, Qdrant, and pgvector provide ACID-compliant storage, filtering, and indexing of high-dimensional embeddings for recommendation systems and semantic search."
    },
    {
        "id": 11,
        "title": "Quantization-Aware Training and Post-Training Quantization",
        "category": "Model Compression",
        "content": "Quantizing 32-bit floating point weights to INT8 or INT4 drastically reduces model memory footprint and accelerates inference throughput on edge devices and neural accelerators."
    },
    {
        "id": 12,
        "title": "Curse of Dimensionality and Distance Concentration",
        "category": "Information Retrieval",
        "content": "As dimensionality increases, the volume of space grows exponentially, causing distance distributions between points to concentrate and making naive spatial partitioning ineffective."
    },
    {
        "id": 13,
        "title": "Maximum Likelihood Estimation of Local Intrinsic Dimension",
        "category": "Statistics & Geometry",
        "content": "Amsaleg et al. proposed an MLE estimator using the ratio of neighbor distances to the k-th neighbor, providing a fast and mathematically grounded assessment of local data density."
    },
    {
        "id": 14,
        "title": "Distributed Vector Search with Consistent Hashing",
        "category": "Distributed Systems",
        "content": "Sharding proximity graphs across multiple server nodes requires careful partitioning of high-layer hubs to avoid bottleneck congestion and maintain high query throughput under load."
    },
    {
        "id": 15,
        "title": "Dense Passage Retrieval for Open-Domain QA",
        "category": "Natural Language Processing",
        "content": "DPR uses dual-encoder architectures (BERT for queries and passages) to map text to dense 768-dimensional vectors, outperforming traditional BM25 keyword matching."
    }
]

class LightweightEmbedder:
    """
    Fast, deterministic vocabulary-based semantic embedder.
    Maps text into dense, normalized semantic embedding vectors.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim
        # Seeded random projection matrix for word tokens
        np.random.seed(42)
        self.vocab_proj: Dict[str, np.ndarray] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z0-9_-]+\b', text.lower())

    def _get_word_vector(self, word: str) -> np.ndarray:
        if word not in self.vocab_proj:
            # Deterministic hash seed
            seed = sum(ord(c) * (31 ** i) for i, c in enumerate(word[:6])) % (2**31 - 1)
            rng = np.random.RandomState(seed)
            vec = rng.normal(0, 1, size=self.dim)
            self.vocab_proj[word] = vec
        return self.vocab_proj[word]

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        
        vec = np.zeros(self.dim, dtype=np.float32)
        for t in tokens:
            vec += self._get_word_vector(t)
            
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec.astype(np.float32)

class SemanticSearchEngine:
    """
    Dual-engine Semantic Search Manager hosting both Stock HNSW and AdaptiveVec.
    """
    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None, dim: int = 64):
        self.dim = dim
        self.embedder = LightweightEmbedder(dim=dim)
        self.documents = documents or SAMPLE_DOCUMENTS
        
        # Build Vector Store
        self.vectors = np.array([
            self.embedder.embed_text(f"{doc['title']} {doc['category']} {doc['content']}")
            for doc in self.documents
        ], dtype=np.float32)
        
        # Initialize Indexes
        self.stock_index = StockHNSW(dim=self.dim, m=8, ef_construction=80, space="cosine")
        self.adaptive_index = AdaptiveHNSW(
            dim=self.dim,
            policy_config=AdaptivePolicyConfig(policy_type="continuous", m_base=8, m_min=4, m_max=16, ef_construction_base=80),
            space="cosine"
        )
        
        self._build_indexes()

    def _build_indexes(self):
        # Calibrate AdaptiveVec
        self.adaptive_index.calibrate(self.vectors)
        
        for vec in self.vectors:
            self.stock_index.insert(vec)
            self.adaptive_index.insert(vec)

    def add_document(self, title: str, category: str, content: str) -> int:
        doc_id = len(self.documents) + 1
        new_doc = {"id": doc_id, "title": title, "category": category, "content": content}
        self.documents.append(new_doc)
        
        vec = self.embedder.embed_text(f"{title} {category} {content}")
        self.stock_index.insert(vec)
        self.adaptive_index.insert(vec)
        return doc_id

    def search(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """Runs search on both indexes and returns comparative results."""
        q_vec = self.embedder.embed_text(query_text)
        
        # Stock search
        stock_results, stock_trace = self.stock_index.search(q_vec, k=k, record_trace=True)
        # Adaptive search
        adapt_results, adapt_trace = self.adaptive_index.search(q_vec, k=k, record_trace=True)
        
        def format_hits(results):
            hits = []
            for dist, idx in results:
                doc = self.documents[idx]
                sim = max(0.0, 1.0 - dist)
                hits.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "content": doc["content"],
                    "similarity": round(float(sim), 4),
                    "distance": round(float(dist), 4),
                    "node_id": idx
                })
            return hits

        stock_stats = self.stock_index.get_stats()
        adapt_stats = self.adaptive_index.get_stats()
        
        memory_savings = 0.0
        if stock_stats["total_edges"] > 0:
            memory_savings = ((stock_stats["total_edges"] - adapt_stats["total_edges"]) / stock_stats["total_edges"]) * 100.0

        return {
            "query": query_text,
            "stock": {
                "hits": format_hits(stock_results),
                "dist_evals": stock_trace.get("total_dist_evals", 0),
                "total_edges": stock_stats["total_edges"],
                "avg_edges_per_node": stock_stats["avg_edges_per_node"]
            },
            "adaptive": {
                "hits": format_hits(adapt_results),
                "dist_evals": adapt_trace.get("total_dist_evals", 0),
                "total_edges": adapt_stats["total_edges"],
                "avg_edges_per_node": adapt_stats["avg_edges_per_node"],
                "used_ef": adapt_trace.get("used_ef", 0)
            },
            "comparison": {
                "edge_savings_pct": round(memory_savings, 2),
                "dist_evals_diff": adapt_trace.get("total_dist_evals", 0) - stock_trace.get("total_dist_evals", 0)
            }
        }
