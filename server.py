"""
FastAPI Server for AdaptiveVec Interactive Studio & Benchmark Suite.
"""

import time
import os
import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from adaptivevec.signals import profile_dataset_signals
from adaptivevec.policy import AdaptivePolicyConfig
from adaptivevec.hnsw_base import StockHNSW
from adaptivevec.adaptive_hnsw import AdaptiveHNSW
from adaptivevec.datasets import load_or_generate_dataset, compute_ground_truth
from adaptivevec.benchmark import BenchmarkHarness
from adaptivevec.semantic_search import SemanticSearchEngine, SAMPLE_DOCUMENTS

app = FastAPI(title="AdaptiveVec API", version="1.0.0", description="Density- and Dimension-Aware HNSW Vector Index")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class AppState:
    def __init__(self):
        self.data: np.ndarray = np.empty((0, 64), dtype=np.float32)
        self.queries: np.ndarray = np.empty((0, 64), dtype=np.float32)
        self.dataset_meta: Dict[str, Any] = {}
        self.stock_index: Optional[StockHNSW] = None
        self.adaptive_index: Optional[AdaptiveHNSW] = None
        self.projection_2d: Optional[np.ndarray] = None
        self.semantic_engine: Optional[SemanticSearchEngine] = None
        self.latest_benchmark_results: List[Dict[str, Any]] = []

state = AppState()

# Request Models
class DatasetGenRequest(BaseModel):
    name: str = "multi_manifold"
    n_samples: int = 2000
    dim: int = 32
    n_queries: int = 50

class BuildIndexRequest(BaseModel):
    policy_type: str = "continuous"
    m_base: int = 16
    m_min: int = 8
    m_max: int = 28
    ef_construction_base: int = 100
    ef_construction_min: int = 35
    ef_construction_max: int = 200
    alpha_lid: float = 0.5
    beta_density: float = 0.5
    sensitivity: float = 0.5

class SearchRequest(BaseModel):
    query_index: Optional[int] = 0
    custom_vector: Optional[List[float]] = None
    k: int = 10
    ef_search: int = 50
    index_type: str = "adaptive"  # "adaptive" or "stock"

class SemanticSearchRequest(BaseModel):
    query: str
    k: int = 5

class AddDocumentRequest(BaseModel):
    title: str
    category: str
    content: str

def compute_pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Fast 2D PCA projection for canvas visualization."""
    if len(vectors) < 2:
        return np.zeros((len(vectors), 2), dtype=np.float32)
    
    # Center data
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    
    # SVD
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    
    # Normalize to [-1, 1]
    max_val = np.max(np.abs(proj))
    if max_val > 1e-7:
        proj = proj / max_val
    return proj.astype(np.float32)

@app.on_event("startup")
def startup_event():
    """Initializes default dataset and semantic search engine."""
    data, queries, meta = load_or_generate_dataset("multi_manifold", n_samples=1500, dim=32, n_queries=30)
    state.data = data
    state.queries = queries
    state.dataset_meta = meta
    state.projection_2d = compute_pca_2d(data)
    
    # Build default indices
    state.stock_index = StockHNSW(dim=32, m=16, ef_construction=100)
    for v in data:
        state.stock_index.insert(v)
        
    cfg = AdaptivePolicyConfig(policy_type="continuous", m_base=16, m_min=8, m_max=28, ef_construction_base=100)
    state.adaptive_index = AdaptiveHNSW(dim=32, policy_config=cfg)
    state.adaptive_index.calibrate(data[:min(300, len(data))])
    for v in data:
        state.adaptive_index.insert(v)
        
    state.semantic_engine = SemanticSearchEngine()

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "dataset": state.dataset_meta,
        "n_samples": len(state.data),
        "dim": state.data.shape[1] if len(state.data) > 0 else 0,
        "stock_built": state.stock_index is not None,
        "adaptive_built": state.adaptive_index is not None,
        "stock_stats": state.stock_index.get_stats() if state.stock_index else None,
        "adaptive_stats": state.adaptive_index.get_stats() if state.adaptive_index else None
    }

@app.post("/api/dataset/generate")
def generate_dataset_endpoint(req: DatasetGenRequest):
    data, queries, meta = load_or_generate_dataset(req.name, n_samples=req.n_samples, dim=req.dim, n_queries=req.n_queries)
    state.data = data
    state.queries = queries
    state.dataset_meta = meta
    state.projection_2d = compute_pca_2d(data)
    
    # Reset indices
    state.stock_index = None
    state.adaptive_index = None
    
    return {
        "message": f"Generated dataset '{meta['name']}' with {len(data)} samples and {len(queries)} queries.",
        "meta": meta,
        "n_samples": len(data),
        "dim": req.dim
    }

@app.post("/api/index/build")
def build_indices(req: BuildIndexRequest):
    if len(state.data) == 0:
        raise HTTPException(status_code=400, detail="No dataset loaded. Generate a dataset first.")
    
    dim = state.data.shape[1]
    
    # 1. Build Stock HNSW Baseline
    stock_start = time.perf_counter()
    stock_idx = StockHNSW(dim=dim, m=req.m_base, ef_construction=req.ef_construction_base)
    for v in state.data:
        stock_idx.insert(v)
    stock_build_time = time.perf_counter() - stock_start
    state.stock_index = stock_idx
    
    # 2. Build AdaptiveVec
    adapt_start = time.perf_counter()
    cfg = AdaptivePolicyConfig(
        policy_type=req.policy_type,
        m_base=req.m_base,
        m_min=req.m_min,
        m_max=req.m_max,
        ef_construction_base=req.ef_construction_base,
        ef_construction_min=req.ef_construction_min,
        ef_construction_max=req.ef_construction_max,
        alpha_lid=req.alpha_lid,
        beta_density=req.beta_density,
        sensitivity=req.sensitivity
    )
    adapt_idx = AdaptiveHNSW(dim=dim, policy_config=cfg)
    adapt_idx.calibrate(state.data[:min(300, len(state.data))])
    for v in state.data:
        adapt_idx.insert(v)
    adapt_build_time = time.perf_counter() - adapt_start
    state.adaptive_index = adapt_idx
    
    stock_stats = stock_idx.get_stats()
    adapt_stats = adapt_idx.get_stats()
    
    edge_savings = 0.0
    if stock_stats["total_edges"] > 0:
        edge_savings = ((stock_stats["total_edges"] - adapt_stats["total_edges"]) / stock_stats["total_edges"]) * 100.0
        
    build_speedup = 0.0
    if stock_build_time > 0:
        build_speedup = ((stock_build_time - adapt_build_time) / stock_build_time) * 100.0

    return {
        "stock": {
            "build_time_s": round(stock_build_time, 4),
            "stats": stock_stats
        },
        "adaptive": {
            "build_time_s": round(adapt_build_time, 4),
            "stats": adapt_stats
        },
        "comparison": {
            "edge_savings_pct": round(edge_savings, 2),
            "build_speedup_pct": round(build_speedup, 2),
            "total_nodes": len(state.data)
        }
    }

@app.get("/api/graph/projection")
def get_graph_projection(max_nodes: int = 1500):
    """Returns 2D projected coordinates with node signals and sampled layer edges."""
    if len(state.data) == 0 or state.projection_2d is None:
        raise HTTPException(status_code=400, detail="Index or dataset not initialized")
    
    n_display = min(max_nodes, len(state.data))
    nodes = []
    
    for i in range(n_display):
        x, y = float(state.projection_2d[i, 0]), float(state.projection_2d[i, 1])
        
        # Extract metadata from Adaptive index if built
        lid = 2.0
        density = 1.0
        degree = 16
        level = 0
        score = 0.0
        
        if state.adaptive_index:
            level = state.adaptive_index.node_levels.get(i, 0)
            sig = state.adaptive_index.node_signals.get(i, {})
            lid = float(sig.get("lid", 2.0))
            density = float(sig.get("density", 1.0))
            score = float(sig.get("score", 0.0))
            degree = len(state.adaptive_index.graphs[0].get(i, [])) if len(state.adaptive_index.graphs) > 0 else 0
        elif state.stock_index:
            level = state.stock_index.node_levels.get(i, 0)
            degree = len(state.stock_index.graphs[0].get(i, [])) if len(state.stock_index.graphs) > 0 else 0

        nodes.append({
            "id": i,
            "x": round(x, 4),
            "y": round(y, 4),
            "level": level,
            "degree": degree,
            "lid": round(lid, 2),
            "density": round(density, 2),
            "score": round(score, 2)
        })
        
    # Extract sampled edges for layer 0 and layer 1
    edges_layer0 = []
    edges_layer1 = []
    
    active_idx = state.adaptive_index or state.stock_index
    if active_idx and len(active_idx.graphs) > 0:
        # Sample layer 0 edges (limit to avoid browser lag)
        for u in range(min(500, n_display)):
            for v in active_idx.graphs[0].get(u, []):
                if v < n_display and u < v:
                    edges_layer0.append({"source": u, "target": v})
                    
        # Layer 1 edges
        if len(active_idx.graphs) > 1:
            for u in active_idx.graphs[1]:
                if u < n_display:
                    for v in active_idx.graphs[1][u]:
                        if v < n_display and u < v:
                            edges_layer1.append({"source": u, "target": v})

    return {
        "nodes": nodes,
        "edges_l0": edges_layer0[:1500],
        "edges_l1": edges_layer1[:500],
        "total_nodes": len(state.data)
    }

@app.post("/api/search")
def search_index(req: SearchRequest):
    if len(state.data) == 0:
        raise HTTPException(status_code=400, detail="Dataset is empty")
    
    if req.custom_vector is not None:
        q_vec = np.array(req.custom_vector, dtype=np.float32)
    elif req.query_index is not None and 0 <= req.query_index < len(state.queries):
        q_vec = state.queries[req.query_index]
    else:
        q_vec = state.queries[0]
        
    # Compute Exact Ground Truth for this single query
    diff = state.data - q_vec
    exact_dists = np.sqrt(np.sum(diff ** 2, axis=1))
    gt_top_k = np.argsort(exact_dists)[:req.k]
    gt_set = set(gt_top_k)
    
    # Run Stock Search
    stock_res = []
    stock_trace = {}
    if state.stock_index:
        stock_res, stock_trace = state.stock_index.search(q_vec, k=req.k, ef=req.ef_search, record_trace=True)
        
    # Run Adaptive Search
    adapt_res = []
    adapt_trace = {}
    if state.adaptive_index:
        adapt_res, adapt_trace = state.adaptive_index.search(q_vec, k=req.k, ef=req.ef_search, record_trace=True)
        
    def calculate_hit_rate(results):
        found = set(node for _, node in results)
        return len(found.intersection(gt_set)) / req.k if req.k > 0 else 0.0

    return {
        "k": req.k,
        "ef_search": req.ef_search,
        "ground_truth": [{"id": int(idx), "dist": round(float(exact_dists[idx]), 4)} for idx in gt_top_k],
        "stock": {
            "results": [{"id": int(node), "dist": round(float(d), 4)} for d, node in stock_res],
            "recall": round(calculate_hit_rate(stock_res), 4),
            "trace": stock_trace
        },
        "adaptive": {
            "results": [{"id": int(node), "dist": round(float(d), 4)} for d, node in adapt_res],
            "recall": round(calculate_hit_rate(adapt_res), 4),
            "trace": adapt_trace
        }
    }

@app.post("/api/benchmark/run")
def run_benchmark_endpoint(ef_search: int = 50):
    if len(state.data) == 0 or len(state.queries) == 0:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    
    harness = BenchmarkHarness(state.data, state.queries, space="l2")
    results = harness.run_full_suite(ef_search=ef_search)
    state.latest_benchmark_results = [r.to_dict() for r in results]
    
    return {
        "results": state.latest_benchmark_results,
        "dataset_name": state.dataset_meta.get("name", "Vector Dataset"),
        "n_samples": len(state.data),
        "dim": state.data.shape[1]
    }

@app.post("/api/semantic/search")
def search_semantic_endpoint(req: SemanticSearchRequest):
    if not state.semantic_engine:
        state.semantic_engine = SemanticSearchEngine()
    return state.semantic_engine.search(req.query, k=req.k)

@app.post("/api/semantic/add")
def add_semantic_document(req: AddDocumentRequest):
    if not state.semantic_engine:
        state.semantic_engine = SemanticSearchEngine()
    doc_id = state.semantic_engine.add_document(req.title, req.category, req.content)
    return {"message": "Document added to semantic index", "id": doc_id, "total_docs": len(state.semantic_engine.documents)}

@app.get("/api/semantic/documents")
def list_semantic_documents():
    if not state.semantic_engine:
        state.semantic_engine = SemanticSearchEngine()
    return {"documents": state.semantic_engine.documents}

# Serve static frontend files
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "AdaptiveVec API is running. Build frontend or use /docs."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
