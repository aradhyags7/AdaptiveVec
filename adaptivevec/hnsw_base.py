"""
Stock HNSW Implementation (Baseline / Control).
Faithful implementation of Algorithms 1-5 from Malkov & Yashunin (2020)
with standard uniform M and efConstruction across the entire index.
"""

import math
import random
import heapq
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Any, Union

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance."""
    diff = a - b
    return float(np.dot(diff, diff) ** 0.5)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine_similarity)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 1.0
    cos_sim = np.dot(a, b) / (norm_a * norm_b)
    return float(max(0.0, 1.0 - cos_sim))

class StockHNSW:
    """
    Standard Hierarchical Navigable Small World (HNSW) Index with uniform parameters.
    """
    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 200,
        m_max: Optional[int] = None,
        m_max0: Optional[int] = None,
        space: str = "l2",
        heuristic: bool = True
    ):
        self.dim = dim
        self.m = m
        self.m_max = m_max if m_max is not None else m
        self.m_max0 = m_max0 if m_max0 is not None else 2 * m
        self.ef_construction = ef_construction
        self.space = space.lower()
        self.heuristic = heuristic
        
        self.m_l = 1.0 / math.log(self.m) if self.m > 1 else 1.0
        self.dist_fn = cosine_distance if self.space == "cosine" else l2_distance
        
        # Data storage
        self.data: List[np.ndarray] = []
        # Multi-layer adjacency lists: graphs[layer][node_id] = list of neighbor IDs
        self.graphs: List[Dict[int, List[int]]] = []
        # Maximum level for each node
        self.node_levels: Dict[int, int] = {}
        
        self.enter_point: Optional[int] = None
        self.max_level: int = -1
        
        # Instrumentation
        self.total_dist_computations: int = 0

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        self.total_dist_computations += 1
        return self.dist_fn(a, b)

    def _dist_nodes(self, idx1: int, idx2: int) -> float:
        return self._distance(self.data[idx1], self.data[idx2])

    def _generate_random_level(self) -> int:
        """Generates random maximum level according to geometric distribution."""
        r = random.random()
        while r == 0.0:
            r = random.random()
        return int(-math.log(r) * self.m_l)

    def _search_layer(
        self,
        query: np.ndarray,
        enter_points: List[int],
        ef: int,
        lc: int,
        record_trace: bool = False
    ) -> Union[List[Tuple[float, int]], Tuple[List[Tuple[float, int]], List[Dict[str, Any]]]]:
        """
        Algorithm 2: SEARCH-LAYER(q, ep, ef, lc)
        Greedy beam search at layer lc.
        Returns dynamic list W of (distance, node_id) sorted by distance.
        """
        visited: Set[int] = set(enter_points)
        # C: min-heap of (dist, node_id) for closest candidate exploration
        candidates: List[Tuple[float, int]] = []
        # W: max-heap of (-dist, node_id) to maintain the ef closest elements found so far
        w_furthest: List[Tuple[float, int]] = []
        
        trace_steps: List[Dict[str, Any]] = []

        for ep in enter_points:
            d = self._distance(query, self.data[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(w_furthest, (-d, ep))
            if record_trace:
                trace_steps.append({"action": "enter", "node": ep, "dist": d, "layer": lc})

        while len(candidates) > 0:
            c_dist, c_node = heapq.heappop(candidates)
            furthest_d = -w_furthest[0][0]
            
            if c_dist > furthest_d:
                break  # All candidates are further than furthest element in W
            
            neighbors = self.graphs[lc].get(c_node, [])
            for e_node in neighbors:
                if e_node not in visited:
                    visited.add(e_node)
                    e_dist = self._distance(query, self.data[e_node])
                    furthest_d = -w_furthest[0][0]
                    
                    if e_dist < furthest_d or len(w_furthest) < ef:
                        heapq.heappush(candidates, (e_dist, e_node))
                        heapq.heappush(w_furthest, (-e_dist, e_node))
                        
                        if record_trace:
                            trace_steps.append({"action": "explore", "from": c_node, "node": e_node, "dist": e_dist, "layer": lc})
                            
                        if len(w_furthest) > ef:
                            removed_d, removed_node = heapq.heappop(w_furthest)
                            if record_trace:
                                trace_steps.append({"action": "prune_w", "node": removed_node, "dist": -removed_d, "layer": lc})

        # Format W as sorted list of (dist, node_id)
        result = sorted([(-neg_d, node) for neg_d, node in w_furthest], key=lambda x: x[0])
        if record_trace:
            return result, trace_steps
        return result

    def _select_neighbors_simple(
        self,
        query: np.ndarray,
        candidates: List[Tuple[float, int]],
        m_limit: int
    ) -> List[int]:
        """Algorithm 3: SELECT-NEIGHBORS-SIMPLE (Take M nearest)."""
        sorted_c = sorted(candidates, key=lambda x: x[0])
        return [node for _, node in sorted_c[:m_limit]]

    def _select_neighbors_heuristic(
        self,
        query: np.ndarray,
        candidates: List[Tuple[float, int]],
        m_limit: int,
        keep_pruned: bool = True
    ) -> List[int]:
        """
        Algorithm 4: SELECT-NEIGHBORS-HEURISTIC
        Selects neighbors to maintain directional diversity and approximate Relative Neighborhood Graph.
        """
        w_sorted = sorted(candidates, key=lambda x: x[0])
        result_nodes: List[int] = []
        result_vectors: List[np.ndarray] = []
        discarded: List[Tuple[float, int]] = []
        
        for dist_q_e, e_node in w_sorted:
            if len(result_nodes) >= m_limit:
                break
            
            e_vec = self.data[e_node]
            # Check if e is closer to query than to any already chosen neighbor in result_nodes
            is_diverse = True
            for r_vec in result_vectors:
                dist_e_r = self._distance(e_vec, r_vec)
                if dist_e_r < dist_q_e:
                    is_diverse = False
                    break
            
            if is_diverse:
                result_nodes.append(e_node)
                result_vectors.append(e_vec)
            else:
                discarded.append((dist_q_e, e_node))
        
        # If not enough diverse neighbors and keep_pruned is True, backfill with closest discarded
        if keep_pruned and len(result_nodes) < m_limit and len(discarded) > 0:
            for _, disc_node in discarded:
                if len(result_nodes) >= m_limit:
                    break
                result_nodes.append(disc_node)
                
        return result_nodes

    def _select_neighbors(
        self,
        query: np.ndarray,
        candidates: List[Tuple[float, int]],
        m_limit: int
    ) -> List[int]:
        if self.heuristic:
            return self._select_neighbors_heuristic(query, candidates, m_limit)
        return self._select_neighbors_simple(query, candidates, m_limit)

    def insert(self, vector: np.ndarray) -> int:
        """
        Algorithm 1: INSERT(hnsw, q, M, Mmax, efConstruction, mL)
        Inserts a new vector into the HNSW graph index.
        """
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        q_idx = len(self.data)
        self.data.append(vector)
        
        # If graph is empty, initialize entry point
        if self.enter_point is None:
            self.enter_point = q_idx
            self.max_level = 0
            self.node_levels[q_idx] = 0
            self.graphs.append({q_idx: []})
            return q_idx
        
        # 1. Select random maximum level for new element
        q_level = self._generate_random_level()
        self.node_levels[q_idx] = q_level
        
        # Ensure graph levels exist
        while len(self.graphs) <= max(self.max_level, q_level):
            self.graphs.append({})
        
        curr_ep = self.enter_point
        top_level = self.max_level
        
        # Phase 1: Top-down greedy 1-NN traversal (ef=1) from top layer to q_level + 1
        for lc in range(top_level, q_level, -1):
            w = self._search_layer(vector, [curr_ep], ef=1, lc=lc)
            curr_ep = w[0][1]
        
        # Phase 2: Insert into levels min(top_level, q_level) down to 0
        for lc in range(min(top_level, q_level), -1, -1):
            w = self._search_layer(vector, [curr_ep], ef=self.ef_construction, lc=lc)
            
            # Select best neighbors for q at layer lc
            max_m = self.m_max0 if lc == 0 else self.m_max
            neighbors = self._select_neighbors(vector, w, self.m)
            
            if q_idx not in self.graphs[lc]:
                self.graphs[lc][q_idx] = []
            
            # Add bidirectional connections
            for neighbor in neighbors:
                self.graphs[lc][q_idx].append(neighbor)
                
                if neighbor not in self.graphs[lc]:
                    self.graphs[lc][neighbor] = []
                self.graphs[lc][neighbor].append(q_idx)
                
                # Shrink connections of neighbor if exceeding Mmax
                neighbor_m_max = self.m_max0 if lc == 0 else self.m_max
                if len(self.graphs[lc][neighbor]) > neighbor_m_max:
                    n_candidates = [(self._distance(self.data[neighbor], self.data[c]), c) 
                                    for c in self.graphs[lc][neighbor]]
                    pruned = self._select_neighbors(self.data[neighbor], n_candidates, neighbor_m_max)
                    self.graphs[lc][neighbor] = pruned
            
            curr_ep = w[0][1] if len(w) > 0 else curr_ep
        
        # Update entry point if new node has a higher level than current max_level
        if q_level > self.max_level:
            self.max_level = q_level
            self.enter_point = q_idx
            
        return q_idx

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        record_trace: bool = False
    ) -> Union[List[Tuple[float, int]], Tuple[List[Tuple[float, int]], Dict[str, Any]]]:
        """
        Algorithm 5: K-NN-SEARCH(hnsw, q, K, ef)
        Executes approximate K-nearest neighbor search.
        """
        if self.enter_point is None or len(self.data) == 0:
            return ([], {}) if record_trace else []
        
        query = np.ascontiguousarray(query, dtype=np.float32)
        ef_val = ef if ef is not None else max(k, self.ef_construction // 2)
        
        curr_ep = self.enter_point
        all_traces: List[Dict[str, Any]] = []
        
        # Top-down greedy search (ef=1) from top layer down to layer 1
        for lc in range(self.max_level, 0, -1):
            if record_trace:
                w, trace = self._search_layer(query, [curr_ep], ef=1, lc=lc, record_trace=True)
                all_traces.extend(trace)
            else:
                w = self._search_layer(query, [curr_ep], ef=1, lc=lc, record_trace=False)
            curr_ep = w[0][1]
        
        # Search layer 0 with dynamic candidate list size ef
        if record_trace:
            w, trace = self._search_layer(query, [curr_ep], ef=ef_val, lc=0, record_trace=True)
            all_traces.extend(trace)
        else:
            w = self._search_layer(query, [curr_ep], ef=ef_val, lc=0, record_trace=False)
            
        top_k = w[:k]
        
        if record_trace:
            trace_info = {
                "steps": all_traces,
                "total_dist_evals": len(all_traces),
                "top_k": top_k
            }
            return top_k, trace_info
        return top_k

    def get_stats(self) -> Dict[str, Any]:
        """Calculates index statistics (memory, edges, layers)."""
        total_nodes = len(self.data)
        if total_nodes == 0:
            return {"total_nodes": 0, "total_edges": 0, "avg_edges_per_node": 0.0, "memory_bytes": 0}
        
        total_edges = 0
        edges_per_layer = {}
        for lc, graph in enumerate(self.graphs):
            layer_edges = sum(len(neighbors) for neighbors in graph.values())
            edges_per_layer[lc] = layer_edges
            total_edges += layer_edges
            
        # Each edge link is stored as a 4-byte node ID
        # Plus dictionary/list overhead approx 16 bytes per entry in Python or 4 bytes in pure C++
        edge_memory_bytes = total_edges * 4
        vector_memory_bytes = total_nodes * self.dim * 4
        
        return {
            "total_nodes": total_nodes,
            "dim": self.dim,
            "max_level": self.max_level,
            "total_edges": total_edges,
            "avg_edges_per_node": total_edges / total_nodes,
            "edges_per_layer": edges_per_layer,
            "edge_memory_bytes": edge_memory_bytes,
            "vector_memory_bytes": vector_memory_bytes,
            "total_memory_mb": (edge_memory_bytes + vector_memory_bytes) / (1024 * 1024),
            "m": self.m,
            "ef_construction": self.ef_construction
        }
