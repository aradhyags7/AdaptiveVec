"""
AdaptiveVec: Density- and Dimension-Aware Vector Index.
Extends Hierarchical Navigable Small World graphs with dynamic, per-node
edge limits (M_i, M_max_i, M_max0_i) and dynamic candidate beam width (efConstruction_i)
driven by fast online estimation of Local Density and Local Intrinsic Dimensionality (LID).
"""

import math
import random
import heapq
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Any, Union

from .signals import estimate_node_signals, compute_mle_lid, compute_local_density, profile_dataset_signals
from .policy import AdaptivePolicy, AdaptivePolicyConfig, NodeParameters

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.dot(diff, diff) ** 0.5)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 1.0
    cos_sim = np.dot(a, b) / (norm_a * norm_b)
    return float(max(0.0, 1.0 - cos_sim))

class AdaptiveHNSW:
    """
    AdaptiveVec Index with per-node dynamic parameter allocation based on
    Local Intrinsic Dimension (LID) and Local Density signals.
    """
    def __init__(
        self,
        dim: int,
        policy_config: Optional[AdaptivePolicyConfig] = None,
        space: str = "l2",
        heuristic: bool = True,
        adaptive_search_ef: bool = True,
        early_exit: bool = True,
        stagnation_patience: int = 6,
        stagnation_epsilon: float = 1e-4
    ):
        self.dim = dim
        self.space = space.lower()
        self.heuristic = heuristic
        self.adaptive_search_ef = adaptive_search_ef
        self.early_exit = early_exit
        self.stagnation_patience = stagnation_patience
        self.stagnation_epsilon = stagnation_epsilon
        
        self.policy = AdaptivePolicy(policy_config)
        self.dist_fn = cosine_distance if self.space == "cosine" else l2_distance
        
        # Base m_l for level generation
        self.m_l = 1.0 / math.log(self.policy.config.m_base) if self.policy.config.m_base > 1 else 1.0
        
        # Storage
        self.data: List[np.ndarray] = []
        self.graphs: List[Dict[int, List[int]]] = []
        self.node_levels: Dict[int, int] = {}
        
        # Per-node adaptive metadata
        self.node_params: Dict[int, NodeParameters] = {}
        self.node_signals: Dict[int, Dict[str, float]] = {}
        
        self.enter_point: Optional[int] = None
        self.max_level: int = -1
        
        # Performance & measurement counters
        self.total_dist_computations: int = 0
        self.signal_estimation_time_ms: float = 0.0

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        self.total_dist_computations += 1
        return self.dist_fn(a, b)

    def calibrate(self, sample_data: np.ndarray) -> Dict[str, Any]:
        """Profiles a representative sample to calibrate policy baseline statistics."""
        profile = profile_dataset_signals(sample_data, sample_size=min(500, len(sample_data)))
        self.policy.update_reference_stats(profile)
        return profile

    def _generate_random_level(self) -> int:
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
        record_trace: bool = False,
        early_exit: bool = False,
        stagnation_patience: int = 6,
        stagnation_epsilon: float = 1e-4
    ) -> Union[List[Tuple[float, int]], Tuple[List[Tuple[float, int]], List[Dict[str, Any]]]]:
        """
        Greedy beam search at layer lc with candidate list capacity ef.
        Includes distance stagnation early-exit criteria to terminate when
        the best discovered candidate distance ceases to improve.
        """
        visited: Set[int] = set(enter_points)
        candidates: List[Tuple[float, int]] = []
        w_furthest: List[Tuple[float, int]] = []
        trace_steps: List[Dict[str, Any]] = []

        best_dist = float("inf")
        stagnation_counter = 0

        for ep in enter_points:
            d = self._distance(query, self.data[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(w_furthest, (-d, ep))
            if d < best_dist:
                best_dist = d
            if record_trace:
                trace_steps.append({"action": "enter", "node": ep, "dist": d, "layer": lc})

        while len(candidates) > 0:
            c_dist, c_node = heapq.heappop(candidates)
            furthest_d = -w_furthest[0][0]
            
            if c_dist > furthest_d:
                break
                
            # Stagnation early-stopping check (active primarily on layer 0 search)
            if early_exit and len(w_furthest) >= min(ef, 10):
                current_best = min(-neg_d for neg_d, _ in w_furthest)
                if best_dist - current_best > stagnation_epsilon:
                    best_dist = current_best
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    if stagnation_counter >= stagnation_patience:
                        if record_trace:
                            trace_steps.append({"action": "early_exit", "node": c_node, "dist": current_best, "layer": lc, "stagnation": stagnation_counter})
                        break

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

        result = sorted([(-neg_d, node) for neg_d, node in w_furthest], key=lambda x: x[0])
        if record_trace:
            return result, trace_steps
        return result

    def _select_neighbors_heuristic(
        self,
        query: np.ndarray,
        candidates: List[Tuple[float, int]],
        m_limit: int,
        keep_pruned: bool = True
    ) -> List[int]:
        """
        Algorithm 4: Relative Neighborhood Graph heuristic edge selection.
        """
        w_sorted = sorted(candidates, key=lambda x: x[0])
        result_nodes: List[int] = []
        result_vectors: List[np.ndarray] = []
        discarded: List[Tuple[float, int]] = []
        
        for dist_q_e, e_node in w_sorted:
            if len(result_nodes) >= m_limit:
                break
            
            e_vec = self.data[e_node]
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
        sorted_c = sorted(candidates, key=lambda x: x[0])
        return [node for _, node in sorted_c[:m_limit]]

    def insert(self, vector: np.ndarray) -> int:
        """
        Adaptive Insertion:
        1. Explores upper layers greedily.
        2. Fast online estimation of local density and LID from routing candidates.
        3. Dynamically assigns M_i, M_max_i, efConstruction_i to the node.
        4. Inserts with personalized edge and candidate budgets.
        """
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        q_idx = len(self.data)
        self.data.append(vector)
        
        if self.enter_point is None:
            self.enter_point = q_idx
            self.max_level = 0
            self.node_levels[q_idx] = 0
            self.graphs.append({q_idx: []})
            
            # Default params for root node
            params = self.policy.evaluate(density=self.policy.config.density_mean, lid=self.policy.config.lid_mean)
            self.node_params[q_idx] = params
            self.node_signals[q_idx] = {"density": self.policy.config.density_mean, "lid": self.policy.config.lid_mean, "score": 0.0}
            return q_idx
        
        q_level = self._generate_random_level()
        self.node_levels[q_idx] = q_level
        
        while len(self.graphs) <= max(self.max_level, q_level):
            self.graphs.append({})
            
        curr_ep = self.enter_point
        top_level = self.max_level
        
        # Phase 1: Top-down greedy descent (ef=1)
        for lc in range(top_level, q_level, -1):
            w = self._search_layer(vector, [curr_ep], ef=1, lc=lc)
            curr_ep = w[0][1]
            
        # Fast Online Signal Estimation:
        # Run a probe search at level min(top_level, q_level) with baseline ef to estimate density and LID
        probe_w = self._search_layer(vector, [curr_ep], ef=min(30, max(15, self.policy.config.ef_construction_base // 4)), lc=min(top_level, q_level))
        if len(probe_w) >= 3:
            candidate_dists = np.array([d for d, _ in probe_w])
            density_est = compute_local_density(candidate_dists)
            lid_est = compute_mle_lid(candidate_dists)
        else:
            density_est = self.policy.config.density_mean
            lid_est = self.policy.config.lid_mean
            
        # Online streaming observation: updates running stats if uncalibrated
        self.policy.observe(density=density_est, lid=lid_est)
        
        # Evaluate Adaptive Policy for node q
        params = self.policy.evaluate(density=density_est, lid=lid_est)
        self.node_params[q_idx] = params
        self.node_signals[q_idx] = {
            "density": density_est,
            "lid": lid_est,
            "score": params.score,
            "m": params.m,
            "ef_c": params.ef_construction
        }
        
        # Phase 2: Insert into levels min(top_level, q_level) down to 0 using adaptive efConstruction
        for lc in range(min(top_level, q_level), -1, -1):
            # Dynamic candidate list size for this node
            w = self._search_layer(vector, [curr_ep], ef=params.ef_construction, lc=lc)
            
            # Select personalized M neighbors for q at layer lc
            m_layer = params.get_m_for_layer(lc)
            neighbors = self._select_neighbors(vector, w, m_layer)
            
            if q_idx not in self.graphs[lc]:
                self.graphs[lc][q_idx] = []
                
            for neighbor in neighbors:
                self.graphs[lc][q_idx].append(neighbor)
                
                if neighbor not in self.graphs[lc]:
                    self.graphs[lc][neighbor] = []
                self.graphs[lc][neighbor].append(q_idx)
                
                # Check neighbor's OWN personalized M_max capacity for layer lc!
                neighbor_params = self.node_params.get(neighbor, params)
                n_limit = neighbor_params.get_m_max_for_layer(lc)
                
                if len(self.graphs[lc][neighbor]) > n_limit:
                    n_candidates = [(self._distance(self.data[neighbor], self.data[c]), c) 
                                    for c in self.graphs[lc][neighbor]]
                    pruned = self._select_neighbors(self.data[neighbor], n_candidates, n_limit)
                    self.graphs[lc][neighbor] = pruned
            
            curr_ep = w[0][1] if len(w) > 0 else curr_ep
            
        if q_level > self.max_level:
            self.max_level = q_level
            self.enter_point = q_idx
            
        return q_idx

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        record_trace: bool = False,
        early_exit: Optional[bool] = None
    ) -> Union[List[Tuple[float, int]], Tuple[List[Tuple[float, int]], Dict[str, Any]]]:
        """
        K-NN Search with optional adaptive query-time ef modulation and early exit.
        """
        if self.enter_point is None or len(self.data) == 0:
            return ([], {}) if record_trace else []
            
        query = np.ascontiguousarray(query, dtype=np.float32)
        use_early_exit = self.early_exit if early_exit is None else early_exit
        
        curr_ep = self.enter_point
        all_traces: List[Dict[str, Any]] = []
        
        # Greedy descent down to layer 1
        for lc in range(self.max_level, 0, -1):
            if record_trace:
                w, trace = self._search_layer(query, [curr_ep], ef=1, lc=lc, record_trace=True)
                all_traces.extend(trace)
            else:
                w = self._search_layer(query, [curr_ep], ef=1, lc=lc, record_trace=False)
            curr_ep = w[0][1]
            
        # Determine search ef
        if ef is not None:
            ef_val = ef
        elif self.adaptive_search_ef and curr_ep in self.node_params:
            # Scale ef according to the local complexity around entry node
            ep_score = self.node_params[curr_ep].score
            base_ef = max(k, self.policy.config.ef_construction_base // 2)
            ef_val = int(np.clip(base_ef * (1.0 + 0.3 * ep_score), k, base_ef * 2))
        else:
            ef_val = max(k, self.policy.config.ef_construction_base // 2)
            
        if record_trace:
            w, trace = self._search_layer(
                query,
                [curr_ep],
                ef=ef_val,
                lc=0,
                record_trace=True,
                early_exit=use_early_exit,
                stagnation_patience=self.stagnation_patience,
                stagnation_epsilon=self.stagnation_epsilon
            )
            all_traces.extend(trace)
        else:
            w = self._search_layer(
                query,
                [curr_ep],
                ef=ef_val,
                lc=0,
                record_trace=False,
                early_exit=use_early_exit,
                stagnation_patience=self.stagnation_patience,
                stagnation_epsilon=self.stagnation_epsilon
            )
            
        top_k = w[:k]
        
        if record_trace:
            trace_info = {
                "steps": all_traces,
                "total_dist_evals": len(all_traces),
                "top_k": top_k,
                "used_ef": ef_val,
                "entry_point": curr_ep
            }
            return top_k, trace_info
        return top_k

    def get_stats(self) -> Dict[str, Any]:
        """Detailed stats on degrees, signals, and memory."""
        total_nodes = len(self.data)
        if total_nodes == 0:
            return {"total_nodes": 0, "total_edges": 0, "avg_edges_per_node": 0.0, "memory_bytes": 0}
            
        total_edges = 0
        edges_per_layer = {}
        layer_0_degrees = []
        
        for lc, graph in enumerate(self.graphs):
            layer_edges = sum(len(neighbors) for neighbors in graph.values())
            edges_per_layer[lc] = layer_edges
            total_edges += layer_edges
            if lc == 0:
                layer_0_degrees = [len(n) for n in graph.values()]
                
        edge_memory_bytes = total_edges * 4
        vector_memory_bytes = total_nodes * self.dim * 4
        
        # Signals summary
        lids = [s["lid"] for s in self.node_signals.values()] if self.node_signals else [0]
        densities = [s["density"] for s in self.node_signals.values()] if self.node_signals else [0]
        assigned_ms = [p.m for p in self.node_params.values()] if self.node_params else [0]
        
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
            "l0_degree_min": int(np.min(layer_0_degrees)) if layer_0_degrees else 0,
            "l0_degree_max": int(np.max(layer_0_degrees)) if layer_0_degrees else 0,
            "l0_degree_median": float(np.median(layer_0_degrees)) if layer_0_degrees else 0.0,
            "lid_avg": float(np.mean(lids)),
            "density_avg": float(np.mean(densities)),
            "m_assigned_avg": float(np.mean(assigned_ms)),
            "policy_type": self.policy.config.policy_type
        }
