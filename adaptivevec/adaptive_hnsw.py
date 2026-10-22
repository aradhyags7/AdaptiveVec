# Adaptive Hierarchical Navigable Small World (HNSW) Python Reference
# Native Python reference implementation with heuristic edge selection
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
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple, Optional, Any, Union
from .policy import AdaptivePolicy, AdaptivePolicyConfig, NodeParameters
@dataclass
class AdaptiveHNSWConfig:
    """Configuration container for AdaptiveHNSW index."""
    dim: int
    policy_config: Optional[AdaptivePolicyConfig] = None
    space: str = "l2"
    heuristic: bool = True
    adaptive_search_ef: bool = True
    early_exit: bool = True
    stagnation_patience: int = 6
    stagnation_epsilon: float = 1e-4
    hubness_regulation: bool = True
    hubness_penalty_weight: float = 0.15
    quantize: bool = False
from .signals import estimate_node_signals, compute_mle_lid, compute_local_density, profile_dataset_signals
from .policy import AdaptivePolicy, AdaptivePolicyConfig, NodeParameters
from .quantization import ScalarQuantizer, ScalarQuantizer4Bit

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
        dim: int = None,
        policy_config: Optional[AdaptivePolicyConfig] = None,
        space: str = "l2",
        heuristic: bool = True,
        adaptive_search_ef: bool = True,
        early_exit: bool = True,
        stagnation_patience: int = 6,
        stagnation_epsilon: float = 1e-4,
        hubness_regulation: bool = True,
        hubness_penalty_weight: float = 0.15,
        quantize: bool = False,
        config: Optional[AdaptiveHNSWConfig] = None,
    ):
        """
        Initialize AdaptiveHNSW.
        Accepts either legacy positional arguments or a single AdaptiveHNSWConfig instance.
        """
        # Resolve configuration
        if isinstance(dim, AdaptiveHNSWConfig):
            cfg = dim
        else:
            cfg = config or AdaptiveHNSWConfig(
                dim=dim,
                policy_config=policy_config,
                space=space,
                heuristic=heuristic,
                adaptive_search_ef=adaptive_search_ef,
                early_exit=early_exit,
                stagnation_patience=stagnation_patience,
                stagnation_epsilon=stagnation_epsilon,
                hubness_regulation=hubness_regulation,
                hubness_penalty_weight=hubness_penalty_weight,
                quantize=quantize,
            )

        # Assign attributes from config
        self.dim = cfg.dim
        self.space = cfg.space.lower()
        self.heuristic = cfg.heuristic
        self.adaptive_search_ef = cfg.adaptive_search_ef
        self.early_exit = cfg.early_exit
        self.stagnation_patience = cfg.stagnation_patience
        self.stagnation_epsilon = cfg.stagnation_epsilon
        self.hubness_regulation = cfg.hubness_regulation
        self.hubness_penalty_weight = cfg.hubness_penalty_weight
        self.quantize = cfg.quantize

        if isinstance(self.quantize, str) and self.quantize.lower() == "sq4":
            self.quantizer = ScalarQuantizer4Bit(dim=self.dim)
        else:
            self.quantizer = ScalarQuantizer(dim=self.dim)
        self.quantized_data: List[np.ndarray] = []

        self.policy = AdaptivePolicy(cfg.policy_config)
        self.dist_fn = cosine_distance if self.space == "cosine" else l2_distance

        # Base m_l for level generation
        self.m_l = 1.0 / math.log(self.policy.config.m_base) if self.policy.config.m_base > 1 else 1.0

        # Storage
        self.data: List[np.ndarray] = []
        self.graphs: List[Dict[int, List[int]]] = []
        self.node_levels: Dict[int, int] = {}
        self.in_degrees: Dict[int, int] = {}
        self.total_in_degrees: int = 0
        
        # Per-node adaptive metadata
        self.node_params: Dict[int, NodeParameters] = {}
        self.node_signals: Dict[int, Dict[str, float]] = {}
        
        # Frontier 2: Dynamic Streaming Node Deletions & Tombstones
        self.deleted: Set[int] = set()
        
        self.enter_point: Optional[int] = None
        self.max_level: int = -1
        
        # Performance & measurement counters
        self.total_dist_computations: int = 0
        self.signal_estimation_time_ms: float = 0.0

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        self.total_dist_computations += 1
        return self.dist_fn(a, b)

    def _query_distance(self, query: np.ndarray, node_idx: int) -> float:
        """Fast distance evaluation using asymmetric SQ8 if quantized or float32."""
        self.total_dist_computations += 1
        if self.quantize and len(self.quantized_data) > node_idx:
            if self.space == "cosine":
                return self.quantizer.asymmetric_cosine_distance(query, self.quantized_data[node_idx])
            return self.quantizer.asymmetric_l2_distance(query, self.quantized_data[node_idx])
        return self.dist_fn(query, self.data[node_idx])

    def calibrate(self, sample_data: np.ndarray) -> Dict[str, Any]:
        """Profiles a representative sample to calibrate policy baseline statistics."""
        if self.quantize and len(sample_data) > 0:
            self.quantizer.train(sample_data)
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
            d = self._query_distance(query, ep)
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
                
            improved = False
            neighbors = self.graphs[lc].get(c_node, [])
            for e_node in neighbors:
                if e_node not in visited:
                    visited.add(e_node)
                    e_dist = self._query_distance(query, e_node)
                    furthest_d = -w_furthest[0][0]
                    
                    if e_dist < furthest_d or len(w_furthest) < ef:
                        heapq.heappush(candidates, (e_dist, e_node))
                        
                        if len(w_furthest) >= ef:
                            old_f = -w_furthest[0][0]
                            heapq.heappop(w_furthest)
                            heapq.heappush(w_furthest, (-e_dist, e_node))
                            new_f = -w_furthest[0][0]
                            if old_f - new_f > stagnation_epsilon:
                                improved = True
                        else:
                            heapq.heappush(w_furthest, (-e_dist, e_node))
                            improved = True
                            
                        if best_dist - e_dist > stagnation_epsilon:
                            best_dist = e_dist
                            improved = True
                            
                        if record_trace:
                            trace_steps.append({"action": "explore", "from": c_node, "node": e_node, "dist": e_dist, "layer": lc})

            # Stagnation early-stopping check (active on layer 0 after beam has filled)
            if early_exit and lc == 0 and len(w_furthest) >= ef:
                if improved:
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    if stagnation_counter >= stagnation_patience:
                        if record_trace:
                            trace_steps.append({"action": "early_exit", "node": c_node, "dist": best_dist, "layer": lc, "stagnation": stagnation_counter})
                        break

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
        Algorithm 4: Relative Neighborhood Graph heuristic edge selection with
        optional Hubness Centrality penalty to prevent topological bottlenecking.
        """
        w_sorted = sorted(candidates, key=lambda x: x[0])
            
        result_nodes: List[int] = []
        result_vectors: List[np.ndarray] = []
        discarded: List[Tuple[float, int]] = []
        
        avg_deg = 1.0
        if self.hubness_regulation and len(self.in_degrees) > 0:
            avg_deg = max(1.0, float(self.total_in_degrees) / float(len(self.in_degrees)))
        
        for dist_q_e, e_node in w_sorted:
            if len(result_nodes) >= m_limit:
                break
            
            e_vec = self.data[e_node]
            is_diverse = True
            for r_node, r_vec in zip(result_nodes, result_vectors):
                dist_e_r = self._distance(e_vec, r_vec)
                if self.hubness_regulation and len(self.in_degrees) > 0:
                    dist_e_r *= (1.0 + self.hubness_penalty_weight * (self.in_degrees.get(r_node, 0) / avg_deg))
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
        if self.quantize:
            self.quantized_data.append(self.quantizer.encode(vector))
        
        if self.enter_point is None:
            self.enter_point = q_idx
            self.max_level = 0
            self.node_levels[q_idx] = 0
            self.graphs.append({q_idx: []})
            
            # Default params for root node
            params = self.policy.evaluate(density=self.policy.config.density_mean, lid=self.policy.config.lid_mean)
            self.node_params[q_idx] = params
            self.node_signals[q_idx] = {"density": self.policy.config.density_mean, "lid": self.policy.config.lid_mean, "score": 0.0}
            self.in_degrees[q_idx] = 0
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
                    
            # Maintain layer-0 in-degree centrality
            if lc == 0:
                old_deg = self.in_degrees.get(q_idx, 0)
                new_deg = len(self.graphs[0].get(q_idx, []))
                self.in_degrees[q_idx] = new_deg
                self.total_in_degrees += (new_deg - old_deg)

                for neighbor in neighbors:
                    old_n_deg = self.in_degrees.get(neighbor, 0)
                    new_n_deg = len(self.graphs[0].get(neighbor, []))
                    self.in_degrees[neighbor] = new_n_deg
                    self.total_in_degrees += (new_n_deg - old_n_deg)
            
            curr_ep = w[0][1] if len(w) > 0 else curr_ep
            
        if q_level > self.max_level:
            self.max_level = q_level
            self.enter_point = q_idx
            
        return q_idx

    def is_deleted(self, node_id: int) -> bool:
        """Checks if a node has been marked deleted."""
        return node_id in self.deleted

    def delete(self, node_id: int, heal: bool = True) -> bool:
        """
        Frontier 2: Deletes a node by setting a tombstone and optionally performing
        localized graph healing to reconnect severed predecessor paths.
        """
        if node_id >= len(self.data) or node_id in self.deleted:
            return False
            
        self.deleted.add(node_id)
        
        if heal:
            self.heal_neighborhood(node_id)
            
        if self.enter_point == node_id:
            self._migrate_enter_point()
            
        return True

    def _migrate_enter_point(self) -> None:
        """Re-homes the index entry point to an active non-deleted node."""
        for lc in range(self.max_level, -1, -1):
            if lc >= len(self.graphs):
                continue
            candidates = [u for u in self.graphs[lc].keys() if u not in self.deleted and len(self.graphs[lc][u]) > 0]
            if candidates:
                best_ep = max(candidates, key=lambda u: len(self.graphs[lc][u]))
                self.enter_point = best_ep
                self.max_level = lc
                return
                
        for u in range(len(self.data)):
            if u not in self.deleted:
                self.enter_point = u
                self.max_level = 0
                return
                
        self.enter_point = None
        self.max_level = -1

    def heal_neighborhood(self, v: int) -> None:
        """
        Frontier 2: Localized Graph Healing.
        Re-wires severed incoming edges to v's surviving neighbors, connects mutual
        neighbors to prevent topological voids, and cleans v's edge lists.
        """
        v_level = self.node_levels.get(v, 0)
        for lc in range(v_level, -1, -1):
            if lc >= len(self.graphs) or v not in self.graphs[lc]:
                continue
                
            v_neighbors = [nb for nb in self.graphs[lc][v] if nb not in self.deleted]
            
            # Sever u -> v and reconnect u to a surviving neighbor of v
            for u in v_neighbors:
                if u in self.deleted or u not in self.graphs[lc]:
                    continue
                if v in self.graphs[lc][u]:
                    self.graphs[lc][u].remove(v)
                    if lc == 0:
                        self.in_degrees[v] = max(0, self.in_degrees.get(v, 0) - 1)
                        self.in_degrees[u] = max(0, self.in_degrees.get(u, 0) - 1)
                        self.total_in_degrees = max(0, self.total_in_degrees - 2)
                        
                    repair_cands = [nb for nb in v_neighbors if nb != u and nb not in self.graphs[lc][u]]
                    if repair_cands:
                        best_rep = min(repair_cands, key=lambda nb: self._distance(self.data[u], self.data[nb]))
                        self.graphs[lc][u].append(best_rep)
                        if lc == 0:
                            self.in_degrees[best_rep] = self.in_degrees.get(best_rep, 0) + 1
                            self.in_degrees[u] = self.in_degrees.get(u, 0) + 1
                            self.total_in_degrees += 2
                            
                        # Reciprocal link best_rep -> u if capacity allows
                        if best_rep in self.graphs[lc] and u not in self.graphs[lc][best_rep]:
                            rep_limit = self.node_params.get(best_rep, NodeParameters(self.policy.config.m_base, self.policy.config.m_base, 2 * self.policy.config.m_base, self.policy.config.ef_construction_base, 0.0, 0.0, 0.0)).get_m_max_for_layer(lc)
                            if len(self.graphs[lc][best_rep]) < rep_limit:
                                self.graphs[lc][best_rep].append(u)
                                if lc == 0:
                                    self.in_degrees[u] = self.in_degrees.get(u, 0) + 1
                                    self.in_degrees[best_rep] = self.in_degrees.get(best_rep, 0) + 1
                                    self.total_in_degrees += 2
                                    
            # Mutual bridging between surviving neighbors of v
            if len(v_neighbors) >= 2:
                for i in range(len(v_neighbors)):
                    w1 = v_neighbors[i]
                    if w1 in self.deleted or w1 not in self.graphs[lc]:
                        continue
                    w1_limit = self.node_params.get(w1, NodeParameters(self.policy.config.m_base, self.policy.config.m_base, 2 * self.policy.config.m_base, self.policy.config.ef_construction_base, 0.0, 0.0, 0.0)).get_m_max_for_layer(lc)
                    for j in range(i + 1, len(v_neighbors)):
                        w2 = v_neighbors[j]
                        if w2 in self.deleted or w2 not in self.graphs[lc]:
                            continue
                        w2_limit = self.node_params.get(w2, NodeParameters(self.policy.config.m_base, self.policy.config.m_base, 2 * self.policy.config.m_base, self.policy.config.ef_construction_base, 0.0, 0.0, 0.0)).get_m_max_for_layer(lc)
                        
                        if len(self.graphs[lc][w1]) < w1_limit and w2 not in self.graphs[lc][w1]:
                            self.graphs[lc][w1].append(w2)
                            if lc == 0:
                                self.in_degrees[w2] = self.in_degrees.get(w2, 0) + 1
                                self.in_degrees[w1] = self.in_degrees.get(w1, 0) + 1
                                self.total_in_degrees += 2
                        if len(self.graphs[lc][w2]) < w2_limit and w1 not in self.graphs[lc][w2]:
                            self.graphs[lc][w2].append(w1)
                            if lc == 0:
                                self.in_degrees[w1] = self.in_degrees.get(w1, 0) + 1
                                self.in_degrees[w2] = self.in_degrees.get(w2, 0) + 1
                                self.total_in_degrees += 2
                                
            # Maintain outgoing links from v to only surviving neighbors (soft transit bridge)
            self.graphs[lc][v] = v_neighbors

    def compact(self) -> Dict[str, Any]:
        """
        Reclaims tombstoned slots, maps surviving nodes to contiguous indices,
        and eliminates memory fragmentation.
        """
        if not self.deleted:
            return {"compacted_nodes": len(self.data), "removed_tombstones": 0}
            
        old_count = len(self.data)
        id_map: Dict[int, int] = {}
        new_count = 0
        for old_id in range(old_count):
            if old_id not in self.deleted:
                id_map[old_id] = new_count
                new_count += 1
                
        new_data = [self.data[old_id] for old_id in range(old_count) if old_id in id_map]
        new_quantized = [self.quantized_data[old_id] for old_id in range(old_count) if old_id in id_map] if self.quantized_data else []
        new_node_levels = {id_map[old_id]: self.node_levels[old_id] for old_id in id_map if old_id in self.node_levels}
        new_node_params = {id_map[old_id]: self.node_params[old_id] for old_id in id_map if old_id in self.node_params}
        new_node_signals = {id_map[old_id]: self.node_signals[old_id] for old_id in id_map if old_id in self.node_signals}
        
        new_graphs = []
        for lc, graph in enumerate(self.graphs):
            new_graph = {}
            for old_u, neighbors in graph.items():
                if old_u in id_map:
                    new_u = id_map[old_u]
                    new_neighbors = [id_map[nb] for nb in neighbors if nb in id_map]
                    new_graph[new_u] = new_neighbors
            new_graphs.append(new_graph)
            
        self.data = new_data
        self.quantized_data = new_quantized
        self.node_levels = new_node_levels
        self.node_params = new_node_params
        self.node_signals = new_node_signals
        self.graphs = new_graphs
        
        num_removed = len(self.deleted)
        self.deleted.clear()
        
        self.in_degrees.clear()
        self.total_in_degrees = 0
        if self.graphs:
            for u, neighbors in self.graphs[0].items():
                self.in_degrees[u] = len(neighbors)
                self.total_in_degrees += len(neighbors)
                
        if self.enter_point is not None and self.enter_point in id_map:
            self.enter_point = id_map[self.enter_point]
        else:
            self._migrate_enter_point()
            
        return {"compacted_nodes": new_count, "removed_tombstones": num_removed}

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
            
        if self.enter_point in self.deleted:
            self._migrate_enter_point()
            if self.enter_point is None:
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
            
        # Determine search ef, expanding if tombstones are present
        if ef is not None:
            ef_val = ef
        elif self.adaptive_search_ef and curr_ep in self.node_params:
            # Scale ef according to the local complexity around entry node
            ep_score = self.node_params[curr_ep].score
            base_ef = max(k, self.policy.config.ef_construction_base // 2)
            ef_val = int(np.clip(base_ef * (1.0 + 0.3 * ep_score), k, base_ef * 2))
        else:
            ef_val = max(k, self.policy.config.ef_construction_base // 2)

        if self.deleted and len(self.data) > 0:
            tombstone_boost = int(ef_val * (len(self.deleted) / len(self.data)))
            ef_val = min(ef_val * 2, ef_val + tombstone_boost)
            
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
            
        if self.quantize:
            # Two-stage re-ranking: evaluate exact float32 distance for top candidate beam excluding deleted
            rerank_count = max(k * 2, min(len(w), 30))
            rerank_candidates = [cand for cand in w if cand[1] not in self.deleted][:rerank_count]
            exact_results = []
            for _, node in rerank_candidates:
                exact_d = self.dist_fn(query, self.data[node])
                exact_results.append((exact_d, node))
            exact_results.sort(key=lambda x: x[0])
            top_k = exact_results[:k]
        else:
            # Filter out deleted nodes
            top_k = [cand for cand in w if cand[1] not in self.deleted][:k]
        
        if record_trace:
            dist_evals = sum(1 for s in all_traces if s.get("action") in ("enter", "explore"))
            trace_info = {
                "steps": all_traces,
                "total_dist_evals": dist_evals,
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
        vector_bytes_per_dim = 1 if self.quantize else 4
        vector_memory_bytes = total_nodes * self.dim * vector_bytes_per_dim
        
        # Signals summary
        lids = [s["lid"] for s in self.node_signals.values()] if self.node_signals else [0]
        densities = [s["density"] for s in self.node_signals.values()] if self.node_signals else [0]
        assigned_ms = [p.m for p in self.node_params.values()] if self.node_params else [0]
        
        active_nodes = total_nodes - len(self.deleted)
        return {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "tombstones": len(self.deleted),
            "tombstone_ratio": float(len(self.deleted) / total_nodes) if total_nodes > 0 else 0.0,
            "dim": self.dim,
            "max_level": self.max_level,
            "total_edges": total_edges,
            "avg_edges_per_node": total_edges / max(1, active_nodes),
            "edges_per_layer": edges_per_layer,
            "edge_memory_bytes": edge_memory_bytes,
            "vector_memory_bytes": vector_memory_bytes,
            "total_memory_mb": (edge_memory_bytes + vector_memory_bytes) / (1024 * 1024),
            "quantized": self.quantize,
            "l0_degree_min": int(np.min(layer_0_degrees)) if layer_0_degrees else 0,
            "l0_degree_max": int(np.max(layer_0_degrees)) if layer_0_degrees else 0,
            "l0_degree_median": float(np.median(layer_0_degrees)) if layer_0_degrees else 0.0,
            "lid_avg": float(np.mean(lids)),
            "density_avg": float(np.mean(densities)),
            "m_assigned_avg": float(np.mean(assigned_ms)),
            "policy_type": self.policy.config.policy_type
        }
