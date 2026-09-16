"""
Adaptive Policy Mapper for AdaptiveVec.

Maps estimated local signals (Density, LID, Variance) to per-node construction parameters:
- M: Number of bi-directional links to establish for node i
- M_max: Maximum allowed connections in upper layers for node i
- M_max0: Maximum allowed connections in layer 0 (default 2 * M)
- efConstruction: Dynamic candidate list size during insertion
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Literal
import math
import numpy as np

PolicyType = Literal["continuous", "quantile", "lid_only", "density_only", "budget_constrained"]

@dataclass
class AdaptivePolicyConfig:
    """Configuration for AdaptiveVec policy mapping."""
    policy_type: PolicyType = "continuous"
    m_base: int = 16
    m_min: int = 8
    m_max: int = 24
    ef_construction_base: int = 120
    ef_construction_min: int = 40
    ef_construction_max: int = 220
    
    # Weighting factors
    alpha_lid: float = 0.5       # Weight of LID signal
    beta_density: float = 0.5    # Weight of Density signal
    sensitivity: float = 0.4     # Sensitivity factor gamma
    
    # Layer Scaling
    lambda_layer: float = 0.75
    m_min_layer: int = 4
    
    # Reference stats (calibrated via dataset profiling or running estimates)
    density_mean: float = 1.0
    density_std: float = 1.0
    lid_mean: float = 4.0
    lid_std: float = 2.0
    
    # Streaming Online Calibration
    enable_streaming_calibration: bool = True
    use_ema: bool = False
    ema_alpha: float = 0.05
    
    # Memory Budget Constraint (in Megabytes, optional)
    target_memory_mb: Optional[float] = None
    bytes_per_link: int = 4

class StreamingStatsTracker:
    """
    Online single-pass running mean and variance estimator using Welford's algorithm
    with optional exponential moving average (EMA) for non-stationary stream tracking.
    """
    def __init__(self, use_ema: bool = False, ema_alpha: float = 0.05):
        self.count: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0
        self.use_ema: bool = use_ema
        self.ema_alpha: float = ema_alpha

    def update(self, x: float) -> Tuple[float, float]:
        self.count += 1
        if self.use_ema and self.count > 10:
            delta = x - self.mean
            self.mean += self.ema_alpha * delta
            self.M2 = (1.0 - self.ema_alpha) * (self.M2 + self.ema_alpha * delta * delta)
            var = max(1e-6, self.M2)
        else:
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2
            var = self.M2 / (self.count - 1) if self.count > 1 else 1.0
            
        std = float(np.sqrt(max(1e-6, var)))
        return self.mean, std

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 1.0
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(1e-6, self.variance)))

@dataclass
class NodeParameters:
    """Parameters assigned to an individual vector node with layer-decoupled scaling."""
    m: int
    m_max: int
    m_max0: int
    ef_construction: int
    score: float
    lid: float
    density: float

    def get_m_for_layer(self, layer: int) -> int:
        """Dynamic edge budget scaled geometrically by layer hierarchy."""
        if layer == 0:
            return self.m
        return max(4, int(math.floor(self.m * (0.75 ** layer))))

    def get_m_max_for_layer(self, layer: int) -> int:
        """Dynamic maximum edge capacity for neighbor pruning at layer lc."""
        if layer == 0:
            return self.m_max0
        return max(4, int(math.floor(self.m_max * (0.75 ** layer))))

class AdaptivePolicy:
    """Policy engine mapping local signals to graph hyper-parameters."""
    
    def __init__(self, config: Optional[AdaptivePolicyConfig] = None):
        self.config = config or AdaptivePolicyConfig()
        self.density_tracker = StreamingStatsTracker(
            use_ema=self.config.use_ema,
            ema_alpha=self.config.ema_alpha
        )
        self.lid_tracker = StreamingStatsTracker(
            use_ema=self.config.use_ema,
            ema_alpha=self.config.ema_alpha
        )
        self.is_calibrated: bool = False
    
    def observe(self, density: float, lid: float) -> Tuple[float, float]:
        """
        Observes a newly discovered candidate neighborhood and dynamically updates
        the running mean and variance without requiring offline calibration sweeps.
        """
        d_mean, d_std = self.density_tracker.update(density)
        l_mean, l_std = self.lid_tracker.update(lid)
        
        if self.config.enable_streaming_calibration and not self.is_calibrated:
            if self.density_tracker.count >= 5:
                self.config.density_mean = d_mean
                self.config.density_std = d_std
                self.config.lid_mean = l_mean
                self.config.lid_std = l_std
        return d_mean, l_mean

    def update_reference_stats(self, profile: Dict[str, Any]) -> None:
        """Updates baseline reference statistics from dataset profiling."""
        self.is_calibrated = True
        if "density_mean" in profile:
            self.config.density_mean = profile["density_mean"]
        if "density_std" in profile:
            self.config.density_std = profile["density_std"] if profile["density_std"] > 1e-6 else 1.0
        if "lid_mean" in profile:
            self.config.lid_mean = profile["lid_mean"]
        if "lid_std" in profile:
            self.config.lid_std = profile["lid_std"] if profile["lid_std"] > 1e-6 else 1.0

    def compute_difficulty_score(self, density: float, lid: float) -> float:
        """
        Computes a normalized difficulty score.
        Positive -> higher intrinsic complexity (needs higher M / efC).
        Negative -> low intrinsic complexity / dense (can save memory with lower M).
        """
        norm_lid = (lid - self.config.lid_mean) / (self.config.lid_std + 1e-6)
        norm_density = (density - self.config.density_mean) / (self.config.density_std + 1e-6)
        
        if self.config.policy_type == "lid_only":
            return float(np.clip(norm_lid, -3.0, 3.0))
        elif self.config.policy_type == "density_only":
            # Higher distance means lower density -> more difficult
            return float(np.clip(norm_density, -3.0, 3.0))
        else:
            combined = (self.config.alpha_lid * norm_lid) + (self.config.beta_density * norm_density)
            return float(np.clip(combined, -3.0, 3.0))

    def evaluate(self, density: float, lid: float) -> NodeParameters:
        """
        Evaluates signals and returns the specific M, M_max, M_max0, and efConstruction for the node.
        """
        score = self.compute_difficulty_score(density, lid)
        
        if self.config.policy_type == "quantile":
            # 3-tier discrete policy
            if score < -0.5:
                # Dense / Low-LID
                m = self.config.m_min
                ef_c = self.config.ef_construction_min
            elif score > 0.5:
                # Sparse / High-LID
                m = self.config.m_max
                ef_c = self.config.ef_construction_max
            else:
                m = self.config.m_base
                ef_c = self.config.ef_construction_base
        else:
            # Continuous linear modulation
            factor = 1.0 + self.config.sensitivity * score
            m_raw = self.config.m_base * factor
            ef_raw = self.config.ef_construction_base * factor
            
            m = int(np.clip(round(m_raw), self.config.m_min, self.config.m_max))
            ef_c = int(np.clip(round(ef_raw), self.config.ef_construction_min, self.config.ef_construction_max))
        
        m_max = m
        m_max0 = 2 * m
        
        return NodeParameters(
            m=m,
            m_max=m_max,
            m_max0=m_max0,
            ef_construction=ef_c,
            score=score,
            lid=lid,
            density=density
        )

def map_signals_to_params(
    density: float,
    lid: float,
    config: Optional[AdaptivePolicyConfig] = None
) -> NodeParameters:
    """Convenience functional helper to evaluate policy."""
    policy = AdaptivePolicy(config)
    return policy.evaluate(density, lid)
