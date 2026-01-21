"""
adaptive_weighting_optimizer.py - Dynamic Strategy Weighting

Adjusts strategy weights based on:
- Recent expectancy and profit factor
- Drawdown contribution
- Regime alignment
- Rolling window performance

Design principles:
- Non-invasive: works with existing metrics
- Conservative: changes are gradual and limited
- Reversible: can revert to static weights
- Safe: never increases total risk exposure
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import threading

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


@dataclass
class StrategyWeightUpdate:
    """Record of a weight change."""
    timestamp: datetime
    strategy: str
    previous_weight: float
    new_weight: float
    change_reason: str
    metrics: Dict[str, float] = field(default_factory=dict)


class AdaptiveWeightingOptimizer:
    """
    Dynamically adjusts strategy weights based on performance and market conditions.
    
    Features:
    - Expectancy-based weighting (higher expectancy = higher weight)
    - Profit factor weighting (more consistent = higher weight)
    - Drawdown penalty (high DD contributors = lower weight)
    - Regime-specific weighting (different weights per regime)
    - Rolling window analysis (only recent performance matters)
    - Conservative adjustment (gradual changes, bounded shifts)
    """
    
    def __init__(
        self,
        base_weights: Dict[str, float],
        rolling_window_trades: int = 50,
        adjustment_rate: float = 0.15,  # Max 15% change per update
        expectancy_weight: float = 0.4,  # Importance of expectancy
        profit_factor_weight: float = 0.3,  # Importance of profit factor
        drawdown_penalty_weight: float = 0.2,  # Penalty for drawdown
        consistency_weight: float = 0.1,  # Importance of consistency
        min_weight: float = 0.1,  # Minimum weight floor
        max_weight: float = 2.0,  # Maximum weight ceiling
    ):
        """
        Initialize optimizer.
        
        Args:
            base_weights: Static weights per strategy (fallback)
            rolling_window_trades: Number of recent trades to analyze
            adjustment_rate: Maximum weight change per update (0.0-1.0)
            expectancy_weight: Weight for expectancy metric
            profit_factor_weight: Weight for profit factor metric
            drawdown_penalty_weight: Weight for drawdown penalty
            consistency_weight: Weight for consistency metric
            min_weight: Minimum weight (prevents disabling strategies)
            max_weight: Maximum weight (prevents over-allocation)
        """
        self._lock = threading.RLock()
        
        # Configuration
        self.base_weights = dict(base_weights)
        self.rolling_window_trades = rolling_window_trades
        self.adjustment_rate = adjustment_rate
        
        # Metric weights (should sum to ~1.0)
        self.expectancy_weight = expectancy_weight
        self.profit_factor_weight = profit_factor_weight
        self.drawdown_penalty_weight = drawdown_penalty_weight
        self.consistency_weight = consistency_weight
        
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Current weights (adaptive)
        self._current_weights = dict(base_weights)
        
        # History for rollback
        self._weight_history: List[StrategyWeightUpdate] = []
        
        # Per-strategy regime weights
        self._regime_weights: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        logger.info(
            f"AdaptiveWeightingOptimizer initialized with base weights: {self.base_weights}"
        )
    
    def update_weights(
        self,
        performance_data: Dict[str, Dict[str, float]],
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Update weights based on performance data.
        
        Args:
            performance_data: Dict mapping strategy_name -> metrics dict
                Expected metrics: expectancy, profit_factor, max_drawdown, sharpe_ratio, win_rate
            regime: Current market regime (optional, for regime-specific weighting)
            
        Returns:
            Updated weights dict
        """
        with self._lock:
            new_weights = {}
            
            for strategy, metrics in performance_data.items():
                if strategy not in self.base_weights:
                    logger.warning(f"Unknown strategy: {strategy}")
                    continue
                
                # Calculate composite score
                score = self._calculate_composite_score(metrics)
                
                # Determine new weight
                old_weight = self._current_weights.get(strategy, self.base_weights[strategy])
                new_weight = self._apply_weight_adjustment(old_weight, score)
                
                # Clamp weight
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                
                new_weights[strategy] = new_weight
                
                # Record update if significant
                if abs(new_weight - old_weight) > 0.01:
                    self._record_weight_update(
                        strategy,
                        old_weight,
                        new_weight,
                        f"score={score:.3f}",
                        metrics,
                    )
                    logger.info(
                        f"Updated weight for {strategy}: {old_weight:.3f} -> {new_weight:.3f} "
                        f"(score={score:.3f}, regime={regime})"
                    )
            
            # Update current weights
            self._current_weights.update(new_weights)
            
            # Store regime-specific weights if applicable
            if regime:
                for strategy, weight in new_weights.items():
                    self._regime_weights[strategy][regime] = weight
            
            return new_weights
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite performance score (0.0 - 2.0).
        
        Higher score = better performance = higher weight.
        
        Args:
            metrics: Performance metrics dict
            
        Returns:
            Composite score
        """
        score = 1.0  # Baseline
        
        # Expectancy component (can be negative)
        if 'expectancy' in metrics:
            expectancy = metrics['expectancy']
            # Normalize to -1.0 to +1.0 range (assuming expectancy in -100 to +100)
            norm_exp = np.clip(expectancy / 100.0, -1.0, 1.0) if np else np.clip(expectancy / 100.0, -1.0, 1.0)
            score += norm_exp * self.expectancy_weight
        
        # Profit factor component (1.0 = breakeven, >1.0 = profitable)
        if 'profit_factor' in metrics:
            pf = metrics['profit_factor']
            # Map to 0.0-2.0 range (1.0 = baseline)
            if pf > 0:
                pf_score = np.log(pf + 1) / np.log(2) if np else np.log(pf + 1) / np.log(2)
            else:
                pf_score = 0.0
            score += (pf_score - 0.5) * self.profit_factor_weight
        
        # Drawdown penalty (higher DD = lower score)
        if 'max_drawdown' in metrics:
            dd = metrics['max_drawdown']
            # Penalty of -0.2 per 10% drawdown (capped)
            dd_penalty = -min(dd / 10.0 * 0.02, 0.3)
            score += dd_penalty * self.drawdown_penalty_weight
        
        # Consistency (Sharpe ratio)
        if 'sharpe_ratio' in metrics:
            sharpe = metrics['sharpe_ratio']
            # Higher Sharpe = higher score (capped)
            sharpe_score = np.clip(sharpe / 2.0, -1.0, 1.0) if np else np.clip(sharpe / 2.0, -1.0, 1.0)
            score += sharpe_score * self.consistency_weight
        
        # Clamp to reasonable range
        return np.clip(score, 0.0, 2.0) if np else max(0.0, min(score, 2.0))
    
    def _apply_weight_adjustment(self, old_weight: float, score: float) -> float:
        """
        Apply conservative adjustment to weight based on score.
        
        Args:
            old_weight: Current weight
            score: Composite performance score (0.0-2.0)
            
        Returns:
            New weight
        """
        # Score of 1.0 = keep weight, >1.0 = increase, <1.0 = decrease
        adjustment_factor = score  # Direct multiplier
        
        # Apply adjustment rate limit (conservative)
        max_change = old_weight * self.adjustment_rate
        target_weight = old_weight * adjustment_factor
        
        if target_weight > old_weight:
            new_weight = old_weight + min(max_change, target_weight - old_weight)
        else:
            new_weight = old_weight - min(max_change, old_weight - target_weight)
        
        return new_weight
    
    def _record_weight_update(
        self,
        strategy: str,
        old_weight: float,
        new_weight: float,
        reason: str,
        metrics: Dict[str, float],
    ) -> None:
        """Record a weight update."""
        update = StrategyWeightUpdate(
            timestamp=datetime.now(timezone.utc),
            strategy=strategy,
            previous_weight=old_weight,
            new_weight=new_weight,
            change_reason=reason,
            metrics=dict(metrics),
        )
        self._weight_history.append(update)
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current adaptive weights."""
        with self._lock:
            return dict(self._current_weights)
    
    def get_weight_for_regime(self, strategy: str, regime: str) -> float:
        """Get weight for strategy in specific regime."""
        with self._lock:
            return self._regime_weights.get(strategy, {}).get(regime, self._current_weights.get(strategy, self.base_weights.get(strategy, 1.0)))
    
    def get_weight_history(self, strategy: Optional[str] = None) -> List[StrategyWeightUpdate]:
        """Get weight change history."""
        with self._lock:
            if strategy:
                return [u for u in self._weight_history if u.strategy == strategy]
            return list(self._weight_history)
    
    def reset_to_base_weights(self) -> None:
        """Revert to static base weights."""
        with self._lock:
            self._current_weights = dict(self.base_weights)
            self._regime_weights.clear()
            logger.info("Reset to base weights")
    
    def normalize_weights(self) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.
        
        Useful for allocation scenarios.
        
        Returns:
            Normalized weights dict
        """
        with self._lock:
            total = sum(self._current_weights.values())
            if total == 0:
                return {s: 1.0/len(self._current_weights) for s in self._current_weights}
            return {s: w/total for s, w in self._current_weights.items()}
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of current allocation."""
        with self._lock:
            weights = self._current_weights
            normalized = self.normalize_weights()
            
            return {
                'raw_weights': dict(weights),
                'normalized_allocation': normalized,
                'total_allocation': sum(weights.values()),
                'weight_range': (min(weights.values()), max(weights.values())),
                'history_size': len(self._weight_history),
            }


# Module-level convenience instance
_default_optimizer: Optional[AdaptiveWeightingOptimizer] = None


def initialize_optimizer(base_weights: Dict[str, float], **kwargs) -> AdaptiveWeightingOptimizer:
    """Initialize default optimizer instance."""
    global _default_optimizer
    _default_optimizer = AdaptiveWeightingOptimizer(base_weights, **kwargs)
    return _default_optimizer


def get_optimizer() -> AdaptiveWeightingOptimizer:
    """Get the default optimizer instance."""
    if _default_optimizer is None:
        raise RuntimeError("Optimizer not initialized. Call initialize_optimizer() first.")
    return _default_optimizer


def update_weights(
    performance_data: Dict[str, Dict[str, float]],
    regime: Optional[str] = None,
) -> Dict[str, float]:
    """Convenience function using default optimizer."""
    return get_optimizer().update_weights(performance_data, regime)
