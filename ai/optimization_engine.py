"""
optimization_engine.py - Unified Optimization Engine

Integrates all optimization components into a single coordinated system:
- Performance decomposition
- Adaptive weighting
- Signal quality scoring
- Strategy retirement

This module acts as the orchestrator, ensuring all components work together
without violating risk constraints.

Design:
- Non-invasive: purely advisory
- Risk-aware: respects existing risk boundaries
- Coordinated: changes are synchronized across components
- Observable: all actions are logged and traceable
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import threading

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


@dataclass
class OptimizationState:
    """Current state of all optimizations."""
    timestamp: datetime
    
    # Weighting state
    current_weights: Dict[str, float]
    weight_changes_this_period: List[str] = field(default_factory=list)
    
    # Signal quality state
    high_quality_signals: int = 0
    average_quality_score: float = 0.0
    
    # Strategy retirement state
    active_strategies: List[str] = field(default_factory=list)
    degraded_strategies: List[str] = field(default_factory=list)
    suspended_strategies: List[str] = field(default_factory=list)
    
    # Performance metrics
    best_strategy: Optional[str] = None
    worst_strategy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'current_weights': self.current_weights,
            'weight_changes_this_period': self.weight_changes_this_period,
            'high_quality_signals': self.high_quality_signals,
            'average_quality_score': self.average_quality_score,
            'active_strategies': self.active_strategies,
            'degraded_strategies': self.degraded_strategies,
            'suspended_strategies': self.suspended_strategies,
            'best_strategy': self.best_strategy,
            'worst_strategy': self.worst_strategy,
        }


class OptimizationEngine:
    """
    Unified optimization engine coordinating all AI advisory components.
    
    Responsibilities:
    - Coordinate updates between components
    - Ensure risk constraints are maintained
    - Track and log optimization actions
    - Provide holistic optimization status
    """
    
    def __init__(
        self,
        base_weights: Dict[str, float],
        risk_max_drawdown: float = 0.20,  # Max portfolio drawdown
        risk_max_position_size: float = 0.05,  # Max per-position
        enable_quality_filtering: bool = True,
        enable_adaptive_weighting: bool = True,
        enable_retirement_management: bool = True,
    ):
        """
        Initialize optimization engine.
        
        Args:
            base_weights: Base strategy weights
            risk_max_drawdown: Maximum allowed portfolio drawdown
            risk_max_position_size: Maximum position size
            enable_quality_filtering: Enable signal quality filtering
            enable_adaptive_weighting: Enable adaptive weights
            enable_retirement_management: Enable strategy retirement
        """
        self._lock = threading.RLock()
        
        # Configuration
        self.base_weights = dict(base_weights)
        self.risk_max_drawdown = risk_max_drawdown
        self.risk_max_position_size = risk_max_position_size
        self.enable_quality_filtering = enable_quality_filtering
        self.enable_adaptive_weighting = enable_adaptive_weighting
        self.enable_retirement_management = enable_retirement_management
        
        # Component initialization (lazy)
        self._decomposer = None
        self._weighting_optimizer = None
        self._quality_scorer = None
        self._retirement_manager = None
        
        # State tracking
        self._last_optimization_state: Optional[OptimizationState] = None
        self._optimization_history: List[OptimizationState] = []
        self._action_log: List[Dict[str, Any]] = []
        
        logger.info(
            f"OptimizationEngine initialized: "
            f"quality={enable_quality_filtering}, "
            f"weighting={enable_adaptive_weighting}, "
            f"retirement={enable_retirement_management}"
        )
    
    def _ensure_components(self) -> None:
        """Lazy-initialize components."""
        if self._decomposer is None:
            try:
                from ai.performance_decomposer import get_decomposer
                self._decomposer = get_decomposer()
            except Exception as e:
                logger.error(f"Failed to initialize decomposer: {e}")
        
        if self._weighting_optimizer is None and self.enable_adaptive_weighting:
            try:
                from ai.adaptive_weighting_optimizer import initialize_optimizer, get_optimizer
                try:
                    # If a global optimizer exists, ensure its base weights align
                    existing = get_optimizer()
                    try:
                        if set(existing.base_weights.keys()) != set(self.base_weights.keys()):
                            logger.info("Reinitializing optimizer to match engine base_weights")
                            self._weighting_optimizer = initialize_optimizer(self.base_weights)
                        else:
                            self._weighting_optimizer = existing
                    except Exception:
                        # Fallback to reinitializing if introspection fails
                        self._weighting_optimizer = initialize_optimizer(self.base_weights)
                except RuntimeError:
                    self._weighting_optimizer = initialize_optimizer(self.base_weights)
            except Exception as e:
                logger.error(f"Failed to initialize weighting optimizer: {e}")
        
        if self._quality_scorer is None and self.enable_quality_filtering:
            try:
                # Create an engine-local scorer instance with a slightly
                # more permissive threshold so engine-level scoring
                # behavior matches optimization expectations while
                # leaving the module-level default untouched.
                from ai.signal_quality_scorer import SignalQualityScorer
                self._quality_scorer = SignalQualityScorer(min_quality_threshold=0.6)
            except Exception as e:
                logger.error(f"Failed to initialize quality scorer: {e}")
        
        if self._retirement_manager is None and self.enable_retirement_management:
            try:
                from ai.strategy_retirement import initialize_manager, get_manager
                try:
                    self._retirement_manager = get_manager()
                except RuntimeError:
                    self._retirement_manager = initialize_manager()
            except Exception as e:
                logger.error(f"Failed to initialize retirement manager: {e}")
            # Fallback: instantiate directly if import paths fail
            if self._retirement_manager is None:
                try:
                    from ai.strategy_retirement import StrategyRetirementManager
                    self._retirement_manager = StrategyRetirementManager()
                except Exception as e:
                    logger.error(f"Fallback failed to create retirement manager: {e}")
    
    def update_optimization(self, regime: Optional[str] = None) -> OptimizationState:
        """
        Perform comprehensive optimization update across all components.
        
        Args:
            regime: Current market regime (optional)
            
        Returns:
            OptimizationState with current state
        """
        with self._lock:
            self._ensure_components()
            
            state = OptimizationState(
                timestamp=datetime.now(timezone.utc),
                current_weights=dict(self.base_weights),
            )
            
            # Step 1: Get performance decomposition
            if self._decomposer:
                all_perf = self._decomposer.get_all_performance()
                
                if all_perf:
                    # Find best/worst by expectancy
                    perf_by_expectancy = sorted(
                        all_perf.values(),
                        key=lambda p: p.expectancy if p.symbol is None and p.regime is None else -999,
                        reverse=True
                    )
                    if perf_by_expectancy:
                        state.best_strategy = perf_by_expectancy[0].strategy_name
                        state.worst_strategy = perf_by_expectancy[-1].strategy_name
            
            # Step 2: Update adaptive weights
            if self._weighting_optimizer and self._decomposer:
                try:
                    # Prepare performance data for optimizer
                    perf_data = {}
                    for key, perf in self._decomposer.get_all_performance().items():
                        # Only use overall strategy performance (no symbol/regime filter)
                        if perf.symbol is None and perf.regime is None:
                            perf_data[perf.strategy_name] = {
                                'expectancy': perf.expectancy,
                                'profit_factor': perf.profit_factor,
                                'max_drawdown': perf.max_drawdown,
                                'sharpe_ratio': perf.sharpe_ratio,
                                'win_rate': perf.win_rate,
                            }
                    
                    if perf_data:
                        new_weights = self._weighting_optimizer.update_weights(perf_data, regime)
                        state.current_weights = new_weights
                        
                        # Log weight changes
                        for strategy, new_weight in new_weights.items():
                            old_weight = self.base_weights.get(strategy, 1.0)
                            if abs(new_weight - old_weight) > 0.01:
                                state.weight_changes_this_period.append(
                                    f"{strategy}: {old_weight:.3f} -> {new_weight:.3f}"
                                )
                except Exception as e:
                    logger.error(f"Error updating adaptive weights: {e}")
            
            # Step 3: Assess strategy retirement
            if self._retirement_manager and self._decomposer:
                try:
                    for strategy_name in self.base_weights.keys():
                        perf = self._decomposer.get_strategy_performance(strategy_name)
                        if perf:
                            self._retirement_manager.assess_strategy(
                                strategy_name,
                                recent_expectancy=perf.expectancy,
                                recent_profit_factor=perf.profit_factor,
                                recent_win_rate=perf.win_rate,
                                recent_trades_count=perf.total_trades,
                                max_drawdown_recent=perf.max_drawdown,
                                consecutive_losses=perf.consecutive_losses,
                                max_consecutive_losses=perf.max_consecutive_losses,
                            )
                    
                    # Gather strategy statuses
                    state.active_strategies = self._retirement_manager.get_active_strategies()
                    state.degraded_strategies = self._retirement_manager.get_degraded_strategies()
                    state.suspended_strategies = self._retirement_manager.get_suspended_strategies()
                except Exception as e:
                    logger.error(f"Error assessing strategy retirement: {e}")
            
            # Step 4: Get signal quality metrics
            if self._quality_scorer:
                try:
                    signal_history = self._quality_scorer.get_signal_history(limit=100)
                    if signal_history:
                        state.high_quality_signals = sum(
                            1 for s in signal_history
                            if s.quality_score >= self._quality_scorer.min_quality_threshold
                        )
                        state.average_quality_score = sum(
                            s.quality_score for s in signal_history
                        ) / len(signal_history)
                except Exception as e:
                    logger.error(f"Error gathering signal quality metrics: {e}")
            
            # Record state
            self._last_optimization_state = state
            self._optimization_history.append(state)
            
            # Keep history bounded
            if len(self._optimization_history) > 1000:
                self._optimization_history = self._optimization_history[-1000:]
            
            logger.info(
                f"Optimization update: {len(state.active_strategies)} active, "
                f"{len(state.degraded_strategies)} degraded, "
                f"{len(state.suspended_strategies)} suspended"
            )
            
            return state
    
    def score_signal(
        self,
        strategy: str,
        symbol: str,
        signal_type: str,
        original_confidence: float,
        confluence_count: int = 1,
        regime: Optional[str] = None,
    ) -> Tuple[float, bool]:
        """
        Score a signal and determine if it should be executed.
        
        Args:
            strategy: Strategy name
            symbol: Trading pair
            signal_type: BUY, SELL, HOLD
            original_confidence: Original confidence (0.0-1.0)
            confluence_count: Number of confirming indicators
            regime: Current market regime
            
        Returns:
            (quality_score, should_execute) tuple
        """
        with self._lock:
            self._ensure_components()
            
            if not self.enable_quality_filtering or not self._quality_scorer:
                # Quality filtering disabled, execute all signals
                return (original_confidence, True)
            
            try:
                metrics = self._quality_scorer.score_signal(
                    strategy=strategy,
                    symbol=symbol,
                    signal_type=signal_type,
                    original_confidence=original_confidence,
                    confluence_count=confluence_count,
                    regime=regime,
                )
                
                should_execute = metrics.quality_score >= self._quality_scorer.min_quality_threshold
                
                logger.debug(
                    f"Signal scored: {strategy}/{symbol} {signal_type} "
                    f"quality={metrics.quality_score:.3f} execute={should_execute}"
                )
                
                return (metrics.quality_score, should_execute)
            except Exception as e:
                logger.error(f"Error scoring signal: {e}")
                # Fallback: execute at reduced confidence
                return (original_confidence * 0.8, True)
    
    def get_strategy_weight(self, strategy: str, regime: Optional[str] = None) -> float:
        """
        Get effective weight for a strategy.
        
        Considers:
        - Adaptive weighting (if enabled)
        - Retirement status (suspended = 0.0)
        - Regime alignment (if applicable)
        
        Args:
            strategy: Strategy name
            regime: Current market regime (optional)
            
        Returns:
            Effective weight (0.0 - max_weight)
        """
        with self._lock:
            self._ensure_components()
            
            # Start with base weight
            weight = self.base_weights.get(strategy, 1.0)
            
            # Check retirement status
            if self._retirement_manager:
                health = self._retirement_manager.get_strategy_health(strategy)
                if health:
                    from ai.strategy_retirement import StrategyStatus
                    
                    if health.status == StrategyStatus.SUSPENDED:
                        return 0.0  # Suspended strategies get zero weight
                    elif health.status == StrategyStatus.DEGRADED:
                        weight *= 0.5  # Degraded strategies get half weight
                    elif health.status == StrategyStatus.RETIRED:
                        return 0.0  # Retired strategies get zero weight
            
            # Apply adaptive weighting
            if self._weighting_optimizer and regime:
                regime_weight = self._weighting_optimizer.get_weight_for_regime(strategy, regime)
                weight = regime_weight
            elif self._weighting_optimizer:
                adaptive_weights = self._weighting_optimizer.get_current_weights()
                if strategy in adaptive_weights:
                    weight = adaptive_weights[strategy]
            
            return max(0.0, weight)
    
    def get_position_size_multiplier(
        self,
        strategy: str,
        signal_quality: float,
        regime: Optional[str] = None,
    ) -> float:
        """
        Calculate position size multiplier based on optimization state.
        
        Considers:
        - Strategy weight
        - Signal quality
        - Strategy retirement status
        
        Args:
            strategy: Strategy name
            signal_quality: Quality score of the signal (0.0-1.0)
            regime: Current market regime
            
        Returns:
            Multiplier (0.0-2.0) to apply to base position size
        """
        with self._lock:
            self._ensure_components()
            
            # Base multiplier from strategy weight
            weight_multiplier = self.get_strategy_weight(strategy, regime)
            
            # Apply signal quality (0.5x to 1.5x)
            quality_multiplier = 0.5 + (signal_quality * 1.0)
            
            # Combined multiplier
            multiplier = weight_multiplier * quality_multiplier
            
            # Clamp to risk bounds
            # Never let optimization increase position beyond 2x base
            multiplier = max(0.0, min(2.0, multiplier))
            
            return multiplier
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        with self._lock:
            self._ensure_components()
            
            status = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'enabled': {
                    'quality_filtering': self.enable_quality_filtering,
                    'adaptive_weighting': self.enable_adaptive_weighting,
                    'retirement_management': self.enable_retirement_management,
                },
                'risk_limits': {
                    'max_drawdown': self.risk_max_drawdown,
                    'max_position_size': self.risk_max_position_size,
                },
                'components_initialized': {
                    'decomposer': self._decomposer is not None,
                    'weighting_optimizer': self._weighting_optimizer is not None,
                    'quality_scorer': self._quality_scorer is not None,
                    'retirement_manager': self._retirement_manager is not None,
                },
            }
            
            if self._last_optimization_state:
                status['last_state'] = self._last_optimization_state.to_dict()
            
            return status
    
    def get_optimization_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent optimization state history."""
        with self._lock:
            recent = self._optimization_history[-limit:]
            return [s.to_dict() for s in recent]


# Module-level convenience instance
_default_engine: Optional[OptimizationEngine] = None


def initialize_engine(
    base_weights: Dict[str, float],
    **kwargs
) -> OptimizationEngine:
    """Initialize default optimization engine."""
    global _default_engine
    _default_engine = OptimizationEngine(base_weights, **kwargs)
    return _default_engine


def get_engine() -> OptimizationEngine:
    """Get the default optimization engine."""
    if _default_engine is None:
        raise RuntimeError("Optimization engine not initialized. Call initialize_engine() first.")
    return _default_engine


def update_optimization(regime: Optional[str] = None) -> OptimizationState:
    """Convenience function using default engine."""
    return get_engine().update_optimization(regime)


def score_signal(
    strategy: str,
    symbol: str,
    signal_type: str,
    original_confidence: float,
    **kwargs,
) -> Tuple[float, bool]:
    """Convenience function using default engine."""
    return get_engine().score_signal(strategy, symbol, signal_type, original_confidence, **kwargs)


def get_strategy_weight(strategy: str, regime: Optional[str] = None) -> float:
    """Convenience function using default engine."""
    return get_engine().get_strategy_weight(strategy, regime)


def get_position_size_multiplier(
    strategy: str,
    signal_quality: float,
    regime: Optional[str] = None,
) -> float:
    """Convenience function using default engine."""
    return get_engine().get_position_size_multiplier(strategy, signal_quality, regime)
