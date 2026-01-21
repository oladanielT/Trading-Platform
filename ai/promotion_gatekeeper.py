"""
Promotion Gatekeeper

Monitors live performance and automatically demotes underperforming strategies.

Demotion Triggers:
- Drawdown exceeds threshold
- Expectancy turns negative
- Regime mismatch persists
- Correlation with existing strategies spikes
- Win rate collapses
- Consecutive losses exceed limit

SAFETY: This is a PROTECTIVE mechanism - demotes only, never promotes.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DemotionReason(Enum):
    """Reasons for automatic demotion."""
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    NEGATIVE_EXPECTANCY = "negative_expectancy"
    WIN_RATE_COLLAPSE = "win_rate_collapse"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    REGIME_MISMATCH = "regime_mismatch"
    HIGH_CORRELATION = "high_correlation"
    INSUFFICIENT_TRADES = "insufficient_trades"
    MANUAL = "manual"


@dataclass
class PromotionDecision:
    """Decision from gatekeeper evaluation."""
    strategy_id: str
    should_demote: bool
    reason: Optional[DemotionReason] = None
    details: str = ""
    metrics_snapshot: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics_snapshot is None:
            self.metrics_snapshot = {}


@dataclass
class DemotionCriteria:
    """Criteria for automatic demotion."""
    max_drawdown: float = 0.15  # 15% max drawdown
    min_expectancy: float = 0.0  # Must be positive
    min_win_rate: float = 0.40  # 40% min win rate
    max_consecutive_losses: int = 5  # Circuit breaker
    min_trades_before_eval: int = 20  # Minimum sample size
    max_correlation: float = 0.85  # Max correlation with other strategies
    regime_mismatch_threshold: int = 10  # Consecutive regime misses


class PromotionGatekeeper:
    """
    Monitors strategy performance and enforces demotion rules.
    
    SAFETY GUARANTEES:
    - Demotes only (never promotes)
    - Automatic demotion based on objective criteria
    - Human can override to keep strategy active
    - Logs all demotion decisions for audit
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize gatekeeper.
        
        Args:
            config: Configuration dict with 'strategy_promotion' section
        """
        self.config = config.get('strategy_promotion', {})
        
        # Load demotion criteria
        criteria_config = self.config.get('demotion_criteria', {})
        self.criteria = DemotionCriteria(
            max_drawdown=criteria_config.get('max_drawdown', 0.15),
            min_expectancy=criteria_config.get('min_expectancy', 0.0),
            min_win_rate=criteria_config.get('min_win_rate', 0.40),
            max_consecutive_losses=criteria_config.get('max_consecutive_losses', 5),
            min_trades_before_eval=criteria_config.get('min_trades_before_eval', 20),
            max_correlation=criteria_config.get('max_correlation', 0.85),
            regime_mismatch_threshold=criteria_config.get('regime_mismatch_threshold', 10)
        )
        
        # Auto-demotion enabled by default
        self.auto_demote = self.config.get('auto_demote', True)
        
        logger.info(
            f"[GATEKEEPER] Initialized (auto_demote={self.auto_demote})"
        )
    
    def evaluate(
        self,
        strategy_id: str,
        metrics: Dict[str, float]
    ) -> PromotionDecision:
        """
        Evaluate strategy performance and determine if demotion needed.
        
        Args:
            strategy_id: Strategy to evaluate
            metrics: Performance metrics dict containing:
                - total_trades: int
                - winning_trades: int
                - losing_trades: int
                - expectancy: float
                - max_drawdown: float
                - consecutive_losses: int
                - win_rate: float (optional)
                - correlation: float (optional)
                
        Returns:
            PromotionDecision with demotion recommendation
        """
        # Check minimum trades requirement
        total_trades = metrics.get('total_trades', 0)
        if total_trades < self.criteria.min_trades_before_eval:
            return PromotionDecision(
                strategy_id=strategy_id,
                should_demote=False,
                details=f"Insufficient trades: {total_trades} < {self.criteria.min_trades_before_eval}"
            )
        
        # Check drawdown
        max_dd = metrics.get('max_drawdown', 0.0)
        if max_dd > self.criteria.max_drawdown:
            return PromotionDecision(
                strategy_id=strategy_id,
                should_demote=True,
                reason=DemotionReason.EXCESSIVE_DRAWDOWN,
                details=f"Drawdown {max_dd:.2%} > {self.criteria.max_drawdown:.2%}",
                metrics_snapshot=metrics
            )
        
        # Check expectancy
        expectancy = metrics.get('expectancy', 0.0)
        if expectancy < self.criteria.min_expectancy:
            return PromotionDecision(
                strategy_id=strategy_id,
                should_demote=True,
                reason=DemotionReason.NEGATIVE_EXPECTANCY,
                details=f"Expectancy {expectancy:.4f} < {self.criteria.min_expectancy}",
                metrics_snapshot=metrics
            )
        
        # Check win rate
        win_rate = metrics.get('win_rate', 0.0)
        if win_rate < self.criteria.min_win_rate:
            return PromotionDecision(
                strategy_id=strategy_id,
                should_demote=True,
                reason=DemotionReason.WIN_RATE_COLLAPSE,
                details=f"Win rate {win_rate:.2%} < {self.criteria.min_win_rate:.2%}",
                metrics_snapshot=metrics
            )
        
        # Check consecutive losses (circuit breaker)
        consecutive_losses = metrics.get('consecutive_losses', 0)
        if consecutive_losses >= self.criteria.max_consecutive_losses:
            return PromotionDecision(
                strategy_id=strategy_id,
                should_demote=True,
                reason=DemotionReason.CONSECUTIVE_LOSSES,
                details=f"Consecutive losses {consecutive_losses} >= {self.criteria.max_consecutive_losses}",
                metrics_snapshot=metrics
            )
        
        # Check correlation (if provided)
        correlation = metrics.get('correlation', 0.0)
        if correlation > self.criteria.max_correlation:
            return PromotionDecision(
                strategy_id=strategy_id,
                should_demote=True,
                reason=DemotionReason.HIGH_CORRELATION,
                details=f"Correlation {correlation:.2f} > {self.criteria.max_correlation:.2f}",
                metrics_snapshot=metrics
            )
        
        # Check regime mismatch (if provided)
        regime_mismatches = metrics.get('regime_mismatches', 0)
        if regime_mismatches >= self.criteria.regime_mismatch_threshold:
            return PromotionDecision(
                strategy_id=strategy_id,
                should_demote=True,
                reason=DemotionReason.REGIME_MISMATCH,
                details=f"Regime mismatches {regime_mismatches} >= {self.criteria.regime_mismatch_threshold}",
                metrics_snapshot=metrics
            )
        
        # All checks passed
        return PromotionDecision(
            strategy_id=strategy_id,
            should_demote=False,
            details="All criteria met",
            metrics_snapshot=metrics
        )
    
    def should_auto_demote(self) -> bool:
        """Check if auto-demotion is enabled."""
        return self.auto_demote
    
    def get_criteria(self) -> DemotionCriteria:
        """Get current demotion criteria."""
        return self.criteria
    
    def evaluate_batch(
        self,
        strategies_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, PromotionDecision]:
        """
        Evaluate multiple strategies at once.
        
        Args:
            strategies_metrics: Dict mapping strategy_id -> metrics
            
        Returns:
            Dict mapping strategy_id -> PromotionDecision
        """
        decisions = {}
        
        for strategy_id, metrics in strategies_metrics.items():
            decision = self.evaluate(strategy_id, metrics)
            decisions[strategy_id] = decision
            
            if decision.should_demote:
                logger.warning(
                    f"[GATEKEEPER] DEMOTION RECOMMENDED: {strategy_id} "
                    f"- {decision.reason.value if decision.reason else 'unknown'}: "
                    f"{decision.details}"
                )
        
        return decisions
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        return {
            'auto_demote_enabled': self.auto_demote,
            'criteria': {
                'max_drawdown': self.criteria.max_drawdown,
                'min_expectancy': self.criteria.min_expectancy,
                'min_win_rate': self.criteria.min_win_rate,
                'max_consecutive_losses': self.criteria.max_consecutive_losses,
                'min_trades_before_eval': self.criteria.min_trades_before_eval,
                'max_correlation': self.criteria.max_correlation,
                'regime_mismatch_threshold': self.criteria.regime_mismatch_threshold
            }
        }
