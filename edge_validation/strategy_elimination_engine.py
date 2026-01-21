from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EliminationDecision:
    strategy: str
    action: str  # RETIRE, SHADOW, DISABLE_REGIME, NONE
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class StrategyEliminationEngine:
    """
    Evaluate strategies for elimination based on expectancy and Sharpe.
    Does not bypass promotion rules; if a promotion_manager is provided
    it will call its demote/retire APIs.
    """

    def __init__(self, promotion_manager: Optional[Any] = None, config: Optional[dict] = None):
        self.promotion_manager = promotion_manager
        self.config = config or {}
        self.min_sharpe = float(self.config.get('min_sharpe', 0.5))

    def evaluate(self, metrics_by_strategy: Dict[str, Dict[str, Any]]) -> List[EliminationDecision]:
        decisions: List[EliminationDecision] = []
        for strat, metrics in metrics_by_strategy.items():
            exp = metrics.get('expectancy', 0.0)
            sharpe = metrics.get('sharpe', 0.0)
            if exp <= 0.0:
                # retire
                reason = f"Expectancy <= 0 ({exp:.4f})"
                decisions.append(EliminationDecision(strategy=strat, action='RETIRE', reason=reason, details={'expectancy': exp, 'sharpe': sharpe}))
                # call promotion manager if available
                try:
                    if self.promotion_manager:
                        # Prefer retire API if exists
                        if hasattr(self.promotion_manager, 'retire'):
                            self.promotion_manager.retire(strat, reason=reason)
                        else:
                            # fallback to demote force retire
                            self.promotion_manager.demote(strat, reason=reason, demoted_by='elimination_engine', force_retire=True)
                except Exception as e:
                    logger.error(f"Failed to retire {strat}: {e}")
                continue
            if sharpe < self.min_sharpe:
                reason = f"Sharpe {sharpe:.3f} < min_sharpe {self.min_sharpe:.3f}"
                decisions.append(EliminationDecision(strategy=strat, action='SHADOW', reason=reason, details={'expectancy': exp, 'sharpe': sharpe}))
                try:
                    if self.promotion_manager:
                        # demote to SHADOW
                        if hasattr(self.promotion_manager, 'demote'):
                            self.promotion_manager.demote(strat, reason=reason, demoted_by='elimination_engine', force_retire=False)
                except Exception as e:
                    logger.error(f"Failed to demote {strat}: {e}")
                continue
            # per-regime disables
            regime_expectancies = metrics.get('by_regime', {})
            for regime, r_exp in regime_expectancies.items():
                if r_exp <= 0.0:
                    reason = f"Negative expectancy in regime {regime}: {r_exp:.4f}"
                    decisions.append(EliminationDecision(strategy=strat, action='DISABLE_REGIME', reason=reason, details={'regime': regime, 'expectancy': r_exp}))
        return decisions
