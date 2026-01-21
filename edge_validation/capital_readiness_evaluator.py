from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class CapitalReadinessReport:
    score: float  # 0.0 - 1.0
    verdict: str  # NO_GO | MICRO_ONLY | LIMITED_LIVE
    blocking_reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class CapitalReadinessEvaluator:
    """
    Compute a capital readiness score and verdict from walk-forward and expectancy metrics.
    This evaluator is analysis-only and does not modify live state.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.min_expectancy_for_live = float(self.config.get('min_expectancy_for_live', 0.01))
        self.min_sharpe_for_micro = float(self.config.get('min_sharpe_for_micro', 0.7))

    def evaluate(self, strategy_metrics: Dict[str, Dict[str, Any]]) -> CapitalReadinessReport:
        # aggregate
        total_score = 0.0
        count = 0
        blocking: List[str] = []
        details: Dict[str, Any] = {}
        for strat, m in strategy_metrics.items():
            exp = float(m.get('expectancy', 0.0))
            sharpe = float(m.get('sharpe', 0.0))
            score = 0.0
            if exp <= 0.0:
                blocking.append(f"{strat}: non-positive expectancy {exp:.4f}")
                score = 0.0
            else:
                # base score from expectancy (sigmoid-ish) capped at 1.0 for 0.05 expectancy
                score = min(1.0, exp / 0.05)
                # boost with sharpe
                score *= min(1.0, sharpe / 1.5)
            total_score += score
            count += 1
            details[strat] = {'expectancy': exp, 'sharpe': sharpe, 'score': score}
        avg_score = (total_score / count) if count else 0.0
        # decide verdict
        if blocking:
            verdict = 'NO_GO'
        elif avg_score < 0.25:
            verdict = 'MICRO_ONLY'
        else:
            verdict = 'LIMITED_LIVE'
        return CapitalReadinessReport(score=avg_score, verdict=verdict, blocking_reasons=blocking, details=details)
