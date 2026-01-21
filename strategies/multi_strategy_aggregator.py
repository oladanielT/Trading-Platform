"""
multi_strategy_aggregator.py - Lightweight multi-strategy combiner.

Combines independent strategy signals for the same asset using a weighted
confidence vote. Intended to be used as the `underlying_strategy` inside
RegimeBasedStrategy so readiness gating and scaling continue to apply
to the aggregated result without altering execution or sizing logic.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

from strategies.base import BaseStrategy, PositionState, SignalOutput, Signal


logger = logging.getLogger(__name__)


class MultiStrategyAggregator(BaseStrategy):
    """Aggregate multiple strategies into a single signal via weighted vote.
    
    Supports optional adaptive weighting based on historical performance.
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        weights: Optional[Dict[str, float]] = None,
        name: Optional[str] = None,
        enable_adaptive_weights: bool = False,
        min_samples_for_adaptation: int = 20,
        adaptation_learning_rate: float = 0.1,
    ):
        """
        Args:
            strategies: List of strategies to aggregate
            weights: Static weights per strategy (default 1.0)
            name: Aggregator name
            enable_adaptive_weights: Enable adaptive weight adjustment (default False)
            min_samples_for_adaptation: Minimum samples before adapting weights (default 20)
            adaptation_learning_rate: Learning rate for weight updates (default 0.1)
        """
        if not strategies:
            raise ValueError("At least one strategy is required")
        super().__init__(name=name or "MultiStrategyAggregator")
        self.strategies = strategies
        self.weights = weights or {}
        self.enable_adaptive_weights = enable_adaptive_weights
        self.min_samples_for_adaptation = min_samples_for_adaptation
        self.adaptation_learning_rate = adaptation_learning_rate
        
        # Adaptive weighting state (per-strategy)
        self._adaptive_weights: Dict[str, float] = {}  # Current adaptive weights
        self._strategy_history: Dict[str, List[Dict]] = defaultdict(list)  # Historical performance
        self._correct_predictions: Dict[str, int] = defaultdict(int)
        self._total_predictions: Dict[str, int] = defaultdict(int)

    def _get_weight(self, strat_name: str) -> float:
        """Get weight for strategy, using adaptive weight if enabled and available."""
        if self.enable_adaptive_weights and strat_name in self._adaptive_weights:
            total_samples = self._total_predictions.get(strat_name, 0)
            if total_samples >= self.min_samples_for_adaptation:
                return self._adaptive_weights[strat_name]
        # Fallback to static weight
        return float(self.weights.get(strat_name, 1.0))
    
    def _calculate_performance_score(self, strat_name: str) -> float:
        """Calculate performance score for a strategy based on historical accuracy."""
        total = self._total_predictions.get(strat_name, 0)
        if total == 0:
            return 1.0  # Neutral score
        
        correct = self._correct_predictions.get(strat_name, 0)
        accuracy = correct / total
        
        # Performance score: accuracy mapped to [0.5, 1.5]
        # 0% accuracy -> 0.5x weight
        # 50% accuracy -> 1.0x weight
        # 100% accuracy -> 1.5x weight
        performance_score = 0.5 + accuracy
        return performance_score
    
    def _update_adaptive_weights(self):
        """Update adaptive weights based on historical performance."""
        if not self.enable_adaptive_weights:
            return
        
        for strat in self.strategies:
            strat_name = strat.name
            total_samples = self._total_predictions.get(strat_name, 0)
            
            if total_samples >= self.min_samples_for_adaptation:
                # Calculate new weight based on performance
                performance_score = self._calculate_performance_score(strat_name)
                static_weight = float(self.weights.get(strat_name, 1.0))
                
                # Adaptive weight = static_weight * performance_score
                # Use learning rate for gradual adjustment
                current_adaptive = self._adaptive_weights.get(strat_name, static_weight)
                target_adaptive = static_weight * performance_score
                
                new_adaptive = (
                    current_adaptive * (1 - self.adaptation_learning_rate) +
                    target_adaptive * self.adaptation_learning_rate
                )
                
                self._adaptive_weights[strat_name] = new_adaptive
                
                logger.info(
                    "[AdaptiveWeights] strategy=%s samples=%d accuracy=%.2f%% "
                    "static_weight=%.3f adaptive_weight=%.3f",
                    strat_name,
                    total_samples,
                    (self._correct_predictions.get(strat_name, 0) / total_samples) * 100,
                    static_weight,
                    new_adaptive,
                )
    
    def record_outcome(self, strategy_name: str, was_correct: bool):
        """Record outcome for a strategy prediction (for adaptive weighting).
        
        Args:
            strategy_name: Name of the strategy
            was_correct: Whether the prediction was correct
        """
        if not self.enable_adaptive_weights:
            return
        
        self._total_predictions[strategy_name] += 1
        if was_correct:
            self._correct_predictions[strategy_name] += 1
        
        # Update adaptive weights after each outcome
        self._update_adaptive_weights()

    def _collect_signal(self, strategy: BaseStrategy, market_data: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[str, float, Dict]:
        """Invoke strategy analyze() if available, else generate_signal()."""
        action = "hold"
        confidence = 0.0
        meta: Dict = {}
        if hasattr(strategy, "analyze"):
            try:
                # Try to pass symbol if analyze accepts it
                import inspect
                sig = inspect.signature(strategy.analyze)
                if "symbol" in sig.parameters:
                    result = strategy.analyze(market_data, symbol=symbol)
                else:
                    result = strategy.analyze(market_data)
                if result:
                    action = result.get("action", "hold")
                    confidence = float(result.get("confidence", 0.0))
                    meta = result
            except Exception as exc:
                logger.error("[MultiStrategy] analyze failed for %s: %s", strategy.name, exc)
        else:
            try:
                sig = strategy.generate_signal(market_data, PositionState())
                action = sig.signal.value.lower()
                confidence = float(sig.confidence)
                meta = sig.to_dict()
            except Exception as exc:
                logger.error("[MultiStrategy] generate_signal failed for %s: %s", strategy.name, exc)
        logger.info(
            "[MultiStrategy] strat=%s action=%s conf=%.3f",
            strategy.name,
            action,
            confidence,
        )
        return action, confidence, meta

    def _combine(self, votes: List[Tuple[str, float, float]]) -> Tuple[str, float]:
        """Combine votes as weighted confidence sums for buy vs sell."""
        buy_sum = 0.0
        sell_sum = 0.0
        total_weight = 0.0
        for action, conf, weight in votes:
            total_weight += weight
            if action == "buy":
                buy_sum += conf * weight
            elif action == "sell":
                sell_sum += conf * weight

        if buy_sum == 0 and sell_sum == 0:
            return "hold", 0.0

        if buy_sum > sell_sum:
            return "buy", buy_sum / max(total_weight, 1e-9)
        elif sell_sum > buy_sum:
            return "sell", sell_sum / max(total_weight, 1e-9)
        else:
            return "hold", 0.0

    def analyze(self, market_data: pd.DataFrame, symbol: Optional[str] = None) -> Optional[Dict]:
        if pd is None:
            raise ImportError("pandas required. Install with: pip install pandas")
        if market_data is None or len(market_data) < 2:
            logger.warning("[MultiStrategy] insufficient market data")
            return None

        contributions: List[Dict] = []
        votes: List[Tuple[str, float, float]] = []

        # Track metadata from all strategies for aggregation
        strategy_metadata = []
        
        for strat in self.strategies:
            action, confidence, meta = self._collect_signal(strat, market_data, symbol)
            weight = self._get_weight(strat.name)
            votes.append((action, confidence, weight))
            
            # Store metadata from this strategy (including stop_loss, entry_price)
            strategy_metadata.append({
                'strategy': strat.name,
                'action': action,
                'confidence': confidence,
                'weight': weight,
                'metadata': meta  # Includes stop_loss, entry_price, etc.
            })
            
            # Include adaptive weight info in contribution
            contrib_info = {
                "strategy": strat.name,
                "action": action,
                "confidence": confidence,
                "weight": weight,
            }
            
            # Add adaptive weight details if enabled
            if self.enable_adaptive_weights:
                static_weight = float(self.weights.get(strat.name, 1.0))
                total_samples = self._total_predictions.get(strat.name, 0)
                accuracy = (
                    self._correct_predictions.get(strat.name, 0) / total_samples
                    if total_samples > 0
                    else 0.0
                )
                contrib_info["adaptive_info"] = {
                    "static_weight": static_weight,
                    "samples": total_samples,
                    "accuracy": accuracy,
                    "using_adaptive": total_samples >= self.min_samples_for_adaptation,
                }
            
            contributions.append(contrib_info)

        final_action, final_conf = self._combine(votes)
        
        # Extract stop_loss and entry_price from the strongest contributing strategy
        # (the one with highest weighted confidence matching the final action)
        best_stop_loss = None
        best_entry_price = None
        best_weighted_conf = 0.0
        
        for strat_meta in strategy_metadata:
            # Only consider strategies that voted for the final action
            if strat_meta['action'] == final_action:
                weighted_conf = strat_meta['confidence'] * strat_meta['weight']
                if weighted_conf > best_weighted_conf:
                    best_weighted_conf = weighted_conf
                    # Extract metadata from this strategy
                    meta = strat_meta.get('metadata', {})
                    if isinstance(meta, dict):
                        # Some strategies wrap details under 'metadata'
                        nested_meta = meta.get('metadata', meta)
                        best_stop_loss = nested_meta.get('stop_loss')
                        best_entry_price = nested_meta.get('entry_price')

        log_msg = (
            "[MultiStrategy] symbol=%s | final=%s %.3f | contributions=%s"
            if symbol
            else "[MultiStrategy] final=%s %.3f | contributions=%s"
        )
        log_args = (
            (symbol, final_action, final_conf, contributions)
            if symbol
            else (final_action, final_conf, contributions)
        )
        logger.info(log_msg, *log_args)

        result = {
            "action": final_action,
            "confidence": final_conf,
            "contributions": contributions,
            "stop_loss": best_stop_loss,
            "entry_price": best_entry_price,
        }
        if symbol:
            result["symbol"] = symbol
        return result

    def generate_signal(self, market_data: pd.DataFrame, position_state: PositionState) -> SignalOutput:
        result = self.analyze(market_data) or {"action": "hold", "confidence": 0.0}
        signal = Signal.HOLD
        if result["action"] == "buy":
            signal = Signal.BUY
        elif result["action"] == "sell":
            signal = Signal.SELL
        
        # Pass through stop_loss and entry_price from strongest strategy
        return self.create_signal(
            signal=signal,
            confidence=result.get("confidence", 0.0),
            contributions=result.get("contributions"),
            stop_loss=result.get("stop_loss"),
            entry_price=result.get("entry_price"),
        )


__all__ = ["MultiStrategyAggregator"]