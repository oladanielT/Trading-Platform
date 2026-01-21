"""
signal_quality_scorer.py - Signal Confluence and Confidence Scoring

Evaluates signal quality based on:
- Confluence count (how many indicators agree)
- Historical success rate in similar setups
- Current regime compatibility
- Time-to-exit predictions

Design:
- Per-strategy signal scoring
- Regime-specific scoring adjustments
- Threshold-based filtering (only high-quality signals execute)
- Historical pattern matching
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import threading

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


@dataclass
class SignalQualityMetrics:
    """Quality metrics for a signal."""
    timestamp: datetime
    strategy: str
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float  # 0.0-1.0 (original)
    confluence_count: int  # Number of confirming signals
    regime_alignment: float  # 0.0-1.0 (how well signal aligns with regime)
    historical_success_rate: float  # 0.0-1.0 (success rate in similar setups)
    quality_score: float  # 0.0-1.0 (composite quality)
    regime: Optional[str] = None
    additional_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metadata is None:
            self.additional_metadata = {}


class SignalQualityScorer:
    """
    Scores signal quality to filter high-confidence signals only.
    
    Uses confluence analysis and historical pattern matching to improve
    signal reliability without changing strategy logic.
    """
    
    def __init__(
        self,
        min_quality_threshold: float = 0.7,
        confluence_weight: float = 0.3,
        regime_alignment_weight: float = 0.3,
        historical_weight: float = 0.4,
    ):
        """
        Initialize scorer.
        
        Args:
            min_quality_threshold: Minimum quality score to pass signals
            confluence_weight: Weight for confluence count
            regime_alignment_weight: Weight for regime alignment
            historical_weight: Weight for historical success rate
        """
        self._lock = threading.RLock()
        
        self.min_quality_threshold = min_quality_threshold
        self.confluence_weight = confluence_weight
        self.regime_alignment_weight = regime_alignment_weight
        self.historical_weight = historical_weight
        
        # Historical signal tracking (per strategy, per symbol, per regime)
        self._signal_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Regime-specific adjustments
        self._regime_adjustments: Dict[str, float] = {
            'TRENDING': 1.1,  # 10% boost in trending
            'RANGING': 0.8,   # 20% reduction in ranging
            'VOLATILE': 0.7,  # 30% reduction in volatility
        }
    
    def score_signal(
        self,
        strategy: str,
        symbol: str,
        signal_type: str,
        original_confidence: float,
        confluence_count: int = 1,
        regime: Optional[str] = None,
        regime_alignment: Optional[float] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> SignalQualityMetrics:
        """
        Score a trading signal based on multiple factors.
        
        Args:
            strategy: Strategy name
            symbol: Trading pair
            signal_type: BUY, SELL, HOLD
            original_confidence: Original confidence from strategy (0.0-1.0)
            confluence_count: Number of confirming signals (1-5+)
            regime: Current market regime
            regime_alignment: How well signal aligns with regime (0.0-1.0)
            additional_metadata: Extra context
            
        Returns:
            SignalQualityMetrics with composite quality score
        """
        with self._lock:
            # Default regime alignment if not provided
            if regime_alignment is None:
                regime_alignment = self._estimate_regime_alignment(strategy, signal_type, regime)
            
            # Get historical success rate
            hist_key = f"{strategy}|{symbol}|{signal_type}|{regime or 'ANY'}"
            hist_success = self._calculate_historical_success_rate(hist_key)
            
            # Calculate composite quality score
            quality_score = (
                (original_confidence * self.confluence_weight) +
                (regime_alignment * self.regime_alignment_weight) +
                (hist_success * self.historical_weight)
            )
            
            # Normalize confluence contribution
            confluence_factor = min(confluence_count / 3.0, 1.0)  # Cap at 1.0
            quality_score = quality_score * (0.5 + 0.5 * confluence_factor)
            
            # Apply regime adjustment
            if regime and regime in self._regime_adjustments:
                quality_score *= self._regime_adjustments[regime]
            
            # Clamp to valid range
            quality_score = max(0.0, min(1.0, quality_score))
            
            # Create metrics object
            metrics = SignalQualityMetrics(
                timestamp=datetime.now(timezone.utc),
                strategy=strategy,
                symbol=symbol,
                signal_type=signal_type,
                confidence=original_confidence,
                confluence_count=confluence_count,
                regime_alignment=regime_alignment,
                historical_success_rate=hist_success,
                quality_score=quality_score,
                regime=regime,
                additional_metadata=additional_metadata or {},
            )
            
            # Record in history
            self._signal_history[hist_key].append(metrics)
            
            logger.debug(
                f"Scored signal: {strategy}/{symbol} {signal_type} "
                f"quality={quality_score:.3f} (confluence={confluence_count}, "
                f"regime_align={regime_alignment:.2f}, hist_success={hist_success:.2f})"
            )
            
            return metrics
    
    def _estimate_regime_alignment(self, strategy: str, signal_type: str, regime: Optional[str]) -> float:
        """
        Estimate how well a signal aligns with current regime.
        
        This is a heuristic that can be overridden by strategy-specific logic.
        """
        if not regime:
            return 0.5  # Neutral
        
        # Strategy-specific alignments
        if 'EMA' in strategy or 'Trend' in strategy:
            if regime == 'TRENDING':
                return 0.9
            elif regime == 'RANGING':
                return 0.4
            else:
                return 0.5
        
        if 'FVG' in strategy or 'Pattern' in strategy:
            if regime == 'TRENDING':
                return 0.8
            elif regime == 'VOLATILE':
                return 0.7
            else:
                return 0.6
        
        # Default
        return 0.5
    
    def _calculate_historical_success_rate(self, key: str) -> float:
        """
        Calculate historical success rate for a signal type.
        
        Args:
            key: Composite key (strategy|symbol|signal_type|regime)
            
        Returns:
            Success rate (0.0-1.0)
        """
        history = self._signal_history.get(key)
        if not history:
            return 0.5  # Neutral if no history
        
        if len(history) < 5:
            return 0.5  # Insufficient data
        
        # For now, assume all recorded signals were "outcomes"
        # In production, this would track actual trade results
        recent_signals = list(history)[-50:]
        
        # This is a placeholder - in real implementation, would compare
        # predicted direction with actual price movement
        successful = sum(1 for sig in recent_signals if sig.quality_score > 0.7)
        return successful / len(recent_signals) if recent_signals else 0.5
    
    def filter_signals(
        self,
        signals: List[SignalQualityMetrics],
        min_quality: Optional[float] = None,
    ) -> List[SignalQualityMetrics]:
        """
        Filter signals to only high-quality ones.
        
        Args:
            signals: List of signal quality metrics
            min_quality: Override minimum quality threshold
            
        Returns:
            Filtered list of high-quality signals
        """
        threshold = min_quality or self.min_quality_threshold
        return [s for s in signals if s.quality_score >= threshold]
    
    def get_signal_history(
        self,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[SignalQualityMetrics]:
        """Get signal history."""
        with self._lock:
            all_signals = []
            for hist in self._signal_history.values():
                all_signals.extend(hist)
            
            # Filter
            if strategy:
                all_signals = [s for s in all_signals if s.strategy == strategy]
            if symbol:
                all_signals = [s for s in all_signals if s.symbol == symbol]
            
            # Sort by timestamp desc
            all_signals.sort(key=lambda s: s.timestamp, reverse=True)
            
            return all_signals[:limit]
    
    def get_quality_summary(self, strategy: str, symbol: str) -> Dict[str, Any]:
        """Get quality summary for a strategy/symbol pair."""
        with self._lock:
            history = self.get_signal_history(strategy=strategy, symbol=symbol, limit=50)
            
            if not history:
                return {'total_signals': 0}
            
            avg_quality = sum(s.quality_score for s in history) / len(history)
            avg_confluence = sum(s.confluence_count for s in history) / len(history)
            high_quality = sum(1 for s in history if s.quality_score >= self.min_quality_threshold)
            
            return {
                'total_signals': len(history),
                'high_quality_count': high_quality,
                'avg_quality_score': avg_quality,
                'avg_confluence_count': avg_confluence,
                'high_quality_percentage': (high_quality / len(history) * 100) if history else 0,
            }
    
    def set_regime_adjustment(self, regime: str, adjustment_factor: float) -> None:
        """
        Set regime-specific quality adjustment factor.
        
        Args:
            regime: Regime name
            adjustment_factor: Multiplier for quality score (1.0 = no change, >1.0 = boost, <1.0 = reduce)
        """
        with self._lock:
            self._regime_adjustments[regime] = adjustment_factor
            logger.info(f"Set regime adjustment: {regime} = {adjustment_factor:.2f}x")
    
    def record_signal_outcome(
        self,
        strategy: str,
        symbol: str,
        signal_type: str,
        regime: Optional[str],
        was_profitable: bool,
        pnl: float = 0.0,
    ) -> None:
        """
        Record actual outcome of a signal (for future success rate calculation).
        
        Args:
            strategy: Strategy name
            symbol: Trading pair
            signal_type: BUY, SELL, HOLD
            regime: Market regime when signal was issued
            was_profitable: Whether the resulting trade was profitable
            pnl: Actual PnL
        """
        with self._lock:
            key = f"{strategy}|{symbol}|{signal_type}|{regime or 'ANY'}"
            history = self._signal_history.get(key)
            
            if history:
                # Update the most recent signal with outcome
                if len(history) > 0:
                    recent_signal = history[-1]
                    recent_signal.additional_metadata['was_profitable'] = was_profitable
                    recent_signal.additional_metadata['pnl'] = pnl


# Module-level convenience instance
_default_scorer: Optional[SignalQualityScorer] = None


def initialize_scorer(**kwargs) -> SignalQualityScorer:
    """Initialize default scorer instance."""
    global _default_scorer
    _default_scorer = SignalQualityScorer(**kwargs)
    return _default_scorer


def get_scorer() -> SignalQualityScorer:
    """Get the default scorer instance."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = SignalQualityScorer()
    return _default_scorer


def score_signal(
    strategy: str,
    symbol: str,
    signal_type: str,
    original_confidence: float,
    **kwargs,
) -> SignalQualityMetrics:
    """Convenience function using default scorer."""
    return get_scorer().score_signal(strategy, symbol, signal_type, original_confidence, **kwargs)
