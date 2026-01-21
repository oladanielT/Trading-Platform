"""
runtime_summary.py - Non-blocking runtime summary aggregator.

Collects lightweight counts of readiness states, regimes, and trades-per-readiness
and emits compact summaries at a configurable interval using the platform logger.

Safety goals:
- No impact on trading logic; purely observational
- Non-blocking: emission happens on a daemon thread
- Defaults are conservative; can be injected where desired (e.g., paper/backtest)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from monitoring.logger import log_event


logger = logging.getLogger(__name__)


class RuntimeSummaryAggregator:
    """Aggregate readiness/regime/trade counts and emit periodic summaries.
    
    Also emits alerts for significant events:
    - Readiness state changes
    - Confidence anomalies
    - Strategy signal deviations
    """

    def __init__(
        self,
        interval_seconds: int = 300,
        logger_fn: Callable[[str, Optional[Dict[str, Any]]], None] = log_event,
        enable_alerts: bool = False,
        readiness_alert_threshold: str = "not_ready",
        confidence_threshold: float = 0.85,
        deviation_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            interval_seconds: Summary emission interval
            logger_fn: Custom logger function
            enable_alerts: Enable alert emission (default False)
            readiness_alert_threshold: Alert when readiness below this level (default "not_ready")
            confidence_threshold: Alert when confidence exceeds this
            deviation_threshold: Alert when strategy deviates by this amount from aggregate
        """
        self.interval_seconds = max(1, int(interval_seconds))
        self.logger_fn = logger_fn
        self.enable_alerts = enable_alerts
        self.readiness_alert_threshold = readiness_alert_threshold.lower()
        self.confidence_threshold = confidence_threshold
        self.deviation_threshold = deviation_threshold

        self._lock = threading.RLock()
        self._last_emit = time.time()
        self._emitting = False

        self._readiness_counts: Dict[str, int] = defaultdict(int)
        self._regime_counts: Dict[str, int] = defaultdict(int)
        self._trades_by_readiness: Dict[str, int] = defaultdict(int)
        self._samples = 0
        
        # Alert tracking
        self._last_readiness_state: Dict[str, str] = {}  # per-symbol
        self._last_confidence: Dict[str, float] = {}  # per-symbol
        self._last_aggregate_action: Dict[str, str] = {}  # per-symbol
        self._strategy_signals: Dict[str, List[Dict]] = defaultdict(list)  # per-symbol

    def record_cycle(self, readiness_state: str, regime: str, action: str, ts: Optional[float] = None) -> None:
        """Record one decision cycle sample.

        Args:
            readiness_state: readiness label (e.g., "ready", "forming")
            regime: regime label (e.g., "trending", "ranging")
            action: final action taken (buy/sell/hold)
            ts: optional timestamp; defaults to time.time()
        """
        now = ts or time.time()
        should_emit = False
        with self._lock:
            self._readiness_counts[readiness_state] += 1
            self._regime_counts[regime] += 1
            if action in ["buy", "sell"]:
                self._trades_by_readiness[readiness_state] += 1
            self._samples += 1
            if (now - self._last_emit) >= self.interval_seconds and not self._emitting:
                should_emit = True
                self._emitting = True
        if should_emit:
            t = threading.Thread(target=self._emit_summary_safe, args=(now,), daemon=True)
            t.start()
    
    def record_analysis(
        self,
        symbol: str,
        readiness_state: str,
        readiness_score: float,
        aggregate_action: str,
        aggregate_confidence: float,
        contributions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Record strategy analysis results for alert detection.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            readiness_state: Current readiness state
            readiness_score: Current readiness score [0, 1]
            aggregate_action: Aggregated action (buy/sell/hold)
            aggregate_confidence: Aggregated confidence [0, 1]
            contributions: List of per-strategy contributions
        """
        if not self.enable_alerts:
            return
        
        with self._lock:
            # Check for readiness degradation
            self._check_readiness_alert(symbol, readiness_state, readiness_score)
            
            # Check for confidence anomalies
            self._check_confidence_alert(symbol, aggregate_action, aggregate_confidence)
            
            # Check for strategy deviations
            if contributions:
                self._check_deviation_alert(symbol, aggregate_action, aggregate_confidence, contributions)
            
            # Update tracking state
            self._last_readiness_state[symbol] = readiness_state
            self._last_confidence[symbol] = aggregate_confidence
            self._last_aggregate_action[symbol] = aggregate_action
            if contributions:
                self._strategy_signals[symbol] = contributions
    
    def _check_readiness_alert(self, symbol: str, state: str, score: float) -> None:
        """Alert when readiness drops to or below threshold."""
        prev_state = self._last_readiness_state.get(symbol)
        
        # Map state to numeric level for comparison
        state_levels = {"ready": 3, "forming": 2, "not_ready": 1}
        threshold_level = state_levels.get(self.readiness_alert_threshold, 0)
        current_level = state_levels.get(state, 0)
        
        # Alert if current <= threshold AND state changed (degradation)
        if (
            current_level <= threshold_level
            and prev_state is not None
            and prev_state != state
        ):
            alert = {
                "type": "readiness_degradation",
                "symbol": symbol,
                "previous_state": prev_state,
                "current_state": state,
                "readiness_score": score,
                "threshold": self.readiness_alert_threshold,
            }
            logger.info("[Alert] Readiness degradation: %s", alert)
    
    def _check_confidence_alert(self, symbol: str, action: str, confidence: float) -> None:
        """Alert when confidence exceeds threshold with unexpected trend."""
        prev_confidence = self._last_confidence.get(symbol, 0.0)
        prev_action = self._last_aggregate_action.get(symbol)
        
        # Alert if high confidence AND action changed (unexpected trend)
        if (
            confidence > self.confidence_threshold
            and prev_action is not None
            and action != prev_action
            and action != "hold"
        ):
            alert = {
                "type": "confidence_anomaly",
                "symbol": symbol,
                "confidence": confidence,
                "threshold": self.confidence_threshold,
                "previous_action": prev_action,
                "current_action": action,
                "confidence_delta": confidence - prev_confidence,
            }
            logger.info("[Alert] Confidence anomaly: %s", alert)
    
    def _check_deviation_alert(
        self,
        symbol: str,
        aggregate_action: str,
        aggregate_confidence: float,
        contributions: List[Dict[str, Any]],
    ) -> None:
        """Alert when strategy signal deviates significantly from aggregate."""
        for contrib in contributions:
            strategy_name = contrib.get("strategy")
            strategy_action = contrib.get("action")
            strategy_confidence = contrib.get("confidence", 0.0)
            
            # Check action deviation (different action)
            if (
                strategy_action != aggregate_action
                and aggregate_action != "hold"
                and strategy_action != "hold"
            ):
                alert = {
                    "type": "strategy_deviation",
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "aggregate_action": aggregate_action,
                    "aggregate_confidence": aggregate_confidence,
                    "strategy_action": strategy_action,
                    "strategy_confidence": strategy_confidence,
                }
                logger.info("[Alert] Strategy deviation: %s", alert)
            
            # Check confidence deviation (significant difference)
            confidence_diff = abs(strategy_confidence - aggregate_confidence)
            if (
                confidence_diff > self.deviation_threshold
                and aggregate_confidence > 0.3
            ):
                alert = {
                    "type": "confidence_deviation",
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "aggregate_confidence": aggregate_confidence,
                    "strategy_confidence": strategy_confidence,
                    "deviation": confidence_diff,
                    "threshold": self.deviation_threshold,
                }
                logger.info("[Alert] Confidence deviation: %s", alert)

    def emit_now(self) -> None:
        """Force an immediate synchronous emission (primarily for tests/demos)."""
        self._emit_summary_safe(time.time(), force=True)

    def _emit_summary_safe(self, ts: float, force: bool = False) -> None:
        try:
            self._emit_summary(ts, force=force)
        finally:
            with self._lock:
                self._emitting = False

    def _emit_summary(self, ts: float, force: bool = False) -> None:
        payload = None
        with self._lock:
            if not force and (ts - self._last_emit) < self.interval_seconds:
                return
            payload = self._build_payload_locked()
            self._reset_locked()
            self._last_emit = ts
        # Emit outside lock to avoid handler delays blocking updates
        self.logger_fn(
            "runtime.summary",
            payload,
        )

    def _build_payload_locked(self) -> Dict[str, Any]:
        return {
            "window_seconds": self.interval_seconds,
            "samples": self._samples,
            "readiness_distribution": dict(self._readiness_counts),
            "regime_distribution": dict(self._regime_counts),
            "trades_by_readiness": dict(self._trades_by_readiness),
        }

    def _reset_locked(self) -> None:
        self._readiness_counts = defaultdict(int)
        self._regime_counts = defaultdict(int)
        self._trades_by_readiness = defaultdict(int)
        self._samples = 0


__all__ = ["RuntimeSummaryAggregator"]
