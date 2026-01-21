"""
Tests for RuntimeSummaryAggregator alert functionality.
"""

import logging
import pytest

from monitoring.runtime_summary import RuntimeSummaryAggregator


class TestAlertsDisabledByDefault:
    """Test that alerts are disabled by default."""
    
    def test_alerts_disabled_by_default(self):
        """Verify alerts are disabled by default."""
        agg = RuntimeSummaryAggregator()
        assert agg.enable_alerts is False
    
    def test_record_analysis_no_op_when_disabled(self, caplog):
        """Verify record_analysis does nothing when alerts disabled."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(enable_alerts=False)
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="not_ready",
            readiness_score=0.1,
            aggregate_action="buy",
            aggregate_confidence=0.9,
        )
        
        # No alerts should be logged
        alert_logs = [rec for rec in caplog.records if "[Alert]" in rec.message]
        assert len(alert_logs) == 0


class TestReadinessAlerts:
    """Test readiness degradation alerts."""
    
    def test_alert_readiness_degradation_to_not_ready(self, caplog):
        """Alert when readiness drops to NOT_READY."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            readiness_alert_threshold="not_ready",  # lowercase
        )
        
        # First: good readiness state
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",  # level 3
            readiness_score=0.9,
            aggregate_action="buy",
            aggregate_confidence=0.6,
        )
        
        # Second: degradation to not_ready
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="not_ready",  # level 1
            readiness_score=0.1,
            aggregate_action="hold",
            aggregate_confidence=0.0,
        )
        
        # Should have readiness alert
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "readiness_degradation" in rec.message
        ]
        assert len(alert_logs) > 0
        assert "BTC/USDT" in alert_logs[0].message
        assert "not_ready" in alert_logs[0].message
    
    def test_alert_readiness_threshold_configurable(self, caplog):
        """Test that readiness threshold is configurable."""
        caplog.set_level(logging.INFO)
        
        # Only alert for very poor readiness (below "not_ready" level)
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            readiness_alert_threshold="not_ready",
        )
        
        # Forming state should NOT trigger alert
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="forming",
            readiness_score=0.5,
            aggregate_action="buy",
            aggregate_confidence=0.6,
        )
        
        alert_logs = [rec for rec in caplog.records if "[Alert]" in rec.message]
        assert len(alert_logs) == 0  # No alert for forming
    
    def test_no_alert_readiness_improvement(self, caplog):
        """No alert when readiness improves."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(enable_alerts=True)
        
        # Start poor
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="not_ready",
            readiness_score=0.1,
            aggregate_action="hold",
            aggregate_confidence=0.0,
        )
        
        # Improve
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.9,
            aggregate_action="buy",
            aggregate_confidence=0.7,
        )
        
        # Should have no readiness alert (improvement is good)
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "readiness_degradation" in rec.message
        ]
        assert len(alert_logs) == 0


class TestConfidenceAlerts:
    """Test confidence anomaly alerts."""
    
    def test_alert_confidence_anomaly(self, caplog):
        """Alert on high confidence with action change."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            confidence_threshold=0.8,
        )
        
        # First: buy signal with moderate confidence
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="buy",
            aggregate_confidence=0.6,
        )
        
        # Second: sudden high confidence sell signal
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="sell",
            aggregate_confidence=0.95,  # High confidence
        )
        
        # Should have confidence anomaly alert
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "confidence_anomaly" in rec.message
        ]
        assert len(alert_logs) > 0
        assert "BTC/USDT" in alert_logs[0].message
        assert "sell" in alert_logs[0].message
    
    def test_no_alert_hold_signal(self, caplog):
        """No alert for hold signal regardless of confidence."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            confidence_threshold=0.8,
        )
        
        # Buy signal
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="buy",
            aggregate_confidence=0.6,
        )
        
        # High confidence hold (should not alert)
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="hold",
            aggregate_confidence=0.95,
        )
        
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "confidence_anomaly" in rec.message
        ]
        assert len(alert_logs) == 0


class TestDeviationAlerts:
    """Test strategy signal deviation alerts."""
    
    def test_alert_strategy_action_deviation(self, caplog):
        """Alert when strategy action differs from aggregate."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            deviation_threshold=0.3,
        )
        
        contributions = [
            {
                "strategy": "FVGStrategy",
                "action": "buy",
                "confidence": 0.7,
            },
            {
                "strategy": "ABCStrategy",
                "action": "sell",  # Different from aggregate
                "confidence": 0.6,
            },
        ]
        
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="buy",
            aggregate_confidence=0.65,
            contributions=contributions,
        )
        
        # Should have deviation alert
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "strategy_deviation" in rec.message
        ]
        assert len(alert_logs) > 0
        assert "ABCStrategy" in alert_logs[0].message
        assert "sell" in alert_logs[0].message
    
    def test_alert_confidence_deviation(self, caplog):
        """Alert when strategy confidence deviates significantly."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            deviation_threshold=0.2,  # Lower threshold
        )
        
        contributions = [
            {
                "strategy": "FVGStrategy",
                "action": "buy",
                "confidence": 0.9,  # Deviation: |0.9 - 0.65| = 0.25
            },
            {
                "strategy": "ABCStrategy",
                "action": "buy",
                "confidence": 0.4,  # Deviation: |0.4 - 0.65| = 0.25
            },
        ]
        
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="buy",
            aggregate_confidence=0.65,
            contributions=contributions,
        )
        
        # Should have confidence deviation alert for both strategies
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "confidence_deviation" in rec.message
        ]
        assert len(alert_logs) >= 2
    
    def test_no_alert_hold_strategy_deviation(self, caplog):
        """No alert when strategy action is hold."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            deviation_threshold=0.3,
        )
        
        contributions = [
            {
                "strategy": "FVGStrategy",
                "action": "buy",
                "confidence": 0.7,
            },
            {
                "strategy": "ABCStrategy",
                "action": "hold",  # Hold doesn't trigger alert
                "confidence": 0.2,
            },
        ]
        
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="buy",
            aggregate_confidence=0.65,
            contributions=contributions,
        )
        
        # Should have NO deviation alert for hold
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "strategy_deviation" in rec.message
        ]
        assert len(alert_logs) == 0
    
    def test_no_alert_low_aggregate_confidence(self, caplog):
        """No deviation alert when aggregate confidence is low."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            deviation_threshold=0.3,
        )
        
        contributions = [
            {
                "strategy": "FVGStrategy",
                "action": "buy",
                "confidence": 0.5,
            },
            {
                "strategy": "ABCStrategy",
                "action": "buy",
                "confidence": 0.1,  # Very different
            },
        ]
        
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="forming",
            readiness_score=0.3,
            aggregate_action="buy",
            aggregate_confidence=0.2,  # Low confidence
            contributions=contributions,
        )
        
        # Should have NO alert (confidence too low to matter)
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "confidence_deviation" in rec.message
        ]
        assert len(alert_logs) == 0


class TestMultiSymbolAlerts:
    """Test alerts for multiple symbols independently."""
    
    def test_independent_alerts_per_symbol(self, caplog):
        """Verify symbols are tracked independently."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            confidence_threshold=0.8,
        )
        
        # BTC: buy at moderate confidence
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="buy",
            aggregate_confidence=0.6,
        )
        
        # ETH: buy at moderate confidence
        agg.record_analysis(
            symbol="ETH/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="buy",
            aggregate_confidence=0.6,
        )
        
        # BTC: high confidence sell (should alert)
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="sell",
            aggregate_confidence=0.95,
        )
        
        # ETH: hold (should NOT alert)
        agg.record_analysis(
            symbol="ETH/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="hold",
            aggregate_confidence=0.95,
        )
        
        # Should have only 1 alert (BTC confidence anomaly)
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "confidence_anomaly" in rec.message
        ]
        assert len(alert_logs) == 1
        assert "BTC/USDT" in alert_logs[0].message


class TestAlertPayload:
    """Test alert payload structure and content."""
    
    def test_readiness_alert_payload(self, caplog):
        """Verify readiness alert includes required fields."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(enable_alerts=True)
        
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.9,
            aggregate_action="buy",
            aggregate_confidence=0.7,
        )
        
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="not_ready",
            readiness_score=0.1,
            aggregate_action="hold",
            aggregate_confidence=0.0,
        )
        
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "readiness_degradation" in rec.message
        ]
        assert len(alert_logs) > 0
        
        msg = alert_logs[0].message
        assert "symbol" in msg.lower()
        assert "readiness_score" in msg.lower() or "score" in msg.lower()
    
    def test_confidence_alert_payload(self, caplog):
        """Verify confidence alert includes required fields."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            confidence_threshold=0.8,
        )
        
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="buy",
            aggregate_confidence=0.6,
        )
        
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="sell",
            aggregate_confidence=0.95,
        )
        
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "confidence_anomaly" in rec.message
        ]
        assert len(alert_logs) > 0
        
        msg = alert_logs[0].message
        assert "symbol" in msg.lower()
        assert "confidence" in msg.lower()
    
    def test_deviation_alert_payload(self, caplog):
        """Verify deviation alert includes strategy and confidence."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            deviation_threshold=0.3,
        )
        
        contributions = [
            {
                "strategy": "FVGStrategy",
                "action": "buy",
                "confidence": 0.7,
            },
            {
                "strategy": "ABCStrategy",
                "action": "sell",
                "confidence": 0.6,
            },
        ]
        
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.8,
            aggregate_action="buy",
            aggregate_confidence=0.65,
            contributions=contributions,
        )
        
        alert_logs = [
            rec for rec in caplog.records
            if "[Alert]" in rec.message and "strategy_deviation" in rec.message
        ]
        assert len(alert_logs) > 0
        
        msg = alert_logs[0].message
        assert "strategy" in msg.lower()
        assert "symbol" in msg.lower()
        assert "confidence" in msg.lower()


class TestAlertIntegration:
    """Integration tests with real workflow scenarios."""
    
    def test_realistic_trading_scenario(self, caplog):
        """Simulate realistic trading scenario with multiple alerts."""
        caplog.set_level(logging.INFO)
        
        agg = RuntimeSummaryAggregator(
            enable_alerts=True,
            confidence_threshold=0.8,
            deviation_threshold=0.3,
        )
        
        # Scenario 1: System starts healthy
        contributions = [
            {"strategy": "FVGStrategy", "action": "buy", "confidence": 0.7},
            {"strategy": "ABCStrategy", "action": "buy", "confidence": 0.65},
        ]
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="ready",
            readiness_score=0.85,
            aggregate_action="buy",
            aggregate_confidence=0.68,
            contributions=contributions,
        )
        
        # Scenario 2: Readiness drops
        contributions = [
            {"strategy": "FVGStrategy", "action": "hold", "confidence": 0.3},
            {"strategy": "ABCStrategy", "action": "hold", "confidence": 0.2},
        ]
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="not_ready",
            readiness_score=0.1,
            aggregate_action="hold",
            aggregate_confidence=0.0,
            contributions=contributions,
        )
        
        # Scenario 3: System recovers with high confidence signal
        contributions = [
            {"strategy": "FVGStrategy", "action": "buy", "confidence": 0.9},
            {"strategy": "ABCStrategy", "action": "sell", "confidence": 0.7},
        ]
        agg.record_analysis(
            symbol="BTC/USDT",
            readiness_state="forming",
            readiness_score=0.5,
            aggregate_action="buy",
            aggregate_confidence=0.85,
            contributions=contributions,
        )
        
        # Should have multiple types of alerts
        alert_logs = [rec for rec in caplog.records if "[Alert]" in rec.message]
        assert len(alert_logs) > 0
        
        # Check for various alert types
        alert_types = set()
        for log in alert_logs:
            if "readiness_degradation" in log.message:
                alert_types.add("readiness")
            if "strategy_deviation" in log.message:
                alert_types.add("deviation")
