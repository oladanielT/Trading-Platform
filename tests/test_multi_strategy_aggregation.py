"""
Tests for MultiStrategyAggregator combining multiple independent strategies.
"""

import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from strategies.multi_strategy_aggregator import MultiStrategyAggregator
from strategies.base import BaseStrategy, SignalOutput, Signal, PositionState


class MockStrategy(BaseStrategy):
    def __init__(self, name, action, confidence):
        super().__init__(name=name)
        self._action = action
        self._confidence = confidence
        self.calls = 0

    def analyze(self, market_data):
        self.calls += 1
        return {"action": self._action, "confidence": self._confidence}

    def generate_signal(self, market_data, position_state: PositionState) -> SignalOutput:
        return self.create_signal(signal=Signal[self._action.upper()], confidence=self._confidence)


def _fake_market(rows=10):
    base = datetime.utcnow()
    data = []
    price = 100.0
    for i in range(rows):
        price += 1
        data.append(
            {
                "timestamp": base + timedelta(minutes=i),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1000,
            }
        )
    return pd.DataFrame(data)


def test_weighted_buy_vs_sell():
    s_buy = MockStrategy("abc", "buy", 0.8)
    s_sell = MockStrategy("cont", "sell", 0.6)
    agg = MultiStrategyAggregator([s_buy, s_sell], weights={"abc": 2.0, "cont": 1.0})

    result = agg.analyze(_fake_market())

    assert result["action"] == "buy"
    assert result["confidence"] > 0.0
    assert any(c["strategy"] == "abc" for c in result["contributions"])
    assert any(c["strategy"] == "cont" for c in result["contributions"])


def test_hold_when_no_votes():
    s_hold = MockStrategy("flat", "hold", 0.2)
    agg = MultiStrategyAggregator([s_hold])
    result = agg.analyze(_fake_market())
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0


def test_strategies_called_independently():
    s1 = MockStrategy("abc", "buy", 0.5)
    s2 = MockStrategy("cont", "sell", 0.4)
    agg = MultiStrategyAggregator([s1, s2])
    agg.analyze(_fake_market())
    assert s1.calls == 1
    assert s2.calls == 1


def test_logging_includes_contributions(caplog):
    import logging

    s1 = MockStrategy("abc", "buy", 0.7)
    s2 = MockStrategy("reclaim", "sell", 0.2)
    agg = MultiStrategyAggregator([s1, s2])
    caplog.set_level(logging.INFO)
    agg.analyze(_fake_market())
    assert any("[MultiStrategy]" in rec.message for rec in caplog.records)


def test_generate_signal_adapter():
    s1 = MockStrategy("abc", "buy", 0.6)
    agg = MultiStrategyAggregator([s1])
    sig = agg.generate_signal(_fake_market(), PositionState())
    assert sig.signal == Signal.BUY
    assert sig.confidence == 0.6
