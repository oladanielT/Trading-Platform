"""
Tests for multi-symbol independent tracking and aggregation.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from strategies.regime_strategy import RegimeBasedStrategy
from strategies.multi_strategy_aggregator import MultiStrategyAggregator
from strategies.regime_detector import MarketRegime
from strategies.base import BaseStrategy, PositionState, Signal


def _market(rows=20, start=100.0, step=1.0):
    base = datetime.utcnow() - timedelta(minutes=rows)
    data = []
    price = start
    for i in range(rows):
        price += step
        data.append(
            {
                "timestamp": base + timedelta(minutes=i),
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 1_000 + i,
            }
        )
    return pd.DataFrame(data)


def test_regime_strategy_tracks_readiness_per_symbol():
    """Verify readiness state is tracked independently per symbol."""
    strategy = RegimeBasedStrategy(
        allowed_regimes=["TRENDING"],
        enable_readiness_gate=False,
        trading_mode="paper",
    )

    with patch.object(strategy.regime_detector, "detect", return_value={
        "regime": MarketRegime.TRENDING,
        "confidence": 0.9,
        "metrics": {"trend_strength": 2.0, "price_slope_pct": 0.2},
        "explanation": "mocked",
    }):
        res_btc = strategy.analyze(_market(), symbol="BTC/USDT")
        res_eth = strategy.analyze(_market(), symbol="ETH/USDT")

    assert res_btc is not None
    assert res_eth is not None
    assert res_btc.get("symbol") == "BTC/USDT"
    assert res_eth.get("symbol") == "ETH/USDT"
    # Each symbol should have independent readiness tracking
    assert "BTC/USDT" in strategy._readiness_score
    assert "ETH/USDT" in strategy._readiness_score


def test_regime_strategy_default_symbol_fallback():
    """Verify single-symbol behavior preserved when symbol not provided."""
    strategy = RegimeBasedStrategy(
        allowed_regimes=["TRENDING"],
        enable_readiness_gate=False,
    )

    with patch.object(strategy.regime_detector, "detect", return_value={
        "regime": MarketRegime.TRENDING,
        "confidence": 0.9,
        "metrics": {"trend_strength": 2.0, "price_slope_pct": 0.2},
        "explanation": "mocked",
    }):
        res = strategy.analyze(_market())

    assert res is not None
    assert "symbol" not in res  # default mode doesn't add symbol to output
    assert strategy._default_symbol in strategy._readiness_score


def test_multi_strategy_aggregator_logs_symbol(caplog):
    """Verify MultiStrategyAggregator includes symbol in logs."""
    import logging

    class DummyStrat(BaseStrategy):
        def __init__(self, name):
            super().__init__(name=name)

        def analyze(self, market_data, symbol=None):
            return {"action": "buy", "confidence": 0.5}

        def generate_signal(self, market_data, position):
            return Signal(
                timestamp=datetime.utcnow(),
                action="hold",
                confidence=0.0,
                rationale="dummy",
            )

    caplog.set_level(logging.INFO)
    agg = MultiStrategyAggregator([DummyStrat("strat1"), DummyStrat("strat2")])
    res = agg.analyze(_market(), symbol="BTC/USDT")

    assert res is not None
    assert res.get("symbol") == "BTC/USDT"
    assert any("symbol=BTC/USDT" in rec.message for rec in caplog.records)


def test_multi_symbol_readiness_independence():
    """Verify different symbols maintain independent readiness scores."""
    strategy = RegimeBasedStrategy(
        allowed_regimes=["TRENDING"],
        enable_readiness_gate=True,
        trading_mode="paper",
    )

    # Create larger dataset to pass EMA requirements (need 200+)
    large_market = _market(rows=250, start=100.0, step=1.0)

    # Mock different metrics for different symbols
    def mock_detect_btc(market_data):
        return {
            "regime": MarketRegime.TRENDING,
            "confidence": 0.95,
            "metrics": {"trend_strength": 3.5, "price_slope_pct": 0.5},
            "explanation": "strong trend",
        }

    def mock_detect_eth(market_data):
        return {
            "regime": MarketRegime.TRENDING,
            "confidence": 0.8,
            "metrics": {"trend_strength": 1.2, "price_slope_pct": 0.1},
            "explanation": "weak trend",
        }

    with patch.object(strategy.regime_detector, "detect", side_effect=mock_detect_btc):
        res_btc = strategy.analyze(large_market, symbol="BTC/USDT")

    with patch.object(strategy.regime_detector, "detect", side_effect=mock_detect_eth):
        res_eth = strategy.analyze(large_market, symbol="ETH/USDT")

    # Different trend strengths should produce different readiness scores
    assert strategy._readiness_score["BTC/USDT"] != strategy._readiness_score["ETH/USDT"]
    # Scores differ but both may still be NOT_READY if insufficient history
    # Instead, verify that each symbol has independent tracking
    assert "BTC/USDT" in strategy._readiness_score
    assert "ETH/USDT" in strategy._readiness_score
    assert "BTC/USDT" in strategy._readiness_state
    assert "ETH/USDT" in strategy._readiness_state
