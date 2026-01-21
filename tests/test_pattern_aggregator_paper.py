"""
Integration tests to ensure pattern strategies aggregate in paper mode with logging.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from strategies.regime_strategy import RegimeBasedStrategy
from strategies.regime_detector import MarketRegime


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


def test_pattern_strategies_enabled_in_paper_mode(caplog):
    caplog.set_level("INFO")
    strategy = RegimeBasedStrategy(
        allowed_regimes=["TRENDING"],
        enable_pattern_strategies=True,
        trading_mode="paper",
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
    assert "contributions" in res
    assert len(res["contributions"]) == 4
    assert any("Pattern strategies enabled" in rec.message for rec in caplog.records)
    assert any("[MultiStrategy] strat=" in rec.message for rec in caplog.records)


def test_live_mode_does_not_enable_patterns():
    strategy = RegimeBasedStrategy(
        allowed_regimes=["TRENDING"],
        enable_pattern_strategies=True,
        trading_mode="live",
        enable_readiness_gate=False,
    )

    # underlying should remain default EMA trend, not aggregator with contributions
    assert strategy.underlying_strategy.name == "EMATrendStrategy"
