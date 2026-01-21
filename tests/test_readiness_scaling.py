"""
Unit tests for readiness-based confidence and position sizing scaling (PHASE 8).

Covers:
- Feature flag default behavior
- Scaling across readiness scores (0.0, 0.5, 1.0)
- Live and paper/backtest mode application when enabled
- Logging of scaling decisions
- Confidence capping when scaling factors exceed 1.0
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

from strategies.regime_strategy import RegimeBasedStrategy
from strategies.regime_detector import MarketRegime


class TestReadinessScaling(unittest.TestCase):
    """Tests for readiness-based scaling of confidence and position sizing."""

    def setUp(self):
        np.random.seed(7)
        self.market_data = self._create_market_data()

    def _create_market_data(self, bars: int = 50) -> pd.DataFrame:
        """Create simple trending market data for testing."""
        timestamps = [datetime.now() - timedelta(minutes=bars - i) for i in range(bars)]
        price = 100.0
        rows = []
        for ts in timestamps:
            price += 0.2  # mild uptrend
            rows.append(
                {
                    "timestamp": ts,
                    "open": price - 0.1,
                    "high": price + 0.2,
                    "low": price - 0.2,
                    "close": price,
                    "volume": 1_000 + np.random.uniform(0, 100),
                }
            )
        return pd.DataFrame(rows)

    def _analyze_with_readiness(self, readiness_score: float, **strategy_kwargs):
        """Run analyze with mocked detector and underlying strategy for a given readiness score."""
        strategy = RegimeBasedStrategy(
            allowed_regimes=["TRENDING"],
            enable_readiness_gate=False,
            reduce_confidence_in_marginal_regimes=False,
            **strategy_kwargs,
        )

        detection = {
            "regime": MarketRegime.TRENDING,
            "confidence": 0.95,
            "metrics": {"trend_strength": 2.5, "price_slope_pct": 0.6},
            "explanation": "mocked",
        }

        with patch.object(strategy, "_calculate_regime_readiness_score", return_value=readiness_score), \
            patch.object(strategy.regime_detector, "detect", return_value=detection), \
            patch.object(strategy.underlying_strategy, "analyze", return_value={"action": "buy", "confidence": 0.8}):
            return strategy.analyze(self.market_data)

    def test_scaling_disabled_preserves_confidence(self):
        result = self._analyze_with_readiness(1.0, enable_readiness_scaling=False, trading_mode="paper")
        self.assertEqual(result["confidence"], 0.8)
        self.assertNotIn("readiness_scale_factor", result)

    def test_scaling_applies_min_factor(self):
        result = self._analyze_with_readiness(0.0, enable_readiness_scaling=True, trading_mode="paper")
        self.assertAlmostEqual(result["confidence"], 0.4, places=5)
        self.assertAlmostEqual(result["readiness_scale_factor"], 0.5, places=5)
        self.assertAlmostEqual(result["position_size_scale"], 0.4, places=5)

    def test_scaling_applies_mid_factor(self):
        result = self._analyze_with_readiness(0.5, enable_readiness_scaling=True, trading_mode="paper")
        self.assertAlmostEqual(result["confidence"], 0.6, places=5)
        self.assertAlmostEqual(result["readiness_scale_factor"], 0.75, places=5)

    def test_scaling_applies_max_factor_live_mode(self):
        result = self._analyze_with_readiness(1.0, enable_readiness_scaling=True, trading_mode="live")
        self.assertAlmostEqual(result["confidence"], 0.8, places=5)
        self.assertAlmostEqual(result["readiness_scale_factor"], 1.0, places=5)

    def test_scaling_logs_info(self):
        strategy_kwargs = {"enable_readiness_scaling": True, "trading_mode": "paper"}
        strategy = RegimeBasedStrategy(
            allowed_regimes=["TRENDING"],
            enable_readiness_gate=False,
            reduce_confidence_in_marginal_regimes=False,
            **strategy_kwargs,
        )

        detection = {
            "regime": MarketRegime.TRENDING,
            "confidence": 0.95,
            "metrics": {"trend_strength": 2.5, "price_slope_pct": 0.6},
            "explanation": "mocked",
        }

        with patch.object(strategy, "_calculate_regime_readiness_score", return_value=0.5), \
            patch.object(strategy.regime_detector, "detect", return_value=detection), \
            patch.object(strategy.underlying_strategy, "analyze", return_value={"action": "buy", "confidence": 0.8}), \
            patch("strategies.regime_strategy.logger") as mock_logger:
            result = strategy.analyze(self.market_data)
            self.assertAlmostEqual(result["confidence"], 0.6, places=5)
            scaling_logs = [c for c in mock_logger.info.call_args_list if "[ReadinessScaling]" in str(c)]
            self.assertGreater(len(scaling_logs), 0)

    def test_scaling_caps_confidence_at_one(self):
        result = self._analyze_with_readiness(
            1.0,
            enable_readiness_scaling=True,
            trading_mode="paper",
            scaling_factor_max=1.5,
            scaling_factor_min=1.0,
        )
        self.assertAlmostEqual(result["confidence"], 1.0, places=5)
        self.assertAlmostEqual(result["readiness_scale_factor"], 1.5, places=5)


if __name__ == "__main__":
    unittest.main()
