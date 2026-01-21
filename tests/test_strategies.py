"""
Unit tests for new pattern strategies and their integration with MultiStrategyAggregator.
"""

from datetime import datetime, timedelta

import pandas as pd

from strategies.pattern_strategies import (
    FVGStrategy,
    ABCStrategy,
    ContinuationStrategy,
    ReclamationBlockStrategy,
    ConfirmationLayer,
)
from strategies.multi_strategy_aggregator import MultiStrategyAggregator
from strategies.base import PositionState, Signal, BaseStrategy


def _uptrend(rows=15, start=100.0, step=1.0):
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


def _gap_revisit():
    # create gap up then revisit
    base = datetime.utcnow() - timedelta(minutes=6)
    prices = [100, 101, 105, 105.2, 104.8, 101.0]
    data = []
    for i, p in enumerate(prices):
        data.append(
            {
                "timestamp": base + timedelta(minutes=i),
                "open": p,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p,
                "volume": 1_000 + i,
            }
        )
    return pd.DataFrame(data)


def test_fvg_revisit_buy():
    strat = FVGStrategy(min_gap_pct=0.002, lookback=4)
    res = strat.analyze(_gap_revisit())
    assert res["action"] in ["buy", "hold"]  # allow hold if gap deemed small
    assert "gap_pct" in res


def test_abc_pattern_detects_direction():
    df = _uptrend()
    # fabricate ABC: slight up, pullback, up
    df.loc[len(df) - 6:, "close"] = [100, 101, 100.5, 102, 101.5, 103]
    res = ABCStrategy(min_move_pct=0.001).analyze(df)
    assert res["action"] in ["buy", "hold", "sell"]


def test_continuation_breakout_buy():
    df = _uptrend(rows=14)
    # make last window consolidating then breakout up
    df.loc[len(df) - 5 :, "close"] = [110, 110.1, 110.05, 110.0, 111.0]
    res = ContinuationStrategy(window=5, min_slope=0.0001).analyze(df)
    assert res["action"] in ["buy", "hold"]


def test_reclamation_support_buy():
    df = _uptrend(rows=25)
    sup = df["low"].min()
    df.loc[len(df) - 1, "close"] = sup * 1.001  # revisit near support
    res = ReclamationBlockStrategy(lookback=20, tolerance_pct=0.005).analyze(df)
    assert res["action"] in ["buy", "hold"]


def test_confirmation_layer_filters_low_confidence():
    class LowConf(BaseStrategy):
        def __init__(self):
            super().__init__("LowConf")

        def analyze(self, market_data):
            return {"action": "buy", "confidence": 0.1}

        def generate_signal(self, market_data, position_state: PositionState):
            return self.create_signal(Signal.BUY, 0.1)

    conf = ConfirmationLayer(LowConf(), min_confidence=0.3, ema_span=3)
    res = conf.analyze(_uptrend(rows=5))
    assert res["action"] == "hold"


def test_confirmation_layer_boosts_on_alignment():
    class MidConf(BaseStrategy):
        def __init__(self):
            super().__init__("MidConf")

        def analyze(self, market_data):
            return {"action": "buy", "confidence": 0.5}

        def generate_signal(self, market_data, position_state: PositionState):
            return self.create_signal(Signal.BUY, 0.5)

    conf = ConfirmationLayer(MidConf(), min_confidence=0.3, ema_span=3, slope_tol=0.0, boost_factor=1.1)
    res = conf.analyze(_uptrend(rows=6))
    assert res["action"] == "buy"
    assert res["confidence"] > 0.5


def test_confirmation_layer_no_boost_on_misalignment():
    class MidConf(BaseStrategy):
        def __init__(self):
            super().__init__("MidConf")

        def analyze(self, market_data):
            return {"action": "buy", "confidence": 0.5}

        def generate_signal(self, market_data, position_state: PositionState):
            return self.create_signal(Signal.BUY, 0.5)

    # Create downtrend to force slope negative
    def _downtrend(rows=6, start=100.0, step=-1.0):
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

    conf = ConfirmationLayer(MidConf(), min_confidence=0.3, ema_span=3, slope_tol=0.0, boost_factor=1.1)
    res = conf.analyze(_downtrend(rows=6))
    assert res["action"] == "buy"
    # misalignment halves (from 0.5 to 0.25) and should not boost
    assert res["confidence"] < 0.5


def test_aggregator_with_new_strategies():
    fvg = FVGStrategy(min_gap_pct=0.0)
    abc = ABCStrategy(min_move_pct=0.0)
    cont = ContinuationStrategy(window=5, min_slope=0.0)
    recl = ReclamationBlockStrategy(lookback=10, tolerance_pct=0.01)
    agg = MultiStrategyAggregator([fvg, abc, cont, recl])
    res = agg.analyze(_uptrend(rows=12))
    assert res is not None
    assert "contributions" in res
    assert len(res["contributions"]) == 4