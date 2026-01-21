import sys, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.pattern_strategies import ContinuationStrategy, FVGStrategy
from strategies.base import PositionState


def make_data(n=500, base=43000.0, up_factor=0.8):
    # seed for reproducibility
    rng = np.random.default_rng(42)
    # upward drift to simulate 80% move
    drift = np.linspace(0, up_factor, n)
    rets = rng.normal(0.0005, 0.01, n) + drift / n
    prices = base * np.exp(np.cumsum(rets))
    ts = [datetime.now() - timedelta(hours=n-i) for i in range(n)]
    o = prices * (1 + rng.normal(0, 0.002, n))
    c = prices
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    v = np.exp(rng.normal(10, 0.5, n))

    # Inject a clear 3-candle bullish FVG near the end
    i = n - 20
    h[i] = min(h[i], l[i+2] * 0.98)  # ensure c1.high < c3.low
    l[i+2] = max(l[i+2], h[i] * 1.02)

    # Inject impulse + consolidation + breakout for continuation
    j = n - 60
    c[j:j+5] *= 1.03  # impulse up ~3%
    v[j:j+5] *= 1.3   # volume surge on impulse
    # consolidation with lower volume
    c[j+5:j+12] *= 0.995
    v[j+5:j+12] *= 0.8
    # breakout
    c[j+12] *= 1.02

    df = pd.DataFrame({
        'timestamp': ts,
        'open': o,
        'high': h,
        'low': l,
        'close': c,
        'volume': v
    })
    return df


def main():
    df = make_data()
    pos = PositionState()

    print("\n=== ContinuationStrategy (relaxed) ===")
    cont = ContinuationStrategy(
        lookback=50,
        min_impulse_pct=1.2,
        min_retracement=0.15,
        max_retracement=0.75,
        min_consolidation_candles=3,
        max_consolidation_candles=35,
        volume_surge_threshold=1.1,
    )
    sig_c = cont.generate_signal(df, pos)
    print("Continuation signal:", sig_c.signal.value, sig_c.confidence, sig_c.metadata)

    print("\n=== FVGStrategy (relaxed) ===")
    fvg = FVGStrategy(
        lookback=100,
        min_gap_size_pct=0.2,
        min_fill_pct=0.3,
        max_fill_pct=0.98,
        require_volume_spike=True,
        volume_spike_multiplier=1.2,
        max_gap_age=100,
        require_momentum=False,
    )
    sig_f = fvg.generate_signal(df, pos)
    print("FVG signal:", sig_f.signal.value, sig_f.confidence, sig_f.metadata)


if __name__ == "__main__":
    main()
