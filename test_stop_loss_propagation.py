"""
Quick test to verify stop-loss propagation through the stack
"""
import sys
sys.path.insert(0, '/home/oladan/Trading-Platform')

import pandas as pd
from datetime import datetime, timedelta
from strategies.pattern_strategies import FVGStrategy, ContinuationStrategy
from strategies.multi_strategy_aggregator import MultiStrategyAggregator
from strategies.base import PositionState

# Create synthetic data with a gap
def create_gap_data():
    base = datetime.utcnow() - timedelta(hours=150)
    data = []
    for i in range(150):
        if i < 50:
            price = 100 + i * 0.5
        elif i < 53:  # Create gap
            price = 105 + (i - 50) * 2
        else:
            price = 105 + (i - 53) * 0.1
        
        data.append({
            "timestamp": base + timedelta(hours=i),
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000 + i,
        })
    return pd.DataFrame(data)

# Test individual FVG strategy
print("="*60)
print("Testing FVG Strategy Stop-Loss Propagation")
print("="*60)

market_data = create_gap_data()
fvg = FVGStrategy(lookback=100, min_gap_size_pct=0.3)
signal = fvg.generate_signal(market_data, PositionState())

print(f"\nFVG Signal: {signal.signal.value}")
print(f"Confidence: {signal.confidence:.3f}")
print(f"Metadata keys: {list(signal.metadata.keys())}")
print(f"Full metadata: {signal.metadata}")
print(f"Entry price: {signal.metadata.get('entry_price')}")
print(f"Stop loss: {signal.metadata.get('stop_loss')}")

# Test multi-strategy aggregator
print("\n" + "="*60)
print("Testing MultiStrategyAggregator Stop-Loss Propagation")
print("="*60)

aggregator = MultiStrategyAggregator(
    strategies=[
        ContinuationStrategy(lookback=50),
        FVGStrategy(lookback=100, min_gap_size_pct=0.3)
    ],
    weights={'ContinuationStrategy': 0.4, 'FVGStrategy': 0.3}
)

agg_signal = aggregator.generate_signal(market_data, PositionState())

print(f"\nAggregated Signal: {agg_signal.signal.value}")
print(f"Confidence: {agg_signal.confidence:.3f}")
print(f"Metadata keys: {list(agg_signal.metadata.keys())}")
print(f"Entry price: {agg_signal.metadata.get('entry_price')}")
print(f"Stop loss: {agg_signal.metadata.get('stop_loss')}")

if agg_signal.metadata.get('stop_loss') is not None:
    print("\n✅ SUCCESS: Stop-loss is propagating through aggregator!")
else:
    print("\n❌ FAILURE: Stop-loss not found in aggregated signal")
