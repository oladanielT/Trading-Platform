#!/usr/bin/env python3
"""Test the improved RangeTradingStrategy with tiered signals"""

import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

from strategies.range_trading import RangeTradingStrategy
from strategies.base import PositionState
import pandas as pd
import numpy as np
import datetime

print("\n" + "="*70)
print("TESTING TIERED RANGE TRADING STRATEGY")
print("="*70)

# Create strategy with new parameters
strat = RangeTradingStrategy(
    rsi_period=14,
    rsi_oversold=45.0,        # Normal buy threshold
    rsi_overbought=55.0,      # Normal sell threshold
    rsi_strong_oversold=35.0, # Strong buy threshold
    rsi_strong_overbought=65.0, # Strong sell threshold
    support_resistance_pct=0.5,  # Very tight
    min_volatility=0.001,
    max_volatility=0.15
)

print(f"\n✅ Strategy created with tiered thresholds:")
print(f"   Normal: oversold < 45, overbought > 55")
print(f"   Strong: oversold < 35, overbought > 65")

# Create test data - scenario from user feedback
# Price at resistance, RSI at 59 (between 55 and 65)
np.random.seed(42)
prices = np.linspace(90000, 92500, 200)
prices[-1] = 92482.54  # Force price at resistance
volumes = np.ones(200) * 1000000
timestamps = [datetime.datetime.now() - datetime.timedelta(hours=i) for i in range(199, -1, -1)]

data = pd.DataFrame({
    'timestamp': timestamps,
    'open': prices + np.random.normal(0, 100, 200),
    'high': prices + 50,
    'low': prices - 50,
    'close': prices,
    'volume': volumes
})

print(f"\n📊 Test scenario:")
print(f"   Price: ${data['close'].iloc[-1]:.2f}")
print(f"   Data range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
print(f"   Candles: {len(data)}")

# Generate signal
print(f"\n🎯 Calling generate_signal():\n")
pos = PositionState()
signal = strat.generate_signal(data, pos)

print(f"\n" + "="*70)
print(f"RESULT:")
print(f"="*70)
print(f"Signal: {signal.signal.value.upper()}")
print(f"Confidence: {signal.confidence:.3f}")
print(f"Reason: {signal.metadata.get('reason', 'N/A')}")

if signal.metadata.get('rsi'):
    print(f"\nDetails:")
    print(f"  RSI: {signal.metadata.get('rsi'):.2f}")
    print(f"  Support: ${signal.metadata.get('support'):.2f}")
    print(f"  Resistance: ${signal.metadata.get('resistance'):.2f}")
    print(f"  Signal strength: {signal.metadata.get('signal_strength', 'N/A')}")

print("\n" + "="*70)
print("KEY IMPROVEMENT:")
print("="*70)
print("With old thresholds (RSI < 65 for sell):")
print("  → Price at resistance, RSI=59 → NO SIGNAL (too strict)")
print("\nWith new tiered thresholds:")
print("  → Normal sell: RSI > 55 (WOULD TRIGGER at RSI=59)")
print("  → Strong sell: RSI > 65 (requires more confirmation)")
print("  → Different confidence levels based on strength")
print("="*70 + "\n")
