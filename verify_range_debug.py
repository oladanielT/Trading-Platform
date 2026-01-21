#!/usr/bin/env python3
"""Verify RangeTradingStrategy debug output appears"""

import sys
import os

# Ensure output is not buffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

print("\n" + "="*70)
print("TESTING RANGE TRADING STRATEGY DEBUG OUTPUT")
print("="*70)

# Import the module - should see module-level debug
print("\n[TEST] Importing strategies.range_trading...")
import strategies.range_trading
print("[TEST] Import complete\n")

# Test instantiation
print("[TEST] Creating RangeTradingStrategy instance...")
from strategies.range_trading import RangeTradingStrategy
from strategies.base import PositionState
import pandas as pd
import numpy as np
import datetime

strat = RangeTradingStrategy(
    rsi_period=14,
    rsi_oversold=35.0,
    rsi_overbought=65.0,
    min_volatility=0.001,
    max_volatility=0.15
)
print(f"[TEST] Strategy created: {strat.name}\n")

# Create realistic test data
print("[TEST] Creating test market data...")
np.random.seed(42)
prices = np.linspace(90000, 92000, 200) + np.random.normal(0, 500, 200)
volumes = np.ones(200) * 1000000
timestamps = [datetime.datetime.now() - datetime.timedelta(hours=i) for i in range(199, -1, -1)]

data = pd.DataFrame({
    'timestamp': timestamps,
    'open': prices + np.random.normal(0, 100, 200),
    'high': prices + abs(np.random.normal(0, 200, 200)),
    'low': prices - abs(np.random.normal(0, 200, 200)),
    'close': prices,
    'volume': volumes
})

print(f"[TEST] Data created: {data.shape[0]} candles\n")

# Generate signal - should see all debug output
print("[TEST] Calling generate_signal() - watch for [RANGE DEBUG] output:\n")
print("v"*70 + " START OF STRATEGY OUTPUT " + "v"*70)
sys.stdout.flush()

pos = PositionState()
signal = strat.generate_signal(data, pos)

print("^"*70 + " END OF STRATEGY OUTPUT " + "^"*70)

print(f"\n[TEST] Signal returned: {signal.signal.value}")
print(f"[TEST] Confidence: {signal.confidence}")
print(f"[TEST] Reason: {signal.metadata.get('reason', 'N/A')}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\nIf you see [RANGE DEBUG] output above, the strategy is working!")
print("If NOT, there's an issue with stdout buffering or exception handling.")
print("="*70 + "\n")
