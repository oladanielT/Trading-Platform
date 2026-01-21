#!/usr/bin/env python3
"""
Sanity check: verify all strategies produce reasonable stop-loss distances
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.ema_trend import EMATrendStrategy
from strategies.pattern_strategies import FVGStrategy, ContinuationStrategy, ReclamationBlockStrategy
from strategies.base import PositionState


def create_mock_data(symbol="BTCUSDT", periods=200):
    """Create mock OHLCV data for testing"""
    np.random.seed(42)
    
    # Generate realistic price movement
    start_price = 100.0
    prices = [start_price]
    
    for _ in range(periods - 1):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV
    data = []
    base_time = datetime.now() - timedelta(hours=periods)
    
    for i, close in enumerate(prices):
        timestamp = base_time + timedelta(hours=i)
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.uniform(1000, 10000)
        
        data.append([timestamp, open_price, high, low, close, volume])
    
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df


def test_stop_distances():
    """Test each strategy produces valid stop-loss distances"""
    
    # Get mock market data
    market_data = create_mock_data(periods=200)
    position_snapshot = PositionState()
    
    strategies = [
        ("EMA", EMATrendStrategy(fast_period=12, slow_period=26, atr_multiplier=2.0)),
        ("FVG", FVGStrategy(lookback=150, min_gap_size_pct=0.3)),
        ("Continuation", ContinuationStrategy(lookback=100)),
        ("ReclamationBlock", ReclamationBlockStrategy(lookback=100)),
    ]
    
    print("=" * 60)
    print("STOP-LOSS DISTANCE AUDIT")
    print("=" * 60)
    print(f"Market data: {len(market_data)} candles")
    print(f"Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")
    print("=" * 60)
    
    all_passed = True
    
    for name, strategy in strategies:
        try:
            signal = strategy.generate_signal(market_data, position_snapshot)
            
            if signal.signal.value in ["BUY", "SELL"]:
                entry = signal.metadata.get('entry_price')
                stop = signal.metadata.get('stop_loss')
                
                if entry is None or stop is None:
                    print(f"❌ {name:15} Missing entry_price or stop_loss")
                    all_passed = False
                    continue
                
                # Calculate distance (for crypto, use % instead of pips)
                distance_pct = abs(entry - stop) / entry * 100
                distance_dollars = abs(entry - stop)
                
                # Sanity checks
                passed = True
                issues = []
                
                if stop == entry:
                    issues.append("Zero-distance stop")
                    passed = False
                
                if distance_pct < 0.1:  # Less than 0.1%
                    issues.append(f"Stop too tight: {distance_pct:.2f}%")
                    passed = False
                
                if distance_pct > 10:  # More than 10%
                    issues.append(f"Stop too wide: {distance_pct:.2f}%")
                    passed = False
                
                # Check direction makes sense
                if signal.signal.value == "BUY" and stop >= entry:
                    issues.append("BUY signal but stop is above entry")
                    passed = False
                
                if signal.signal.value == "SELL" and stop <= entry:
                    issues.append("SELL signal but stop is below entry")
                    passed = False
                
                # Report
                status = "✓" if passed else "❌"
                print(f"{status} {name:15} {signal.signal.value:4} | "
                      f"Entry: ${entry:8.2f} | Stop: ${stop:8.2f} | "
                      f"Distance: {distance_pct:5.2f}% (${distance_dollars:6.2f})")
                
                if issues:
                    for issue in issues:
                        print(f"    └─ {issue}")
                    all_passed = False
            else:
                print(f"⊘  {name:15} No signal (action={signal.signal.value})")
        
        except Exception as e:
            print(f"❌ {name:15} Exception: {str(e)[:60]}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready for paper trading")
    else:
        print("❌ ISSUES FOUND - Fix before paper trading")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    passed = test_stop_distances()
    sys.exit(0 if passed else 1)
