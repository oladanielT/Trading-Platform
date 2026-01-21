"""
Test script to demonstrate PHASE 5 regime transition diagnostics.

This script creates market conditions that simulate a volatile market
transitioning to a trending market, showing how the diagnostics track:
- Regime persistence (consecutive cycles in same regime)
- EMA slope stability (consecutive cycles with same slope direction)
- Trend strength acceleration (consecutive increases in trend strength)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.regime_strategy import RegimeBasedStrategy

# Set random seed for reproducibility
np.random.seed(42)

def create_transitioning_market_data(num_bars=200):
    """
    Create synthetic market data that transitions from volatile to trending.
    
    First 100 bars: Volatile, choppy movement
    Last 100 bars: Strong uptrend with increasing momentum
    """
    timestamps = []
    prices = []
    
    start_time = datetime.now() - timedelta(hours=num_bars)
    base_price = 100.0
    
    # Phase 1: Volatile, choppy market (bars 0-99)
    current_price = base_price
    for i in range(100):
        timestamps.append(start_time + timedelta(hours=i))
        # Random walk with high volatility
        change = np.random.normal(0, 2.0)  # High volatility, no trend
        current_price += change
        prices.append(current_price)
    
    # Phase 2: Transition to trending (bars 100-199)
    # Gradually increasing trend with decreasing noise
    for i in range(100, num_bars):
        timestamps.append(start_time + timedelta(hours=i))
        # Trend component increases over time
        trend_component = 0.3 * (1 + (i - 100) / 100.0)
        # Noise decreases over time
        noise_scale = 1.5 * (1 - (i - 100) / 150.0)
        noise = np.random.normal(0, noise_scale)
        
        current_price += trend_component + noise
        prices.append(current_price)
    
    # Create OHLCV data
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Simple OHLC generation around close price
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = close + np.random.normal(0, 0.5)
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def main():
    """Run regime transition diagnostic test."""
    print("=" * 80)
    print("PHASE 5: Regime Transition Diagnostics Test")
    print("=" * 80)
    print()
    print("This test demonstrates the new diagnostic tracking features:")
    print("1. Regime persistence (consecutive cycles in same regime)")
    print("2. EMA slope stability (consecutive cycles with same slope sign)")
    print("3. Trend strength acceleration (consecutive increases)")
    print()
    print("Market Simulation:")
    print("  - Bars 0-99: Volatile, choppy market")
    print("  - Bars 100-199: Transition to strong uptrend")
    print()
    print("=" * 80)
    print()
    
    # Create strategy with volatile override enabled
    strategy = RegimeBasedStrategy(
        allowed_regimes=['TRENDING'],
        allow_volatile_trend_override=True,
        volatile_trend_strength_min=3.0,
        volatile_confidence_scale=0.5
    )
    
    # Create market data
    full_data = create_transitioning_market_data(num_bars=200)
    
    # Analyze in windows to see regime transitions
    window_size = 100
    step_size = 10  # Analyze every 10 bars
    
    print("Analyzing market in sliding windows...")
    print()
    
    for i in range(0, len(full_data) - window_size + 1, step_size):
        window_data = full_data.iloc[i:i+window_size].copy()
        
        # Analyze this window
        result = strategy.analyze(window_data)
        
        print(f"\n--- Bar {i+window_size-1} (window {i}-{i+window_size-1}) ---")
        if result:
            print(f"Action: {result['action']}")
            print(f"Confidence: {result.get('confidence', 0):.3f}")
            print(f"Regime: {result['regime']}")
            print(f"Regime Confidence: {result.get('regime_confidence', 0):.3f}")
            
            # Access internal diagnostic state (normally only visible in logs)
            print(f"\nInternal Diagnostics:")
            print(f"  Regime persistence: {strategy._regime_persistence_count} cycles")
            print(f"  EMA slope stability: {strategy._slope_stability_count} cycles")
            print(f"  Trend accel count: {strategy._trend_accel_count} cycles")
            print(f"  Last trend delta: {strategy._last_trend_delta:+.3f}")
        else:
            print("Analysis returned None (insufficient data)")
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print("Interpretation:")
    print("- Look for increasing regime_persistence when market stabilizes")
    print("- ema_slope_stable >= 3-4 suggests direction is forming")
    print("- trend_accel_count >= 3 with positive delta suggests volatile→trend transition")
    print()
    print("These diagnostics help identify when a volatile market is BEGINNING to trend,")
    print("which could inform future trading decisions (not in this phase).")
    print()


if __name__ == '__main__':
    main()
