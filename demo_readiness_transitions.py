"""
Demo script for PHASE 6: Regime Readiness Gate

Demonstrates readiness state transitions:
- NOT_READY → FORMING → READY

Shows how readiness score is calculated from component metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.regime_strategy import RegimeBasedStrategy, ReadinessState

np.random.seed(42)


def create_market_scenario(num_bars=150, scenario='choppy'):
    """
    Create market data for different scenarios.
    
    Args:
        num_bars: Number of bars
        scenario: 'choppy', 'emerging', or 'trending'
    """
    timestamps = []
    prices = []
    
    start_time = datetime.now() - timedelta(hours=num_bars)
    base_price = 100.0
    current_price = base_price
    
    if scenario == 'choppy':
        # High volatility, no trend
        for i in range(num_bars):
            timestamps.append(start_time + timedelta(hours=i))
            change = np.random.normal(0, 2.0)
            current_price += change
            prices.append(current_price)
    
    elif scenario == 'emerging':
        # Moderate trend starting to form
        for i in range(num_bars):
            timestamps.append(start_time + timedelta(hours=i))
            trend = 0.5
            noise = np.random.normal(0, 1.2)
            current_price += trend + noise
            prices.append(current_price)
    
    elif scenario == 'trending':
        # Strong trend
        for i in range(num_bars):
            timestamps.append(start_time + timedelta(hours=i))
            trend = 1.0 * (1 + i / 200.0)  # Accelerating trend
            noise = np.random.normal(0, 0.6)
            current_price += trend + noise
            prices.append(current_price)
    
    # Create OHLCV
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
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
    print("=" * 80)
    print("PHASE 6: Regime Readiness Gate - Demonstration")
    print("=" * 80)
    print()
    print("This demo shows readiness state transitions as market conditions change:")
    print("  1. NOT_READY - Choppy market, no clear direction")
    print("  2. FORMING - Trend beginning to emerge")
    print("  3. READY - Strong trend established")
    print()
    print("=" * 80)
    print()
    
    # Create strategy
    strategy = RegimeBasedStrategy(
        allowed_regimes=['TRENDING'],
        allow_volatile_trend_override=True,
        volatile_trend_strength_min=3.0
    )
    
    scenarios = [
        ('choppy', 'NOT_READY expected'),
        ('emerging', 'FORMING expected'),
        ('trending', 'READY expected')
    ]
    
    for scenario_name, expectation in scenarios:
        print(f"\n{'─' * 80}")
        print(f"SCENARIO: {scenario_name.upper()} ({expectation})")
        print(f"{'─' * 80}\n")
        
        # Create market data for this scenario
        data = create_market_scenario(num_bars=150, scenario=scenario_name)
        
        # Analyze
        result = strategy.analyze(data)
        
        if result:
            print(f"Market Analysis Results:")
            print(f"  Signal: {result['action'].upper()}")
            print(f"  Regime: {result['regime']}")
            print(f"  Regime Confidence: {result.get('regime_confidence', 0):.3f}")
            print()
            
            # Display readiness information
            print(f"Readiness Assessment:")
            print(f"  State: {strategy._readiness_state.value.upper()}")
            print(f"  Score: {strategy._readiness_score:.3f}")
            print()
            
            # Show component breakdown
            print(f"Component Metrics:")
            print(f"  Regime Persistence: {strategy._regime_persistence_count} cycles")
            print(f"  EMA Slope Stability: {strategy._slope_stability_count} cycles")
            print(f"  Trend Acceleration: {strategy._trend_accel_count} cycles")
            print(f"  Trend Strength: {strategy._last_trend_strength:.3f}")
            print()
            
            # Calculate individual component scores
            persistence_score = min(strategy._regime_persistence_count / 8.0, 1.0)
            stability_score = min(strategy._slope_stability_count / 5.0, 1.0)
            accel_score = min(strategy._trend_accel_count / 4.0, 1.0)
            strength_score = min(strategy._last_trend_strength / 3.0, 1.0) if strategy._last_trend_strength > 0 else 0.0
            
            print(f"Component Scores (each 0.0-1.0):")
            print(f"  Persistence: {persistence_score:.3f} (weight: 0.25)")
            print(f"  Stability:   {stability_score:.3f} (weight: 0.25)")
            print(f"  Acceleration: {accel_score:.3f} (weight: 0.25)")
            print(f"  Strength:    {strength_score:.3f} (weight: 0.25)")
            print(f"  {'─' * 40}")
            print(f"  Total Score: {strategy._readiness_score:.3f}")
            print()
            
            # Interpretation
            state = strategy._readiness_state
            if state == ReadinessState.NOT_READY:
                print("🔴 NOT_READY: Market conditions not favorable for trend trading")
                print("   - Consider waiting for more stability")
                print("   - Watch for regime persistence to increase")
            elif state == ReadinessState.FORMING:
                print("🟡 FORMING: Trend showing early signs of developing")
                print("   - Direction emerging but not fully confirmed")
                print("   - Monitor for progression to READY state")
            elif state == ReadinessState.READY:
                print("🟢 READY: Strong indicators that trend is established")
                print("   - All metrics aligned favorably")
                print("   - Conditions conducive for trend following")
        else:
            print("Analysis returned None (insufficient data)")
        
        print()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  • Readiness score combines 4 components with equal weights (0.25 each)")
    print("  • State transitions: NOT_READY → FORMING → READY")
    print("  • Score ranges: NOT_READY (0-0.33), FORMING (0.34-0.66), READY (0.67-1.0)")
    print("  • This is OBSERVATION ONLY - does not affect trading signals")
    print("  • Provides enhanced visibility into market conditions")
    print()


if __name__ == '__main__':
    main()
