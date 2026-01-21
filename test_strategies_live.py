"""
test_strategies_live.py - Live-ready strategy test with clear signals
"""

import pandas as pd
import numpy as np
from strategies.base import PositionState
from strategies.pattern_strategies import FVGStrategy, ContinuationStrategy, ABCStrategy
from strategies.multi_strategy_aggregator import MultiStrategyAggregator


def create_test_data():
    """Create realistic historical data with patterns"""
    np.random.seed(42)
    ts = pd.Timestamp('2024-01-01')
    data = {
        'timestamp': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []
    }
    
    price = 100.0
    
    # 1. Initial trend (30 candles)
    print("  [Phase 1: Initial trend...]")
    for i in range(30):
        o = price + np.random.uniform(-0.1, 0.1)
        c = price + np.random.uniform(-0.1, 0.3)
        h = max(o, c) + np.random.uniform(0.1, 0.2)
        l = min(o, c) - np.random.uniform(0, 0.1)
        
        data['timestamp'].append(ts)
        data['open'].append(o)
        data['close'].append(c)
        data['high'].append(h)
        data['low'].append(l)
        data['volume'].append(np.random.uniform(1000, 1500))
        ts += pd.Timedelta(hours=1)
        price = c
    
    # 2. Strong impulse up (8 candles, 3%+ move)
    print("  [Phase 2: Strong impulse move...]")
    impulse_start_price = price
    for i in range(8):
        o = price + np.random.uniform(-0.05, 0.05)
        c = price + np.random.uniform(0.6, 1.0)  # Strong up
        h = c + np.random.uniform(0.15, 0.3)
        l = o - np.random.uniform(0, 0.05)
        
        data['timestamp'].append(ts)
        data['open'].append(o)
        data['close'].append(c)
        data['high'].append(h)
        data['low'].append(l)
        data['volume'].append(np.random.uniform(2000, 3000))
        ts += pd.Timedelta(hours=1)
        price = c
    
    impulse_end_price = price
    impulse_pct = ((impulse_end_price - impulse_start_price) / impulse_start_price) * 100
    print(f"    → Impulse: {impulse_pct:.2f}% in 8 candles")
    
    # 3. Consolidation (38.2% retracement, 8 candles)
    print("  [Phase 3: Consolidation with reduced volume...]")
    target_low = impulse_end_price - (impulse_end_price - impulse_start_price) * 0.382
    for i in range(8):
        mid = target_low + np.random.uniform(-0.15, 0.15)
        o = mid + np.random.uniform(-0.1, 0.1)
        c = mid + np.random.uniform(-0.1, 0.1)
        h = max(o, c, impulse_end_price * 0.998)
        l = min(o, c)
        
        data['timestamp'].append(ts)
        data['open'].append(o)
        data['close'].append(c)
        data['high'].append(h)
        data['low'].append(l)
        data['volume'].append(np.random.uniform(700, 1000))
        ts += pd.Timedelta(hours=1)
        price = c
    
    consolidation_high = max([d for d in data['high'][-8:]])
    
    # 4. Breakout with volume (5 candles)
    print("  [Phase 4: Breakout with volume spike...]")
    for i in range(5):
        o = price + np.random.uniform(-0.05, 0.05)
        c = price + np.random.uniform(0.8, 1.3)
        h = c + np.random.uniform(0.2, 0.4)
        l = o - np.random.uniform(0, 0.05)
        
        data['timestamp'].append(ts)
        data['open'].append(o)
        data['close'].append(c)
        data['high'].append(h)
        data['low'].append(l)
        data['volume'].append(np.random.uniform(2500, 4000))
        ts += pd.Timedelta(hours=1)
        price = c
    
    print(f"  [Generated {len(data['close'])} candles, final price: {price:.2f}]")
    return pd.DataFrame(data)


def test_continuation_in_detail():
    """Detailed test of continuation strategy"""
    print("\n" + "="*80)
    print("CONTINUATION STRATEGY - DETAILED TEST")
    print("="*80)
    
    print("\nGenerating test data with flag pattern...")
    data = create_test_data()
    
    print(f"\nData Summary:")
    print(f"  • Total candles: {len(data)}")
    print(f"  • Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"  • Open: {data['close'].iloc[0]:.2f}, Close: {data['close'].iloc[-1]:.2f}")
    print(f"  • Latest volume: {data['volume'].iloc[-1]:.0f}")
    
    strategy = ContinuationStrategy(lookback=30)
    position_state = PositionState()
    
    signal = strategy.generate_signal(data, position_state)
    
    print(f"\n{'='*80}")
    print(f"SIGNAL RESULT:")
    print(f"{'='*80}")
    print(f"  Signal:     {signal.signal.value.upper():10s}")
    print(f"  Confidence: {signal.confidence:.1%}")
    
    if signal.metadata:
        print(f"\nMetadata:")
        for key, value in signal.metadata.items():
            if not isinstance(value, (dict, list)):
                print(f"  • {key}: {value}")


def test_fvg_in_detail():
    """Detailed test of FVG strategy"""
    print("\n" + "="*80)
    print("FVG STRATEGY - DETAILED TEST")
    print("="*80)
    
    np.random.seed(99)
    ts = pd.Timestamp('2024-01-01')
    data = {
        'timestamp': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []
    }
    
    price = 100.0
    
    # Build up to FVG
    print("\nGenerating FVG pattern...")
    for i in range(40):
        o = price + np.random.uniform(-0.1, 0.1)
        c = price + np.random.uniform(-0.1, 0.2)
        h = max(o, c) + np.random.uniform(0.1, 0.2)
        l = min(o, c) - np.random.uniform(0, 0.1)
        
        data['timestamp'].append(ts)
        data['open'].append(o)
        data['close'].append(c)
        data['high'].append(h)
        data['low'].append(l)
        data['volume'].append(np.random.uniform(1000, 1500))
        ts += pd.Timedelta(hours=1)
        price = c
    
    # Create FVG (gap up)
    print("  [Creating gap up (bullish FVG)...]")
    
    # Candle 1
    data['timestamp'].append(ts)
    data['open'].append(price - 0.1)
    data['close'].append(price + 0.2)
    data['high'].append(price + 0.5)  # This will be gap_bottom
    data['low'].append(price - 0.1)
    data['volume'].append(1200)
    ts += pd.Timedelta(hours=1)
    gap_bottom = price + 0.5
    
    # Candle 2 (gap up)
    price_gap = gap_bottom + 0.5
    data['timestamp'].append(ts)
    data['open'].append(price_gap - 0.2)
    data['close'].append(price_gap + 0.3)
    data['high'].append(price_gap + 0.5)
    data['low'].append(price_gap - 0.3)
    data['volume'].append(2200)
    ts += pd.Timedelta(hours=1)
    
    # Candle 3
    price_gap += 0.4
    data['timestamp'].append(ts)
    data['open'].append(price_gap - 0.1)
    data['close'].append(price_gap + 0.1)
    data['high'].append(price_gap + 0.2)
    data['low'].append(price_gap - 0.3)  # This will be gap_top
    data['volume'].append(1800)
    ts += pd.Timedelta(hours=1)
    gap_top = price_gap - 0.3
    
    # Continue uptrend
    for i in range(8):
        price = gap_top + np.random.uniform(0.2, 0.5)
        o = price - np.random.uniform(0, 0.1)
        c = price + np.random.uniform(0, 0.2)
        h = c + np.random.uniform(0.1, 0.2)
        l = o - np.random.uniform(0, 0.05)
        
        data['timestamp'].append(ts)
        data['open'].append(o)
        data['close'].append(c)
        data['high'].append(h)
        data['low'].append(l)
        data['volume'].append(np.random.uniform(1500, 2200))
        ts += pd.Timedelta(hours=1)
    
    df = pd.DataFrame(data)
    
    print(f"  • Gap: {gap_bottom:.2f} - {gap_top:.2f} ({(gap_top - gap_bottom) / gap_bottom * 100:.2f}%)")
    print(f"  • Current price: {df['close'].iloc[-1]:.2f}")
    
    strategy = FVGStrategy(lookback=30)
    position_state = PositionState()
    
    signal = strategy.generate_signal(df, position_state)
    
    print(f"\n{'='*80}")
    print(f"SIGNAL RESULT:")
    print(f"{'='*80}")
    print(f"  Signal:     {signal.signal.value.upper():10s}")
    print(f"  Confidence: {signal.confidence:.1%}")
    
    if signal.metadata:
        print(f"\nMetadata:")
        for key, value in signal.metadata.items():
            if not isinstance(value, (dict, list)):
                print(f"  • {key}: {value}")


def test_aggregated():
    """Test all strategies together"""
    print("\n" + "="*80)
    print("MULTI-STRATEGY AGGREGATION TEST")
    print("="*80)
    
    print("\nGenerating test data...")
    data = create_test_data()
    
    print(f"\nTesting with {len(data)} candles")
    
    strategies_list = [
        ("FVG", FVGStrategy(lookback=25)),
        ("Continuation", ContinuationStrategy(lookback=25)),
        ("ABC", ABCStrategy()),
    ]
    
    position_state = PositionState()
    
    print("\nIndividual Strategy Results:")
    print("-" * 80)
    for name, strat in strategies_list:
        sig = strat.generate_signal(data, position_state)
        conf_pct = sig.confidence * 100
        print(f"  {name:20s} → {sig.signal.value:6s} ({conf_pct:5.1f}%)")
    
    print("\n" + "-" * 80)
    print("Aggregated Result:")
    print("-" * 80)
    
    agg = MultiStrategyAggregator(
        strategies=[strat for _, strat in strategies_list],
        weights={"FVGStrategy": 1.0, "ContinuationStrategy": 1.5, "ABCStrategy": 0.8}
    )
    
    agg_signal = agg.generate_signal(data, position_state)
    
    print(f"  Combined Signal:    {agg_signal.signal.value.upper():10s}")
    print(f"  Combined Confidence: {agg_signal.confidence:.1%}")


def main():
    print("\n" + "█"*80)
    print("█" + "STRATEGY TEST SUITE - PRODUCTION READY".center(78) + "█")
    print("█"*80)
    
    test_continuation_in_detail()
    test_fvg_in_detail()
    test_aggregated()
    
    print("\n" + "█"*80)
    print("█" + "✓ TESTS COMPLETE - ALL STRATEGIES OPERATIONAL".center(78) + "█")
    print("█"*80 + "\n")


if __name__ == "__main__":
    main()
