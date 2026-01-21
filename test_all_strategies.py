"""
test_all_strategies.py - Comprehensive test of all proper strategies
Tests individual strategies, aggregation, and regime gating
"""

import pandas as pd
import numpy as np
from strategies.base import PositionState
from strategies.pattern_strategies import (
    FVGStrategy, ContinuationStrategy, ABCStrategy, 
    ReclamationBlockStrategy, ConfirmationLayer
)
from strategies.multi_strategy_aggregator import MultiStrategyAggregator
from strategies.regime_strategy import RegimeBasedStrategy
from strategies.ema_trend import EMATrendStrategy


def create_bullish_with_fvg():
    """Create data with bullish trend + FVG pattern"""
    np.random.seed(42)
    data_dict = {
        'timestamp': [],
        'open': [], 'high': [], 'low': [], 'close': [], 'volume': []
    }
    
    ts = pd.Timestamp('2024-01-01')
    price = 100.0
    
    # Initial data
    for i in range(30):
        o = price + np.random.uniform(-0.1, 0.1)
        c = price + np.random.uniform(-0.1, 0.2)
        h = max(o, c) + np.random.uniform(0.1, 0.2)
        l = min(o, c) - np.random.uniform(0, 0.1)
        
        data_dict['timestamp'].append(ts)
        data_dict['open'].append(o)
        data_dict['close'].append(c)
        data_dict['high'].append(h)
        data_dict['low'].append(l)
        data_dict['volume'].append(np.random.uniform(1000, 1500))
        
        ts += pd.Timedelta(hours=1)
        price = c
    
    # Create FVG (gap up)
    o = price + 0.3
    c = price + 0.8
    h = c + 0.2
    l = o - 0.1
    
    data_dict['timestamp'].append(ts)
    data_dict['open'].append(o)
    data_dict['close'].append(c)
    data_dict['high'].append(h)
    data_dict['low'].append(l)
    data_dict['volume'].append(1200)
    ts += pd.Timedelta(hours=1)
    
    # Gap up candle
    o = c - 0.1
    c = c + 0.3
    h = c + 0.1
    l = o - 0.2
    
    data_dict['timestamp'].append(ts)
    data_dict['open'].append(o)
    data_dict['close'].append(c)
    data_dict['high'].append(h)
    data_dict['low'].append(l)
    data_dict['volume'].append(2000)
    ts += pd.Timedelta(hours=1)
    price = c
    
    # More uptrend
    for i in range(15):
        o = price + np.random.uniform(-0.1, 0.1)
        c = price + np.random.uniform(0.3, 0.7)
        h = c + np.random.uniform(0.1, 0.3)
        l = o - np.random.uniform(0, 0.1)
        
        data_dict['timestamp'].append(ts)
        data_dict['open'].append(o)
        data_dict['close'].append(c)
        data_dict['high'].append(h)
        data_dict['low'].append(l)
        data_dict['volume'].append(np.random.uniform(1500, 2500))
        
        ts += pd.Timedelta(hours=1)
        price = c
    
    return pd.DataFrame(data_dict)


def create_flag_pattern():
    """Create data with flag/pennant pattern"""
    np.random.seed(123)
    data_dict = {
        'timestamp': [],
        'open': [], 'high': [], 'low': [], 'close': [], 'volume': []
    }
    
    ts = pd.Timestamp('2024-01-01')
    price = 100.0
    
    # Initial data
    for i in range(25):
        o = price + np.random.uniform(-0.1, 0.1)
        c = price + np.random.uniform(-0.1, 0.15)
        h = max(o, c) + np.random.uniform(0.1, 0.2)
        l = min(o, c) - np.random.uniform(0, 0.1)
        
        data_dict['timestamp'].append(ts)
        data_dict['open'].append(o)
        data_dict['close'].append(c)
        data_dict['high'].append(h)
        data_dict['low'].append(l)
        data_dict['volume'].append(np.random.uniform(1000, 1500))
        
        ts += pd.Timedelta(hours=1)
        price = c
    
    # Strong impulse (10+ candles)
    impulse_start = price
    for i in range(12):
        o = price + np.random.uniform(-0.1, 0.05)
        c = price + np.random.uniform(0.6, 1.0)  # Strong up
        h = c + np.random.uniform(0.1, 0.3)
        l = o - np.random.uniform(0, 0.1)
        
        data_dict['timestamp'].append(ts)
        data_dict['open'].append(o)
        data_dict['close'].append(c)
        data_dict['high'].append(h)
        data_dict['low'].append(l)
        data_dict['volume'].append(np.random.uniform(2000, 3000))
        
        ts += pd.Timedelta(hours=1)
        price = c
    
    impulse_top = price
    
    # Consolidation (38.2% retracement)
    consolidation_target = impulse_top - (impulse_top - impulse_start) * 0.382
    for i in range(10):
        mid = consolidation_target
        o = mid + np.random.uniform(-0.2, 0.2)
        c = mid + np.random.uniform(-0.2, 0.2)
        h = max(o, c, impulse_top * 0.995)
        l = min(o, c)
        
        data_dict['timestamp'].append(ts)
        data_dict['open'].append(o)
        data_dict['close'].append(c)
        data_dict['high'].append(h)
        data_dict['low'].append(l)
        data_dict['volume'].append(np.random.uniform(600, 900))  # Low volume
        
        ts += pd.Timedelta(hours=1)
        price = c
    
    consolidation_high = max(data_dict['high'][-10:])
    
    # Breakout
    for i in range(6):
        o = price + np.random.uniform(-0.1, 0.1)
        c = price + np.random.uniform(0.7, 1.2)  # Strong breakout
        h = c + np.random.uniform(0.2, 0.4)
        l = o - np.random.uniform(0, 0.05)
        
        data_dict['timestamp'].append(ts)
        data_dict['open'].append(o)
        data_dict['close'].append(c)
        data_dict['high'].append(h)
        data_dict['low'].append(l)
        data_dict['volume'].append(np.random.uniform(2500, 4000))  # High volume
        
        ts += pd.Timedelta(hours=1)
        price = c
    
    return pd.DataFrame(data_dict)


def test_individual_strategies():
    """Test each strategy independently"""
    print("\n" + "="*80)
    print("TEST 1: INDIVIDUAL STRATEGY TESTS")
    print("="*80)
    
    position_state = PositionState()
    
    # Test with FVG data
    print("\n[Dataset A: Bullish Trend with FVG]")
    fvg_data = create_bullish_with_fvg()
    print(f"Data: {len(fvg_data)} candles | Price: {fvg_data['close'].iloc[0]:.2f} → {fvg_data['close'].iloc[-1]:.2f}")
    
    strategies = [
        ("FVGStrategy", FVGStrategy(lookback=30)),
        ("ContinuationStrategy", ContinuationStrategy(lookback=30)),
        ("ABCStrategy", ABCStrategy()),
        ("ReclamationBlockStrategy", ReclamationBlockStrategy(lookback=20)),
        ("EMATrendStrategy", EMATrendStrategy(fast_period=21, slow_period=89))
    ]
    
    for name, strat in strategies:
        try:
            sig = strat.generate_signal(fvg_data, position_state)
            conf_pct = sig.confidence * 100
            print(f"  {name:25s} → {sig.signal.value:6s} ({conf_pct:5.1f}%)")
        except Exception as e:
            print(f"  {name:25s} → ERROR: {str(e)[:40]}")
    
    # Test with flag data
    print("\n[Dataset B: Flag Pattern]")
    flag_data = create_flag_pattern()
    print(f"Data: {len(flag_data)} candles | Price: {flag_data['close'].iloc[0]:.2f} → {flag_data['close'].iloc[-1]:.2f}")
    
    for name, strat in strategies:
        try:
            sig = strat.generate_signal(flag_data, position_state)
            conf_pct = sig.confidence * 100
            print(f"  {name:25s} → {sig.signal.value:6s} ({conf_pct:5.1f}%)")
        except Exception as e:
            print(f"  {name:25s} → ERROR: {str(e)[:40]}")


def test_multi_strategy_aggregator():
    """Test strategies combined via aggregator"""
    print("\n" + "="*80)
    print("TEST 2: MULTI-STRATEGY AGGREGATOR")
    print("="*80)
    
    position_state = PositionState()
    flag_data = create_flag_pattern()
    
    print(f"\nData: {len(flag_data)} candles | Price: {flag_data['close'].iloc[0]:.2f} → {flag_data['close'].iloc[-1]:.2f}")
    
    # Create aggregator with multiple strategies
    strategies = [
        FVGStrategy(lookback=30),
        ContinuationStrategy(lookback=30),
        EMATrendStrategy(fast_period=21, slow_period=89),
        ABCStrategy(),
        ReclamationBlockStrategy(lookback=20)
    ]
    
    agg = MultiStrategyAggregator(
        strategies=strategies,
        weights={
            "FVGStrategy": 1.2,
            "ContinuationStrategy": 1.5,
            "EMATrendStrategy": 1.0,
            "ABCStrategy": 0.8,
            "ReclamationBlockStrategy": 0.9
        },
        enable_adaptive_weights=False
    )
    
    sig = agg.generate_signal(flag_data, position_state)
    
    print(f"\n✓ Aggregated Signal: {sig.signal.value.upper()}")
    print(f"✓ Combined Confidence: {sig.confidence:.3f} ({sig.confidence*100:.1f}%)")
    print(f"\nMetadata:")
    for key, value in sig.metadata.items():
        if not isinstance(value, (list, dict)):
            print(f"  • {key}: {value}")


def test_regime_gating():
    """Test strategies with regime gate"""
    print("\n" + "="*80)
    print("TEST 3: REGIME-GATED STRATEGY")
    print("="*80)
    
    position_state = PositionState()
    flag_data = create_flag_pattern()
    
    print(f"\nData: {len(flag_data)} candles | Price: {flag_data['close'].iloc[0]:.2f} → {flag_data['close'].iloc[-1]:.2f}")
    
    # Create aggregator
    base_strategies = [
        FVGStrategy(lookback=30),
        ContinuationStrategy(lookback=30),
        EMATrendStrategy(fast_period=21, slow_period=89)
    ]
    
    agg = MultiStrategyAggregator(strategies=base_strategies)
    
    # Wrap with regime gate
    regime_strat = RegimeBasedStrategy(
        underlying_strategy=agg,
        allow_volatile_trend_override=True,
        volatile_trend_strength_min=2.0,
        trading_mode='live'
    )
    
    sig = regime_strat.generate_signal(flag_data, position_state)
    
    print(f"\n✓ Regime-Gated Signal: {sig.signal.value.upper()}")
    print(f"✓ Confidence: {sig.confidence:.3f} ({sig.confidence*100:.1f}%)")
    print(f"\nMetadata:")
    for key, value in sig.metadata.items():
        if not isinstance(value, (list, dict)):
            print(f"  • {key}: {value}")


def test_confirmation_layer():
    """Test signal validation with confirmation layer"""
    print("\n" + "="*80)
    print("TEST 4: CONFIRMATION LAYER WRAPPER")
    print("="*80)
    
    position_state = PositionState()
    flag_data = create_flag_pattern()
    
    print(f"\nData: {len(flag_data)} candles | Price: {flag_data['close'].iloc[0]:.2f} → {flag_data['close'].iloc[-1]:.2f}")
    
    # Create base strategy
    base = EMATrendStrategy(fast_period=21, slow_period=89)
    
    # Wrap with confirmation
    confirmed = ConfirmationLayer(
        underlying=base,
        min_confidence=0.3,
        ema_span=9,
        boost_factor=1.1
    )
    
    sig_base = base.generate_signal(flag_data, position_state)
    sig_confirmed = confirmed.generate_signal(flag_data, position_state)
    
    print(f"\n  Base Signal:      {sig_base.signal.value:6s} (conf: {sig_base.confidence:.3f})")
    print(f"  Confirmed Signal: {sig_confirmed.signal.value:6s} (conf: {sig_confirmed.confidence:.3f})")
    
    if sig_base.confidence != sig_confirmed.confidence:
        change = ((sig_confirmed.confidence - sig_base.confidence) / max(sig_base.confidence, 0.01)) * 100
        direction = "↑" if change > 0 else "↓"
        print(f"  Adjustment: {direction} {abs(change):.1f}%")


def main():
    """Run all tests"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  COMPREHENSIVE STRATEGY TEST SUITE".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    try:
        test_individual_strategies()
    except Exception as e:
        print(f"\n❌ Error in individual tests: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_multi_strategy_aggregator()
    except Exception as e:
        print(f"\n❌ Error in aggregator test: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_regime_gating()
    except Exception as e:
        print(f"\n❌ Error in regime test: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_confirmation_layer()
    except Exception as e:
        print(f"\n❌ Error in confirmation test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  ✓ ALL TESTS COMPLETE".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")


if __name__ == "__main__":
    main()
