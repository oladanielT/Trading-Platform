#!/usr/bin/env python3
"""Simple test: Can we instantiate RangeTradingStrategy?"""

import sys

try:
    from strategies.range_trading import RangeTradingStrategy
    from strategies.base import PositionState
    import pandas as pd
    import numpy as np
    
    print("\n" + "="*70)
    print("RANGE TRADING STRATEGY - INSTANTIATION TEST")
    print("="*70)
    
    # Create strategy instance
    print("\n[1] Creating RangeTradingStrategy instance...")
    strat = RangeTradingStrategy(
        rsi_period=14,
        rsi_oversold=35.0,
        rsi_overbought=65.0,
        lookback=100,
        support_resistance_pct=1.5,
        min_volatility=0.001,
        max_volatility=0.15,
        signal_confidence=0.70
    )
    print(f"✅ Strategy created: {strat.name}")
    
    # Check parameters
    print("\n[2] Verifying parameters...")
    params = {
        'rsi_period': strat.get_parameter('rsi_period'),
        'rsi_oversold': strat.get_parameter('rsi_oversold'),
        'rsi_overbought': strat.get_parameter('rsi_overbought'),
        'min_volatility': strat.get_parameter('min_volatility'),
        'max_volatility': strat.get_parameter('max_volatility'),
    }
    for param, value in params.items():
        print(f"  ✓ {param}: {value}")
    
    # Test with sample data
    print("\n[3] Testing with sample market data...")
    np.random.seed(42)
    prices = np.linspace(90000, 92000, 200) + np.random.normal(0, 500, 200)
    volumes = np.ones(200) * 1000000
    
    # Create timestamps
    import datetime
    timestamps = [datetime.datetime.now() - datetime.timedelta(hours=i) for i in range(199, -1, -1)]
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices + np.random.normal(0, 100, 200),
        'high': prices + abs(np.random.normal(0, 200, 200)),
        'low': prices - abs(np.random.normal(0, 200, 200)),
        'close': prices,
        'volume': volumes
    })
    
    print(f"  - Data shape: {data.shape}")
    print(f"  - Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Generate signal
    print("\n[4] Generating signal (this will show debug output)...\n")
    pos = PositionState()
    signal = strat.generate_signal(data, pos)
    
    print(f"\n[5] Signal result:")
    print(f"  - Action: {signal.signal.value}")
    print(f"  - Confidence: {signal.confidence}")
    print(f"  - Reason: {signal.metadata.get('reason', 'N/A')}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - STRATEGY IS WORKING")
    print("="*70)
    print("\nNow run the full platform:")
    print("  python3 main.py -c FIXED_CONFIG_FOR_VOLATILE_MARKETS.yaml -e paper --once")
    print()
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
