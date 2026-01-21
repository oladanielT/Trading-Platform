"""
Example: Using RegimeBasedStrategy

Demonstrates how to use the regime-based meta-strategy to filter
trading signals based on market conditions.

This example shows:
1. Creating a RegimeDetector
2. Creating a RegimeBasedStrategy
3. Analyzing regime in different market conditions
4. Getting regime-filtered signals
"""

import sys
from pathlib import Path

# Add strategies directory to path
sys.path.insert(0, str(Path(__file__).parent / 'strategies'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.regime_detector import RegimeDetector, MarketRegime
from strategies.regime_strategy import RegimeBasedStrategy
from strategies.ema_trend import EMATrendStrategy
from strategies.base import PositionState


def generate_sample_data(n_candles: int = 200, regime_type: str = 'trending') -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing.
    
    Args:
        n_candles: Number of candles to generate
        regime_type: Type of market to simulate ('trending', 'ranging', 'volatile')
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    timestamps = [datetime.now() - timedelta(hours=n_candles-i) for i in range(n_candles)]
    
    if regime_type == 'trending':
        # Upward trend with low noise
        trend = np.linspace(100, 150, n_candles)
        noise = np.random.randn(n_candles) * 2
        close = trend + noise
        
    elif regime_type == 'ranging':
        # Oscillating price
        close = 100 + 10 * np.sin(np.linspace(0, 8*np.pi, n_candles)) + np.random.randn(n_candles) * 1
        
    elif regime_type == 'volatile':
        # High volatility random walk
        returns = np.random.randn(n_candles) * 0.05  # 5% volatility
        close = 100 * np.exp(np.cumsum(returns))
        
    else:  # quiet
        # Low volatility
        close = 100 + np.random.randn(n_candles) * 0.5
    
    # Generate OHLC from close
    high = close + np.abs(np.random.randn(n_candles) * 1)
    low = close - np.abs(np.random.randn(n_candles) * 1)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    
    volume = np.random.randint(1000, 10000, n_candles)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


def example_1_regime_detection():
    """Example 1: Basic regime detection."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Regime Detection")
    print("=" * 80)
    
    detector = RegimeDetector(lookback_period=100)
    
    # Test different market conditions
    for regime_type in ['trending', 'ranging', 'volatile', 'quiet']:
        print(f"\n{regime_type.upper()} Market:")
        print("-" * 40)
        
        data = generate_sample_data(n_candles=200, regime_type=regime_type)
        result = detector.detect(data)
        
        print(f"Detected Regime: {result['regime'].value}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Explanation: {result['explanation']}")
        print(f"\nMetrics:")
        for key, value in result['metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def example_2_regime_strategy():
    """Example 2: Using RegimeBasedStrategy."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Regime-Based Strategy Filtering")
    print("=" * 80)
    
    # Create underlying strategy (EMA trend)
    ema_strategy = EMATrendStrategy(
        fast_period=20,
        slow_period=50,
        signal_confidence=0.7
    )
    
    # Create regime-based meta-strategy
    regime_strategy = RegimeBasedStrategy(
        underlying_strategy=ema_strategy,
        allowed_regimes=[MarketRegime.TRENDING],
        regime_confidence_threshold=0.5
    )
    
    # Test on different market conditions
    position_state = PositionState()
    
    for regime_type in ['trending', 'ranging', 'volatile']:
        print(f"\n{regime_type.upper()} Market:")
        print("-" * 40)
        
        data = generate_sample_data(n_candles=200, regime_type=regime_type)
        
        # Get signal from regime-based strategy
        signal = regime_strategy.generate_signal(data, position_state)
        
        print(f"Signal: {signal.signal.value}")
        print(f"Confidence: {signal.confidence:.2f}")
        print(f"Regime: {signal.metadata.get('regime')}")
        print(f"Decision: {signal.metadata.get('decision')}")
        print(f"Reason: {signal.metadata.get('decision_reason')}")


def example_3_configuration():
    """Example 3: Strategy configuration and customization."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Strategy Configuration")
    print("=" * 80)
    
    # Create custom regime detector
    detector = RegimeDetector(
        lookback_period=150,
        volatility_threshold_high=0.04,
        volatility_threshold_low=0.008,
        trend_threshold_strong=2.5,
        trend_threshold_weak=0.3
    )
    
    # Create strategy allowing both trending and quiet regimes
    strategy = RegimeBasedStrategy(
        regime_detector=detector,
        allowed_regimes=[MarketRegime.TRENDING, MarketRegime.QUIET],
        reduce_confidence_in_marginal_regimes=True
    )
    
    # Print configuration
    config = strategy.get_configuration()
    print("\nStrategy Configuration:")
    print("-" * 40)
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"{key}: {', '.join(str(x) for x in value)}")
        else:
            print(f"{key}: {value}")


def example_4_regime_status():
    """Example 4: Monitoring regime status without trading."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Regime Status Monitoring")
    print("=" * 80)
    
    strategy = RegimeBasedStrategy(
        allowed_regimes=[MarketRegime.TRENDING]
    )
    
    # Generate trending data
    data = generate_sample_data(n_candles=200, regime_type='trending')
    
    # Get regime status (no signal generation)
    status = strategy.get_regime_status(data)
    
    print("\nCurrent Regime Status:")
    print("-" * 40)
    print(f"Regime: {status['regime']}")
    print(f"Confidence: {status['confidence']:.2f}")
    print(f"Is Allowed: {status['is_allowed']}")
    print(f"Allowed Regimes: {', '.join(status['allowed_regimes'])}")
    print(f"\nMetrics:")
    for key, value in status['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")


def example_5_dynamic_regime_changes():
    """Example 5: Dynamically changing allowed regimes."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Dynamic Regime Configuration")
    print("=" * 80)
    
    strategy = RegimeBasedStrategy(
        allowed_regimes=[MarketRegime.TRENDING]
    )
    
    data = generate_sample_data(n_candles=200, regime_type='ranging')
    position_state = PositionState()
    
    # Test with original settings (only TRENDING allowed)
    print("\nWith TRENDING only:")
    signal1 = strategy.generate_signal(data, position_state)
    print(f"Signal: {signal1.signal.value} (confidence: {signal1.confidence:.2f})")
    
    # Change to allow ranging markets
    strategy.set_allowed_regimes([MarketRegime.TRENDING, MarketRegime.RANGING])
    
    print("\nWith TRENDING and RANGING allowed:")
    signal2 = strategy.generate_signal(data, position_state)
    print(f"Signal: {signal2.signal.value} (confidence: {signal2.confidence:.2f})")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("REGIME-BASED STRATEGY EXAMPLES")
    print("=" * 80)
    
    try:
        example_1_regime_detection()
        example_2_regime_strategy()
        example_3_configuration()
        example_4_regime_status()
        example_5_dynamic_regime_changes()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
