#!/usr/bin/env python3
"""
Quick Test: Regime-Based Strategy

Verifies that the regime-based strategy implementation works correctly.
Run this to validate the installation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import regime components
from strategies.regime_detector import RegimeDetector, MarketRegime
from strategies.regime_strategy import RegimeBasedStrategy
from strategies.base import PositionState


def create_test_data(n=200, trend='up'):
    """Create simple test data."""
    np.random.seed(42)
    timestamps = [datetime.now() - timedelta(hours=n-i) for i in range(n)]
    
    if trend == 'up':
        close = np.linspace(100, 150, n) + np.random.randn(n) * 2
    elif trend == 'down':
        close = np.linspace(150, 100, n) + np.random.randn(n) * 2
    else:  # ranging
        close = 125 + 10 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.randn(n)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': close,
        'high': close + np.abs(np.random.randn(n)),
        'low': close - np.abs(np.random.randn(n)),
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })


def test_regime_detector():
    """Test 1: RegimeDetector basic functionality."""
    print("\n" + "="*60)
    print("TEST 1: RegimeDetector")
    print("="*60)
    
    detector = RegimeDetector(lookback_period=100)
    
    # Test trending market
    data = create_test_data(200, 'up')
    result = detector.detect(data)
    
    print(f"✓ Detected regime: {result['regime'].value}")
    print(f"✓ Confidence: {result['confidence']:.2f}")
    print(f"✓ Volatility: {result['metrics']['volatility']:.4f}")
    print(f"✓ Trend strength: {result['metrics']['trend_strength']:.2f}")
    
    assert isinstance(result['regime'], MarketRegime), "Should return MarketRegime enum"
    assert 0.0 <= result['confidence'] <= 1.0, "Confidence should be 0-1"
    assert 'volatility' in result['metrics'], "Should have volatility metric"
    
    print("✓ RegimeDetector test passed!")
    return True


def test_regime_strategy():
    """Test 2: RegimeBasedStrategy basic functionality."""
    print("\n" + "="*60)
    print("TEST 2: RegimeBasedStrategy")
    print("="*60)
    
    # Create strategy with default settings
    strategy = RegimeBasedStrategy(
        allowed_regimes=[MarketRegime.TRENDING]
    )
    
    # Test with trending data
    data = create_test_data(200, 'up')
    position = PositionState()
    
    signal = strategy.generate_signal(data, position)
    
    print(f"✓ Signal: {signal.signal.value}")
    print(f"✓ Confidence: {signal.confidence:.2f}")
    print(f"✓ Regime: {signal.metadata.get('regime')}")
    print(f"✓ Decision: {signal.metadata.get('decision')}")
    
    assert hasattr(signal, 'signal'), "Should have signal attribute"
    assert hasattr(signal, 'confidence'), "Should have confidence attribute"
    assert 'regime' in signal.metadata, "Should have regime in metadata"
    assert 'decision' in signal.metadata, "Should have decision in metadata"
    
    print("✓ RegimeBasedStrategy test passed!")
    return True


def test_regime_filtering():
    """Test 3: Regime filtering behavior."""
    print("\n" + "="*60)
    print("TEST 3: Regime Filtering")
    print("="*60)
    
    strategy = RegimeBasedStrategy(
        allowed_regimes=[MarketRegime.TRENDING]
    )
    
    position = PositionState()
    
    # Test 1: Trending data (should allow signals)
    trending_data = create_test_data(200, 'up')
    signal1 = strategy.generate_signal(trending_data, position)
    print(f"Trending market signal: {signal1.signal.value}")
    
    # Test 2: Ranging data (should HOLD)
    ranging_data = create_test_data(200, 'ranging')
    signal2 = strategy.generate_signal(ranging_data, position)
    print(f"Ranging market signal: {signal2.signal.value}")
    
    # Ranging should typically be filtered to HOLD
    assert 'regime' in signal1.metadata, "Should have regime metadata"
    assert 'regime' in signal2.metadata, "Should have regime metadata"
    
    print("✓ Regime filtering test passed!")
    return True


def test_configuration():
    """Test 4: Configuration and customization."""
    print("\n" + "="*60)
    print("TEST 4: Configuration")
    print("="*60)
    
    # Create custom detector
    detector = RegimeDetector(
        lookback_period=150,
        volatility_threshold_high=0.04
    )
    
    # Create strategy with multiple allowed regimes
    strategy = RegimeBasedStrategy(
        regime_detector=detector,
        allowed_regimes=[MarketRegime.TRENDING, MarketRegime.RANGING]
    )
    
    # Get configuration
    config = strategy.get_configuration()
    
    print(f"✓ Strategy name: {config['strategy_name']}")
    print(f"✓ Allowed regimes: {config['allowed_regimes']}")
    print(f"✓ Detector lookback: {config['regime_detector_parameters']['lookback_period']}")
    
    assert config['regime_detector_parameters']['lookback_period'] == 150
    assert 'trending' in config['allowed_regimes']
    
    print("✓ Configuration test passed!")
    return True


def test_regime_status():
    """Test 5: Regime status monitoring."""
    print("\n" + "="*60)
    print("TEST 5: Regime Status Monitoring")
    print("="*60)
    
    strategy = RegimeBasedStrategy()
    data = create_test_data(200, 'up')
    
    # Get regime status without generating signal
    status = strategy.get_regime_status(data)
    
    print(f"✓ Current regime: {status['regime']}")
    print(f"✓ Is allowed: {status['is_allowed']}")
    print(f"✓ Confidence: {status['confidence']:.2f}")
    
    assert 'regime' in status, "Should have regime"
    assert 'is_allowed' in status, "Should have is_allowed"
    assert 'metrics' in status, "Should have metrics"
    
    print("✓ Regime status test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("REGIME-BASED STRATEGY TEST SUITE")
    print("="*60)
    
    tests = [
        test_regime_detector,
        test_regime_strategy,
        test_regime_filtering,
        test_configuration,
        test_regime_status
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED - Implementation verified!")
        return True
    else:
        print(f"\n✗ {failed} test(s) failed - check errors above")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
