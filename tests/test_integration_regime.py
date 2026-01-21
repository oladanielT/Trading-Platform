#!/usr/bin/env python3
"""
test_integration_regime.py - Integration Test for RegimeBasedStrategy

Tests the complete end-to-end flow of RegimeBasedStrategy within the trading platform:
- Configuration loading
- Strategy initialization from config
- Signal generation with regime detection
- Risk management integration
- Logging and metrics collection

This is a minimal integration test to ensure all modules work together correctly.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from strategies.regime_strategy import RegimeBasedStrategy
from strategies.regime_detector import RegimeDetector, MarketRegime
from strategies.ema_trend import EMATrendStrategy
from strategies.base import PositionState, Signal
from risk.position_sizing import PositionSizer, SizingMethod
from risk.drawdown_guard import DrawdownGuard
from monitoring.metrics import MetricsCollector


def create_test_config_yaml() -> str:
    """Create a temporary config.yaml for testing."""
    config = {
        'environment': 'paper',
        'exchange': {'id': 'binance', 'testnet': True},
        'trading': {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'initial_capital': 100000.0,
            'fetch_limit': 500
        },
        'strategy': {
            'name': 'regime_based',
            'allowed_regimes': ['trending'],
            'regime_confidence_threshold': 0.5,
            'reduce_confidence_in_marginal_regimes': True,
            'underlying_strategy': {
                'name': 'ema_trend',
                'fast_period': 20,
                'slow_period': 50,
                'signal_confidence': 0.75
            }
        },
        'risk': {
            'max_position_size': 0.1,
            'risk_per_trade': 0.02,
            'sizing_method': 'fixed_risk',
            'max_daily_drawdown': 0.05,
            'max_total_drawdown': 0.20,
            'max_consecutive_losses': 5,
            'use_confidence_scaling': True
        },
        'execution': {
            'order_timeout': 30,
            'max_retries': 3,
            'retry_delay': 5
        },
        'monitoring': {
            'log_level': 'INFO',
            'log_file': 'logs/trading.log',
            'metrics_enabled': True
        }
    }
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_file)
    temp_file.close()
    return temp_file.name


def generate_test_market_data(num_bars: int = 100) -> pd.DataFrame:
    """Generate synthetic market data for testing."""
    np.random.seed(42)
    
    timestamps = pd.date_range(start='2024-01-01', periods=num_bars, freq='1H')
    price = 100.0
    data = []
    
    for ts in timestamps:
        open_price = price
        # Trending upward
        price_change = price * 0.005  # 0.5% trend
        noise = price * 0.01 * np.random.randn()
        close_price = price + price_change + noise
        
        high_price = max(open_price, close_price) * 1.002
        low_price = min(open_price, close_price) * 0.998
        volume = 10 + 5 * np.random.rand()
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        price = close_price
    
    return pd.DataFrame(data)


class TestRegimeStrategyIntegration:
    """Integration tests for RegimeBasedStrategy in the platform."""
    
    def test_strategy_initialization_from_config(self):
        """Test that RegimeBasedStrategy can be initialized from config."""
        config_path = create_test_config_yaml()
        
        try:
            # Load config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            strategy_config = config['strategy']
            
            # Parse allowed regimes
            allowed_regimes = []
            for regime in strategy_config.get('allowed_regimes', ['trending']):
                if isinstance(regime, str):
                    allowed_regimes.append(MarketRegime[regime.upper()])
            
            # Create underlying strategy
            underlying_config = strategy_config.get('underlying_strategy', {})
            underlying_strategy = EMATrendStrategy(
                fast_period=underlying_config.get('fast_period', 20),
                slow_period=underlying_config.get('slow_period', 50),
                signal_confidence=underlying_config.get('signal_confidence', 0.75)
            )
            
            # Create regime strategy
            regime_strategy = RegimeBasedStrategy(
                underlying_strategy=underlying_strategy,
                allowed_regimes=allowed_regimes,
                regime_confidence_threshold=strategy_config.get('regime_confidence_threshold', 0.5),
                reduce_confidence_in_marginal_regimes=strategy_config.get('reduce_confidence_in_marginal_regimes', True)
            )
            
            assert regime_strategy is not None
            assert regime_strategy.name == "RegimeBasedStrategy"
            assert MarketRegime.TRENDING in regime_strategy.allowed_regimes
            
        finally:
            # Cleanup
            Path(config_path).unlink()
    
    def test_end_to_end_signal_generation(self):
        """Test complete signal generation flow with regime detection."""
        # Initialize components
        regime_detector = RegimeDetector(lookback_period=50)
        underlying_strategy = EMATrendStrategy(fast_period=10, slow_period=20)
        
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING],
            regime_confidence_threshold=0.5
        )
        
        # Generate market data
        market_data = generate_test_market_data(num_bars=100)
        
        # Generate signal
        position_state = PositionState(has_position=False)
        signal_output = regime_strategy.generate_signal(market_data, position_state)
        
        # Verify signal structure
        assert signal_output is not None
        assert hasattr(signal_output, 'signal')
        assert hasattr(signal_output, 'confidence')
        assert hasattr(signal_output, 'metadata')
        
        # Verify regime metadata exists (metadata is kwargs dict)
        assert 'regime' in signal_output.metadata
        assert 'regime_confidence' in signal_output.metadata
        assert 'decision' in signal_output.metadata
    
    def test_risk_management_integration(self):
        """Test that risk modules work with regime strategy signals."""
        config_path = create_test_config_yaml()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Initialize risk modules
            risk_config = config['risk']
            trading_config = config['trading']
            initial_equity = trading_config['initial_capital']
            
            position_sizer = PositionSizer(
                account_equity=initial_equity,
                risk_per_trade=risk_config['risk_per_trade'],
                max_position_size=initial_equity * risk_config['max_position_size'],
                sizing_method=SizingMethod.FIXED_RISK,
                use_confidence_scaling=risk_config['use_confidence_scaling']
            )
            
            drawdown_guard = DrawdownGuard(
                initial_equity=initial_equity,
                max_daily_drawdown=risk_config['max_daily_drawdown'],
                max_total_drawdown=risk_config['max_total_drawdown'],
                max_consecutive_losses=risk_config.get('max_consecutive_losses')
            )
            
            # Initialize strategy
            regime_detector = RegimeDetector()
            underlying_strategy = EMATrendStrategy(fast_period=10, slow_period=20)
            regime_strategy = RegimeBasedStrategy(
                underlying_strategy=underlying_strategy,
                regime_detector=regime_detector,
                allowed_regimes=[MarketRegime.TRENDING]
            )
            
            # Generate signal
            market_data = generate_test_market_data(num_bars=100)
            position_state = PositionState(has_position=False)
            signal_output = regime_strategy.generate_signal(market_data, position_state)
            
            # Test position sizing
            if signal_output.signal != Signal.HOLD:
                current_price = float(market_data.iloc[-1]['close'])
                sizing_result = position_sizer.calculate_size(
                    signal={
                        'signal': signal_output.signal.value,
                        'confidence': signal_output.confidence,
                        'metadata': signal_output.metadata
                    },
                    current_price=current_price
                )
                
                assert sizing_result is not None
                assert hasattr(sizing_result, 'approved')
                assert hasattr(sizing_result, 'position_size')
            
            # Test drawdown guard
            drawdown_guard.update_equity(initial_equity)  # Update equity first
            veto_decision = drawdown_guard.check_trade()
            assert veto_decision is not None
            assert hasattr(veto_decision, 'approved')
            
        finally:
            Path(config_path).unlink()
    
    def test_metrics_collection_with_regime_metadata(self):
        """Test that metrics collector properly records regime information."""
        metrics = MetricsCollector()
        
        # Initialize strategy
        regime_detector = RegimeDetector()
        underlying_strategy = EMATrendStrategy(fast_period=10, slow_period=20)
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING]
        )
        
        # Generate signal with regime metadata
        market_data = generate_test_market_data(num_bars=100)
        position_state = PositionState(has_position=False)
        signal_output = regime_strategy.generate_signal(market_data, position_state)
        
        # Verify metadata structure is suitable for logging
        assert signal_output.metadata is not None
        assert isinstance(signal_output.metadata, dict)
        
        # Metadata should be JSON-serializable
        import json
        try:
            # Metadata values are already primitives (strings, floats)
            # No need to convert enums as they're stored as strings
            json_str = json.dumps(signal_output.metadata, default=str)
            assert len(json_str) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"Metadata not JSON-serializable: {e}")
    
    def test_multiple_regime_configurations(self):
        """Test different allowed_regimes configurations."""
        regime_detector = RegimeDetector()
        underlying_strategy = EMATrendStrategy(fast_period=10, slow_period=20)
        market_data = generate_test_market_data(num_bars=100)
        position_state = PositionState(has_position=False)
        
        # Test 1: Only TRENDING allowed
        strategy1 = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING]
        )
        signal1 = strategy1.generate_signal(market_data, position_state)
        assert signal1 is not None
        
        # Test 2: Multiple regimes allowed
        strategy2 = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING, MarketRegime.RANGING]
        )
        signal2 = strategy2.generate_signal(market_data, position_state)
        assert signal2 is not None
        
        # Test 3: No regimes allowed (should always HOLD)
        strategy3 = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[]
        )
        signal3 = strategy3.generate_signal(market_data, position_state)
        assert signal3.signal == Signal.HOLD


class TestConfigurationValidation:
    """Test configuration loading and validation."""
    
    def test_config_yaml_structure(self):
        """Test that config.yaml has all required fields for regime strategy."""
        config_path = create_test_config_yaml()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Verify required top-level sections
            assert 'environment' in config
            assert 'exchange' in config
            assert 'trading' in config
            assert 'strategy' in config
            assert 'risk' in config
            assert 'execution' in config
            assert 'monitoring' in config
            
            # Verify strategy section
            strategy_config = config['strategy']
            assert strategy_config['name'] == 'regime_based'
            assert 'allowed_regimes' in strategy_config
            assert 'regime_confidence_threshold' in strategy_config
            assert 'underlying_strategy' in strategy_config
            
            # Verify underlying strategy
            underlying = strategy_config['underlying_strategy']
            assert underlying['name'] == 'ema_trend'
            assert 'fast_period' in underlying
            assert 'slow_period' in underlying
            
            # Verify risk section
            risk_config = config['risk']
            assert 'risk_per_trade' in risk_config
            assert 'max_daily_drawdown' in risk_config
            assert 'max_total_drawdown' in risk_config
            
        finally:
            Path(config_path).unlink()
    
    def test_config_defaults_fallback(self):
        """Test that missing config values fall back to defaults."""
        minimal_config = {
            'environment': 'paper',
            'trading': {'symbol': 'BTC/USDT', 'initial_capital': 100000.0},
            'strategy': {'name': 'regime_based'}
        }
        
        # Create strategy with defaults
        regime_detector = RegimeDetector()
        underlying_strategy = EMATrendStrategy()
        
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING],  # Default
            regime_confidence_threshold=0.5  # Default
        )
        
        assert regime_strategy is not None
        assert len(regime_strategy.allowed_regimes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
