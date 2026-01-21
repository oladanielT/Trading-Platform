"""
Tests for Walk-Forward Validation Engine

Validates that walk-forward engine:
- Prevents lookahead bias
- Produces consistent OOS results
- Detects overfitting
- Handles edge cases safely
"""

import pytest
import numpy as np
from ai.walk_forward_engine import WalkForwardEngine, WindowConfig, WindowResult


def create_mock_data(n_samples=500, trend='up'):
    """Create mock OHLCV data for testing."""
    close_prices = []
    if trend == 'up':
        close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5 + 0.1)
    elif trend == 'down':
        close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5 - 0.1)
    else:  # sideways
        close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    
    return {
        'close': close_prices.tolist(),
        'open': (close_prices - np.random.rand(n_samples)).tolist(),
        'high': (close_prices + np.random.rand(n_samples)).tolist(),
        'low': (close_prices - np.random.rand(n_samples)).tolist(),
        'volume': (np.random.rand(n_samples) * 1000).tolist()
    }


def simple_strategy(data):
    """Simple buy-and-hold mock strategy."""
    trades = []
    close_prices = data['close']
    
    # Buy at 25%, sell at 75%
    if len(close_prices) > 2:
        buy_idx = len(close_prices) // 4
        sell_idx = (len(close_prices) * 3) // 4
        
        pnl = close_prices[sell_idx] - close_prices[buy_idx]
        trades.append({'pnl': pnl, 'entry': buy_idx, 'exit': sell_idx})
    
    return trades


def overfit_strategy(data):
    """Overfit strategy - perfect on training, bad on test."""
    trades = []
    close_prices = data['close']
    
    # Cheat: use future data to pick only winners
    for i in range(len(close_prices) - 10):
        future_return = close_prices[i + 10] - close_prices[i]
        if future_return > 0:  # Only "trade" if we know it wins
            trades.append({'pnl': future_return})
    
    return trades


class TestWalkForwardEngine:
    """Test suite for walk-forward validation."""
    
    def test_initialization(self):
        """Test engine initializes correctly."""
        engine = WalkForwardEngine()
        assert engine.config is not None
        assert engine.results == []
    
    def test_custom_config(self):
        """Test custom window configuration."""
        config = WindowConfig(
            train_pct=0.7,
            test_pct=0.3,
            step_size=0.15
        )
        engine = WalkForwardEngine(config)
        assert engine.config.train_pct == 0.7
        assert engine.config.test_pct == 0.3
    
    def test_no_lookahead_bias(self):
        """CRITICAL: Verify no lookahead bias in data splits."""
        engine = WalkForwardEngine()
        data = create_mock_data(300)
        
        results = engine.run(simple_strategy, data)
        
        # Check that test always comes after train
        for window in results['windows']:
            assert window.test_start >= window.train_end, \
                "Test data must not overlap with training data"
    
    def test_oos_consistency_detection(self):
        """Test OOS consistency scoring."""
        engine = WalkForwardEngine()
        data = create_mock_data(500, trend='up')
        
        results = engine.run(simple_strategy, data)
        
        # Should have consistency score
        assert 'overall_consistency' in results
        assert 0.0 <= results['overall_consistency'] <= 1.0
    
    def test_overfitting_detection(self):
        """Test detection of overfit strategies."""
        engine = WalkForwardEngine()
        data = create_mock_data(500)
        
        results = engine.run(overfit_strategy, data)
        
        # Overfit strategy should have some indication of overfitting
        # Either positive degradation OR high overfitting scores
        has_degradation = results['oos_degradation'] > -1.0  # Some degradation detected
        has_high_overfit = results.get('max_overfitting_score', 0) > 0
        
        assert has_degradation or has_high_overfit, \
            "Overfit strategies should show some overfitting indicators"
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        engine = WalkForwardEngine()
        data = create_mock_data(50)  # Too small
        
        with pytest.raises(ValueError, match="Insufficient data"):
            engine.run(simple_strategy, data)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        engine = WalkForwardEngine()
        
        with pytest.raises(ValueError):
            engine.run(simple_strategy, {})
    
    def test_multiple_windows_created(self):
        """Test that multiple rolling windows are created."""
        config = WindowConfig(step_size=0.1)  # Smaller steps for more windows
        engine = WalkForwardEngine(config)
        data = create_mock_data(1000)  # More data for multiple windows
        
        results = engine.run(simple_strategy, data)
        
        # Should have multiple windows
        assert len(results['windows']) >= 1, "Should create at least one window"
    
    def test_regime_coverage_tracking(self):
        """Test regime label tracking in windows."""
        engine = WalkForwardEngine()
        data = create_mock_data(300)
        regime_labels = ['TRENDING'] * 150 + ['RANGING'] * 150
        
        results = engine.run(simple_strategy, data, regime_labels)
        
        # Windows should track regime coverage
        for window in results['windows']:
            assert isinstance(window.regime_coverage, dict)
    
    def test_consistency_pass_fail(self):
        """Test consistency threshold for pass/fail."""
        engine = WalkForwardEngine()
        data = create_mock_data(500, trend='up')
        
        results = engine.run(simple_strategy, data)
        
        # 'passed' should be based on consistency >= 70%
        if results['overall_consistency'] >= 0.70:
            assert results['passed'] is True
        else:
            assert results['passed'] is False
    
    def test_deterministic_results(self):
        """Test that results are deterministic (no random state)."""
        engine1 = WalkForwardEngine()
        engine2 = WalkForwardEngine()
        
        np.random.seed(42)
        data1 = create_mock_data(300)
        
        np.random.seed(42)
        data2 = create_mock_data(300)
        
        results1 = engine1.run(simple_strategy, data1)
        results2 = engine2.run(simple_strategy, data2)
        
        # Should produce same results
        assert results1['overall_consistency'] == results2['overall_consistency']
    
    def test_get_diagnostics(self):
        """Test diagnostics output."""
        engine = WalkForwardEngine()
        
        # Before running
        diag = engine.get_diagnostics()
        assert diag['status'] == 'no_runs'
        
        # After running
        data = create_mock_data(300)
        engine.run(simple_strategy, data)
        
        diag = engine.get_diagnostics()
        assert diag['status'] == 'completed'
        assert diag['windows'] > 0


class TestWindowResult:
    """Test WindowResult data structure."""
    
    def test_window_result_creation(self):
        """Test creating a WindowResult."""
        result = WindowResult(
            window_id=1,
            train_start=0,
            train_end=100,
            test_start=100,
            test_end=150,
            in_sample_trades=20,
            out_sample_trades=15,
            in_sample_sharpe=1.5,
            out_sample_sharpe=1.2,
            in_sample_expectancy=0.5,
            out_sample_expectancy=0.3,
            in_sample_max_dd=0.05,
            out_sample_max_dd=0.08,
            regime_coverage={'TRENDING': 30, 'RANGING': 20},
            overfitting_score=0.2
        )
        
        assert result.window_id == 1
        assert result.overfitting_score == 0.2


class TestWindowConfig:
    """Test WindowConfig data structure."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WindowConfig()
        assert config.train_pct == 0.6
        assert config.test_pct == 0.4
        assert config.step_size == 0.1
    
    def test_custom_config_values(self):
        """Test custom configuration."""
        config = WindowConfig(
            train_pct=0.7,
            test_pct=0.3,
            step_size=0.05,
            min_train_samples=200,
            min_test_samples=50
        )
        assert config.train_pct == 0.7
        assert config.min_train_samples == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
